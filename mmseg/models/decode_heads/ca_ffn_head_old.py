# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
# from mmcv.runner import auto_fp16, force_fp32

from timm.models.layers import trunc_normal_

from mmseg.models.losses import accuracy
import torch.distributed as dist
import numpy as np

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        # attns = []
        for mod in self.layers:
            # output, attn = mod(output, memory, tgt_mask=tgt_mask,
            #              memory_mask=memory_mask,
            #              tgt_key_padding_mask=tgt_key_padding_mask,
            #              memory_key_padding_mask=memory_key_padding_mask)
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            # attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        # return output, attn
        return output

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # tgt2, attn2 = self.multihead_attn(
        #     tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        
        tgt2 = self.multihead_attn(
            tgt, memory.transpose(0, 1), memory.transpose(0, 1))
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        # return tgt, attn2
        return tgt
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size()
        Nk = xk.size()[1]
        Nv = xv.size()[1]
              
        xq = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        xk = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        xv = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(xq, xk, xv)

        attn = attn.transpose(1, 2).reshape(B, Nq, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        return attn

@MODELS.register_module()
class CA_FFN_Head(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=3,
            use_proj=True,
            CE_loss=False,
            crop_train=False,
            shrink_ratio=None,
            **kwargs,
    ):
        super(CA_FFN_Head, self).__init__(
            in_channels=in_channels, **kwargs)

        # self.image_size = img_size
        sp_times = int(np.log2(dist.get_world_size()))
        sp_temp = sp_times // 2
        sp_h = 2 ** sp_temp
        sp_w = 2 ** (sp_times - sp_temp)
        self.image_size = (img_size//sp_h, img_size//sp_w)
        
        self.use_stages = use_stages
        self.crop_train = crop_train
        nhead = num_heads
        dim = embed_dims
        input_proj = []
        proj_norm = []
        atm_decoders = []
                
        for i in range(self.use_stages):
            # FC layer to change ch
            if use_proj:
                proj = nn.Linear(self.in_channels, dim)
                trunc_normal_(proj.weight, std=.02)
            else:
                proj = nn.Identity()
            self.add_module("input_proj_{}".format(i + 1), proj)
            input_proj.append(proj)
            # norm layer
            if use_proj:
                norm = nn.LayerNorm(dim)
            else:
                norm = nn.Identity()
            self.add_module("proj_norm_{}".format(i + 1), norm)
            proj_norm.append(norm)
            # decoder layer
            if i == 0:
                continue
            else:
                decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
                decoder = TPN_Decoder(decoder_layer, num_layers)
                self.add_module("decoder_{}".format(i + 1), decoder)
                atm_decoders.append(decoder)

        self.input_proj = input_proj
        self.proj_norm = proj_norm
        self.decoder = atm_decoders

        self.CE_loss = CE_loss
        # delattr(self, 'conv_seg')
        self.ffn = nn.Linear(dim, dim)
        self.conv_seg = nn.Linear(dim, self.num_classes)
        trunc_normal_(self.ffn.weight, std=.02)
        trunc_normal_(self.conv_seg.weight, std=.02)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward(self, x):
        laterals = []
        hw = x[0].size()[-2:]
        
        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            lateral = norm_(proj_(self.d4_to_d3(x_)))        
            if idx == 0:
                _q = lateral
            else:
                laterals.append(lateral) 

        for idx, (lateral, decoder_) in enumerate(zip(laterals, self.decoder)):

            _q = decoder_(_q, lateral.transpose(0, 1))
                

        out = F.interpolate(self.d3_to_d4(_q, hw),
                            size = self.image_size,
                            mode='bilinear', align_corners=False)

        out = out.permute(0,2,3,1).contiguous()
        out = self.cls_seg(self.ffn(out))
        out = out.permute(0,3,1,2).contiguous()
        
        return out

    def d3_to_d4(self, t, hw):
        n, _, c = t.size()
        return t.transpose(1, 2).reshape(n, c, hw[0], hw[1])

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
