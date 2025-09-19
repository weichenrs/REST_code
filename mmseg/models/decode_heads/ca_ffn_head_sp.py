# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

from torch import Tensor
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional

from timm.models.layers import trunc_normal_

from mmseg.models.losses import accuracy
import torch.distributed as dist
import numpy as np

global _SEQUENCE_PARALLEL_GROUP
_SEQUENCE_PARALLEL_GROUP = dist.new_group(range(0, dist.get_world_size()))

import deepspeed.comm as deepdist
from typing import Any, Tuple

try:
    from mmcv.ops import point_sample
except ModuleNotFoundError:
    point_sample = None
    
from mmseg.utils import ConfigType, SampleList
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict

class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: deepdist.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = deepdist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        deepdist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


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
        
        del self.self_attn
        self.self_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1
        )
        # del self.dropout1, self.norm1
        
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            tgt, memory.transpose(0, 1), memory.transpose(0, 1))
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

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

        self.scatter_idx = 1
        self.gather_idx = 2

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
        
        xq = _SeqAllToAll.apply(_SEQUENCE_PARALLEL_GROUP, xq, self.scatter_idx, self.gather_idx)
        xk = _SeqAllToAll.apply(_SEQUENCE_PARALLEL_GROUP, xk, self.scatter_idx, self.gather_idx)
        xv = _SeqAllToAll.apply(_SEQUENCE_PARALLEL_GROUP, xv, self.scatter_idx, self.gather_idx)

        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        # attn_mask = torch.zeros(Nq).bool().cuda()
        # attn_mask[0:2,] = 1
        # attn = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask)
        
        attn = F.scaled_dot_product_attention(xq, xk, xv)
        
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        attn = _SeqAllToAll.apply(_SEQUENCE_PARALLEL_GROUP, attn, self.gather_idx, self.scatter_idx)

        attn = attn.transpose(1, 2).reshape(B, Nq, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        return attn

@MODELS.register_module()
class CA_FFN_Head_sp(BaseDecodeHead):
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
            sample=False,
            pretrained=None,
            **kwargs,
    ):
        super(CA_FFN_Head_sp, self).__init__(
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
        
        self.pretrained = pretrained
        
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
        self.dropout = None
        self.sample = sample
        # self.sample = True
        
        trunc_normal_(self.ffn.weight, std=.02)
        trunc_normal_(self.conv_seg.weight, std=.02)

    def init_weights(self):
        
        if self.pretrained is not None :
            checkpoint = CheckpointLoader.load_checkpoint(
                self.pretrained, logger=None, map_location='cpu')
            
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            from collections import OrderedDict
            new_ckpt = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module'):
                    new_k = k.replace('module.', '')
                else:
                    new_k = k
    
                if new_k.startswith('backbone'):
                    new_k = new_k.replace('backbone.', '')
                    continue
                if new_k.startswith('decode_head'):
                    new_k = new_k.replace('decode_head.', '')
                
                new_ckpt[new_k] = v
            state_dict = new_ckpt

            load_state_dict(self, state_dict, strict=False, logger=None)
        
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.0)

    def forward(self, x):
        laterals = []
        # x.reverse()
        bs = x[0].size()[0]
        hw = x[0].size()[-2:]
        
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        sample = self.training and self.sample > 1
            
        for idx, (x_, proj_, norm_) in enumerate(zip(x, self.input_proj, self.proj_norm)):
            
            # if dist.get_rank() == 0:
            #     import pdb;pdb.set_trace()
            # dist.barrier()

            if idx == 0:
                if sample:
                    sampled_points = torch.rand(bs, int(hw[0]*hw[1]/self.sample), 2, device=x_.device)
                    x_ = point_sample(x_, sampled_points)
                    x_ = x_.transpose(-1, -2)
                else:
                    x_ = self.d4_to_d3(x_)
                    
                lateral = norm_(proj_(x_))
                _q = lateral
            else:
                lateral = norm_(proj_(self.d4_to_d3(x_)))       
                laterals.append(lateral) 
                
        for idx, (lateral, decoder_) in enumerate(zip(laterals, self.decoder)):

            _q = decoder_(_q, lateral.transpose(0, 1))

        if sample:
            out = self.cls_seg(self.ffn(_q))
            out = out.transpose(-1,-2).contiguous()
            return out, sampled_points

        else:
            # import torch.distributed as dist
            # if dist.get_rank() == 0:
            #     import pdb;pdb.set_trace()
            # dist.barrier()
            
            # out = F.interpolate(self.d3_to_d4(_q, hw),
            #         size = self.image_size,
            #         mode='bilinear', align_corners=False)
            # out = out.permute(0,2,3,1).contiguous()
            # out = self.cls_seg(self.ffn(out))
            # out = out.permute(0,3,1,2).contiguous()
            
            # out = self.d3_to_d4(_q, hw)
            # out = out.permute(0,2,3,1).contiguous()
            # out = F.interpolate(out,
            #         size = self.image_size,
            #         mode='bilinear', align_corners=False)
            # out = self.cls_seg(self.ffn(out))
            # out = out.permute(0,3,1,2).contiguous()
            
            out = self.d3_to_d4(_q, hw)
            out = F.interpolate(out, size = self.image_size, mode='bicubic')
            out = out.permute(0,2,3,1).contiguous()
            out = self.cls_seg(self.ffn(out))
            out = out.permute(0,3,1,2).contiguous()
            return out
        
    def d3_to_d4(self, t, hw):
        n, _, c = t.size()
        return t.transpose(1, 2).reshape(n, c, hw[0], hw[1])

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.sample:
            seg_logits, sampled_points = self.forward(inputs)
            losses = self.loss_by_feat_sample(seg_logits, batch_data_samples, sampled_points)
        else:
            seg_logits = self.forward(inputs)
            losses = self.loss_by_feat(seg_logits, batch_data_samples)

        return losses
    
    def loss_by_feat_sample(self, seg_logits: Tensor,
                     batch_data_samples: SampleList,
                     sampled_points) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        
        seg_label = point_sample(seg_label.float(), sampled_points, mode='nearest')
        seg_label = seg_label.squeeze(1).long()
                        
        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=None,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=None,
                    ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss