import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
# from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy
from .atm_head import *


def convert_true_idx(orig, new):
    assert (~orig).sum() == len(
        new), "mask_idx and new pos mismatch!!! orig:{}, new:{} ".format((~orig).sum(), len(new))
    orig_new = torch.zeros_like(orig)
    orig_new[~orig] = new
    return orig_new


@HEADS.register_module()
class new_PruneHead(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            layers_per_decoder=2,
            num_heads=4,
            thresh=1.0,
            **kwargs,
    ):
        super(new_PruneHead, self).__init__(
            in_channels=in_channels, **kwargs)
        self.thresh = thresh
        self.image_size = img_size
        nhead = num_heads
        dim = self.channels
        proj = nn.Linear(self.in_channels, dim)
        trunc_normal_(proj.weight, std=.02)
        norm = nn.LayerNorm(dim)
        decoder_layer = TPN_DecoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
        decoder = TPN_Decoder(decoder_layer, layers_per_decoder)

        self.input_proj = proj
        self.proj_norm = norm
        self.decoder = decoder
        self.q = nn.Embedding(self.num_classes, dim)

        self.class_embed = nn.Linear(dim, 1 + 1)
        delattr(self, 'conv_seg')

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def per_img_forward(self, q, x):
        x = self.proj_norm(self.input_proj(x))#x.shape=[4,1024,768]
        q, attn = self.decoder(q, x.transpose(0, 1))#q.shape=[150,4,384]
        cls = self.class_embed(q.transpose(0, 1))#cls.shape=[4,150,2]
        pred = cls.softmax(dim=-1)[..., :-1] * attn.sigmoid()#pred.shape=[4,150,1024]
        return attn, cls,pred

    def forward(self, inputs, inference=False, mask_idx=None, canvas=None):
        mask_idx = mask_idx.cuda()
        if inference:
            #x = self._transform_inputs(inputs)
            x=inputs
            canvas_copy = canvas.clone()
            if x.dim() == 4:
                x = self.d4_to_d3(x)
            B, hw, ch = x.shape
            if mask_idx.sum():
                cls = []
                pred = []
                qs = self.q.weight.repeat(B, 1, 1).transpose(0, 1)
                for b in range(B):
                    q = qs[:, b].unsqueeze(1)
                    x_ = x[b][~mask_idx[b]]
                    attn_, cls_,pred_= self.per_img_forward(q, x_.unsqueeze(0))
                    cls.append(cls_)
                    pred.append(pred_[0])
                    canvas_copy[b][:, ~mask_idx[b]] = attn_[0]

                cls = torch.cat(cls, dim=0)
                self.results = {"attn": canvas_copy}
                self.results.update({"pred_logits": cls})
            else:
                q = self.q.weight.repeat(B, 1, 1).transpose(0, 1) # q.shape [cls, b, ch]
                attn, cls,pred= self.per_img_forward(q, x) # x.shape [b, hw, ch]
                canvas_copy = attn
                self.results = {"attn": canvas_copy}
                self.results.update({"pred_logits": cls})

            if self.thresh == 1:
                return mask_idx, canvas_copy
            else:
                with torch.no_grad():
                    for b, pred_ in enumerate(pred):#enumerate给出索引
                        val, ind = pred_.max(dim=0)#ind是val的索引
                        pos = val > self.thresh
                        # keep top5 or smaller per class每一类保留5个
                        for per_cls in ind[pos].unique():
                            per_cls_topk = (
                                ind[pos] == per_cls).sum().clamp_max(5)#看每一类有多少过线的，并限制在5以下，如果只有一个
                            topk_v, topk_idx = val[pos][ind[pos] == per_cls].topk(#值>thresh中的某一类
                                per_cls_topk)#topk按从大到小排序
                            pos_idx = pos.nonzero()#获取非零元素的索引值
                            ind_idx = pos_idx[ind[pos] == per_cls]#每一类的索引值
                            topk_idx = ind_idx[topk_idx]#该类要提出的索引值
                            pos[topk_idx] = False#要剔除的索引值为False

                            # per_cls_idx = pos_idx[ind[pos] == per_cls]
                            # pos[per_cls_idx] = pos[per_cls_idx].index_fill_(
                            #     0, topk_idx, False)
                        pos_true = convert_true_idx(mask_idx[b], pos)#mask_idx的False字段，重新按照pos分配，原本为false的过了线就位True,裁剪的为False，原本为True的现在是False
                        mask_idx[b] = torch.bitwise_or(
                            mask_idx[b], pos_true)#原本是True的现在是True,原本为False的，裁剪的为False，过线的为True
                return mask_idx, canvas_copy

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)
