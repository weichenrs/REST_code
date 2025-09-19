# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .cascade_decode_head import BaseCascadeDecodeHead

import numpy as np

class _EfficientAttention(nn.Module):
    
    def __init__(self, in_channels, head_count):
        super().__init__()
        self.in_channels = in_channels
        self.head_count = head_count

    def forward(self, query_feats, key_feats):
        n, _, _  = query_feats.size()

        keys = key_feats.reshape((n, self.in_channels, -1))
        queries = query_feats.reshape(n, self.in_channels, -1)
        values= key_feats.reshape((n, self.in_channels, -1))
        head_key_channels = self.in_channels // self.head_count
        head_value_channels = self.in_channels // self.head_count
        
        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=2)
            query = F.softmax(queries[
                :,
                i * head_key_channels: (i + 1) * head_key_channels,
                :
            ], dim=1)
            value = values[
                :,
                i * head_value_channels: (i + 1) * head_value_channels,
                :
            ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, -1)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = aggregated_values + query_feats.reshape(n, self.in_channels, -1)

        return attention

class _LargeKernelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 15), padding=(0, 7), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (15, 1), padding=(7, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 31), padding=(0, 15), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (31, 1), padding=(15, 0), groups=dim)
        # self.conv3_1 = nn.Conv2d(dim, dim, (1, 63), padding=(0, 31), groups=dim)
        # self.conv3_2 = nn.Conv2d(dim, dim, (63, 1), padding=(31, 0), groups=dim)

    def forward(self, x):
        attn = self.conv0(x)
        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)
        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)
        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        out = torch.cat((x, attn, attn_0, attn_1, attn_2), 1)

        return out

class EffcientAttentionBlock(_EfficientAttention):
    def __init__(self, in_channels, head_count):
        super(EffcientAttentionBlock, self).__init__(
            in_channels=in_channels, 
            head_count=head_count, )

    def forward(self, query_feats, key_feats):
        """Forward function."""
        output = super(EffcientAttentionBlock, self).forward(query_feats, key_feats)
        return output

class LargeKernelAttentionBlock(_LargeKernelAttention):
    def __init__(self, dim):
        super(LargeKernelAttentionBlock, self).__init__(
            dim = dim)

    def forward(self, input):
        """Forward function."""
        output = super(LargeKernelAttentionBlock, self).forward(input)
        return output

class GatherHardRegion(nn.Module):
    def __init__(self, refine_port):
        super(GatherHardRegion, self).__init__()
        self.refine_port = refine_port
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, _, _ = probs.size()
        channels = feats.size(1)
        probs = probs.reshape(batch_size, num_classes, -1)
        feats = feats.reshape(batch_size, channels, -1)
        probs, _ = torch.topk(self.softmax(probs), k=2, dim=1, sorted=True)
        probs = probs[:,0] - probs[:,1]
        _, probs = probs.sort(descending=True)
        hard_region = probs[:,:np.uint64(self.refine_port*probs[0].numel())]
        hard_feat = feats.gather(2, hard_region.unsqueeze(1).repeat(1,channels,1))
        return hard_feat, feats, hard_region

@MODELS.register_module()
class EHSHead(BaseCascadeDecodeHead):
    def __init__(self, dim_model=64, eli_heads=1, refine_port=0.1, **kwargs):
        super(EHSHead, self).__init__(**kwargs)
        self.dim_model = dim_model
        self.proj_1 = nn.Conv2d(self.in_channels, self.dim_model, 1)
        self.activation = F.gelu
        self.spatial_gating_unit = LargeKernelAttentionBlock(self.dim_model)
        self.proj_2 = nn.Conv2d(self.dim_model*5 , self.channels, 1)

        self.eli_heads = eli_heads
        self.refine_port = refine_port
        self.gather_hard_region = GatherHardRegion(self.refine_port)
        self.effcient_attention = EffcientAttentionBlock(in_channels=self.in_channels, head_count=self.eli_heads)

    def forward(self, feats, prev_output):
        """Forward function."""
        feats = self._transform_inputs(feats)
        feats = self.proj_1(feats)
        feats = self.activation(feats)
        feats = self.spatial_gating_unit(feats)
        feats = self.proj_2(feats)

        batch_size, channels, h, w = feats.size()
        hard_feats, all_feats, hard_region = self.gather_hard_region(feats, prev_output)
        output_feats = self.effcient_attention(hard_feats, all_feats)
        att_feats = feats.clone().reshape(batch_size, channels, -1)
        att_feats = att_feats.scatter_(2, hard_region.unsqueeze(1).repeat(1,channels,1), output_feats)
        att_feats = att_feats.reshape(batch_size, channels, h, w)
        output = self.cls_seg(att_feats)
        return output