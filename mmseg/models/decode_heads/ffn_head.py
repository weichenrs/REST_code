# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class FFNHead(BaseDecodeHead):
    """
    """

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        self.ffn = nn.Linear(self.in_channels, self.channels)
        self.conv_seg = nn.Linear(self.channels, self.out_channels)
        
    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)

        x = x.permute(0,2,3,1).contiguous()
        feats = self.ffn(x)

        return feats

    def forward(self, inputs):
        """Forward function."""
        
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
        output = output.permute(0,3,1,2).contiguous()
        return output
