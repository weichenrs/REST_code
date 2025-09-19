# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict

@MODELS.register_module()
class FCNHead(BaseDecodeHead):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 num_convs=2,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 pretrained=None,
                 freeze=False,
                 **kwargs):
        assert num_convs >= 0 and dilation > 0 and isinstance(dilation, int)
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        super().__init__(**kwargs)
        if num_convs == 0:
            assert self.in_channels == self.channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        self.pretrained = pretrained
        self.freeze = freeze
        
    def _freeze(self):
        if self.freeze == True:
            all_modules = [self.convs, self.conv_seg]
            for m in all_modules:
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
                    
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
                    continue
                    # new_k = new_k.replace('backbone.', '')
                if new_k.startswith('decode_head'):
                    continue
                    # new_k = new_k.replace('decode_head.', '')
                if new_k.startswith('auxiliary_head'):
                    new_k = new_k.replace('auxiliary_head.', '')
                
                new_ckpt[new_k] = v
            state_dict = new_ckpt
            # import torch.distributed as dist
            # if dist.get_rank() == 0:
            #     import pdb;pdb.set_trace()
            # dist.barrier()
            load_state_dict(self, state_dict, strict=False, logger=None)
        
        else:
            pass
        self._freeze()
        
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
        feats = self.convs(x)
        if self.concat_input:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        return feats

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
