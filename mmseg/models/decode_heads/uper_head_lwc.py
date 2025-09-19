# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
import torch.nn.functional as F
from .lib_upernet.nn import SynchronizedBatchNorm2d
# def BN_convert_float(module):
#     if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
#         module.float()
#     for child in module.children():
#         BN_convert_float(child)
#     return module
def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        SynchronizedBatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


@MODELS.register_module()
class UPerHead_lwc(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), pretrained=None, freeze=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            # we use the feature map size instead of input image size, so down_scale = 1.0
            # self.ppm_pooling.append(PrRoIPool2D(scale, scale, 1.))  # TODO
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))

            self.ppm_conv.append(nn.Sequential(
                nn.Conv2d(self.in_channels[-1], self.channels, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in self.in_channels[:-1]:  # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, self.channels, kernel_size=1, bias=False),
                SynchronizedBatchNorm2d(self.channels),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(self.in_channels) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(self.channels, self.channels, 1),
            ))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_fusion = conv3x3_bn_relu(len(self.in_channels) * self.channels, self.channels, 1)

        # input: Fusion out, input_dim: self.channels
        self.object_head = nn.Sequential(
            conv3x3_bn_relu(self.channels, self.channels, 1),
            nn.Conv2d(self.channels, self.out_channels, kernel_size=1, bias=True)
        )
        # del self.conv_seg
        # BN_convert_float(self)
        self.pretrained = pretrained
        self._init_weights()
    #     self.freeze = freeze

    # def _freeze(self):
    #     if self.freeze == True:
    #         all_modules = [self.psp_modules, self.bottleneck, self.lateral_convs, 
    #                        self.fpn_convs, self.fpn_bottleneck, self.conv_seg]
    #         for m in all_modules:
    #             m.eval()
    #             for param in m.parameters():
    #                 param.requires_grad = False
    
    def _init_weights(self):
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()
        
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
                    # continue
                    new_k = new_k.replace('decode_head.', '')
                if new_k.startswith('auxiliary_head'):
                    continue
                    # new_k = new_k.replace('auxiliary_head.', '')
                
                new_ckpt[new_k] = v
            state_dict = new_ckpt

            load_state_dict(self, state_dict, strict=False, logger=None)
    
        else:
            pass
        
        # self._freeze()
        
    # def psp_forward(self, inputs):
    #     """Forward function of PSP module."""
    #     x = inputs[-1]
    #     psp_outs = [x]
    #     psp_outs.extend(self.psp_modules(x))
    #     psp_outs = torch.cat(psp_outs, dim=1)
    #     output = self.bottleneck(psp_outs)
    #     return output

    # def _forward_feature(self, inputs):
    #     """Forward function for feature maps before classifying each pixel with
    #     ``self.cls_seg`` fc.

    #     Args:
    #         inputs (list[Tensor]): List of multi-level img features.

    #     Returns:
    #         feats (Tensor): A tensor of shape (batch_size, self.channels,
    #             H, W) which is feature map for last layer of decoder head.
    #     """
    #     inputs = self._transform_inputs(inputs)
        
    #     # build laterals
    #     laterals = [
    #         lateral_conv(inputs[i])
    #         for i, lateral_conv in enumerate(self.lateral_convs)
    #     ]

    #     laterals.append(self.psp_forward(inputs))

    #     # build top-down path
    #     used_backbone_levels = len(laterals)
    #     for i in range(used_backbone_levels - 1, 0, -1):
    #         prev_shape = laterals[i - 1].shape[2:]
    #         laterals[i - 1] = laterals[i - 1] + resize(
    #             laterals[i],
    #             size=prev_shape,
    #             mode='bilinear',
    #             align_corners=self.align_corners)

    #     # build outputs
    #     fpn_outs = [
    #         self.fpn_convs[i](laterals[i])
    #         for i in range(used_backbone_levels - 1)
    #     ]
    #     # append psp feature
    #     fpn_outs.append(laterals[-1])

    #     for i in range(used_backbone_levels - 1, 0, -1):
    #         fpn_outs[i] = resize(
    #             fpn_outs[i],
    #             size=fpn_outs[0].shape[2:],
    #             mode='bilinear',
    #             align_corners=self.align_corners)
    #     fpn_outs = torch.cat(fpn_outs, dim=1)
    #     feats = self.fpn_bottleneck(fpn_outs)

    #     # if torch.isnan(feats).any():
    #     # import torch.distributed as dist
    #     # if dist.get_rank() == 0:
    #     #     import pdb;pdb.set_trace()
    #     # dist.barrier()

    #     return feats

    # def forward(self, inputs):
    #     """Forward function."""
    #     output = self._forward_feature(inputs)
    #     output = self.cls_seg(output)
    #     return output

    def forward(self, x):
        seg_size = x[-1].shape[-2:]
    
        conv5 = x[-1]
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):

            ppm_out.append(pool_conv(F.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))

        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(x) - 1)):
            conv_x = x[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = F.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)
        out = self.object_head(x)
        return F.interpolate(out, size=seg_size, mode='bilinear', align_corners=False)
