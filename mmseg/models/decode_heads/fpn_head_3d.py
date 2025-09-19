# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule, ModuleList

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead

class FAM_3D_module(BaseModule):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super().__init__(init_cfg=None)
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fam = ModuleList()
        for pool_size in pool_sizes:
            self.fam.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool3d(pool_size),
                    nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        outputs = []
        for fam in self.fam:
            fam_out = fam(x)
            fam_out = F.interpolate(fam_out, size=(x.size(2), x.size(3), x.size(4)), mode='trilinear',
                                                align_corners=True)
            outputs.append(fam_out)
        return outputs

class FAM_3D(BaseModule):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__(init_cfg=None)
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fam_modules = FAM_3D_module(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv3d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            nn.BatchNorm3d(self.out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        if len(self.pool_sizes) == 0:
            out = self.final(x)
            return out

        out = self.fam_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out

class FPN_3D(BaseModule):
    def __init__(self, channels=2048, out_channels=256, pool_sizes=[1, 2, 3, 6]):
        super().__init__(init_cfg=None)
        self.FAM_head = FAM_3D(in_channels=channels, out_channels=out_channels, pool_sizes=pool_sizes)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv3d(channels // 2, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv3d(channels // 4, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv3d(channels // 8, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.fuse_all = nn.Sequential(
            nn.Conv3d(out_channels * 4, out_channels, 1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )

        self.conv_x1 = nn.Conv3d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        x1 = self.FAM_head(input_fpn[-1])

        x = F.interpolate(x1, size=(x1.size(2), x1.size(3) * 2, x1.size(4) * 2), mode='trilinear',
                                      align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = F.interpolate(x2, size=(x1.size(2), x2.size(3) * 2, x2.size(4) * 2), mode='trilinear',
                                      align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = F.interpolate(x3, size=(x1.size(2), x3.size(3) * 2, x3.size(4) * 2), mode='trilinear',
                                      align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-3:], mode='trilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-3:], mode='trilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-3:], mode='trilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x

@MODELS.register_module()
class FPNHead_3d(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, pool_sizes=[1, 2, 3, 6], **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        
        self.decoder = FPN_3D(channels=self.in_channels[-1], out_channels=self.channels, pool_sizes=pool_sizes)
        self.conv = nn.Sequential(
                nn.Conv3d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(self.channels),
                nn.ReLU(inplace=True) )
        
        self.conv_seg = nn.Conv3d(self.channels, self.num_classes, kernel_size=1)
        
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout3d(self.dropout_ratio)
        else:
            self.dropout = None
            
    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        x = self.decoder(x)
        x = self.conv(x)
        x = self.cls_seg(x)
        x = F.interpolate(x, size=(x.size(2) * 4, x.size(3) * 4, x.size(4) * 4), 
                               mode='trilinear', align_corners=True)
        x = torch.mean(x, dim=2)
        
        return x
