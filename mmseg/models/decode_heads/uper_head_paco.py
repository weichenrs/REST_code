# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
from .psp_head import PPM
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from .paco import PaCoLoss

def BN_convert_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        BN_convert_float(child)
    return module

@MODELS.register_module()
class UPerHead_paco(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), pretrained=None, freeze=False, alpha=0.01, 
                    temperature=0.07, K=8192, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.mlp = nn.Sequential(
             ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
             ConvModule(self.channels, self.channels, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,inplace=False),
             nn.Conv2d(self.channels, 128, 1))
        
        self.alpha = alpha
        self.temperature = temperature
        self.K = K
        self.paco_loss = PaCoLoss(alpha=self.alpha, num_classes=self.num_classes, temperature=self.temperature)

        BN_convert_float(self)
        self.pretrained = pretrained
        self.freeze = freeze
        
    def _freeze(self):
        if self.freeze == True:
            all_modules = [self.psp_modules, self.bottleneck, self.lateral_convs, 
                           self.fpn_convs, self.fpn_bottleneck, self.conv_seg]
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
                    # continue
                    new_k = new_k.replace('decode_head.', '')
                if new_k.startswith('auxiliary_head'):
                    continue
                    # new_k = new_k.replace('auxiliary_head.', '')
                
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
        
    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)
        
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        seg_logit = self.cls_seg(output)
        embed = self.mlp(output)
        if self.training:
            return seg_logit, embed 
        else:
            return seg_logit

    def forward(self, inputs):
        """Forward function."""
        if not self.training:
            output = self._forward_feature(inputs)
            return output
        
        # else:
        #     seg_logit, embed = self.forward(inputs)
        #     losses = self.losses(seg_logit, gt_semantic_seg)

        #     # reduced_label
        #     n, c, h, w = embed.shape
        #     reduced_seg_label = F.interpolate(gt_semantic_seg.to(torch.float32), size=(h, w), mode='nearest')
        #     reduced_seg_label = reduced_seg_label.long()

        #     # paco loss
        #     loss_paco = []
        #     for i in range(n):
        #         embed_s = embed[i].flatten(1).transpose(0,1).contiguous().view(-1, c)
        #         embed_s = F.normalize(embed_s, dim=1)
        #         seg_logit_t = seg_logit[i].flatten(1).transpose(0,1).contiguous().view(-1, self.num_classes)
        #         seg_label = torch.where(reduced_seg_label[i]>=self.num_classes, self.num_classes, reduced_seg_label[i])
        #         seg_label = seg_label.view(-1,)
        #         t = embed_s.size(0) if self.K == -1 else self.K 
        #         sample_index = torch.randperm(embed_s.size(0))[:t]
        #         loss_paco.append(self.paco_loss(embed_s[sample_index], seg_label[sample_index], seg_logit_t[sample_index]))
        #     losses['paco_loss'] = sum(loss_paco) / n
        #     return losses

