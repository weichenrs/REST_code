# Copyright (c) OpenMMLab. All rights reserved.
from .beit import BEiT
from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .cgnet import CGNet
from .ddrnet import DDRNet
from .erfnet import ERFNet
from .fast_scnn import FastSCNN
from .hrnet import HRNet
from .icnet import ICNet
from .mae import MAE
from .mit import MixVisionTransformer
from .mobilenet_v2 import MobileNetV2
from .mobilenet_v3 import MobileNetV3
from .mscan import MSCAN
from .pidnet import PIDNet
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnext import ResNeXt
from .stdc import STDCContextPathNet, STDCNet
from .swin import SwinTransformer
from .timm_backbone import TIMMBackbone
from .twins import PCPVT, SVT
from .unet import UNet
from .vit import VisionTransformer
from .vpd import VPD

# from .sp_vit import SPVisionTransformer
from .my_vit import MYVisionTransformer
# from .sp_swin import SPSwinTransformer
# from .tp_swin import TPSwinTransformer
# from .vit_ds import VisionTransformer_ds

# from .my_vit_fa import MYVisionTransformer_fa
# from .my_vit_ds import MYVisionTransformer_ds
# from .my_vit_fa_ds import MYVisionTransformer_fa_ds
# from .my_vit_fa_nods import MYVisionTransformer_fa_nods

# from .my_vit_ds_pos import MYVisionTransformer_ds_pos
# from .my_vit_ds_pos_eval import MYVisionTransformer_ds_pos_eval

# from .my_vit_ds_pos_hw import MYVisionTransformer_ds_pos_hw
# from .my_vit_fa_ds_hw import MYVisionTransformer_fa_ds_hw
# from .my_vit_ds_pos_hw_cls import MYVisionTransformer_ds_pos_hw_cls
# from .sp_vit_hw import SPVisionTransformer_hw

# from .eva2 import EVA2
# from .eva2_nocls import EVA2_nocls
# from .eva2_wocls import EVA2_wocls
# from .eva2_woape import EVA2_woape
# from .eva2_woclsape import EVA2_woclsape

# from .swin_spattn import SwinTransformer_sp
from .swin_splayer import SwinTransformer_sp_layer
from .swin_splayer_nores import SwinTransformer_sp_layer_nores
from .swin_splayer_show import SwinTransformer_sp_layer_show
# from .my_vit_ds_pos_hw_part import MYVisionTransformer_ds_pos_hw_part

from .swin_3d import SwinTransformer_3d

from .convnext_splayer import ConvNeXt_sp_layer
from .focalnet import FocalNet

from .vmamba.vmamba_sp import MM_VSSM_sp

from .swin_prune import Swin_prune
from .swin_prune_rope import Swin_prune_rope
from .swin_rope import SwinTransformer_rope
__all__ = [
    'ResNet', 'ResNetV1c', 'ResNetV1d', 'ResNeXt', 'HRNet', 'FastSCNN',
    'ResNeSt', 'MobileNetV2', 'UNet', 'CGNet', 'MobileNetV3',
    'VisionTransformer', 'SwinTransformer', 'MixVisionTransformer',
    'BiSeNetV1', 'BiSeNetV2', 'ICNet', 'TIMMBackbone', 'ERFNet', 'PCPVT',
    'SVT', 'STDCNet', 'STDCContextPathNet', 'BEiT', 'MAE', 'PIDNet', 'MSCAN',
    'DDRNet', 'VPD',
    # 'SPVisionTransformer', 
    'MYVisionTransformer',
    # 'VisionTransformer_ds', 
    # 'MYVisionTransformer_fa', 'MYVisionTransformer_ds',
    # 'MYVisionTransformer_fa_ds', 'MYVisionTransformer_fa_nods',
    # 'MYVisionTransformer_ds_pos', 'MYVisionTransformer_ds_pos_eval',
    # 'MYVisionTransformer_ds_pos_hw', 'MYVisionTransformer_fa_ds_hw',
    # 'SPVisionTransformer_hw', 
    # 'EVA2', 'EVA2_nocls', 'EVA2_wocls', 'EVA2_woape', 'EVA2_woclsape'
    # 'SwinTransformer_sp', 
    'SwinTransformer_sp_layer',
    # 'MYVisionTransformer_ds_pos_hw_cls',
    # 'MYVisionTransformer_ds_pos_hw_part',
    'SwinTransformer_3d',
    'ConvNeXt_sp_layer',
    'FocalNet',
    'MM_VSSM_sp',
    'Swin_prune',
    'Swin_prune_rope',
    'SwinTransformer_rope',
    'SwinTransformer_sp_layer_nores'
]
