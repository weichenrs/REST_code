# A script to visualize the ERF.
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs (https://arxiv.org/abs/2203.06717)
# Github source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------'

from mmseg.models import SwinTransformer

class SwinForERF(SwinTransformer):

    def __init__(self):
        super().__init__(                 
                pretrain_img_size=224,
                in_channels=3,
                embed_dims=192,
                patch_size=4,
                window_size=7,
                mlp_ratio=4,
                depths=(2, 2, 18, 2),
                num_heads=(6, 12, 24, 48),
                strides=(4, 2, 2, 2),
                out_indices=(0, 1, 2, 3),
                qkv_bias=True,
                qk_scale=None,
                patch_norm=True,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.1,
                use_abs_pos_embed=False,
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='LN'),
                with_cp=False,
                pretrained=None,
                frozen_stages=-1,
                init_cfg=None)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)
        
        return outs[-1]
        # return out