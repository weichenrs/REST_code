from .swin_zcg import SwinTransformer_zcg
from mmseg.registry import MODELS
import torch
import torch.nn.functional as F
# from ..utils_mm import resize
from ..utils.wrappers import resize
# from mmcv.transforms import Resize
# from .visualization import showmypic
import cv2
from ..decode_heads.new_prune import *

# def booltoint(input):
#     B,hw=input.shape
#     x = torch.zeros((B, hw)).cuda()
#     for i in range(len(input)):
#         for j in range(len(input[i])):
#             if input[i][j]==False:
#                 x[i][j]=0
#             if input[i][j]==True:
#                 x[i][j]=1
#     return x.unsqueeze(1)

# def inttobool(input):
#     input=input.squeeze(1)
#     B,hw = input.shape
#     x = torch.zeros((B, hw))!=0
#     x = x.cuda()
#     for i in range(len(input)):
#         for j in range(len(input[i])):
#             if input[i][j] == 0:
#                 x[i][j] = False
#             if input[i][j] !=0:
#                 x[i][j] = True
#     return x

@MODELS.register_module()
class Swin_prune(SwinTransformer_zcg):
    def __init__(self,
                 num_classes,
                 channels,
                 in_channels,
                 freeze=False,
                 thresh=1.0,
                 **kwargs,
                 ):
        super(Swin_prune, self).__init__(
            **kwargs,
        )
        self.channels=channels
        self.in_channels=in_channels
        self.num_classes = num_classes
        self.prune_heads = nn.ModuleList()
        for in_channels in self.in_channels:
            prune_head = new_PruneHead(
                img_size=512,
                in_channels=in_channels,
                channels=self.channels,
                num_classes=num_classes,
                thresh=thresh,
                layers_per_decoder=3,
                loss_decode=dict(type='ATMLoss', num_classes=1, dec_layers=1, loss_weight=1.0))
            self.prune_heads.append(prune_head)

    def forward(self, inputs):
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs)

        hw = hw_shape[0] * hw_shape[1]
        mask_idx = torch.zeros(
            (B, hw), device=x.device) != 0
        
        # mask_idx = mask_idx.cuda()
        
        # import torch.distributed as dist
        # if dist.get_rank() == 0:
        

        
        # dist.barrier()
        
        canvas = torch.zeros_like(mask_idx).unsqueeze(
            1).repeat(1, self.num_classes, 1).float()
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.stages):
            # total += (~mask_idx).sum()
            if mask_idx[:, -hw:].sum() == 0:
                x, hw_shape, out_x, out_hw_shape = layer(x=x,hw_shape=hw_shape)
            else:
                x, hw_shape, out_x, out_hw_shape = layer(x=x, hw_shape=hw_shape,mask_idx=mask_idx)

            if i in self.out_indices:
                idx = self.out_indices.index(i)
                if i != self.out_indices[-1]:#循环更新，最后才输出
                    mask_idx, canvas = self.prune_heads[idx](
                        out_x, inference=True, mask_idx=mask_idx, canvas=canvas)
                    
                    # import pdb;pdb.set_trace()
                    
                    mask_idx_new = mask_idx.float().reshape(B,1,out_hw_shape[0],out_hw_shape[1])
                    # mask_idx_new = booltoint(mask_idx).reshape(B,1,out_hw_shape[0],out_hw_shape[1])
                    mask_idx_new = resize(mask_idx_new, size=(hw_shape[0],hw_shape[1]),
                                                  mode='nearest').reshape(B,1,-1)#添加了一个unsqueence,特征维度需要剔除B,C
                    mask_idx = mask_idx_new.bool()
                    del mask_idx_new
                    
                    canvas=resize(canvas.reshape(B,self.num_classes,out_hw_shape[0],out_hw_shape[1]),
                                  size=(hw_shape[0],hw_shape[1]),mode='nearest').reshape(B,self.num_classes,-1)
                    # mask_idx = inttobool(mask_idx_new)

                else:
                    mask_idx, canvas = self.prune_heads[idx](
                        out_x,inference=True, mask_idx=mask_idx, canvas=canvas)
                out =out_x

                B, _, C = out.shape
                out = out.reshape(B, out_hw_shape[0], out_hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
                outs.append(mask_idx)
                # for s in range(len(out)):
                #     showmypic(out[s],inputs[s])

        return tuple(outs)