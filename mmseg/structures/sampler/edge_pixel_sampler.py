# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_pixel_sampler import BasePixelSampler
from .builder import PIXEL_SAMPLERS

import cv2
import numpy as np

@PIXEL_SAMPLERS.register_module()
class EDGEPixelSampler(BasePixelSampler):
    """Online Hard Example Mining Sampler for segmentation.

    Args:
        context (nn.Module): The context of sampler, subclass of
            :obj:`BaseDecodeHead`.
        thresh (float, optional): The threshold for hard example selection.
            Below which, are prediction with low confidence. If not
            specified, the hard examples will be pixels of top ``min_kept``
            loss. Default: None.
        min_kept (int, optional): The minimum number of predictions to keep.
            Default: 100000.
    """

    def __init__(self, context, edge_buff=1, edge_weight=1):
        super(EDGEPixelSampler, self).__init__()
        self.context = context
        self.edge_buff = edge_buff
        self.edge_weight = edge_weight

    def sample(self, seg_logit, seg_label):
        """Sample pixels that have high loss or with low prediction confidence.
        Args:
            seg_logit (torch.Tensor): segmentation logits, shape (N, C, H, W)
            seg_label (torch.Tensor): segmentation label, shape (N, 1, H, W)
        Returns:
            torch.Tensor: segmentation weight, shape (N, H, W)
        """
        with torch.no_grad():
            assert seg_logit.shape[2:] == seg_label.shape[2:]
            assert seg_label.shape[1] == 1

            seg_label = seg_label.squeeze(1).long()
            seg_weight = seg_logit.new_ones(size=seg_label.size())
            seg_label = seg_label.cpu().numpy().astype(np.uint8)

            for num_batch in range(seg_label.shape[0]):
                tmp_label = seg_label[num_batch]
                tmp_weight = self.edge_weight * self.sobel_edge_buff(tmp_label, self.edge_buff)
                tmp_weight[tmp_label == self.context.ignore_index] = 0
                seg_weight[num_batch] += torch.Tensor(tmp_weight.astype(np.float32)).cuda()

            return seg_weight

    def sobel_edge_buff(self, img, edge_buff=1):
        mysobel_x = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=1, dy=0, ksize=edge_buff)
        mysobel_y = cv2.Sobel(img, ddepth=cv2.CV_8U, dx=0, dy=1, ksize=edge_buff)
        mysobel = mysobel_x + mysobel_y 
        mysobel = mysobel > 0

        return mysobel