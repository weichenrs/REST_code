# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class PASTISDataset(BaseSegDataset):
    """URUR dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=("1", "2", "3", "4", "5", "6", 
                 "7", "8", "9", "10", "11", "12", 
                 "13", "14", "15", "16", "17", "18",),

        palette=[
                [128, 64, 128], [244, 35, 232], [70, 70, 70], 
                [102, 102, 156], [190, 153, 153], [153, 153, 153], 
                [250, 170, 30], [220, 220, 0], [107, 142, 35], 
                [152, 251, 152], [70, 130, 180],[220, 20, 60], 
                [255, 0, 0], [0, 0, 142], [0, 0, 70],
                [0, 60, 100], [0, 80, 100], [0, 0, 230], 
                ]
        )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)