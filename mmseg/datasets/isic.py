# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class isicDataset(BaseSegDataset):
    """FBP dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background', 'object'),
        palette=[[0, 0, 0], [255, 255, 255]]
        )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='_segmentation.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)