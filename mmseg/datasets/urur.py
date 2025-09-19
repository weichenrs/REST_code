# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class URURDataset(BaseSegDataset):
    """URUR dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=("building", "farmland", "greenhouse", "woodland", 
                 "bareland", "water", "road", "others"),

        palette=[

                    [230, 230, 230], #building
                    [95, 163, 7], #farmland
                    [100, 100, 100], #greenhouse
                    [200, 230, 160], #woodland
                    [255, 255, 100], #bareland
                    [150, 200, 250], #water
                    [240, 100, 80], #road
                    [255, 255, 255], #others
                    # [0, 0, 0], #others
                 ]
        )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)