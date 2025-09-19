# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class UAVIDDataset(BaseSegDataset):
    """UAVID dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    METAINFO = dict(
        classes=('Background clutter', 'Building', 'Road', 'Tree',
                 'Low vegetation', 'Moving car', 'Static car', 'Human'),
        palette=[
                    [0,0,0], 
                    [128,0,0], 
                    [128,64,128], 
                    [0,128,0], 
                    [128,128,0], 
                    [64,0,128], 
                    [192,0,192], 
                    [64,64,0], 
                 ])


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
