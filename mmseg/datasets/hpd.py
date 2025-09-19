# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class HPDDataset(BaseSegDataset):
    """HPD dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=(
                'agricultural land', 'forest', 'water', 'meadow', 'road', 'dense residential area',
                'sparse residential area', 'public area', 'construction area'
                ),

        palette=[
                    [250, 250,  75], 
                    [  0, 150,   0], 
                    [  0, 150, 200], 
                    [200, 200,   0], 
                    [250, 150, 150], 
                    [200,   0,   0],
                    [250,   0, 100], 
                    [250, 200,   0], 
                    [200, 200, 200],
                 ]
        )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)