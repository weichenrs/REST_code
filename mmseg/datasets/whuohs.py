# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class WHUOHSDataset(BaseSegDataset):
    """WHUOHS dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=(
                    "Paddy field", "Dry farm", "Woodland", 
                    "Shrubbery", "Sparse woodland", "Other forest land", 
                    "High-covered grassland", "Medium-covered grassland", "Low-covered grassland", 
                    "River/canal", "Lake", "Reservoir/pond", 
                    "Beach land", "Shoal", "Urban built-up", 
                    "Rural settlement", "Other construction land", "Sand", 
                    "Gobi", "Saline/alkali soil", "Marshland", 
                    "Bare land", "Bare rock", "Ocean"
                ),
        
        palette=[
                    [188, 207, 251],
                    [51, 223, 199],
                    [34, 113, 5],
                    [168, 237, 115],
                    [79, 213, 0],
                    [69, 210, 0],
                    [111, 113, 0],
                    [166, 164, 0],
                    [255, 242, 3],
                    [113, 174, 251],
                    [0, 91, 228],
                    [5, 37, 113],
                    [121, 139, 244],
                    [0, 168, 132],
                    [114, 0, 3],
                    [252, 126, 125],
                    [251, 188, 189],
                    [254, 186, 229],
                    [248, 0, 197],
                    [226, 5, 168],
                    [167, 0, 131],
                    [113, 0, 75],
                    [254, 110, 222],
                    [160, 160, 160],
                ]
        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)