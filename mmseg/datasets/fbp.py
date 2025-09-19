# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()
class FBPDataset(BaseSegDataset):
    """FBP dataset.

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('industrial area','paddy field','irrigated field','dry cropland','garden land',
                'arbor forest','shrub forest','park land','natural meadow','artificial meadow',
                'river','urban residential','lake','pond','fish pond','snow','bareland',
                'rural residential','stadium','square','road','overpass','railway station','airport'),

        palette=[
                    [200,   0,   0], # industrial area
                    [0, 200,   0], # paddy field
                    [150, 250,   0], # irrigated field
                    [150, 200, 150], # dry cropland
                    [200,   0, 200], # garden land
                    [150,   0, 250], # arbor forest
                    [150, 150, 250], # shrub forest
                    [200, 150, 200], # park land
                    [250, 200,   0], # natural meadow
                    [200, 200,   0], # artificial meadow
                    [0,   0, 200], # river
                    [250,   0, 150], # urban residential
                    [0, 150, 200], # lake
                    [0, 200, 250], # pond
                    [150, 200, 250], # fish pond
                    [250, 250, 250], # snow
                    [200, 200, 200], # bareland
                    [200, 150, 150], # rural residential
                    [250, 200, 150], # stadium
                    [150, 150,   0], # square
                    [250, 150, 150], # road
                    [250, 150,   0], # overpass
                    [250, 200, 250], # railway station
                    [200, 150,   0] # airport
                    # [0,   0,   0] # unlabeled
                 ]
        )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='_24label.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)