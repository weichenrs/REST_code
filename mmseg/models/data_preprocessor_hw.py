# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Any, Dict, List, Optional, Sequence

import torch
from mmengine.model import BaseDataPreprocessor

from mmseg.registry import MODELS
from mmseg.utils import stack_batch

import numpy as np
import torch.distributed as dist

@MODELS.register_module()
class SPSegDataPreProcessor_hw(BaseDataPreprocessor):
    """Image pre-processor for segmentation tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It won't do normalization if ``mean`` is not specified.
    2. It does normalization and color space conversion after stacking batch.
    3. It supports batch augmentations like mixup and cutmix.


    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the input size with defined ``pad_val``, and pad seg map
        with defined ``seg_pad_val``.
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations like Mixup and Cutmix during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        padding_mode (str): Type of padding. Default: constant.
            - constant: pads with a constant value, this value is specified
              with pad_val.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
        test_cfg (dict, optional): The padding size config in testing, if not
            specify, will use `size` and `size_divisor` params as default.
            Defaults to None, only supports keys `size` or `size_divisor`.
    """

    def __init__(
        self,
        mean: Sequence[Number] = None,
        std: Sequence[Number] = None,
        size: Optional[tuple] = None,
        size_divisor: Optional[int] = None,
        pad_val: Number = 0,
        seg_pad_val: Number = 255,
        bgr_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        batch_augments: Optional[List[dict]] = None,
        test_cfg: dict = None,
    ):
        super().__init__()
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr2rgb` and `rgb2bgr` cannot be set to True at the same time')
        self.channel_conversion = rgb_to_bgr or bgr_to_rgb

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            self.register_buffer('mean',
                                 torch.tensor(mean).view(-1, 1, 1), False)
            self.register_buffer('std',
                                 torch.tensor(std).view(-1, 1, 1), False)
        else:
            self._enable_normalize = False

        # TODO: support batch augmentations.
        self.batch_augments = batch_augments

        # Support different padding methods in testing
        self.test_cfg = test_cfg

    def forward(self, data: dict, training: bool = False) -> Dict[str, Any]:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            Dict: Data in the same format as the model input.
        """             
        data = self.cast_data(data)  # type: ignore 
        inputs = data['inputs']
        data_samples = data.get('data_samples', None)
        
        # TODO: whether normalize should be after stack_batch
        if self.channel_conversion and inputs[0].size(0) == 3:
            inputs = [_input[[2, 1, 0], ...] for _input in inputs]

        inputs = [_input.float() for _input in inputs]
        if self._enable_normalize:
            inputs = [(_input - self.mean) / self.std for _input in inputs]

        if training:
            assert data_samples is not None, ('During training, ',
                                              '`data_samples` must be define.')
            inputs, data_samples = stack_batch(
                inputs=inputs,
                data_samples=data_samples,
                size=self.size,
                size_divisor=self.size_divisor,
                pad_val=self.pad_val,
                seg_pad_val=self.seg_pad_val)

            if self.batch_augments is not None:
                inputs, data_samples = self.batch_augments(
                    inputs, data_samples)
        else:
            img_size = inputs[0].shape[1:]
            assert all(input_.shape[1:] == img_size for input_ in inputs),  \
                'The image size in a batch should be the same.'
            # pad images when testing
            if self.test_cfg:
                inputs, padded_samples = stack_batch(
                    inputs=inputs,
                    size=self.test_cfg.get('size', None),
                    size_divisor=self.test_cfg.get('size_divisor', None),
                    pad_val=self.pad_val,
                    seg_pad_val=self.seg_pad_val)
                for data_sample, pad_info in zip(data_samples, padded_samples):
                    data_sample.set_metainfo({**pad_info})
            else:
                inputs = torch.stack(inputs, dim=0)

        # sp_times = np.uint8(np.log2(dist.get_world_size()))
        sp_times = int(np.log2(dist.get_world_size()))
        sp_temp = sp_times // 2
        sp_h = 2 ** sp_temp
        sp_w = 2 ** (sp_times - sp_temp)
        ind_h = dist.get_rank()//sp_h 
        ind_w = dist.get_rank()%sp_h
        # sp_h=2
        # sp_w=4
        # [0 2 4 6 
        #  1 3 5 7]
        # [00 10 20 30
        #  01 11 21 31]
        # if dist.get_rank() == 0:
        #     import pdb;pdb.set_trace()
        # dist.barrier()

        inputs = torch.chunk(inputs, sp_h, dim=-2)[ind_w] if sp_h > 1 else inputs
        inputs = torch.chunk(inputs, sp_w, dim=-1)[ind_h] if sp_w > 1 else inputs
        
        # inputs = torch.chunk( torch.chunk(inputs, sp_h, dim=-2)[ind_h], sp_w, dim=-1 )[ind_w]

        for data_sample in data_samples:
            temp = data_sample.gt_sem_seg.data
            del data_sample.gt_sem_seg.data
            # data_sample.gt_sem_seg.data = torch.chunk( torch.chunk(temp, sp_h, dim=-2)[ind_h], sp_w, dim=-1 )[ind_w]
            temp = torch.chunk(temp, sp_h, dim=-2)[ind_w] if sp_h > 1 else temp
            data_sample.gt_sem_seg.data = torch.chunk(temp, sp_w, dim=-1)[ind_h] if sp_w > 1 else temp
            data_sample.set_metainfo({'img_shape': (data_sample.img_shape[0]//sp_h, 
                                                    data_sample.img_shape[1]//sp_w),
                                      'ori_shape': (data_sample.ori_shape[0]//sp_h, 
                                                    data_sample.ori_shape[1]//sp_w),
                                        })
            
        return dict(inputs=inputs, data_samples=data_samples)