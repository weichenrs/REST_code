# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
import torch

import torch.distributed as dist
import numpy as np

import os
@MODELS.register_module()
class SPEncoderDecoder_show(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.init_basket()    # 初始化
    def init_basket(self):
        # self.feat_vecs = torch.tensor([]).cuda()            # 特征向量
        # self.feat_vec_labels = torch.tensor([]).cuda()      # 特征向量的类别
        # self.feat_vec_domlabels = torch.tensor([]).cuda()   # 特征向量的域信息
        self.mem_vecs = None                                # 聚类中心的向量
        self.mem_vec_labels = None                          # 聚类中心的类别
    def input2basket(self, feature_map, gt_cuda):
            b, c, h, w = feature_map.shape
            features = F.normalize(feature_map.clone(), dim=1)
            gt_cuda = gt_cuda.clone()
            H, W = gt_cuda.size()[-2:]
            gt_cuda[gt_cuda == 255] = self.num_classes
            gt_cuda = F.one_hot(gt_cuda, num_classes=self.num_classes + 1)
            gt = gt_cuda.view(b, -1, self.num_classes + 1)
            denominator = gt.sum(1).unsqueeze(dim=1)
            denominator = denominator.sum(0)  # batchwise sum
            denominator = denominator.squeeze()
            
            # if dist.get_rank() == 0:
            #     import pdb; pdb.set_trace()
            # dist.barrier()
            
            features = F.interpolate(features, [H, W], mode='bilinear', align_corners=True)
            # 这里是将feature采样到跟标签一样的大小。当然也可以将标签采样到跟feature一样的大小
            features = features.view(b, c, -1)
            

            nominator = torch.matmul(features, gt.type(torch.float32))
            nominator = torch.t(nominator.sum(0))  # batchwise sum
            
            feat_vecs = torch.tensor([])            # 特征向量
            feat_vec_labels = torch.tensor([])      # 特征向量的类别
            feat_vec_domlabels = torch.tensor([])   # 特征向量的域信息
            for slot in range(24):
                if denominator[slot] != 0:
                    cls_vec = nominator[slot] / denominator[slot]  # mean vector
                    cls_label = (torch.zeros(1, 1) + slot)
                    dom_label = (torch.zeros(1, 1))
                    
                    cls_vec = cls_vec.detach().cpu()
                    feat_vecs = torch.cat((feat_vecs, cls_vec.unsqueeze(dim=0)), dim=0)
                    feat_vec_labels = torch.cat((feat_vec_labels, cls_label), dim=0)
                    feat_vec_domlabels = torch.cat((feat_vec_domlabels, dom_label), dim=0)
                    
            namedir = 'proj/spvit/work_dirs/visattn/0927/512new/'+str(dist.get_rank())+'/feat_vecs/'
            os.makedirs(namedir, exist_ok=True)
            np.save(namedir + str(int( len( os.listdir(namedir) ) )) + '.npy', feat_vecs.numpy())
            
            namedir = 'proj/spvit/work_dirs/visattn/0927/512new/'+str(dist.get_rank())+'/feat_vec_labels/'
            os.makedirs(namedir, exist_ok=True)
            np.save(namedir + str(int( len( os.listdir(namedir) ) )) + '.npy', feat_vec_labels.numpy())

            namedir = 'proj/spvit/work_dirs/visattn/0927/512new/'+str(dist.get_rank())+'/feat_vec_domlabels/'
            os.makedirs(namedir, exist_ok=True)
            np.save(namedir + str(int( len( os.listdir(namedir) ) )) + '.npy', feat_vec_domlabels.numpy())


    def _init_decode_head(self, decode_head: ConfigType) -> None:
        """Initialize ``decode_head``"""
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels

    def _init_auxiliary_head(self, auxiliary_head: ConfigType) -> None:
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(MODELS.build(head_cfg))
            else:
                self.auxiliary_head = MODELS.build(auxiliary_head)

    def extract_feat(self, inputs: Tensor) -> List[Tensor]:
        """Extract features from images."""
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, inputs: Tensor,
                      batch_img_metas: List[dict]) -> Tensor:
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(inputs)
        seg_logits = self.decode_head.predict(x, batch_img_metas,
                                              self.test_cfg)

        return seg_logits

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.loss(inputs, data_samples,
                                            self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples, self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.loss(inputs, data_samples,
                                                self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(inputs)
        y = torch.stack([data_sample.gt_sem_seg.data for data_sample in data_samples], dim=0)
        # for feat in x:
        #     self.input2basket(x, y)
        # import pdb; pdb.set_trace()
        self.input2basket(x[-1], y)

        # import pdb; pdb.set_trace()
        losses = dict()
        # with torch.autograd.graph.save_on_cpu(pin_memory=True):
        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)

        return losses

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        if data_samples is not None:
            batch_img_metas = [
                data_sample.metainfo for data_sample in data_samples
            ]
        else:
            batch_img_metas = [
                dict(
                    ori_shape=inputs.shape[2:],
                    img_shape=inputs.shape[2:],
                    pad_shape=inputs.shape[2:],
                    padding_size=[0, 0, 0, 0])
            ] * inputs.shape[0]

        seg_logits = self.inference(inputs, batch_img_metas)
        # dist.broadcast(seg_logits, src=0)
        return self.postprocess_result(seg_logits, data_samples)

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        x = self.extract_feat(inputs)
        return self.decode_head.forward(x)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        tempws = dist.get_world_size()
        
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                
                # if dist.get_rank() == 0:
                #     import pdb; pdb.set_trace()
                    
                sp_times = int(np.log2(dist.get_world_size()))
                sp_temp = sp_times // 2
                sp_h = 2 ** sp_temp
                sp_w = 2 ** (sp_times - sp_temp)
                ind_h = dist.get_rank()//sp_h 
                ind_w = dist.get_rank()%sp_h           
                
                crop_img = torch.chunk(crop_img, sp_h, dim=-2)[ind_w] if sp_h > 1 else crop_img
                crop_img = torch.chunk(crop_img, sp_w, dim=-1)[ind_h] if sp_w > 1 else crop_img
                # batch_img_metas[0]['img_shape'] = (crop_img.shape[2]//sp_h, crop_img.shape[3]//sp_w)
                
                batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                
                # if dist.get_rank() == 0:
                #     import pdb; pdb.set_trace()
                    
                pred_list = [None] * tempws
                # dist.gather_object(crop_seg_logit, pred_list, dst=dist.get_rank())
                dist.gather_object(crop_seg_logit, pred_list if dist.get_rank() == 0 else None, dst=0)
                if dist.get_rank() == 0:
                    alllist = []
                    for j in range(sp_w):
                        templist = []
                        for i in range(sp_h):
                            templist.append(pred_list[i + sp_h * j].cuda(0))
                        # temp = np.concatenate([x for x in templist], 0)
                        temp = torch.cat([x for x in templist], -2)
                        alllist.append(temp)
                    # crop_seg_logit = np.concatenate([x for x in alllist], 1)
                    crop_seg_logit = torch.cat([x for x in alllist], -1)

                    preds += F.pad(crop_seg_logit,
                                (int(x1), int(preds.shape[3] - x2), int(y1),
                                    int(preds.shape[2] - y2)))

                    count_mat[:, :, y1:y2, x1:x2] += 1
                    
        if dist.get_rank() == 0:
            assert (count_mat == 0).sum() == 0
            seg_logits = preds / count_mat
        else:
            seg_logits = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
            
        # if dist.get_rank() == 0:
        #     import pdb; pdb.set_trace()
            
        return seg_logits

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        seg_logits = self.encode_decode(inputs, batch_img_metas)

        return seg_logits

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        assert self.test_cfg.get('mode', 'whole') in ['slide', 'whole'], \
            f'Only "slide" or "whole" test mode are supported, but got ' \
            f'{self.test_cfg["mode"]}.'
        ori_shape = batch_img_metas[0]['ori_shape']
        if not all(_['ori_shape'] == ori_shape for _ in batch_img_metas):
            print_log(
                'Image shapes are different in the batch.',
                logger='current',
                level=logging.WARN)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(inputs, batch_img_metas)
        else:
            seg_logit = self.whole_inference(inputs, batch_img_metas)

        return seg_logit

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(inputs[0], batch_img_metas[0], rescale)
        for i in range(1, len(inputs)):
            cur_seg_logit = self.inference(inputs[i], batch_img_metas[i],
                                           rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(inputs)
        seg_pred = seg_logit.argmax(dim=1)
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
