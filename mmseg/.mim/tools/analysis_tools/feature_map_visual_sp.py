# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from typing import Type

import mmcv
import torch
import torch.nn as nn

from mmengine.model import revert_sync_batchnorm
from mmengine.structures import PixelData
import math

import os 
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['RANK'] = '0'
# os.environ['LOCAL_RANK'] = '0'
# os.environ['WORLD_SIZE'] = '1'
# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '21352'
# print(int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK']))

# torch.distributed.init_process_group('nccl',world_size=int(os.environ['WORLD_SIZE']),rank=int(os.environ['LOCAL_RANK']))
import deepspeed
deepspeed.init_distributed(dist_backend='nccl', world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['LOCAL_RANK']))

import numpy as np 
import torch.distributed as dist

from mmseg.apis import inference_model, init_model
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmseg.visualization import SegLocalVisualizer_sp


class Recorder:
    """record the forward output feature map and save to data_buffer."""

    def __init__(self) -> None:
        self.data_buffer = list()

    def __enter__(self, ):
        self._data_buffer = list()

    def record_data_hook(self, model: nn.Module, input: Type, output: Type):
        self.data_buffer.append(output)

    def __exit__(self, *args, **kwargs):
        pass


def visualize(args, model, recorder, result):

    seg_visualizer = SegLocalVisualizer_sp(
        vis_backends=[dict(
            # type='WandbVisBackend'
            type='LocalVisBackend'
            )],
        save_dir='temp_dir_1226',
        alpha=1)
    seg_visualizer.dataset_meta = dict(
        classes=model.dataset_meta['classes'],
        palette=model.dataset_meta['palette'])

    image = mmcv.imread(args.img, 'color')
    
    sp_times = int(np.log2(dist.get_world_size()))
    sp_temp = sp_times // 2
    sp_h = 2 ** sp_temp
    sp_w = 2 ** (sp_times - sp_temp)
    ind_h = dist.get_rank()//sp_h 
    ind_w = dist.get_rank()%sp_h
                
    image = np.split(image, sp_h, axis=0)[ind_w] if sp_h > 1 else image
    image = np.split(image, sp_w, axis=1)[ind_h] if sp_w > 1 else image

    if args.gt_mask:
        sem_seg = mmcv.imread(args.gt_mask, 'unchanged')
        
        # sem_seg = np.split(sem_seg, sp_h, axis=0)[ind_w] if sp_h > 1 else sem_seg
        # sem_seg = np.split(sem_seg, sp_w, axis=1)[ind_h] if sp_w > 1 else sem_seg
        
        sem_seg = torch.from_numpy(sem_seg-1)
        gt_mask = dict(data=sem_seg)
        gt_mask = PixelData(**gt_mask)
        result.gt_sem_seg = gt_mask
        
        seg_visualizer.add_datasample(
            name='predict',
            image=image,
            data_sample=result,
            draw_gt=True,
            draw_pred=True,
            wait_time=0,
            out_file=None,
            show=False)
    else:
        seg_visualizer.add_datasample(
            name='predict',
            image=image,
            data_sample=result,
            draw_gt=False,
            draw_pred=True,
            wait_time=0,
            out_file=None,
            show=False)

    # add feature map to wandb visualizer
    for i in range(len(recorder.data_buffer)):
        feature = recorder.data_buffer[i][0]  # remove the batch
        
        # tmpsize = round(math.sqrt(feature.shape[0]))     
        feature = feature.transpose(0, 1).reshape(-1, 32, 32)

        drawn_img = seg_visualizer.draw_featmap(
            feature, image, channel_reduction='select_max')
        seg_visualizer.add_image(f'feature_map_max{i}', drawn_img)
        
        # drawn_img = seg_visualizer.draw_featmap(
        #     feature, image)
        # seg_visualizer.add_image(f'feature_map_mean{i}', drawn_img)
        
        # drawn_img = seg_visualizer.draw_featmap(
        #     feature, image, channel_reduction=None)
        # seg_visualizer.add_image(f'feature_map{i}', drawn_img)
        
    seg_visualizer.add_image('image', image)


def main():
    parser = ArgumentParser(
        description='Draw the Feature Map During Inference')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--gt_mask', default=None, help='Path of gt mask file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    # parser.add_argument(
    #     '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()

    register_all_modules()
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device='cuda:'+os.environ['LOCAL_RANK'], 
                       cfg_options={
                                    # 'model.data_preprocessor.type':'SegDataPreProcessor',
                                    'model.data_preprocessor.type':'SPSegDataPreProcessor_hw_cam',
                                    'model.data_preprocessor.size':(1024, 1024),
                                    'model.data_preprocessor.mean':[127.5, 127.5, 127.5], 
                                    'model.data_preprocessor.std':[127.5, 127.5, 127.5], 
                                    'model.data_preprocessor.bgr_to_rgb':True,
                                    'model.data_preprocessor.pad_val':0,
                                    'model.data_preprocessor.seg_pad_val':255
                                    }
                        )   
    
    # if args.device == 'cpu':
    #     model = revert_sync_batchnorm(model)

    # show all named module in the model and use it in source list below
    for name, module in model.named_modules():
        print(name)

    source = [
        # 'decode_head.fusion.stages.0.query_project.activate',
        # 'decode_head.context.stages.0.key_project.activate',
        # 'decode_head.context.bottleneck.activate'
        'backbone.layers.11.ffn.layers.2'
    ]
    source = dict.fromkeys(source)

    count = 0
    recorder = Recorder()
    # registry the forward hook
    for name, module in model.named_modules():
        if name in source:
            count += 1
            module.register_forward_hook(recorder.record_data_hook)
            if count == len(source):
                break

    with recorder:
        # test a single image, and record feature map to data_buffer
        result = inference_model(model, args.img)
        
    # if dist.get_rank() == 0:
    #     import pdb;pdb.set_trace()
    # dist.barrier()

    visualize(args, model, recorder, result[0])


if __name__ == '__main__':
    main()
