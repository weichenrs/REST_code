_base_ = [
    '../../_base_/models/upernet_swin.py', '../../_base_/datasets/whuohs_test_sp.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]

size = 6400
overlap = 0
# size = 1024
# overlap = 0
stride = size - overlap

crop_size = (size, size)
data_preprocessor = dict(
    # type='SPSegDataPreProcessor_hw',
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0]*32,
    std=[10000.0]*32,
    bgr_to_rgb=False
    )
num_classes = 24
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
checkpoint_file = None
# checkpoint_file = 'best_mIoU_iter_78000.pth'
# checkpoint_file = 'work_dirs/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512.py/0530/best_mIoU_iter_78000.pth'
# checkpoint_file = 'best_mIoU_iter_80000_0531_server3.pth'
# checkpoint_file = 'proj/spvit/work_dirs/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-512x512.py/0625/best_mIoU_iter_14000.pth'
# checkpoint_file = 'proj/spvit/work_dirs/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_urur-512x512.py/0626/best_mIoU_iter_32000.pth'

model = dict(
    type='SPEncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer_sp_layer',
        in_channels=32,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        # frozen_stages=3,
        frozen_stages=-1
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=num_classes,
        pretrained = checkpoint_file
    ),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=num_classes,
        pretrained = checkpoint_file
    )
    # ,test_cfg=dict(mode='slide', crop_size=(2048, 2048), stride=(512, 512))
    ,test_cfg=dict(mode='slide', crop_size=(size, size), stride=(stride, stride))
)

custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        num_layers=24,
        layer_decay_rate=1.0,
        depths=(2, 2, 18, 2),
        custom_keys={
            'bias': dict(decay_mult=0.),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    constructor='LayerDecayOptimizerConstructorSwin'
    )


param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=10000,
        by_epoch=False,
    )
]

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=10000, val_interval=500)
# val_cfg = dict(type='SPValLoop', fp16=True)
# test_cfg = dict(type='SPTestLoop', fp16=True)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop', fp16=True)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500,
                    save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    )

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=4, num_workers=4)    
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader



find_unused_parameters = True

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator