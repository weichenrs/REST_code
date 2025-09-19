_base_ = [
    '../../_base_/models/upernet_convnext.py',
    '../../_base_/datasets/whuohs_2048x2048_sp.py', 
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k.py'
]
crop_size = (2048, 2048)
data_preprocessor = dict(
    type='SPSegDataPreProcessor_hw',
    size=crop_size,
    mean=[0]*32,
    std=[10000.0]*32,
    bgr_to_rgb=False
    )
num_classes = 24
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-xlarge_3rdparty_in21k_20220301-08aa5ddc.pth'  # noqa
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth'
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-large_3rdparty-fcmae_in1k_20230104-bf38df92.pth'
# checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-huge_3rdparty-fcmae_in1k_20230104-fe43ae6c.pth'
# checkpoint_file = 'proj/spvit/work_dirs/whuohs/convnext/convnext-large_upernet_8xb2-amp-80k_whuohs-512x512.py/0723/best_mIoU_iter_44000.pth'
checkpoint_file = 'proj/spvit/work_dirs/whuohs/convnext-large_upernet_8xb2-amp-80k_whuohs-512x512.py/1027/best_mIoU_iter_34000.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=32,
        # type='mmpretrain.ConvNeXt',
        type = 'ConvNeXt_sp_layer',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        gap_before_final_norm=False,
        use_grn=True,
        # frozen_stages=3,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=num_classes,
    ),
    auxiliary_head=dict(in_channels=768, num_classes=num_classes),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)),
    # test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512))
)

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
        # type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05),
        # type='AdamW', lr=1e-6, betas=(0.9, 0.999), weight_decay=0.05),
        # type='AdamW', lr=5e-6, betas=(0.9, 0.999), weight_decay=0.05),
        # type='AdamW', lr=8e-5, betas=(0.9, 0.999), weight_decay=0.05),
        # type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.95,
        'decay_type': 'stage_wise',
        'num_layers': 12
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=10000,
        eta_min=0.0,
        by_epoch=False,
    )
]

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=10000, val_interval=500)
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)

default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500, 
                    max_keep_ckpts=1, save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    )

# By default, models are trained on 8 GPUs with 2 images per GPU
num_batch_size = 3
train_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader



find_unused_parameters = True

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator