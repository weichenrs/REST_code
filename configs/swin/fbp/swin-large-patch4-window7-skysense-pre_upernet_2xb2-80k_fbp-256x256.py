_base_ = [
    '../../_base_/models/upernet_swin.py', '../../_base_/datasets/fbp_256x256_new.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]
# crop_size = (256, 256)
crop_size = (2048, 2048)
data_preprocessor = dict(size=crop_size)
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# checkpoint_file = 'swin_large_skysense_20240516.pth'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'
# SwinTransformer_rope
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=24,
    ),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=24,
    )
    ,test_cfg=dict(mode='slide', crop_size=crop_size, stride=crop_size)
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys={
#             'bias': dict(decay_mult=0.),
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#             'norm': dict(decay_mult=0.)
#         }))

custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)
# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
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
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                    save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    )

# By default, models are trained on 8 GPUs with 2 images per GPU
# train_dataloader = dict(batch_size=8)
# val_dataloader = dict(batch_size=8)
# test_dataloader = val_dataloader
num_batch_size = 64
train_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
val_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
test_dataloader = dict(batch_size=1, num_workers=1)
