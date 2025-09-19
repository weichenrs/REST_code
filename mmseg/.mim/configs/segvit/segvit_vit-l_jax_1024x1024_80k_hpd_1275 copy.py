_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/hpd-1024x1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
in_channels = 1024

img_size = 1024
data_preprocessor = dict(
    size=(img_size, img_size),
    type='SegDataPreProcessor',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[127.5, 127.5, 127.5], 
    std=[127.5, 127.5, 127.5], 
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
out_indices = [7, 15, 23]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(1024, 1024),
        embed_dims=1024,
        num_layers=24,
        drop_path_rate=0.3,
        num_heads=16,
        with_cp=True,
        out_indices=out_indices),
    decode_head=dict(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=9,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=16,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=9, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(896, 896) ),
    test_cfg=dict(mode='whole'),
)

optim_wrapper = dict(
    _delete_=True,
    # type='DeepSpeedOptimWrapper',
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01),
        # type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'ln': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
    #         'pos_embed': dict(decay_mult=0.),
    #         'cls_token': dict(decay_mult=0.),
    #         'norm': dict(decay_mult=0.)
        }),
    clip_grad=dict(max_norm=35, norm_type=2), # amp  inf
    # accumulative_counts=2
    )

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]

find_unused_parameters=False

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=5000)
default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', interval=100)
    )

train_dataloader = dict(batch_size=1, num_workers=1)
val_dataloader = dict(batch_size=2, num_workers=2)
test_dataloader = val_dataloader


# jax use different img norm cfg
# img_norm_cfg = dict(
#     mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5], to_rgb=True)
# crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]

# data = dict(
#     samples_per_gpu=1,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))