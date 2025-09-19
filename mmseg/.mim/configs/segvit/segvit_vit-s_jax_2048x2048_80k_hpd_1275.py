_base_ = [
    '../_base_/models/seg_vit-b16.py',
    # '../_base_/datasets/hpd_4096x4096_sp.py', 
    '../_base_/datasets/hpd_2048x2048.py', 
    # '../_base_/datasets/hpd_sp.py', 
    # '../_base_/default_runtime.py',
    '../_base_/tensorboard_runtime.py',
    # '../_base_/schedules/sp_schedule_80k.py'
    '../_base_/schedules/schedule_80k.py'
]
in_channels = 384

img_size = 2048
data_preprocessor = dict(
    size=(img_size, img_size),
    type='SegDataPreProcessor',
    # type='SPSegDataPreProcessor_hw',
    # mean=[123.675, 116.28, 103.53],
    # std=[58.395, 57.12, 57.375],
    mean=[127.5, 127.5, 127.5], 
    std=[127.5, 127.5, 127.5], 
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
# out_indices = [7, 15, 23]

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
# checkpoint = 'pretrain/vit_large_p32_384-9b920ba8.pth'
checkpoint = 'pretrain/vit_small_p32_384.pth'
# out_indices = [7, 15, 23]
out_indices = [5, 7, 11]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        # type='MYVisionTransformer_ds',
        # type='MYVisionTransformer_fa_ds_hw', #2.5s
        # type='MYVisionTransformer_ds_pos', #21s
        # type='MYVisionTransformer_ds_pos_hw', #cwwhu2 21s    #cwwhu 2048bs2 3.8s 4096bs8 16-21s 84096 30s
        # type='MYVisionTransformer_ds_pos_eval',
        # type='MYVisionTransformer_ds_pos_hw_eval', #2s
        img_size=(img_size, img_size),
        patch_size=32,
        embed_dims=in_channels,
        num_layers=12,
        drop_path_rate=0.3,
        num_heads=8,
        with_cls_token=False,
        # with_cp=True,
        out_indices=out_indices
        ),
    decode_head=dict(
        # type='ATMHead_sp_hw',
        img_size=img_size,
        in_channels=in_channels,
        num_classes=9,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=8,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=9, dec_layers=len(out_indices), loss_weight=1.0),
    ),
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
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=3000),
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

# train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=40000, val_interval=4000)
# val_cfg = dict(type='SPValLoop', fp16=True)
# test_cfg = dict(type='SPTestLoop', fp16=True)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(batch_size=1, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    visualization=dict(type='SegVisualizationHook', interval=50)
    )

