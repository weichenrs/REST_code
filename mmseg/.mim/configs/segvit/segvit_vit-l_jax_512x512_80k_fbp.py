_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/fbp_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
in_channels = 1024

img_size = 512
data_preprocessor = dict(
    size=(img_size, img_size),
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
out_indices = [7, 15, 23]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        img_size=(512, 512),
        embed_dims=1024,
        num_layers=24,
        drop_path_rate=0.3,
        num_heads=16,
        # with_cp=True,
        out_indices=out_indices),
    decode_head=dict(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=24,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=16,
        use_stages=len(out_indices),
        loss_decode=dict(
            type='ATMLoss', num_classes=24, dec_layers=len(out_indices), loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(384, 384)),
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
    clip_grad=dict(max_norm=35, norm_type=2),
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
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    # visualization=dict(type='SegVisualizationHook')
    )

# env_cfg = dict(
#     dist_cfg=dict(init_backend='deepspeed')
# )