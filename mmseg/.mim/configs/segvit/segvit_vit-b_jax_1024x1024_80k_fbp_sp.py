_base_ = [
    '../_base_/models/seg_vit-b16.py',
    '../_base_/datasets/fbp_1024x1024_sp.py', '../_base_/default_runtime.py',
    '../_base_/schedules/sp_schedule_80k.py'
]
in_channels = 768

img_size = 1024
data_preprocessor = dict(
    size=(img_size, img_size),
    type='SPSegDataPreProcessor_hw',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # mean=[127.5, 127.5, 127.5], 
    # std=[127.5, 127.5, 127.5], 
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

# checkpoint = './pretrained/vit_large_p16_384_20220308-d4efb41d.pth'
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
# out_indices = [7, 15, 23]

checkpoint = 'pretrain/vit_base_patch16_224.pth'
out_indices = [5, 7, 11]
model = dict(
    pretrained=checkpoint,
    backbone=dict(
        type='MYVisionTransformer_ds_pos_hw',
        img_size=(1024, 1024),
        # embed_dims=1024,
        # num_layers=24,
        drop_path_rate=0.3,
        num_heads=16,
        with_cls_token=False,
        # with_cp=True,
        out_indices=out_indices
        ),
    decode_head=dict(
        type='ATMHead_sp',
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

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=40000, val_interval=4000)
# val_cfg = dict(type='SPValLoop', fp16=True)
# test_cfg = dict(type='SPTestLoop', fp16=True)
val_cfg = dict(type='SPValLoop')
test_cfg = dict(type='SPTestLoop')

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=2, num_workers=4)
test_dataloader = val_dataloader
    
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    # visualization=dict(type='SegVisualizationHook')
    )

