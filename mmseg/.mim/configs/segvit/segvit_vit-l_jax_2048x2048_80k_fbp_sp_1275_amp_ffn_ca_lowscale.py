_base_ = [
    '../_base_/datasets/fbp_2048x2048_sp.py', '../_base_/default_runtime.py',
    '../_base_/schedules/sp_schedule_80k.py'
]
in_channels = 1024

img_size = 2048
data_preprocessor = dict(
    size=(img_size, img_size),
    # type='SPSegDataPreProcessor',
    type='SPSegDataPreProcessor_hw',
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

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_large_p16_384_20220308-d4efb41d.pth'
# out_indices = [7, 15, 23]
out_indices = [23]
model = dict(
    type='EncoderDecoder',
    pretrained=checkpoint,
    backbone=dict(
        type='MYVisionTransformer_ds_pos_hw', 
        img_size=(2048, 2048),
        patch_size=16,
        in_channels=3,
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        drop_path_rate=0.3,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        out_indices=out_indices,
        final_norm=False,
        norm_cfg=dict(type='LN', eps=1e-6, requires_grad=True),
        with_cls_token=False,
        interpolate_mode='bicubic',
        with_cp=True,
        ),
    neck=dict(type='VITMSNeck', embed_dim=1024, rescales=[1, 0.5, 0.25]),
    decode_head=dict(
        type='CA_FFN_Head_sp',
        # type='CA_FFN_Head',
        use_stages=3,
        num_layers=1,
        img_size=img_size,
        in_channels=in_channels,
        num_classes=24,
        channels=in_channels,
        embed_dims=in_channels // 2,
        num_heads=16,
        sample=False, # 4
    ),
    # decode_head=dict(
    #     type='ATMHead_sp_hw',
    #     img_size=img_size,
    #     in_channels=in_channels,
    #     num_classes=24,
    #     channels=in_channels,
    #     embed_dims=in_channels // 2,
    #     num_heads=16,
    #     use_stages=len(out_indices),
    #     loss_decode=dict(
    #         type='ATMLoss', num_classes=24, dec_layers=len(out_indices), loss_weight=1.0),
    # ),
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

find_unused_parameters=True

train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=40000, val_interval=4000)
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)
# val_cfg = dict(type='SPValLoop')
# test_cfg = dict(type='SPTestLoop')

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
    
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    # visualization=dict(type='SegVisualizationHook')
    )



val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
