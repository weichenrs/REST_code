_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    # '../_base_/datasets/fbp_1024x1024_sp.py', 
    '../_base_/datasets/fbp_1024x1024.py', 
    # '../_base_/datasets/fbp_1024x1024_crop_vt.py', 
    '../_base_/default_runtime.py',
    # '../_base_/wandb_runtime.py',
    # '../_base_/schedules/sp_schedule_80k.py'
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (1024, 1024)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    )

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/vit_base_patch16_224.pth',
    backbone=dict(
        # type='MYVisionTransformer_fa_nods',
        # type='MYVisionTransformer_fa',
        img_size=crop_size, 
        # with_cp=True,
        with_cls_token=False,
        ),
    # decode_head=dict(num_classes=24),
    decode_head=dict(
        type='UPerHead_nopsp',
        num_classes=24,
        # pool_scales=( (2, 4), (4, 8), (6, 12), (8, 16) ),
        # pool_scales=( (2, 8), (3, 9), (4, 16), (6, 24) ),
        ),
    auxiliary_head=dict(num_classes=24),
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(896, 896))
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(384, 384))
    test_cfg=dict(mode='whole')
    )

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    # dtype='bfloat16',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    # clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
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

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# vis_backends = [dict(type='WandbVisBackend')]
train_cfg = dict(max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop', fp16=True)
test_cfg = dict(type='TestLoop', fp16=True)

default_hooks = dict(
    # timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    # param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, 
                    max_keep_ckpts=5, save_best='mIoU'),
    # sampler_seed=dict(type='DistSamplerSeedHook'),
    # visualization=dict(type='SegVisualizationHook')
    )

# vis_backends = [dict(type='WandbVisBackend')]