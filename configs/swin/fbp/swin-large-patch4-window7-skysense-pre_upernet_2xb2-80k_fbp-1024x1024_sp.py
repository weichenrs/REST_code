_base_ = [
    '../../_base_/models/upernet_swin.py', 
    '../../_base_/datasets/fbp_1024x1024_sp_new.py',
    # '../../_base_/datasets/fbp_1024x1024_sp_new_tv.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'
]

crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SPSegDataPreProcessor_hw',
    size=crop_size)
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
# checkpoint_file = 'swin_large_skysense_20240516.pth'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220308-d5bdebaf.pth'
# checkpoint_file = 'best_mIoU_iter_78000.pth'
# checkpoint_file = 'work_dirs/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512.py/0530/best_mIoU_iter_78000.pth'
# checkpoint_file = 'best_mIoU_iter_80000_0531_server3.pth'
# checkpoint_file = 'work_dirs/fbp/swin/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_decay_new_bs8.py/0609/best_mIoU_iter_32000.pth'
# checkpoint_file = 'proj/spvit/work_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-1024x1024_sp_layer_new_data_bs8.py/0912/best_mIoU_iter_6500.pth'
# checkpoint_file =  'proj/spvit/work_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_decay_new_tv_1110.py/1110/best_mIoU_iter_32000.pth'

checkpoint_file = 'work_dirs/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512.py/iter_80.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer_sp_layer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        frozen_stages=3,
    ),
    decode_head=dict(
        in_channels=[192, 384, 768, 1536],
        num_classes=24,
        pretrained = checkpoint_file
    ),
    auxiliary_head=dict(
        in_channels=768, 
        num_classes=24,
        pretrained = checkpoint_file
    )
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
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')

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
num_batch_size = 8
train_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
val_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
test_dataloader = dict(batch_size=1, num_workers=1)

# env_cfg = dict(
    # dist_cfg=dict(init_backend='deepspeed')
    # seed=0
# )

find_unused_parameters = True

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator