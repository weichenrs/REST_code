# optimizer
optimizer=dict(
        type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.01)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.),
            'ln': dict(decay_mult=0.),
            'head': dict(lr_mult=10.),
        }),
    clip_grad=dict(max_norm=35, norm_type=2),
    )
# learning policy
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=2),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=100,
        by_epoch=True,
    )
]
# training schedule for 80k
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=1)
val_cfg = dict(type='SPValLoop')
test_cfg = dict(type='SPTestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1, 
                    max_keep_ckpts=1, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    )

# early_stop_cfg = dict(
#         type='EarlyStoppingHook', 
#         monitor='mIoU', 
#         rule=None, 
#         min_delta=0.01, 
#         patience=3
#         )
# custom_hooks=[early_stop_cfg]
