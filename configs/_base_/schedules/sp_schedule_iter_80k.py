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

# training schedule for 80k
train_cfg = dict(type='SPIterBasedTrainLoop', max_iters=80000, val_interval=5000)
val_cfg = dict(type='SPValLoop', fp16=True)
test_cfg = dict(type='SPTestLoop', fp16=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, 
                    save_best='mIoU'),
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

