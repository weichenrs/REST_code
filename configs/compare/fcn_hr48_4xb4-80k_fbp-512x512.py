_base_ = [
    '../_base_/models/fcn_hr18.py', '../_base_/datasets/fbp_512x512_new.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

num_classes = 24
model = dict(
    pretrained='conda_cache/torch/hub/checkpoints/hrnetv2_w48-d2186c55.pth',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    data_preprocessor=data_preprocessor, 
    decode_head=dict(
        num_classes=num_classes,
        in_channels=[48, 96, 192, 384], 
        channels=sum([48, 96, 192, 384])
        ),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(512, 512))
    )

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=2000)
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000,
                    max_keep_ckpts=1, save_best='mIoU'),
    )

num_batch_size = 16
train_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
val_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
test_dataloader = dict(batch_size=1, num_workers=1)