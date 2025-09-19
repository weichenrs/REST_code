_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/whuohs_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[0]*32,
    std=[10000.0]*32,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
num_classes = 24
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='conda_cache/torch/hub/checkpoints/resnet101-5d3b4d8f.pth', 
    backbone=dict(in_channels=32,depth=101),
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes),
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