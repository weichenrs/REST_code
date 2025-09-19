# dataset settings
dataset_type = 'FBPDataset_new'
# data_root = 'data/FBP_new/ori_3band'
# data_root = 'data/FBP_new/crop2048_3band_ovlp'
data_root = 'data/FBP_new/crop512_3band'
# data_root = 'data/FBP_new/crop1024_3band'
# crop_size = (6908, 7300)
# crop_size = (1024, 1024)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 1.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    # sampler=dict(type='SPInfiniteSampler', shuffle=True),
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            # img_path='train/img', 
            # seg_map_path='train/annotation'),
            img_path='test/img', 
            seg_map_path='test/annotation'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val/img', 
            seg_map_path='val/annotation'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/img', 
            seg_map_path='test/annotation'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
