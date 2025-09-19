# dataset settings
# _base_ = './fbp.py'
dataset_type = 'isicDataset'
data_root = '../../data/isic/tiled_1024'
data_test_root = data_root
crop_size = (1024, 1024)

custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_binary', reduce_zero_label=False),
    # dict(
    #     type='RandomResize',
    #     scale=(1024, 1024),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations_binary', reduce_zero_label=False),
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
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='SPInfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='image/train', 
            seg_map_path='label/train'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_test_root,
        data_prefix=dict(
            img_path='image/test', 
            seg_map_path='label/test'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_test_root,
        data_prefix=dict(
            img_path='image/test', 
            seg_map_path='label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
