# dataset settings
dataset_type = 'isicDataset'
data_root = '../../data/isic'
# crop_size = (6908, 7300)
# crop_size = (7300, 6908)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=False),
#     dict(
#         type='RandomResize',
#         scale=crop_size,
#         ratio_range=(0.5, 1.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]

custom_imports = dict(imports=['mmcv_custom'], allow_failed_imports=False)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=crop_size, keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations_binary', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
img_ratios = [1.0]
tta_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=False)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
# train_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     sampler=dict(type='SPInfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='train/img', 
#             seg_map_path='train/annotation'),
#         pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='image/val', 
            seg_map_path='label/val'),
        pipeline=test_pipeline))
train_dataloader = val_dataloader
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='SPDefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='image/test', 
            seg_map_path='label/test'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = val_evaluator
# test_evaluator = dict(type='IoUMetric_sp_hw', iou_metrics=['mIoU', 'mFscore'], format_only=True)
