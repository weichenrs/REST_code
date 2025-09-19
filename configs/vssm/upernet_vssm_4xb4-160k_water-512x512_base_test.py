_base_ = [
'../swin/water/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_water-512x512_testwhole.py'
]

val_evaluator = dict(type='IoUMetric_binary', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

model = dict(
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=None,
        dims=128,
        depths=(2, 2, 15, 2),
        ssm_d_state=1,
        ssm_dt_rank="auto",
        ssm_ratio=2.0,
        ssm_conv=3,
        ssm_conv_bias=False,
        forward_type="v05_noz", # v3_noz,
        mlp_ratio=4.0,
        downsample_version="v3",
        patchembed_version="v2",
        drop_path_rate=0.6,
        norm_layer="ln2d",
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=2,
    ),
    auxiliary_head=dict(
        in_channels=512, 
        num_classes=2,
    )
    )
