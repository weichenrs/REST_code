_base_ = [
    # '../swin/fbp/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_fbp-512x512_decay_new_bs8.py'
    'proj/spvit/configs/swin/uavid/swin-large-patch4-window7-skysense-pre_upernet_2xb2-80k_uavid-4096x4096_sp_layer.py'
]

checkpoint_file = 'proj/spvit/work_dirs/uavid/upernet_vssm_4xb4-160k_uavid-512x512_base.py/1119/best_mIoU_iter_74000.pth'

num_classes = 8
model = dict(
    backbone=dict(
        type='MM_VSSM_sp',
        out_indices=(0, 1, 2, 3),
        # pretrained="../../ckpts/classification/outs/vssm/vssmbasedp05/vssmbase_dp05_ckpt_epoch_260.pth",
        # pretrained='vssm_base_0229_ckpt_epoch_237.pth',
        pretrained=checkpoint_file,
        # copied from classification/configs/vssm/vssm_base_224.yaml
        dims=128,
        # depths=(2, 2, 27, 2),
        # ssm_d_state=16,
        # ssm_dt_rank="auto",
        # ssm_ratio=2.0,
        # mlp_ratio=0.0,
        # downsample_version="v1",
        # patchembed_version="v1",
        # # forward_type="v0", # if you want exactly the same
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
        num_classes=num_classes,
        pretrained = checkpoint_file
    ),
    auxiliary_head=dict(
        in_channels=512, 
        num_classes=num_classes,
        pretrained = checkpoint_file
    )
    )
# train_dataloader = dict(batch_size=4) # as gpus=4

num_batch_size = 4
train_dataloader = dict(batch_size=num_batch_size, num_workers=num_batch_size)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader