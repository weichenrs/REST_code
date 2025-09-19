# Training Guide

This document provides comprehensive instructions for training REST models on your datasets.

## 🚀 Quick Start Training

### Single GPU Training

```bash
cd rest
python tools/train.py configs/rest/rest_water_swin_large.py
```

### Multi-GPU Training (Recommended)

```bash
# Using distributed training script
bash tools/dist_train.sh configs/rest/rest_water_swin_large.py 4

# Or using torch.distributed.launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --launcher pytorch
```

## 📋 Training Configuration

### Available Configurations

| Config File | Dataset | Backbone | Purpose |
|-------------|---------|----------|---------|
| `rest_water_swin_large.py` | GLH-Water | Swin-Large | Water segmentation |
| `baseline_fbp_swin_large.py` | Five-Billion-Pixels | Swin-Large | Multi-class segmentation |
| `rest_whu_swin_base.py` | WHU-OHS | Swin-Base | Overhead scenes |
| `rest_uavid_vit.py` | UAVid | ViT | UAV imagery |

### Key Configuration Parameters

```python
# Model configuration
model = dict(
    type='REST',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
    ),
    decode_head=dict(
        type='RESTHead',
        num_classes=2,  # Adjust for your dataset
        use_spim=True,  # Enable Spatial Parallel Interaction Mechanism
    )
)

# Training configuration
optimizer = dict(type='AdamW', lr=0.00006, weight_decay=0.01)
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)

# SPIM-specific settings
spim_config = dict(
    parallel_gpus=4,        # Number of GPUs for parallel processing
    chunk_size=1024,        # Chunk size for divide-and-conquer
    overlap_ratio=0.1,      # Overlap ratio between chunks
    sync_method='allgather' # Synchronization method
)
```

## 🔧 Training Strategies

### 1. Standard Training

For regular-sized images that fit in GPU memory:

```bash
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --work-dir work_dirs/rest_water \
    --gpu-id 0
```

### 2. Large Image Training with SPIM

For whole-scene remote sensing imagery that exceeds GPU memory:

```bash
# Enable SPIM for large image processing
python tools/train.py \
    configs/rest/rest_water_spim.py \
    --work-dir work_dirs/rest_water_spim \
    --cfg-options model.decode_head.use_spim=True
```

### 3. Multi-Scale Training

Train with multiple scales for better generalization:

```bash
python tools/train.py \
    configs/rest/rest_multiscale.py \
    --work-dir work_dirs/rest_multiscale
```

### 4. Fine-tuning from Pre-trained Models

```bash
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --work-dir work_dirs/rest_finetune \
    --cfg-options load_from=checkpoints/REST_water_swin_large.pth
```

## ⚙️ Advanced Training Options

### Resume Training

```bash
# Resume from the latest checkpoint
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --work-dir work_dirs/rest_water \
    --resume-from work_dirs/rest_water/latest.pth

# Resume from specific checkpoint
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --work-dir work_dirs/rest_water \
    --resume-from work_dirs/rest_water/epoch_50.pth
```

### Custom Learning Rate Schedule

```bash
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options \
    optimizer.lr=0.0001 \
    lr_config.policy=step \
    lr_config.step=[100,150]
```

### Mixed Precision Training

```bash
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options fp16.loss_scale=512.0
```

## 🎯 Training Best Practices

### 1. Batch Size Guidelines

| GPU Memory | Recommended Batch Size | Notes |
|------------|----------------------|-------|
| 8GB | 1-2 | Use gradient accumulation |
| 12GB | 2-4 | Standard training |
| 24GB+ | 4-8 | Optimal performance |

### 2. Learning Rate Schedule

```python
# Recommended for most datasets
lr_config = dict(
    policy='poly',
    power=1.0,
    min_lr=0.0,
    by_epoch=False
)

# For fine-tuning
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11],
    gamma=0.1
)
```

### 3. Data Augmentation

```python
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=10),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
```

## 📊 Monitoring Training

### TensorBoard Logging

```bash
# Enable TensorBoard logging
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options log_config.hooks[1].type=TensorboardLoggerHook

# View logs
tensorboard --logdir work_dirs/rest_water/tf_logs
```

### Weights & Biases Integration

```bash
# Enable W&B logging
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options log_config.hooks[2].type=WandbLoggerHook \
    log_config.hooks[2].init_kwargs.project=rest_project
```

### Key Metrics to Monitor

1. **Loss Curves**: Training and validation loss
2. **mIoU**: Mean Intersection over Union
3. **Accuracy**: Overall pixel accuracy
4. **Memory Usage**: GPU memory consumption
5. **Training Speed**: Iterations per second

## 🔍 Debugging Training Issues

### Common Training Problems

1. **Out of Memory (OOM)**:
   ```bash
   # Reduce batch size
   --cfg-options data.samples_per_gpu=1
   
   # Enable gradient accumulation
   --cfg-options data.samples_per_gpu=1 optimizer_config.grad_clip.max_norm=1.0
   ```

2. **Slow Training**:
   ```bash
   # Increase number of workers
   --cfg-options data.workers_per_gpu=4
   
   # Use multiple GPUs
   bash tools/dist_train.sh configs/rest/rest_water_swin_large.py 4
   ```

3. **NaN Loss**:
   ```bash
   # Reduce learning rate
   --cfg-options optimizer.lr=0.00003
   
   # Enable gradient clipping
   --cfg-options optimizer_config.grad_clip.max_norm=35
   ```

### Validation During Training

```bash
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options evaluation.interval=1000 \
    evaluation.metric=mIoU \
    evaluation.save_best=mIoU
```

## 🎛️ Hyperparameter Tuning

### Learning Rate Grid Search

```bash
for lr in 0.00003 0.00006 0.0001; do
    python tools/train.py \
        configs/rest/rest_water_swin_large.py \
        --work-dir work_dirs/rest_lr_${lr} \
        --cfg-options optimizer.lr=${lr}
done
```

### Batch Size Experiments

```bash
for bs in 2 4 8; do
    python tools/train.py \
        configs/rest/rest_water_swin_large.py \
        --work-dir work_dirs/rest_bs_${bs} \
        --cfg-options data.samples_per_gpu=${bs}
done
```

## 🏗️ Custom Training Pipeline

### Creating Custom Configs

1. **Copy base config**:
   ```bash
   cp configs/rest/rest_water_swin_large.py configs/rest/my_custom_config.py
   ```

2. **Modify for your dataset**:
   ```python
   # Change dataset
   data = dict(
       train=dict(data_root='data/my_dataset'),
       val=dict(data_root='data/my_dataset'),
       test=dict(data_root='data/my_dataset')
   )
   
   # Change number of classes
   model = dict(
       decode_head=dict(num_classes=10)  # Your number of classes
   )
   ```

3. **Train with custom config**:
   ```bash
   python tools/train.py configs/rest/my_custom_config.py
   ```

### Multi-Dataset Training

```python
# configs/rest/multi_dataset.py
dataset_A = dict(
    type='GLHWaterDataset',
    data_root='data/GLH-Water',
    # ... other configs
)

dataset_B = dict(
    type='FBPDataset', 
    data_root='data/FBP_new',
    # ... other configs
)

data = dict(
    train=dict(
        type='ConcatDataset',
        datasets=[dataset_A, dataset_B]
    )
)
```

## 📈 Training Optimization Tips

### 1. SPIM Configuration for Large Images

```python
# Optimal SPIM settings for different GPU configurations
spim_configs = {
    '2_gpus': dict(parallel_gpus=2, chunk_size=2048, overlap_ratio=0.1),
    '4_gpus': dict(parallel_gpus=4, chunk_size=1024, overlap_ratio=0.15), 
    '8_gpus': dict(parallel_gpus=8, chunk_size=512, overlap_ratio=0.2),
}
```

### 2. Memory-Efficient Training

```bash
# Enable gradient checkpointing
python tools/train.py \
    configs/rest/rest_water_swin_large.py \
    --cfg-options model.backbone.use_checkpoint=True
```

### 3. Distributed Training Optimization

```bash
# Optimize for InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=ib0

# Launch distributed training
bash tools/dist_train.sh configs/rest/rest_water_swin_large.py 8
```

## 🔄 Training Workflows

### Development Workflow

```bash
# 1. Quick test with small dataset
python tools/train.py configs/rest/rest_debug.py --work-dir work_dirs/debug

# 2. Hyperparameter search
python tools/train.py configs/rest/rest_search.py --work-dir work_dirs/search

# 3. Full training
bash tools/dist_train.sh configs/rest/rest_final.py 4 --work-dir work_dirs/final
```

### Production Workflow

```bash
# 1. Data validation
python tools/misc/browse_dataset.py configs/rest/rest_water_swin_large.py

# 2. Distributed training with monitoring
bash tools/dist_train.sh configs/rest/rest_water_swin_large.py 8 \
    --cfg-options log_config.hooks[1].type=TensorboardLoggerHook

# 3. Model validation
python tools/test.py configs/rest/rest_water_swin_large.py \
    work_dirs/rest_water/latest.pth --eval mIoU
```

## 📝 Training Logs and Outputs

### Output Structure

```
work_dirs/rest_water/
├── rest_water_swin_large.py    # Config file copy
├── tf_logs/                    # TensorBoard logs
├── 20250101_120000.log        # Training log
├── latest.pth                  # Latest checkpoint
├── best_mIoU_epoch_XX.pth     # Best model
├── epoch_1.pth                # Epoch checkpoints
├── epoch_2.pth
└── ...
```

### Log Analysis

```bash
# Extract training metrics
python tools/analysis/analyze_logs.py work_dirs/rest_water/20250101_120000.log

# Plot training curves
python tools/analysis/plot_training_curves.py work_dirs/rest_water/tf_logs
```

## 💡 Tips for Better Performance

1. **Use appropriate backbone**: Swin-Large for best performance, Swin-Base for efficiency
2. **Enable SPIM**: Essential for large remote sensing images
3. **Multi-GPU training**: Significant speedup with proper scaling
4. **Data augmentation**: Crucial for generalization
5. **Learning rate scheduling**: Use polynomial decay for best results
6. **Regular validation**: Monitor overfitting with validation metrics

## 📚 Additional Resources

- [Configuration System](docs/CONFIG.md)
- [Model Zoo](docs/MODEL_ZOO.md)
- [Evaluation Guide](EVALUATION.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)

---

**Happy Training! 🎯 For more help, check our [training FAQ](docs/FAQ.md#training) or [open an issue](https://github.com/weichenrs/REST_code/issues)!**