# Evaluation Guide

This document provides comprehensive instructions for evaluating REST models on various datasets and metrics.

## 🚀 Quick Start Evaluation

### Basic Model Testing

```bash
cd rest

# Test pre-trained model on GLH-Water dataset
bash tools/test.sh configs/rest/rest_water_swin_large.py checkpoints/REST_water_swin_large.pth

# Test on Five-Billion-Pixels dataset
bash tools/test.sh configs/baseline/baseline_fbp_swin_large.py checkpoints/baseline_fbp_swin_large.pth
```

### Custom Test Script

```bash
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --show-dir results/visualization
```

## 📊 Evaluation Metrics

### Supported Metrics

| Metric | Description | Formula | Usage |
|--------|-------------|---------|-------|
| **mIoU** | Mean Intersection over Union | `TP/(TP+FP+FN)` per class, then average | Primary metric |
| **mAcc** | Mean Accuracy | `TP/(TP+FN)` per class, then average | Class-wise accuracy |
| **aAcc** | Overall Accuracy | `(TP+TN)/(TP+TN+FP+FN)` overall | Pixel-level accuracy |
| **F1-Score** | F1 Score | `2*Precision*Recall/(Precision+Recall)` | Harmonic mean |
| **Kappa** | Cohen's Kappa | Agreement measure accounting for chance | Statistical agreement |

### Metric Configuration

```python
# In your config file
evaluation = dict(
    interval=1000,           # Evaluation interval during training
    metric=['mIoU', 'mAcc'], # Metrics to compute
    pre_eval=True,           # Pre-compute confusion matrix
    save_best='mIoU',        # Save best model based on this metric
    rule='greater'           # Higher is better for mIoU
)
```

## 🔧 Evaluation Options

### 1. Standard Evaluation

```bash
# Basic evaluation with mIoU
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU

# Multiple metrics
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU mAcc aAcc
```

### 2. Multi-Scale Testing

```bash
# Test with multiple scales for better performance
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --aug-test \
    --cfg-options data.test.pipeline[1].img_ratios=[0.5,0.75,1.0,1.25,1.5]
```

### 3. Test-Time Augmentation (TTA)

```bash
# Enable TTA for improved accuracy
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --aug-test \
    --cfg-options data.test.pipeline[1].flip=True
```

### 4. Distributed Testing

```bash
# Multi-GPU testing for faster evaluation
bash tools/dist_test.sh \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    4 \
    --eval mIoU
```

## 📈 Comprehensive Evaluation

### Per-Class Analysis

```bash
# Generate detailed per-class results
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --options "jsonfile_prefix=results/rest_water_results"
```

### Confusion Matrix Generation

```bash
# Generate and visualize confusion matrix
python tools/analysis/confusion_matrix.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --show-path results/confusion_matrix.png
```

### Speed Benchmarking

```bash
# Measure inference speed
python tools/benchmark.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --shape 1024 1024 \
    --num-warmup 50 \
    --num-iters 200
```

## 🎯 Dataset-Specific Evaluation

### GLH-Water Dataset Evaluation

```bash
# Water segmentation specific metrics
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --cfg-options \
    test_cfg.mode='slide' \
    test_cfg.crop_size=[1024,1024] \
    test_cfg.stride=[768,768]
```

**Expected Performance:**
- mIoU: >85%
- Water IoU: >90%
- Overall Accuracy: >95%

### Five-Billion-Pixels Dataset Evaluation

```bash
# Multi-class segmentation evaluation
python tools/test.py \
    configs/baseline/baseline_fbp_swin_large.py \
    checkpoints/baseline_fbp_swin_large.pth \
    --eval mIoU mAcc \
    --show-dir results/fbp_results
```

**Expected Performance:**
- mIoU: >75%
- mAcc: >80%
- Overall Accuracy: >85%

### WHU-OHS Dataset Evaluation

```bash
# High-resolution overhead scene evaluation
python tools/test.py \
    configs/rest/rest_whu_swin_large.py \
    checkpoints/rest_whu_swin_large.pth \
    --eval mIoU \
    --cfg-options test_cfg.crop_size=[512,512]
```

### UAVid Dataset Evaluation

```bash
# UAV imagery evaluation
python tools/test.py \
    configs/rest/rest_uavid_vit.py \
    checkpoints/rest_uavid_vit.pth \
    --eval mIoU mAcc
```

## 🔍 Advanced Evaluation Features

### 1. SPIM Evaluation for Large Images

```bash
# Evaluate with Spatial Parallel Interaction Mechanism
python tools/test.py \
    configs/rest/rest_water_spim.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --cfg-options \
    model.decode_head.use_spim=True \
    model.decode_head.spim_config.parallel_gpus=4
```

### 2. Memory Usage Analysis

```bash
# Monitor GPU memory usage during evaluation
python tools/analysis/memory_analysis.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --input-shape 1024 1024
```

### 3. Throughput Analysis

```bash
# Analyze throughput scalability with multiple GPUs
python tools/analysis/throughput_analysis.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --gpu-nums 1 2 4 8
```

## 📊 Results Visualization

### 1. Prediction Visualization

```bash
# Generate prediction visualizations
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --show-dir results/predictions \
    --opacity 0.5
```

### 2. Error Analysis

```bash
# Visualize prediction errors
python tools/analysis/error_analysis.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --output-dir results/error_analysis
```

### 3. Feature Map Visualization

```bash
# Visualize intermediate feature maps
python tools/visualization/feature_visualization.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --input-img data/sample.jpg \
    --output-dir results/features
```

## 📋 Evaluation Reports

### Automatic Report Generation

```bash
# Generate comprehensive evaluation report
python tools/analysis/generate_report.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --output-dir results/reports \
    --include-visualizations
```

### Custom Metrics Implementation

```python
# tools/analysis/custom_metrics.py
import numpy as np
from mmseg.core import eval_metrics

def water_detection_metrics(results, gt_seg_maps, num_classes, ignore_index):
    """Custom metrics for water detection tasks."""
    # Implementation for water-specific metrics
    water_precision = compute_water_precision(results, gt_seg_maps)
    water_recall = compute_water_recall(results, gt_seg_maps) 
    water_f1 = 2 * water_precision * water_recall / (water_precision + water_recall)
    
    return {
        'water_precision': water_precision,
        'water_recall': water_recall,
        'water_f1': water_f1
    }
```

## 🔄 Batch Evaluation

### Multiple Model Comparison

```bash
# Compare multiple models
python tools/analysis/model_comparison.py \
    --models \
    configs/rest/rest_water_swin_large.py:checkpoints/REST_water_swin_large.pth \
    configs/baseline/baseline_fbp_swin_large.py:checkpoints/baseline_fbp_swin_large.pth \
    --output-dir results/comparison
```

### Cross-Dataset Evaluation

```bash
# Test model trained on one dataset on another dataset
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --cfg-options data.test.data_root=data/WHU-OHS
```

## 🎛️ Evaluation Configuration

### Test-time Configuration

```python
# In your config file
test_cfg = dict(
    mode='slide',           # Sliding window inference
    crop_size=(1024, 1024), # Crop size for sliding window
    stride=(768, 768),      # Stride for sliding window
    batched_slide=True,     # Batch sliding windows
    use_spim=True          # Enable SPIM for large images
)
```

### Data Pipeline for Testing

```python
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
```

## 🔧 Troubleshooting Evaluation

### Common Issues

1. **Out of Memory during Testing**:
   ```bash
   # Reduce crop size
   --cfg-options test_cfg.crop_size=[512,512]
   
   # Use CPU for post-processing
   --cfg-options model.test_cfg.mode='slide'
   ```

2. **Slow Inference**:
   ```bash
   # Use multiple GPUs
   bash tools/dist_test.sh config checkpoint 4 --eval mIoU
   
   # Optimize crop size and stride
   --cfg-options test_cfg.stride=[682,682]
   ```

3. **Memory Efficient Testing**:
   ```bash
   # Enable SPIM for memory efficiency
   --cfg-options model.decode_head.use_spim=True
   ```

### Debug Mode

```bash
# Run evaluation in debug mode
python tools/test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --eval mIoU \
    --cfg-options log_level=DEBUG \
    --show-dir results/debug
```

## 📊 Performance Benchmarks

### Expected Results

| Dataset | Model | Backbone | mIoU | mAcc | FPS |
|---------|-------|----------|------|------|-----|
| GLH-Water | REST | Swin-L | 87.5 | 94.2 | 12.3 |
| Five-Billion-Pixels | REST | Swin-L | 76.8 | 82.1 | 8.7 |
| WHU-OHS | REST | Swin-L | 79.2 | 85.6 | 15.1 |
| UAVid | REST | ViT-L | 71.4 | 78.3 | 10.5 |

### Scalability Analysis

```bash
# Test scalability with different image sizes
for size in 512 1024 2048 4096; do
    python tools/benchmark.py \
        configs/rest/rest_water_swin_large.py \
        checkpoints/REST_water_swin_large.pth \
        --shape $size $size \
        --output results/scalability_${size}.json
done
```

## 📚 Additional Evaluation Tools

### 1. Statistical Analysis

```bash
# Generate statistical analysis of results
python tools/analysis/statistical_analysis.py \
    results/rest_water_results.pkl \
    --output-dir results/statistics
```

### 2. Qualitative Analysis

```bash
# Generate qualitative analysis report
python tools/analysis/qualitative_analysis.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --num-samples 100 \
    --output-dir results/qualitative
```

### 3. Robustness Testing

```bash
# Test model robustness to various conditions
python tools/analysis/robustness_test.py \
    configs/rest/rest_water_swin_large.py \
    checkpoints/REST_water_swin_large.pth \
    --test-conditions noise blur brightness contrast
```

## 💡 Evaluation Tips

1. **Use appropriate metrics**: mIoU for general segmentation, F1 for imbalanced datasets
2. **Enable multi-scale testing**: Improves performance at the cost of speed
3. **Use SPIM for large images**: Essential for whole-scene evaluation
4. **Monitor memory usage**: Optimize crop size and stride accordingly
5. **Generate visualizations**: Helps understand model behavior
6. **Compare with baselines**: Use consistent evaluation protocols

## 📝 Results Interpretation

### Understanding Metrics

- **mIoU > 80%**: Excellent performance
- **mIoU 70-80%**: Good performance  
- **mIoU 60-70%**: Moderate performance
- **mIoU < 60%**: Need improvement

### Common Performance Issues

1. **Low water class IoU**: Check data balance and loss weighting
2. **Poor boundary accuracy**: Consider boundary loss or post-processing
3. **Inconsistent results**: Verify test set quality and model stability

## 📚 Additional Resources

- [MMSegmentation Evaluation](https://mmsegmentation.readthedocs.io/en/latest/tutorials/training_tricks.html)
- [Metric Definitions](docs/METRICS.md)
- [Visualization Tools](tools/visualization/)
- [Analysis Scripts](tools/analysis/)

---

**Ready to evaluate! 📊 For troubleshooting, check our [evaluation FAQ](docs/FAQ.md#evaluation) or [open an issue](https://github.com/weichenrs/REST_code/issues)!**