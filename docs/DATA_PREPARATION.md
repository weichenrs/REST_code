# Data Preparation Guide

This document provides comprehensive instructions for preparing datasets to work with the REST framework.

## 📋 Supported Datasets

REST supports various remote sensing datasets for semantic segmentation:

| Dataset | Type | Classes | Platform | Spectral | Size | Official Link |
|---------|------|---------|----------|----------|------|---------------|
| **GLH-Water** | Water Segmentation | 2 | Satellite | RGB | Large-scale | [Link](https://jack-bo1220.github.io/project/GLH-water.html) |
| **Five-Billion-Pixels** | Multi-class | 24 | Satellite | RGB | 150,000+ images | [Link](https://x-ytong.github.io/project/Five-Billion-Pixels.html) |
| **WHU-OHS** | Overhead | 7 | Satellite | RGB | High-resolution | [Link](http://irsip.whu.edu.cn/resources/WHU_OHS_show.php) |
| **UAVid** | Urban Scenes | 8 | Drone/UAV | RGB | Video sequences | [Link](https://uavid.nl/) |

## 🗂️ Directory Structure

After preparation, organize your data as follows:

```
REST/
├── data/
│   ├── GLH-Water/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   ├── annotations/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── splits/
│   │       ├── train.txt
│   │       ├── val.txt
│   │       └── test.txt
│   ├── FBP_new/
│   │   ├── img_dir/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── ann_dir/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── splits/
│   ├── WHU-OHS/
│   │   └── ... (similar structure)
│   └── UAVid/
│       └── ... (similar structure)
```

## 📊 Dataset-Specific Instructions

### 1. GLH-Water Dataset

**Download:**
```bash
# Download from official website
wget https://jack-bo1220.github.io/project/GLH-water.html
# Or use provided data
wget https://github.com/weichenrs/REST_code/releases/download/data/waterdata.zip
unzip waterdata.zip -d data/
```

**Characteristics:**
- **Focus**: Water body segmentation
- **Classes**: Water, Non-water
- **Format**: RGB images with binary masks
- **Resolution**: Variable (typically high-resolution satellite imagery)

**Preprocessing:**
```bash
python tools/dataset_converters/glh_water.py \
    --dataset-path data/GLH-Water \
    --output-path data/GLH-Water/processed
```

### 2. Five-Billion-Pixels Dataset

**Download:**
```bash
# Download sample data (included in repository)
wget https://github.com/weichenrs/REST_code/releases/download/data/data.zip
unzip data.zip -d data/

# For full dataset, visit official website
# https://x-ytong.github.io/project/Five-Billion-Pixels.html
```

**Characteristics:**
- **Focus**: Large-scale multi-class segmentation
- **Classes**: 24 semantic categories
- **Format**: RGB images with multi-class masks
- **Size**: 150,000+ annotated images

**Class Mapping:**
```python
CLASSES = ['background', 'building', 'road', 'water', 'barren', 'forest', 
          'agricultural', 'playground', 'pond', 'parking', 'residential',
          'industrial', 'port', 'farm', 'airport', 'golf', 'stadium', 
          'park', 'overpass', 'railway', 'river', 'bridge', 'pier', 'island']
PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], ...]  # RGB color map
```

### 3. WHU-OHS Dataset

**Download:**
```bash
# Visit official website to download
# http://irsip.whu.edu.cn/resources/WHU_OHS_show.php

# Extract to data directory
unzip WHU-OHS.zip -d data/WHU-OHS/
```

**Characteristics:**
- **Focus**: Overhead scene parsing
- **Classes**: 7 categories (building, road, water, vegetation, etc.)
- **Resolution**: High-resolution satellite imagery
- **Annotation**: Pixel-level labels

### 4. UAVid Dataset

**Download:**
```bash
# Download from official website
# https://uavid.nl/

# Extract sequences
unzip UAVid.zip -d data/UAVid/
```

**Characteristics:**
- **Focus**: Urban scene understanding from UAV perspective
- **Classes**: 8 categories
- **Format**: Video sequences with frame-level annotations
- **Platform**: Drone/UAV imagery

## 🔧 Data Processing Scripts

### Convert Annotations

Convert various annotation formats to MMSegmentation-compatible format:

```bash
# Convert GLH-Water annotations
python tools/dataset_converters/glh_water_converter.py

# Convert Five-Billion-Pixels annotations  
python tools/dataset_converters/fbp_converter.py

# Convert WHU-OHS annotations
python tools/dataset_converters/whu_ohs_converter.py

# Convert UAVid annotations
python tools/dataset_converters/uavid_converter.py
```

### Generate Dataset Splits

Create train/validation/test splits:

```bash
python tools/dataset_converters/generate_splits.py \
    --dataset GLH-Water \
    --split-ratio 0.7 0.2 0.1 \
    --output data/GLH-Water/splits/
```

### Data Validation

Verify data integrity and format:

```bash
python tools/dataset_converters/validate_dataset.py \
    --dataset-path data/GLH-Water \
    --config configs/rest/rest_water_swin_large.py
```

## 📝 Custom Dataset Support

To use your own dataset with REST:

### 1. Dataset Structure

Organize your data following the standard structure:

```
custom_dataset/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── image002.jpg
│   └── val/
│       ├── image003.jpg
│       └── image004.jpg
├── annotations/
│   ├── train/
│   │   ├── image001.png
│   │   └── image002.png
│   └── val/
│       ├── image003.png
│       └── image004.png
└── splits/
    ├── train.txt
    └── val.txt
```

### 2. Create Dataset Configuration

Create a dataset config file in `configs/_base_/datasets/`:

```python
# configs/_base_/datasets/custom_dataset.py
dataset_type = 'CustomDataset'
data_root = 'data/custom_dataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], 
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

# Define your classes and color palette
CLASSES = ('background', 'class1', 'class2', ...)
PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], ...]
```

### 3. Register Dataset

Register your dataset in the codebase:

```python
# rest/datasets/custom_dataset.py
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset as MMCustomDataset

@DATASETS.register_module()
class CustomDataset(MMCustomDataset):
    CLASSES = ('background', 'class1', 'class2', ...)
    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], ...]
    
    def __init__(self, **kwargs):
        super(CustomDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
```

## 🎯 Data Quality Guidelines

### Image Requirements

- **Format**: JPEG, PNG, TIFF supported
- **Size**: No strict limits (REST handles large images efficiently)
- **Channels**: RGB (3-channel) or multispectral
- **Quality**: High resolution recommended for better performance

### Annotation Requirements

- **Format**: PNG (grayscale) with pixel values as class indices
- **Classes**: 0 should be background/ignore class
- **Consistency**: Same spatial dimensions as corresponding images
- **Quality**: Accurate pixel-level annotations

### Best Practices

1. **Data Balance**: Ensure reasonable class distribution
2. **Quality Control**: Manually inspect samples for annotation quality
3. **Validation**: Use separate validation set for model selection
4. **Augmentation**: Consider data augmentation for small datasets

## 🚀 Quick Validation

After preparing your data, run a quick validation:

```bash
# Test data loading
python tools/misc/print_config.py configs/rest/rest_water_swin_large.py

# Visualize data samples
python tools/misc/browse_dataset.py \
    configs/rest/rest_water_swin_large.py \
    --output-dir vis_data

# Verify dataset statistics
python tools/analysis/dataset_analysis.py \
    --config configs/rest/rest_water_swin_large.py \
    --out analysis_results.json
```

## 💡 Tips and Troubleshooting

### Common Issues

1. **Path Issues**: Ensure all paths in config files are correct
2. **Class Mismatch**: Verify class definitions match annotations
3. **Memory Issues**: For very large images, consider using SPIM efficiently
4. **Format Issues**: Check image and annotation formats are supported

### Performance Optimization

1. **Data Loading**: Use multiple workers for data loading
2. **Caching**: Enable data caching for repeated experiments
3. **Preprocessing**: Optimize preprocessing pipelines for your data
4. **Storage**: Use fast storage (SSD) for better I/O performance

## 📚 Additional Resources

- [MMSegmentation Data Preparation](https://mmsegmentation.readthedocs.io/en/latest/tutorials/data_pipeline.html)
- [Dataset Format Conversion Tools](tools/dataset_converters/)
- [Data Visualization Tools](tools/analysis/)
- [Configuration Examples](configs/_base_/datasets/)

---

**Need help with data preparation? Check our [FAQ](docs/FAQ.md) or [open an issue](https://github.com/weichenrs/REST_code/issues)!**