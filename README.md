# REST: Holistic Learning for End-to-End Semantic Segmentation of Whole-Scene Remote Sensing Imagery (IEEE TPAMI 2025)

[![TPAMI 2025](https://img.shields.io/badge/TPAMI-2025-blue.svg)](https://ieeexplore.ieee.org/document/XXXXX)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the IEEE TPAMI 2025 paper "REST: Holistic Learning for End-to-End Semantic Segmentation of Whole-Scene Remote Sensing Imagery".

## 🏠 Overview

REST (Robust End-to-end semantic Segmentation architecture for whole-scene remoTe sensing imagery) is the **first intrinsically end-to-end framework** for truly holistic segmentation of whole-scene remote sensing imagery (WRI). Unlike conventional deep learning methods that struggle with GPU memory limitations and resort to suboptimal cropping or fusion strategies, REST enables seamless processing of large-scale remote sensing imagery without performance degradation.

**Key innovations:**
- **Spatial Parallel Interaction Mechanism (SPIM)**: Novel approach combining parallel computation with divide-and-conquer strategy to overcome GPU memory constraints while achieving global context awareness
- **Plug-and-play compatibility**: Supports a wide range of encoders and decoders, enabling integration with mainstream segmentation methods and advanced foundation models
- **True end-to-end processing**: No need for cropping or fusion - processes entire scenes holistically
- **Near-linear scalability**: Theoretical and experimental validation of throughput scalability with additional GPUs

## 🎯 Key Features

- ✅ **First end-to-end framework** for holistic whole-scene remote sensing imagery segmentation
- ✅ **Spatial Parallel Interaction Mechanism (SPIM)** for efficient large-scale image processing
- ✅ **Plug-and-play architecture** supporting various encoders/decoders and foundation models  
- ✅ **GPU memory efficient** - eliminates need for cropping or fusion strategies
- ✅ **Multi-platform support** - satellite, drone, multispectral, and hyperspectral imagery
- ✅ **Scalable performance** - near-linear throughput scaling with additional GPUs
- ✅ **Versatile applications** - single-class to multi-class segmentation scenarios

## 📊 Performance

### Quantitative Results

| Dataset | Method | mIoU | F1-Score | OA |
|---------|--------|------|----------|-----|
| GLH-Water | REST | **89.15（IoU）** | **94.26** | **-** |
| Five-Billion-Pixels | REST | **72.95** | **-** | **92.78** |
| WHU-OHS | REST | **22.81** | **-** | **49.26** |
| UAVid | REST | **73.16** | **-** | **82.30** |

### Qualitative Results

<div align="center">
<img src="assets/results_demo.png" width="800px">
</div>

*REST enables holistic segmentation of entire remote sensing scenes without cropping, demonstrating superior performance across satellite and drone platforms with both multispectral and hyperspectral imagery.*

## 🚀 Quick Start

### Installation

Please refer to [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Data Preparation

1. Download the datasets:
   - [GLH-Water Dataset](https://jack-bo1220.github.io/project/GLH-water.html)
   - [Five-Billion-Pixels Dataset](https://x-ytong.github.io/project/Five-Billion-Pixels.html)
   - [WHU-OHS Dataset](http://irsip.whu.edu.cn/resources/WHU_OHS_show.php)
   - [UAVid Dataset](https://uavid.nl/)

2. Download pre-trained models:
   ```bash
   # Download all models
   cd rest/checkpoints
   python download_model.py
   
   # Or download individually
   wget https://github.com/weichenrs/REST_code/releases/download/models/REST_water_swin_large.pth -O checkpoints/REST_water_swin_large.pth
   wget https://github.com/weichenrs/REST_code/releases/download/models-0.1/baseline_fbp_swin_large.pth -O checkpoints/baseline_fbp_swin_large.pth
   ```

3. Organize your data structure:
   ```
   REST/
   ├── data/
   │   ├── GLH-Water/
   │   ├── FBP_new/
   │   ├── WHU-OHS/
   │   └── UAVid/
   ├── rest/
   │   ├── checkpoints/
   │   │   ├── REST_water_swin_large.pth
   │   │   └── baseline_fbp_swin_large.pth
   │   └── ...
   └── ...
   ```

### Inference

Run inference on sample images:
```bash
# Test on GLH-Water dataset
bash test.sh configs/rest/rest_water_swin_large.py checkpoints/REST_water_swin_large.pth

# Test on Five-Billion-Pixels dataset
bash test.sh configs/baseline/baseline_fbp_swin_large.py checkpoints/baseline_fbp_swin_large.pth
```

### Training

Train your own REST model:
```bash
# Single GPU training
python tools/train.py configs/rest/rest_water_swin_large.py

# Multi-GPU training
bash tools/dist_train.sh configs/rest/rest_water_swin_large.py 4
```

## 📁 Model Zoo

We provide pre-trained models for different datasets and configurations:

| Model | Dataset | Backbone | mIoU | Config | Download |
|-------|---------|----------|------|--------|----------|
| REST | GLH-Water | Swin-Large | XX.X | [config](configs/rest/rest_water_swin_large.py) | [model](https://github.com/weichenrs/REST_code/releases/download/models/REST_water_swin_large.pth) |
| Baseline | Five-Billion-Pixels | Swin-Large | XX.X | [config](configs/baseline/baseline_fbp_swin_large.py) | [model](https://github.com/weichenrs/REST_code/releases/download/models-0.1/baseline_fbp_swin_large.pth) |

## 📖 Documentation

- [Installation Guide](INSTALL.md): Detailed installation instructions
- [Data Preparation](docs/DATA_PREPARATION.md): How to prepare your datasets
- [Training Guide](docs/TRAINING.md): Comprehensive training instructions
- [Evaluation Guide](docs/EVALUATION.md): Model evaluation and metrics
- [API Documentation](docs/API.md): Detailed API reference

## 🛠️ Project Structure

```
REST/
├── configs/                 # Configuration files
│   ├── _base_/             # Base configurations
│   ├── rest/               # REST model configs
│   └── baseline/           # Baseline model configs
├── data/                   # Dataset directory
├── docs/                   # Documentation
├── rest/                   # Main source code
│   ├── core/              # Core components
│   ├── models/            # Model definitions
│   ├── datasets/          # Dataset loaders
│   ├── utils/             # Utility functions
│   └── checkpoints/       # Model checkpoints
├── tools/                  # Training and testing scripts
└── assets/                # Images and resources
```

## 🔬 Technical Details

### Architecture

REST addresses the fundamental challenge of semantic segmentation for whole-scene remote sensing imagery, which is typically characterized by large image sizes that exceed GPU memory limits. Traditional methods resort to suboptimal cropping or fusion strategies, leading to performance degradation and loss of global context.

**Core Components:**

1. **Spatial Parallel Interaction Mechanism (SPIM)**: 
   - Combines parallel computation with divide-and-conquer strategy
   - Enables processing of large WRI while maintaining global context awareness
   - Overcomes GPU memory constraints without sacrificing performance

2. **Plug-and-Play Architecture**:
   - Compatible with mainstream semantic segmentation encoders and decoders
   - Supports integration with advanced foundation models
   - Flexible framework adaptable to various network architectures

3. **End-to-End Processing**:
   - First framework to achieve truly holistic segmentation without cropping
   - Direct optimization for entire scene understanding
   - Eliminates information loss from patch-based approaches

### Key Innovations

1. **Memory-Efficient Processing**: Unlike conventional methods limited by GPU memory, REST processes entire scenes through innovative parallel interaction mechanisms.

2. **Scalable Architecture**: Theoretical analysis and experiments demonstrate near-linear throughput scalability as additional GPUs are employed.

3. **Universal Compatibility**: Supports diverse imagery types (multispectral, hyperspectral) and platforms (satellite, drone) in a unified framework.

4. **Robust Performance**: Consistently outperforms cropping-based and fusion-based methods across single-class to multi-class segmentation scenarios.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 Citation

If you find REST useful in your research, please consider citing:

```bibtex
@article{rest2025,
  title={REST: Holistic Learning for End-to-End Semantic Segmentation of Whole-Scene Remote Sensing Imagery},
  author={Chen, Wei and Bruzzone, Lorenzo and Dang, Bo and Gao, Yuan and Deng, Youming and Yu, Jin-Gang and Yuan, Liangqi and Li, Yansheng},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  volume={},
  number={},
  pages={1-18},
  publisher={IEEE},
  doi={10.1109/TPAMI.2025.3609767}}
}
```

## 📞 Contact

- **Author**: [Wei Chen](mailto:weichenrs@whu.edu.cn)
- **Institution**: Wuhan University
- **Project Page**: [https://weichenrs.github.io/REST/](https://weichenrs.github.io/REST/)

## 🙏 Acknowledgements

This work is built upon several excellent open-source projects:
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) - Comprehensive segmentation toolbox
- [SkySense](https://github.com/Jack-bo1220/SkySense) - Powerful remote sensing foundation model SkySense
- [VMamba](https://github.com/MzeroMiko/VMamba) - VMamba implementation

We thank the remote sensing community for providing high-quality datasets and the authors of foundation models for enabling our plug-and-play architecture design.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
⭐ If you find this project helpful, please consider giving it a star! ⭐
</div>
