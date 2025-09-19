# Installation Guide

This document provides detailed installation instructions for the REST project.

## 📋 Requirements

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04/20.04/22.04 recommended)
- **Python**: 3.9+
- **CUDA**: 11.2+ (for GPU support)
- **GPU Memory**: 8GB+ recommended for training
- **Disk Space**: 50GB+ (for datasets and models)

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 16GB | 32GB+ |
| **GPU** | GTX 1080Ti (11GB) | RTX 3090/4090 (24GB+) |
| **Storage** | 50GB free space | 100GB+ SSD |

## 🐍 Environment Setup

### Option 1: Conda Environment (Recommended)

1. **Create and activate conda environment**:
   ```bash
   conda create -n rest python=3.9 -y
   conda activate rest
   ```

2. **Install PyTorch and related packages**:
   ```bash
   # For CUDA 12.1
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
   
   # For CUDA 11.8 (alternative)
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
   
   # For CPU-only installation
   pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install OpenMMLab packages**:
   ```bash
   # Install MMCV
   pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
   
   # Install OpenMMLab tools
   pip install -U openmim
   mim install mmpretrain mmengine mmdet
   ```

4. **Install additional dependencies**:
   ```bash
   pip install ftfy regex timm einops prettytable fvcore
   pip install numpy==1.23.0
   pip install opencv-python pillow matplotlib seaborn
   pip install tensorboard wandb  # For logging (optional)
   ```

### Option 2: Docker Environment

We provide a Docker image for easy setup:

```bash
# Pull the Docker image
docker pull your-registry/rest:latest

# Run the container
docker run -it --gpus all -v /path/to/your/data:/workspace/data your-registry/rest:latest

# Or build from Dockerfile
docker build -t rest:latest .
docker run -it --gpus all -v /path/to/your/data:/workspace/data rest:latest
```

## 📦 Project Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/weichenrs/REST_code.git
   cd REST_code
   
   # Unzip if you downloaded the zip file
   unzip rest.zip
   cd rest
   ```

2. **Set executable permissions**:
   ```bash
   chmod 777 tools/dist_test.sh
   chmod 777 tools/dist_train.sh
   ```

3. **Install VMamba dependency**:
   ```bash
   git clone https://github.com/MzeroMiko/VMamba.git
   cd VMamba
   pip install -r requirements.txt
   cd kernels/selective_scan && pip install .
   cd ../../..  # Return to main directory
   ```

4. **Install REST package (optional)**:
   ```bash
   pip install -e .
   ```

## 📊 Data and Model Setup

### Download Data

1. **Download datasets**:
   ```bash
   # Create data directory
   mkdir -p data
   
   # Download and extract data
   wget https://github.com/weichenrs/REST_code/releases/download/data/data.zip
   wget https://github.com/weichenrs/REST_code/releases/download/data/waterdata.zip
   
   unzip data.zip
   unzip waterdata.zip
   ```

2. **Download pre-trained models**:
   ```bash
   # Method 1: Using Python script (recommended)
   cd rest/checkpoints
   python download_model.py
   
   # Method 2: Using wget
   wget https://github.com/weichenrs/REST_code/releases/download/models/REST_water_swin_large.pth -O rest/checkpoints/REST_water_swin_large.pth
   wget https://github.com/weichenrs/REST_code/releases/download/models-0.1/baseline_fbp_swin_large.pth -O rest/checkpoints/baseline_fbp_swin_large.pth
   
   # Method 3: Direct download from GitHub releases
   # Visit: https://github.com/weichenrs/REST_code/releases
   ```

### Verify Data Structure

After setup, your directory structure should look like:

```
REST_code/
├── data/
│   ├── FBP_new/                 # Five-Billion-Pixels sample data
│   ├── GLH-Water/               # GLH-Water dataset (if downloaded)
│   ├── WHU-OHS/                 # WHU-OHS dataset (if downloaded)
│   └── UAVid/                   # UAVid dataset (if downloaded)
├── rest/
│   ├── checkpoints/
│   │   ├── REST_water_swin_large.pth
│   │   ├── baseline_fbp_swin_large.pth
│   │   └── download_model.py
│   ├── configs/
│   ├── core/
│   ├── datasets/
│   ├── models/
│   └── utils/
├── tools/
│   ├── train.py
│   ├── test.py
│   ├── dist_train.sh
│   └── dist_test.sh
└── VMamba/
```

## ✅ Verification

### Test Installation

1. **Check Python environment**:
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import mmcv; print(f'MMCV version: {mmcv.__version__}')"
   ```

2. **Test model loading**:
   ```bash
   cd rest
   python -c "
   import torch
   model_path = 'checkpoints/REST_water_swin_large.pth'
   checkpoint = torch.load(model_path, map_location='cpu')
   print('Model loaded successfully!')
   print(f'Model keys: {list(checkpoint.keys())}')
   "
   ```

3. **Run a quick test**:
   ```bash
   # Test with sample data
   python tools/test.py configs/rest/rest_water_swin_large.py checkpoints/REST_water_swin_large.pth --show-dir results/
   ```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Version Mismatch**:
   ```bash
   # Check CUDA version
   nvcc --version
   nvidia-smi
   
   # Install matching PyTorch version
   # Visit: https://pytorch.org/get-started/locally/
   ```

2. **MMCV Installation Failed**:
   ```bash
   # Try installing without pre-built wheels
   pip install mmcv==2.1.0 --no-binary mmcv
   
   # Or build from source
   git clone https://github.com/open-mmlab/mmcv.git
   cd mmcv
   pip install -e .
   ```

3. **Permission Denied**:
   ```bash
   # Fix script permissions
   chmod +x tools/*.sh
   
   # Or run with bash explicitly
   bash tools/dist_train.sh
   ```

4. **Out of Memory (OOM)**:
   - Reduce batch size in config files
   - Use gradient accumulation
   - Try mixed precision training

### Environment-Specific Issues

#### Ubuntu 18.04/20.04
```bash
# Install build essentials
sudo apt update
sudo apt install build-essential git curl wget

# Install CUDA (if needed)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

#### macOS (CPU-only)
```bash
# Install via Homebrew
brew install python@3.9
pip install torch torchvision torchaudio

# Note: GPU training not supported on macOS
```

#### Windows (with WSL2 recommended)
```bash
# Use Windows Subsystem for Linux 2
# Follow Ubuntu installation steps within WSL2
```

## 📚 Additional Resources

- **PyTorch Installation**: https://pytorch.org/get-started/locally/
- **MMSegmentation Docs**: https://mmsegmentation.readthedocs.io/
- **CUDA Installation**: https://developer.nvidia.com/cuda-downloads
- **Conda Environments**: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

## 🆘 Getting Help

If you encounter issues during installation:

1. **Check the logs**: Most error messages provide helpful information
2. **Search existing issues**: Check our [GitHub Issues](https://github.com/weichenrs/REST_code/issues)
3. **Create a new issue**: Provide detailed error messages and system information
4. **Join our community**: [Discord/Slack link] for real-time help

## 🎯 Next Steps

After successful installation:

1. **Quick Start**: Follow the [README.md](README.md) for basic usage
2. **Data Preparation**: See [docs/DATA_PREPARATION.md] for dataset setup
3. **Training**: Check [docs/TRAINING.md] for training instructions
4. **Evaluation**: Refer to [docs/EVALUATION.md] for model evaluation

---

**Installation completed successfully? 🎉 You're ready to use REST!**