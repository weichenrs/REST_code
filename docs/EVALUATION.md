# Evaluation Guide

\---

## Step 1 — Download Pretrained Model Checkpoints

Three methods are available:

### Method A — Python Script (Recommended)

```bash
python checkpoints/download\_model.py
```

### Method B — wget

```bash
wget https://github.com/weichenrs/REST\_code/releases/download/models/REST\_water\_swin\_large.pth \\
     -O checkpoints/REST\_water\_swin\_large.pth

wget https://github.com/weichenrs/REST\_code/releases/download/models-0.1/baseline\_fbp\_swin\_large.pth \\
     -O checkpoints/baseline\_fbp\_swin\_large.pth
```

### Method C — Direct Browser Download

Visit the URLs above and save the `.pth` files to the `checkpoints/` directory manually.

\---

## Checkpoint Summary

|Model|Dataset|Filename|
|-|-|-|
|REST (Swin-Large)|GLH-Water|`REST\_water\_swin\_large.pth`|
|Baseline (Swin-Large)|Five-Billion-Pixels|`baseline\_fbp\_swin\_large.pth`|

\---

## Step 2 — Edit `test.sh`

Open `test.sh` and verify the following fields:

|Field|Description|
|-|-|
|Config file path|Must match the model being evaluated|
|Checkpoint path|Path to the downloaded `.pth` file|
|Visualization output path|Directory to save prediction visualizations|

\---

## Step 3 — Launch Evaluation

### Single-GPU Evaluation

```bash
python tools/test.py configs/xxmodel/yy.py \\
    checkpoints/REST\_water\_swin\_large.pth \\
    --show-dir /path/to/vis\_output
```

### Multi-GPU Distributed Evaluation

```bash
bash tools/dist\_test.sh configs/xxmodel/yy.py \\
    checkpoints/REST\_water\_swin\_large.pth \\
    <NUM\_GPUS>
```

\---

## Metrics

REST reports standard semantic segmentation metrics as implemented in MMSegmentation:

|Metric|Description|
|-|-|
|**mIoU**|Mean Intersection over Union across all classes|
|**mAcc**|Mean per-class pixel accuracy|
|**aAcc**|Overall pixel accuracy|

Results are printed to the terminal and saved in the working directory upon completion.

\---

## References

* MMSegmentation documentation: https://mmsegmentation.readthedocs.io/en/latest/
* REST\_code model releases: https://github.com/weichenrs/REST\_code/releases

