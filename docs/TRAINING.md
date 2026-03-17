# Training Guide

> Reference repository: [https://github.com/weichenrs/REST_code](https://github.com/weichenrs/REST_code)

---

## Step 1 — Prepare Pretrained Backbone Weights

Before training, make sure the pretrained backbone referenced in your config's `pretrained` field is downloaded and accessible. Refer to the [MMSegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/) for guidance on setting pretrained paths.

---

## Step 2 — Edit `train.sh`

Open `train.sh` and adjust the following fields:

| Field | Description |
|---|---|
| Config file path | Path to your chosen config, e.g. `configs/xxmodel/yy.py` |
| Working directory | Output directory for logs and checkpoints |
| `pretrained` in config | Path or URL to the backbone pretrained weights |

---

## Step 3 — Launch Training

### Single-GPU Training

```bash
python tools/train.py configs/xxmodel/yy.py \
    --work-dir /path/to/work_dir
```

### Multi-GPU Distributed Training

```bash
bash tools/dist_train.sh configs/xxmodel/yy.py <NUM_GPUS> \
    --work-dir /path/to/work_dir
```

Replace `<NUM_GPUS>` with the number of GPUs available (e.g. `4`).

---

## Notes

- Logs and checkpoints are saved to the specified `--work-dir`.
- To resume training from a checkpoint, add `--resume` to the command.
- For full training options (mixed precision, custom hooks, etc.), refer to:  
  https://mmsegmentation.readthedocs.io/en/latest/
