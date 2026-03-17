# Installation Guide

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 |
| CUDA | 12.1 |
| PyTorch | 2.1.1 |
| torchvision | 0.16.1 |
| torchaudio | 2.1.1 |

---

## Step 1 — Create and Activate Conda Environment

```bash
conda create -n rest python=3.9 -y
conda activate rest
```

---

## Step 2 — Unzip and Enter the Project Directory

```bash
unzip REST_code.zip
cd rest
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
mim install mmpretrain mmengine mmdet
```

---

## Step 4 — Set Script Permissions

```bash
chmod 777 tools/dist_test.sh
chmod 777 tools/dist_train.sh
```

---

## Step 5 — Install VMamba (Selective Scan Kernel)

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba
pip install -r requirements.txt

# Build and install the selective scan CUDA kernel
cd kernels/selective_scan && pip install .
```

After this step, return to the `rest/` project root before proceeding.

---

## References

- MMSegmentation docs: https://mmsegmentation.readthedocs.io/en/latest/
- VMamba repository: https://github.com/MzeroMiko/VMamba
