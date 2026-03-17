# Data Preparation

---

## Option A — Quick-Start: Download Packaged Data

Pre-packaged archives are provided for getting started immediately:

```bash
# General segmentation data
wget https://github.com/weichenrs/REST_code/releases/download/data/data.zip

# Water segmentation data
wget https://github.com/weichenrs/REST_code/releases/download/data/waterdata.zip
```

Extract both archives into the project root:

```bash
unzip data.zip
unzip waterdata.zip
```

A set of example images is also included under `data/FBP_new/` for quickly verifying your environment before downloading the full datasets.

---

## Option B — Full Public Datasets

The following publicly available remote sensing datasets are used in this study:

| Dataset | Task | URL |
|---|---|---|
| **GLH-Water** | Water body segmentation | https://jack-bo1220.github.io/project/GLH-water.html |
| **Five-Billion-Pixels (FBP)** | Land cover segmentation | https://x-ytong.github.io/project/Five-Billion-Pixels.html |
| **WHU-OHS** | Hyperspectral segmentation | http://irsip.whu.edu.cn/resources/WHU_OHS_show.php |
| **UAVid** | UAV scene segmentation | https://uavid.nl/ |

Download each dataset from the links above and place them under the `data/` directory.

---

## Recommended Directory Layout

```
rest/
├── data/
│   ├── FBP_new/        # Five-Billion-Pixels examples / full dataset
│   ├── GLH-Water/      # GLH-Water dataset
│   ├── WHU-OHS/        # WHU-OHS dataset
│   └── UAVid/          # UAVid dataset
├── configs/
├── checkpoints/
├── tools/
└── ...
```

---

## Update Dataset Paths in Config

After placing datasets in `data/`, open the dataset config file and update the `data_root` fields to match your local paths:

```
configs/_base_/datasets/zzzdataset.py
```

Each dataset entry has a `data_root` key — set it to the actual directory path on your machine.
