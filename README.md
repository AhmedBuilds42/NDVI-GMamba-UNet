# NDVI-GMamba-UNet

### Multi-Scale NDVI-Guided Mamba U-Net for Multispectral Crop–Weed Segmentation

NDVI-GMamba-UNet is a multispectral semantic segmentation framework designed for crop–weed segmentation in agricultural imagery.

The model combines a **U-Net encoder–decoder architecture**, **Mamba-based sequence modeling**, and **multi-scale NDVI guidance**. NDVI information is introduced at multiple encoder resolutions to provide vegetation-related spectral information to the feature representation.

---

## Overview

Accurate crop–weed segmentation can be challenging when RGB information alone is insufficient to distinguish vegetation with similar visual appearance.

This project investigates whether **multispectral information**, particularly the Normalized Difference Vegetation Index (NDVI), can guide feature extraction at multiple spatial scales.

The main components are:

* 5-band multispectral input
* Multi-scale NDVI extraction
* NDVI-guided feature gating
* Mamba sequence modeling
* U-Net-style skip connections
* Cross-scale encoder–decoder processing
* Combined Cross-Entropy and Dice loss

---

## Architecture

The network receives five spectral bands:

```text
Blue
Green
Red
Red Edge
NIR
```

The input is processed using a hierarchical encoder:

```text
5 × 512 × 512
        │
        ▼
   Stem Conv
        │
        ▼
96 × 128 × 128
        │
   NDVI-Guided
    Mamba Block
        │
        ▼
192 × 64 × 64
        │
   NDVI-Guided
    Mamba Block
        │
        ▼
384 × 32 × 32
        │
   NDVI-Guided
    Mamba Block
        │
        ▼
768 × 16 × 16
        │
   NDVI-Guided
    Mamba Block
        │
        ▼
     Decoder
        │
        ▼
Segmentation Map
```

The encoder features are passed to the decoder through U-Net skip connections.

---

## NDVI Guidance

NDVI is calculated from the Near-Infrared and Red bands:

$$
NDVI = \frac{NIR - Red}{NIR + Red + \epsilon}
$$

where:

* `NIR` is the near-infrared band
* `Red` is the red band
* $\epsilon$ is a small numerical stability constant

A multi-scale NDVI pyramid is generated:

```text
512 × 512 NDVI
      │
      ├── 4× pooling  → 128 × 128
      ├── 8× pooling  → 64 × 64
      ├── 16× pooling → 32 × 32
      └── 32× pooling → 16 × 16
```

Each NDVI scale is matched with the corresponding encoder stage.

---

## NDVI-Guided Mamba Block

Each encoder stage uses an NDVI-guided feature modulation mechanism.

The NDVI feature is projected into the feature dimension and converted into a gating signal:

$$
G = \sigma(\gamma W_{VI}(VI))
$$

The normalized feature representation is then modulated by the gate:

$$
X' = LN(X) \odot G
$$

When the Mamba implementation is available, the gated representation is processed using a residual Mamba block:

$$
Y = X' + Mamba(X')
$$

This allows the model to combine vegetation-related spectral guidance with sequence-based feature modeling.

---

## Dataset

The project uses a multispectral agricultural image benchmark containing image tiles with five spectral bands:

| Band | Description   |
| ---- | ------------- |
| B    | Blue          |
| G    | Green         |
| R    | Red           |
| RE   | Red Edge      |
| NIR  | Near Infrared |

The dataset contains approximately **783 image tiles**, with each tile having a spatial size of:

```text
512 × 512 pixels
```

Ground-truth segmentation masks are provided as pixel-wise class-index matrices.

The repository does **not** include the dataset itself.

Please obtain the dataset from its original source and place it in the appropriate local data directory.

---
## Results

The current model was evaluated on the validation split using pixel-level semantic segmentation metrics. Class 3 is excluded from evaluation because it is treated as a no-data/void class.

| Class             |        IoU (%) |            F1 (%) |
| ----------------- | -------------: | ----------------: |
| Background / Soil |          98.20 |             99.08 |
| Crop              |          81.63 |             89.81 |
| Weed              |          86.05 |             92.29 |
| **Mean**          | **88.67 mIoU** | **93.75 Mean F1** |

### Evaluation Summary

* Total valid samples: **783**
* Training samples: **627**
* Evaluation samples: **156**
* Image size: **512 × 512**
* Input: **5-band multispectral imagery**
* Evaluated classes: **Background/Soil, Crop, Weed**
* Ignored class: **Class 3 (no-data/void)**

The current results are preliminary and will be finalized after using an identical, explicitly saved train/validation split for training and evaluation.

## Dataset Structure

The expected local structure is approximately:

```text
data/
└── Multispectral Image Benchmark Dataset/
    └── ImageData_Split/
        └── ImageData_Split/
            ├── sample_folder_1/
            │   ├── blue.tif
            │   ├── green.tif
            │   ├── red.tif
            │   ├── rededge.tif
            │   ├── nir.tif
            │   └── label_matrix_*.csv
            │
            ├── sample_folder_2/
            └── ...
```

The exact folder and file names should match the dataset provided by the original dataset source.

---

## Classes

The current implementation uses four output channels.

The fourth label is treated as a **void/no-data class** and is ignored during training loss and evaluation.

Therefore, the supervised/evaluated classes are currently:

```text
Class 0
Class 1
Class 2
```

with:

```text
Class 3 → ignored / no-data
```

The exact semantic names of the classes should be confirmed against the official dataset documentation before interpreting them as specific crop, weed, or soil categories.

---

## Preprocessing

The multispectral bands are:

1. Loaded from TIFF files.
2. Converted to `float32`.
3. Invalid `NaN` and infinite values are replaced with zero.
4. The five bands are stacked into a five-channel tensor.
5. The input values are scaled by `255.0`.
6. NDVI is calculated from the Red and NIR bands.
7. Multi-scale NDVI maps are generated for the four encoder stages.

No pretrained ImageNet weights are used in the current implementation.

---

## Loss Function

The training objective combines Cross-Entropy loss and class-wise Dice loss:

$$
L = L_{CE} + L_{Dice}
$$

The no-data class is excluded using:

```text
ignore_index = 3
```

The Dice component is calculated over the supervised classes.

---

## Training

Current training configuration:

| Parameter         |                Value |
| ----------------- | -------------------: |
| Input size        |            512 × 512 |
| Input bands       |                    5 |
| Output channels   |                    4 |
| Epochs            |                   20 |
| Batch size        |                    8 |
| Optimizer         |                AdamW |
| Learning rate     |                0.001 |
| Validation ratio  |                 0.30 |
| Split seed        |                   42 |
| Gradient clipping |                  1.0 |
| Checkpoint        | Best validation loss |

Run training with:

```bash
python train.py
```

The best model is saved as:

```text
best_model.pth
```

Model checkpoints are intentionally excluded from the Git repository.

---

## Evaluation

Evaluation can be performed using:

```bash
python evaluate.py
```

The evaluation script reports:

* Per-class IoU
* Per-class F1 score
* Mean IoU
* Mean F1 score
* Class distribution

The no-data class is excluded from the reported segmentation metrics.

---

## Additional Utilities

The repository also contains utilities for dataset analysis and evaluation.

```text
analyzedataset.py
```

Used for inspecting the dataset.

```text
identify class3.py
```

Used to investigate the relationship between class 3 and invalid/no-data pixels.

```text
visualize_predictions.py
```

Used for visualizing segmentation predictions.

```text
run within plot.py
```

Provides a spatially structured splitting strategy.

```text
run cross year.py
```

Provides a cross-year evaluation/training workflow.

```text
debug.py
```

Contains debugging and inspection utilities.

---

## Repository Structure

```text
NDVI-GMamba-UNet/
│
├── model/
│   ├── __init__.py
│   ├── ig_mamba.py
│   ├── main.py
│   └── unet_arch.py
│
├── utils/
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   ├── physics.py
│   └── splits.py
│
├── train.py
├── evaluate.py
├── test.py
├── debug.py
├── analyzedataset.py
├── identify class3.py
├── visualize_predictions.py
├── run within plot.py
├── run cross year.py
├── requirements.txt
└── .gitignore
```

---

## Requirements

The main dependencies are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

For the real Mamba implementation, the required Mamba dependencies must also be installed successfully.

If `mamba_ssm` is unavailable, the current implementation can fall back to a non-Mamba feature-gating mode. Results obtained in that fallback mode should **not** be described as Mamba-based results.

---

## Reproducibility

The current experiments use a fixed split seed:

```text
seed = 42
```

and deterministic folder ordering before the split.

For more rigorous experimental comparison, future experiments should additionally control the global Python, NumPy, and PyTorch random seeds and save the exact train/validation folder lists.

---

## Current Status

This repository contains the implementation and experimental pipeline for NDVI-guided Mamba-based multispectral segmentation.

### Completed

* [x] Multispectral dataset loader
* [x] NDVI computation
* [x] Multi-scale NDVI pyramid
* [x] NDVI-guided feature gating
* [x] Mamba integration
* [x] U-Net encoder–decoder
* [x] Combined CE + Dice loss
* [x] IoU and F1 evaluation
* [x] Dataset analysis utilities
* [x] Spatial splitting utilities
* [x] Cross-year splitting utilities

### In progress

* [ ] Final benchmark experiments
* [ ] Spatially disjoint evaluation
* [ ] Cross-year evaluation
* [ ] Ablation experiments
* [ ] Qualitative prediction analysis
* [ ] Final performance reporting

---

## Limitations

Several aspects of the current implementation require further experimental validation:

1. The dataset is spatially structured, so random tile-level splitting may not fully measure generalization to unseen spatial regions.
2. The exact preprocessing assumptions should be verified against the original multispectral data metadata.
3. Mamba-based results depend on successful installation and use of `mamba_ssm`.
4. The current training and evaluation scripts use separate split configurations and should use an identical saved split for strict reproducibility.
5. Final performance should be reported only after the evaluation protocol has been fixed.

---

## Author

**Ahmed Abdullah**

BS Artificial Intelligence
HITEC University, Taxila, Pakistan

GitHub: [@AhmedBuilds42](https://github.com/AhmedBuilds42)
