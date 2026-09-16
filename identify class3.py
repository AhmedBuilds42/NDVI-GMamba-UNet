"""
Standalone diagnostic: what IS class 3?

Run this directly (not part of the training pipeline). It samples folders
from the main training set and checks three independent signals for
class-3 pixels:

  1. Border proximity -- is class 3 concentrated near tile edges? That
     pattern usually means padding / no-data, not a real semantic class.
  2. Overlap with raw NaN pixels -- does class 3 line up with the NaN
     values found in Bug 6 (raw .tif nodata markers)? If so, class 3 may
     literally BE the dataset's own no-data label, baked into the mask.
  3. Spectral signature -- mean/std of each raw band for class-3 pixels
     vs. classes 0/1/2. Near-zero across all bands = no-data. A distinct
     but non-zero signature = a real class (open water, structures, etc).

Usage: edit root_dir / n_folders below, then run with the project's venv:
    python identify_class3.py
"""
import os
import numpy as np
import pandas as pd
import tifffile

from utils.dataset import AgriDataset


def analyze(root_dir, n_folders=60, border_px=5):
    all_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
    valid = [f for f in all_folders if AgriDataset._has_required_files(f)]
    sample = valid[:n_folders]
    print(f"Analyzing {len(sample)} folders out of {len(valid)} valid folders...\n")

    border_hits = 0
    class3_total = 0
    nan_overlap = 0
    nan_total_class3 = 0

    band_sums = {c: np.zeros(5, dtype=np.float64) for c in range(4)}
    band_sqsums = {c: np.zeros(5, dtype=np.float64) for c in range(4)}
    band_counts = {c: 0 for c in range(4)}

    band_keys = ['blue', 'green', 'red', 'rededge', 'nir']

    for folder in sample:
        band_map, mask_file = AgriDataset._load_band_map(folder)
        mask = AgriDataset._load_mask_array(os.path.join(folder, mask_file))
        H, W = mask.shape

        # --- Border proximity check ---
        border_mask = np.zeros_like(mask, dtype=bool)
        border_mask[:border_px, :] = True
        border_mask[-border_px:, :] = True
        border_mask[:, :border_px] = True
        border_mask[:, -border_px:] = True

        c3_mask = (mask == 3)
        class3_total += c3_mask.sum()
        border_hits += (c3_mask & border_mask).sum()

        # --- Load raw bands (RAW -- do NOT nan_to_num here, we want to see
        # the actual NaN locations before the training pipeline sanitizes them) ---
        raw_bands = []
        any_nan = np.zeros((H, W), dtype=bool)
        for key in band_keys:
            path = os.path.join(folder, band_map[key])
            band = tifffile.imread(path).astype(np.float32)
            raw_bands.append(band)
            any_nan |= np.isnan(band) | np.isinf(band)

        nan_overlap += (c3_mask & any_nan).sum()
        nan_total_class3 += c3_mask.sum()

        # --- Spectral signature per class (sanitized bands, so stats aren't NaN) ---
        for c in range(4):
            cls_mask = (mask == c)
            if not cls_mask.any():
                continue
            for bi, band in enumerate(raw_bands):
                clean = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
                vals = clean[cls_mask]
                band_sums[c][bi] += vals.sum()
                band_sqsums[c][bi] += (vals ** 2).sum()
            band_counts[c] += cls_mask.sum()

    print("=" * 60)
    print("1. BORDER PROXIMITY")
    print("=" * 60)
    pct_border = 100.0 * border_hits / max(class3_total, 1)
    print(f"  Class-3 pixels within {border_px}px of tile edge: {pct_border:.2f}%")
    print(f"  (If this is high, e.g. >30-50%, class 3 is likely padding/no-data)")

    print("\n" + "=" * 60)
    print("2. OVERLAP WITH RAW NaN PIXELS")
    print("=" * 60)
    pct_nan = 100.0 * nan_overlap / max(nan_total_class3, 1)
    print(f"  Class-3 pixels that coincide with a raw NaN/Inf band value: {pct_nan:.2f}%")
    print(f"  (If this is high, class 3 may literally be the dataset's own no-data label)")

    print("\n" + "=" * 60)
    print("3. SPECTRAL SIGNATURE (mean ± std per band, sanitized values)")
    print("=" * 60)
    class_names = ["Class 0 (Background/Soil)", "Class 1 (Crop)", "Class 2 (Weed)", "Class 3 (unknown)"]
    for c in range(4):
        n = band_counts[c]
        if n == 0:
            print(f"  {class_names[c]}: no pixels in sample")
            continue
        means = band_sums[c] / n
        stds = np.sqrt(np.maximum(band_sqsums[c] / n - means ** 2, 0))
        print(f"  {class_names[c]} (n={n:,} px):")
        for bi, key in enumerate(band_keys):
            print(f"    {key:<8}: {means[bi]:8.2f} ± {stds[bi]:6.2f}")

    print("\n" + "=" * 60)
    print("INTERPRETATION GUIDE")
    print("=" * 60)
    print("  - High border %% + high NaN overlap %% -> class 3 is almost certainly")
    print("    no-data/padding baked into the mask, not a real class. Consider")
    print("    excluding it from loss/metrics rather than treating it as 'Weed'-adjacent.")
    print("  - Low border %%, low NaN overlap, but near-zero across ALL bands ->")
    print("    likely a masked/black region for another reason (e.g. shadow, water absorbing NIR).")
    print("  - Distinct non-zero signature, different from classes 0-2 -> probably")
    print("    a real 4th semantic class (e.g. open water, bare structures/roads).")


if __name__ == "__main__":
    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    analyze(root_dir, n_folders=60, border_px=5)