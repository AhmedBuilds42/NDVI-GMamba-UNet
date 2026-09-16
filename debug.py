"""
Run this once to find what's causing NaN loss.
Edit `folder` to point at any one valid sample folder.
"""
import os
import numpy as np
import pandas as pd
import tifffile

folder = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split\smalldata_10_1"

print("=" * 60)
print("1. Raw band value ranges (before any normalization)")
print("=" * 60)
band_files = {
    'blue': [f for f in os.listdir(folder) if 'blue' in f.lower() and f.endswith('.tif')][0],
    'green': [f for f in os.listdir(folder) if 'green' in f.lower() and f.endswith('.tif')][0],
    'red': [f for f in os.listdir(folder) if 'red' in f.lower() and 'rededge' not in f.lower() and f.endswith('.tif')][0],
    'rededge': [f for f in os.listdir(folder) if 'rededge' in f.lower() and f.endswith('.tif')][0],
    'nir': [f for f in os.listdir(folder) if 'nir' in f.lower() and f.endswith('.tif')][0],
}

bands = {}
for name, fname in band_files.items():
    arr = tifffile.imread(os.path.join(folder, fname))
    bands[name] = arr.astype(np.float32)
    print(f"{name:8s} dtype={arr.dtype!s:10s} min={arr.min():>10.2f} max={arr.max():>10.2f} "
          f"mean={arr.mean():>10.2f} has_nan={np.isnan(arr.astype(np.float32)).any()} "
          f"has_inf={np.isinf(arr.astype(np.float32)).any()}")

print()
print("=" * 60)
print("2. NDVI check: (nir - red) / (nir + red + eps)")
print("=" * 60)
nir, red = bands['nir'], bands['red']
denom = nir + red + 1e-8
ndvi = (nir - red) / denom
print(f"denom min={denom.min():.6f}  (should never be ~0 unless both bands are 0)")
print(f"ndvi  min={ndvi.min():.4f} max={ndvi.max():.4f} "
      f"has_nan={np.isnan(ndvi).any()} has_inf={np.isinf(ndvi).any()}")

print()
print("=" * 60)
print("3. Mask (label_matrix csv) check")
print("=" * 60)
mask_file = [f for f in os.listdir(folder) if 'label_matrix' in f.lower() and f.endswith('.csv')][0]
mask_df = pd.read_csv(os.path.join(folder, mask_file))
mask_arr = mask_df.values
print(f"shape={mask_arr.shape} dtype={mask_arr.dtype}")
print(f"unique values: {np.unique(mask_arr)}")
print(f"has_nan={pd.isna(mask_arr).any()}")

print()
print("=" * 60)
print("4. What img/255.0 normalization currently produces")
print("=" * 60)
for name, arr in bands.items():
    normed = arr / 255.0
    print(f"{name:8s} normed min={normed.min():.4f} max={normed.max():.4f}  "
          f"{'>>> OUT OF [0,1] RANGE <<<' if normed.max() > 1.5 or normed.min() < -0.5 else 'looks ok'}")