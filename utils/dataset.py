import os
import torch
import tifffile
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.physics import extract_vi_pyramid


class AgriDataset(Dataset):
    """
    Loads 5-band multispectral tiles (Blue/Green/Red/RedEdge/NIR) plus a
    per-pixel class-index mask, and builds a multi-scale NDVI pyramid.

    Mask source: label_matrix_*.csv (NOT color_map_image.png).
    color_map_image.png is a colorized visualization of a continuous value
    (rendered at a different resolution, 354x354, and containing 400+ smoothly
    varying RGBA colors) -- it is not usable as a discrete class mask.
    label_matrix_*.csv is the real per-pixel class-index ground truth: its
    first row is a coordinate header (dropped on load), and the remaining
    512x512 grid of integers are class indices directly usable with
    CrossEntropyLoss.

    Can be constructed two ways:
      - AgriDataset(root_dir, ...)          -> scans root_dir for all folders
      - AgriDataset(root_dir, folder_list=[...]) -> uses exactly this list
        (e.g. the train or val split produced by utilsweed/splits.py). This is
        what train.py / evaluate.py should use so train and val never
        overlap.
    """

    def __init__(self, root_dir, num_classes=None, folder_list=None):
        if folder_list is not None:
            all_folders = folder_list
        else:
            all_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))]

        # Filter out folders missing required files ONCE, up front.
        # This avoids returning fake all-zero samples mid-training, which
        # would silently corrupt the loss/gradients for those batches.
        self.folders = []
        for folder in all_folders:
            if self._has_required_files(folder):
                self.folders.append(folder)
            else:
                print(f"⚠️ Skipping folder (missing files): {folder}")

        if len(self.folders) == 0:
            raise RuntimeError(
                f"No valid folders found under {root_dir}. "
                f"Check that band files (blue/green/red/rededge/nir .tif) "
                f"and a label_matrix_*.csv mask exist in each subfolder."
            )

        print(f"✅ Loaded dataset: {len(self.folders)} valid samples "
              f"({len(all_folders) - len(self.folders)} skipped)")

        # Optional: sanity-check that the CSV masks don't contain more
        # classes than the modelweed expects. Off by default since scanning
        # every mask up front costs time; enable if you want the safety net.
        if num_classes is not None:
            self._validate_class_range(num_classes)

    @staticmethod
    def _load_mask_array(mask_path, expected_size=512):
        """
        Loads a label_matrix_*.csv into a (expected_size, expected_size)
        int64 array.

        Different folders in this dataset disagree on whether the CSV has a
        leading coordinate-header row: the main training folders do (513
        total rows: 1 header + 512 data), but the 2021 Test folders don't
        (512 total rows, no header) -- using pd.read_csv()'s default
        "row 0 is always a header" behavior silently eats a real data row
        on files that don't have one, producing a 511-row mask that then
        crashes downstream against a 512x512 prediction.

        Fix: read with header=None (nothing assumed dropped), then trim any
        extra leading rows/cols down to expected_size. Works for both cases:
        513 rows -> trims the 1 header row; 512 rows -> trims 0.
        """
        raw = pd.read_csv(mask_path, header=None).values
        h, w = raw.shape

        if h < expected_size or w < expected_size:
            raise ValueError(
                f"Mask {mask_path} has shape {raw.shape}, smaller than the "
                f"expected {expected_size}x{expected_size} -- can't safely trim."
            )

        if h > expected_size:
            raw = raw[h - expected_size:, :]
        if w > expected_size:
            raw = raw[:, w - expected_size:]

        return raw.astype(np.int64)

    def _validate_class_range(self, num_classes):
        max_seen = -1
        for folder in self.folders:
            _, mask_file = self._load_band_map(folder)
            arr = self._load_mask_array(os.path.join(folder, mask_file))
            max_seen = max(max_seen, int(arr.max()))
        if max_seen >= num_classes:
            raise ValueError(
                f"Mask contains class index {max_seen}, but num_classes={num_classes}. "
                f"Update num_classes to {max_seen + 1} in IGMambaUNet."
            )

    @staticmethod
    def _has_required_files(folder):
        files = os.listdir(folder)
        band_map = {'blue': None, 'green': None, 'red': None, 'rededge': None, 'nir': None}
        mask_file = None

        for f in files:
            f_lower = f.lower()
            if 'blue' in f_lower and f.endswith('.tif'):
                band_map['blue'] = f
            elif 'green' in f_lower and f.endswith('.tif'):
                band_map['green'] = f
            elif 'rededge' in f_lower and f.endswith('.tif'):
                band_map['rededge'] = f
            elif 'red' in f_lower and f.endswith('.tif'):
                band_map['red'] = f
            elif 'nir' in f_lower and f.endswith('.tif'):
                band_map['nir'] = f
            elif 'label_matrix' in f_lower and f.endswith('.csv'):
                mask_file = f

        return None not in band_map.values() and mask_file is not None

    @staticmethod
    def _load_band_map(folder):
        files = os.listdir(folder)
        band_map = {'blue': None, 'green': None, 'red': None, 'rededge': None, 'nir': None}
        mask_file = None

        for f in files:
            f_lower = f.lower()
            if 'blue' in f_lower and f.endswith('.tif'):
                band_map['blue'] = f
            elif 'green' in f_lower and f.endswith('.tif'):
                band_map['green'] = f
            elif 'rededge' in f_lower and f.endswith('.tif'):
                band_map['rededge'] = f
            elif 'red' in f_lower and f.endswith('.tif'):
                band_map['red'] = f
            elif 'nir' in f_lower and f.endswith('.tif'):
                band_map['nir'] = f
            elif 'label_matrix' in f_lower and f.endswith('.csv'):
                mask_file = f

        return band_map, mask_file

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        folder = self.folders[idx]
        band_map, mask_file = self._load_band_map(folder)

        # Load 5 bands
        img_stack = []
        for key in ['blue', 'green', 'red', 'rededge', 'nir']:
            path = os.path.join(folder, band_map[key])
            band = tifffile.imread(path).astype(np.float32)
            # UAV orthomosaic exports commonly use NaN as a "nodata" marker
            # for pixels outside flight coverage / stitching gaps. Left
            # unhandled, a single NaN pixel poisons NDVI and the loss for
            # the whole batch. Replace with 0 (treated as bare/no-signal).
            if np.isnan(band).any() or np.isinf(band).any():
                band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
            img_stack.append(band)

        img = torch.from_numpy(np.stack(img_stack, axis=0)) / 255.0

        # Load mask from label_matrix_*.csv. Some folders have a leading
        # coordinate-header row, some don't -- _load_mask_array detects and
        # strips it dynamically so both formats land on 512x512. See its
        # docstring for why this matters (2021 Test folders lack a header).
        mask_path = os.path.join(folder, mask_file)
        mask_arr = self._load_mask_array(mask_path)
        mask = torch.from_numpy(mask_arr).long()

        # Physics: build the NDVI pyramid via the VI extractor module.
        # (img assumed 512x512 input -> stem uses stride-4 patchify, so
        # pool factors 4/8/16/32 match encoder stage resolutions
        # 128/64/32/16 -- see utilsweed/physics.py for details.)
        nir, red = img[4], img[2]
        vi_pyr = extract_vi_pyramid(nir, red, pool_factors=(4, 8, 16, 32))
        return img, mask, vi_pyr