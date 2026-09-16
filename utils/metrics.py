import os
import numpy as np
import pandas as pd


def calculate_iou_and_f1(pred_mask, true_mask, num_classes=4, ignore_index=None):
    """
    Computes Intersection over Union (IoU) and F1-score per class.

    ignore_index: if set (e.g. 3, the no-data class -- see identify_class3.py),
    pixels labeled with that index are dropped from the computation entirely
    before scoring, and that class's IoU/F1 come back as NaN so nanmean()
    naturally excludes it from the overall average.
    """
    ious = []
    f1_scores = []

    pred_mask = pred_mask.view(-1)
    true_mask = true_mask.view(-1)

    if ignore_index is not None:
        valid = true_mask != ignore_index
        pred_mask = pred_mask[valid]
        true_mask = true_mask[valid]

    for cls in range(num_classes):
        if cls == ignore_index:
            ious.append(float('nan'))
            f1_scores.append(float('nan'))
            continue
        pred_inds = (pred_mask == cls)
        target_inds = (true_mask == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            ious.append(float('nan'))  # Class not present in this tile
            f1_scores.append(float('nan'))
        else:
            iou = intersection / union
            ious.append(iou)
            total_elements = pred_inds.sum().item() + target_inds.sum().item()
            f1 = (2.0 * intersection) / (total_elements + 1e-8)
            f1_scores.append(f1)

    return ious, f1_scores


def compute_class_distribution(folders, num_classes=4):
    """
    Reads every label_matrix_*.csv in `folders` and counts pixels per class.
    Used to:
      1. Check for class imbalance (e.g. does one class dominate, making a
         low loss trivially easy rather than a sign of good discrimination).
      2. Sanity-check that train/val splits have similar class distributions
         -- if val is missing a class entirely, its per-class IoU for that
         class will be meaningless (all-NaN).
      3. Suggest inverse-frequency class weights for AgriLoss.

    Imports AgriDataset's static band/mask matcher to stay consistent with
    how train.py/evaluate.py locate the mask file -- avoids re-implementing
    (and potentially diverging from) that matching logic here.

    Returns: np.ndarray of shape (num_classes,), raw pixel counts.
    """
    from utils.dataset import AgriDataset

    counts = np.zeros(num_classes, dtype=np.int64)
    for folder in folders:
        _, mask_file = AgriDataset._load_band_map(folder)
        if mask_file is None:
            continue
        arr = AgriDataset._load_mask_array(os.path.join(folder, mask_file))
        for c in range(num_classes):
            counts[c] += int((arr == c).sum())
    return counts


def suggest_class_weights(counts, eps=1e-8, ignore_index=None):
    """
    Simple inverse-frequency weighting: weight_c = total / (num_real_classes * count_c),
    normalized so weights average to 1 OVER THE REAL CLASSES ONLY. Feed the
    result into AgriLoss(weight=torch.tensor(weights)) if class imbalance
    turns out to be severe. Not applied automatically -- this only computes
    the suggestion.

    ignore_index: if set, that class is excluded from the total/mean used
    for normalization, and its own weight is returned as 0 (CrossEntropyLoss
    with ignore_index already skips it regardless, but 0 keeps the array
    self-consistent if it's ever inspected directly).
    """
    counts = counts.astype(np.float64)
    num_classes = len(counts)
    real_idx = [c for c in range(num_classes) if c != ignore_index]

    real_counts = counts[real_idx]
    total = real_counts.sum()
    raw = total / (len(real_idx) * (real_counts + eps))
    normalized_real = raw / raw.mean()

    weights = np.zeros(num_classes, dtype=np.float64)
    for i, c in enumerate(real_idx):
        weights[c] = normalized_real[i]
    return weights


def print_class_distribution(name, counts, class_names=None):
    total = counts.sum()
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(counts))]
    print(f"  {name} class distribution:")
    for i, c in enumerate(class_names):
        pct = 100.0 * counts[i] / total if total > 0 else 0.0
        print(f"    {c:<18}: {counts[i]:>12,d} px  ({pct:5.2f}%)")