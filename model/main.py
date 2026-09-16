"""
Determine which physical site (Beverley vs Kojonup) each tile belongs to,
using the coordinate header row already present in label_matrix_*.csv --
NOT by guessing from the smalldata_X_Y row number, which may be an
unrelated image-tiling grid index rather than the trial's physical ROW.

Beverley and Kojonup are genuinely separate locations in Western Australia
(confirmed via 240502 Plot data.xlsx), so their UTM coordinates should be
clearly, unambiguously different -- this lets us build a real cross-site
split grounded in actual geography.

Usage: python check_site_coordinates.py
"""
import os
import pandas as pd
import numpy as np

from utils.dataset import AgriDataset


def get_header_coords(mask_path):
    """
    Reads just the first row of a label_matrix CSV (the coordinate header
    that AgriDataset normally drops) and returns it as raw values.
    """
    with open(mask_path, 'r') as f:
        header_line = f.readline().strip()
    values = header_line.split(',')
    return values


def analyze(root_dir, sample_per_row=2):
    all_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
    valid = [f for f in all_folders if AgriDataset._has_required_files(f)]

    print(f"Sampling coordinate headers from {len(valid)} valid folders...\n")

    results = []
    for folder in valid:
        name = os.path.basename(folder)
        parts = name.replace('smalldata_', '').split('_')
        if len(parts) != 2:
            continue
        row, col = int(parts[0]), int(parts[1])

        _, mask_file = AgriDataset._load_band_map(folder)
        header = get_header_coords(os.path.join(folder, mask_file))
        results.append((row, col, header[:3]))  # first few header values

    results.sort(key=lambda r: (r[0], r[1]))

    print("=" * 70)
    print(f"{'Row':>4}{'Col':>5}   Header (first 3 values)")
    print("=" * 70)
    seen_rows = set()
    for row, col, header in results:
        # Print only the first couple of columns per row to keep this readable
        if row not in seen_rows or col <= 2:
            print(f"{row:>4}{col:>5}   {header}")
        seen_rows.add(row)

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("  Look for a clear jump/discontinuity in the coordinate values as")
    print("  row number increases -- that's the boundary between two physical")
    print("  sites (Beverley vs Kojonup). Coordinates should be roughly")
    print("  continuous WITHIN a site and jump sharply AT the site boundary.")
    print("  Paste this output back and I'll identify the split point.")


if __name__ == "__main__":
    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    analyze(root_dir)