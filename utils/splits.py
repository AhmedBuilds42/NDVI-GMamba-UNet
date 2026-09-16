import os
import re
import random


def get_folder_split(root_dir, val_ratio=0.2, seed=42):
    """
    Deterministically splits the smalldata_X_Y folders under root_dir into
    train/val sets. Both train.py and evaluate.py must call this with the
    SAME root_dir, val_ratio, and seed to guarantee they use disjoint,
    consistent sets -- otherwise evaluate.py risks re-measuring performance
    on folders the modelweed was trained on (data leakage).

    os.listdir() order is not guaranteed across platforms/runs, so we sort
    first, then shuffle with a seeded RNG -- this makes the split fully
    reproducible run-to-run.

    Returns: (train_folders, val_folders) -- lists of absolute folder paths.
    """
    all_folders = sorted([
        os.path.join(root_dir, d) for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    rng = random.Random(seed)
    shuffled = all_folders[:]
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    val_folders = shuffled[:n_val]
    train_folders = shuffled[n_val:]

    return train_folders, val_folders


_TILE_PATTERN = re.compile(r'smalldata_(\d+)_(\d+)')


def _group_by_row(root_dir):
    """Groups smalldata_X_Y folder paths by their row index X."""
    all_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d))]
    by_row = {}
    unparsed = []
    for f in all_folders:
        m = _TILE_PATTERN.match(os.path.basename(f))
        if m:
            row = int(m.group(1))
            by_row.setdefault(row, []).append(f)
        else:
            unparsed.append(f)
    return by_row, unparsed


def get_within_plot_split(root_dir, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    WITHIN-PLOT split: holds out CONTIGUOUS spatial row blocks for val/test,
    instead of randomly scattering individual tiles (get_folder_split).

    Why this matters: check_site_coordinates.py confirmed ImageData_Split's
    label_matrix header is a pixel-offset index (row * 512), not real GPS --
    meaning every tile here is part of ONE continuous field, grid-tiled.
    Neighbouring tiles are highly spatially correlated (same crop rows, same
    lighting, same day). A random split puts a val tile directly next to a
    near-identical train tile, leaking that correlation and inflating scores.
    A contiguous block split only has correlation at the 1-2 rows sitting
    exactly on a block boundary -- much closer to genuinely testing
    generalization to an unseen part of the field, matching the "leakage-free
    spatial block" principle BAWSeg's methodology describes.

    Test block is taken from the far end of the row range, val block from
    the next-highest rows, train gets everything else. seed is unused here
    (block assignment is deterministic by row order, not random) but kept
    in the signature for interface consistency with get_folder_split.

    Returns: (train_folders, val_folders, test_folders, info_dict) where
    info_dict reports which row numbers landed in each split, for logging.
    """
    by_row, unparsed = _group_by_row(root_dir)
    if unparsed:
        print(f"⚠️ get_within_plot_split: {len(unparsed)} folders didn't match "
              f"the smalldata_X_Y pattern and were excluded entirely.")

    rows_sorted = sorted(by_row.keys())
    total = sum(len(v) for v in by_row.values())

    test_rows = []
    running = 0
    for row in reversed(rows_sorted):
        if total > 0 and running / total < test_ratio:
            test_rows.append(row)
            running += len(by_row[row])
        else:
            break

    remaining_rows = [r for r in rows_sorted if r not in test_rows]
    val_rows = []
    running = 0
    for row in reversed(remaining_rows):
        if total > 0 and running / total < val_ratio:
            val_rows.append(row)
            running += len(by_row[row])
        else:
            break

    train_rows = [r for r in remaining_rows if r not in val_rows]

    train_folders = [f for r in train_rows for f in by_row[r]]
    val_folders = [f for r in val_rows for f in by_row[r]]
    test_folders = [f for r in test_rows for f in by_row[r]]

    info = {
        'train_rows': sorted(train_rows),
        'val_rows': sorted(val_rows),
        'test_rows': sorted(test_rows),
    }
    return train_folders, val_folders, test_folders, info


def get_cross_year_train_folders(root_dir, exclude_rows=(3,)):
    """
    Returns ImageData_Split folders EXCLUDING the given row(s), so a modelweed
    trained on this set has never seen the physical location that 2021 Test
    evaluates on -- required for a genuine cross-year protocol (same place,
    different year).

    Background: check_train_test_overlap.py found that 9 of the 10 row-3
    folder NAMES used in 2021 Test also appeared in the original random
    80/20 TRAIN split. Since row 3 is the same continuous-field grid
    position both times, that overlap very likely means the modelweed already
    saw that ground during training under the old split -- which would
    contaminate any "cross-year generalization" claim made from that setup.
    Excluding row 3 entirely from training removes that risk.

    Returns: (kept_folders, excluded_folders)
    """
    by_row, unparsed = _group_by_row(root_dir)
    kept = [f for row, folders in by_row.items() if row not in exclude_rows for f in folders]
    excluded = [f for row, folders in by_row.items() if row in exclude_rows for f in folders]
    return kept, excluded