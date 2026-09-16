import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

# ======================================================================
# Paths Configuration
# ======================================================================
ROOT_DIR = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset"
PATH_2021_TEST = os.path.join(ROOT_DIR, "2021 Test")
PATH_IMAGE_SPLIT = os.path.join(ROOT_DIR, "ImageData_Split")

BANDS = ["Blue", "Green", "Red", "NIR", "RedEdge", "RGB"]


def find_all_smalldata_folders(base_path):
    """Recursively finds all directories starting with 'smalldata_'."""
    found = {}
    for root, dirs, _ in os.walk(base_path):
        for d in dirs:
            if d.startswith("smalldata_"):
                found[d] = os.path.join(root, d)
    return found


def load_image(filepath):
    """Loads an image into a numpy array."""
    if not os.path.exists(filepath):
        return None
    try:
        if HAS_TIFFFILE and filepath.lower().endswith((".tif", ".tiff")):
            return tifffile.imread(filepath)
        else:
            with Image.open(filepath) as img:
                return np.array(img)
    except Exception:
        return None


def compare_sample_data():
    print("=" * 95)
    print("🔬 AUTO-LOCATING & COMPARING SAMPLES: '2021 Test' vs 'ImageData_Split'")
    print("=" * 95)

    # 1. Discover all smalldata folders automatically
    folders_2021 = find_all_smalldata_folders(PATH_2021_TEST)
    folders_split = find_all_smalldata_folders(PATH_IMAGE_SPLIT)

    print(f"📁 Discovered folders:")
    print(f"   • Found in '2021 Test'       : {len(folders_2021)} subfolders")
    print(f"   • Found in 'ImageData_Split' : {len(folders_split)} subfolders")

    common_keys = sorted(list(set(folders_2021.keys()).intersection(set(folders_split.keys()))))
    print(f"   • Shared matching tiles      : {len(common_keys)} -> {common_keys}\n")

    if not common_keys:
        print("❌ No matching 'smalldata_*' folders found between the two paths.")
        return

    comparison_rows = []

    # 2. Iterate through shared folders and compare
    for folder in common_keys:
        dir_2021 = folders_2021[folder]
        dir_split = folders_split[folder]
        suffix = folder.replace("smalldata_", "")

        # A. Compare Spectral Bands
        for band in BANDS:
            fname = f"{band}_{suffix}.tif"
            f_2021 = os.path.join(dir_2021, fname)
            f_split = os.path.join(dir_split, fname)

            img_2021 = load_image(f_2021)
            img_split = load_image(f_split)

            if img_2021 is None or img_split is None:
                comparison_rows.append({
                    "Sample": folder,
                    "File": fname,
                    "Shape_2021": "Missing" if img_2021 is None else str(img_2021.shape),
                    "Shape_Split": "Missing" if img_split is None else str(img_split.shape),
                    "Pixel_Match": "❌ Missing File",
                    "Max_Diff": "N/A",
                    "Range_2021": "N/A",
                    "Range_Split": "N/A"
                })
                continue

            shapes_match = (img_2021.shape == img_split.shape)
            exact_match = np.array_equal(img_2021, img_split) if shapes_match else False

            if shapes_match:
                diff = np.abs(img_2021.astype(np.float64) - img_split.astype(np.float64))
                max_diff = np.max(diff)
                match_status = "✅ 100% MATCH" if exact_match else ("⚠️ Near Match" if max_diff < 1e-4 else "❌ Different")
            else:
                max_diff = "Shape Mismatch"
                match_status = "❌ Shape Mismatch"

            comparison_rows.append({
                "Sample": folder,
                "File": fname,
                "Shape_2021": str(img_2021.shape),
                "Shape_Split": str(img_split.shape),
                "Pixel_Match": match_status,
                "Max_Diff": f"{max_diff:.6f}" if isinstance(max_diff, (int, float)) else max_diff,
                "Range_2021": f"[{np.min(img_2021):.2f}, {np.max(img_2021):.2f}]",
                "Range_Split": f"[{np.min(img_split):.2f}, {np.max(img_split):.2f}]"
            })

        # B. Compare Ground Truth Label Matrix
        csv_name = f"label_matrix_{suffix}.csv"
        c_2021 = os.path.join(dir_2021, csv_name)
        c_split = os.path.join(dir_split, csv_name)

        if os.path.exists(c_2021) and os.path.exists(c_split):
            try:
                df_2021 = pd.read_csv(c_2021, header=None).values
                df_split = pd.read_csv(c_split, header=None).values
                csv_match = np.array_equal(df_2021, df_split)
                max_c_diff = np.max(np.abs(df_2021.astype(float) - df_split.astype(float)))
                match_status = "✅ 100% MATCH" if csv_match else "❌ Different"
                diff_str = f"{max_c_diff:.4f}"
            except Exception:
                match_status = "⚠️ Error reading CSV"
                diff_str = "Error"
                df_2021 = df_split = np.array([])

            comparison_rows.append({
                "Sample": folder,
                "File": csv_name,
                "Shape_2021": str(df_2021.shape),
                "Shape_Split": str(df_split.shape),
                "Pixel_Match": match_status,
                "Max_Diff": diff_str,
                "Range_2021": f"[{np.nanmin(df_2021)}, {np.nanmax(df_2021)}]" if df_2021.size else "N/A",
                "Range_Split": f"[{np.nanmin(df_split)}, {np.nanmax(df_split)}]" if df_split.size else "N/A"
            })
        else:
            comparison_rows.append({
                "Sample": folder,
                "File": csv_name,
                "Shape_2021": "Found" if os.path.exists(c_2021) else "Missing",
                "Shape_Split": "Found" if os.path.exists(c_split) else "Missing",
                "Pixel_Match": "⚠️ Only in one folder",
                "Max_Diff": "N/A",
                "Range_2021": "N/A",
                "Range_Split": "N/A"
            })

    # 3. Print Results
    df_results = pd.DataFrame(comparison_rows)
    print("=" * 95)
    print("📋 SAMPLE PIXEL COMPARISON TABLE (First 20 Files):")
    print("=" * 95)
    print(df_results[["Sample", "File", "Shape_2021", "Pixel_Match", "Max_Diff", "Range_2021", "Range_Split"]].head(20).to_string(index=False))

    total = len(df_results)
    matches = len(df_results[df_results["Pixel_Match"] == "✅ 100% MATCH"])

    print("\n" + "=" * 95)
    print(f"📊 SUMMARY OF PIXEL COMPARISON:")
    print(f"   • Total Files Compared  : {total}")
    print(f"   • 100% Exact Matches    : {matches} / {total}")
    if total == matches:
        print("\n🎉 CONCLUSION: The files in '2021 Test' are 100% EXACT COPIES of 'ImageData_Split'!")
    else:
        print(f"\n⚠️ Mismatches or unique files found: {total - matches}")
    print("=" * 95)

    df_results.to_csv("pixel_comparison_report.csv", index=False)
    print("💾 Saved complete comparison report to: pixel_comparison_report.csv")


if __name__ == "__main__":
    compare_sample_data()