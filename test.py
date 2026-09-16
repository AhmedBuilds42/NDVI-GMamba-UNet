import torch
import numpy as np
from torch.utils.data import DataLoader
from model.unet_arch import IGMambaUNet
from utils.dataset import AgriDataset
from utils.metrics import (
    calculate_iou_and_f1,
    compute_class_distribution,
    print_class_distribution,
)
import tqdm


def test():
    """
    Final evaluation on the 2021 Test set -- a genuinely separate folder on
    disk (not a random split of the main 783 training tiles), so this is a
    real held-out test, not a re-shuffle of the same data.

    IMPORTANT: run this ONCE at the end for reporting. Don't use it to guide
    training decisions or hyperparameter tuning -- that's what evaluate.py
    (the 80/20 val split) is for. If you start iterating based on numbers
    from this script, it stops being a real test set.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 4

    # Note the actual on-disk path: "2021test" then a nested "2021 Test"
    # (space, capital T) -- confirmed via directory listing.
    test_root = (
        r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\2021 Test\2021 Test"
    )

    class_names = ["Background/Soil", "Crop", "Weed", "Class 3"]

    dataset = AgriDataset(test_root, num_classes=num_classes)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # Class balance check -- this dataset has a different folder-naming
    # pattern (smalldata_3_X, i.e. all row 3) and many extra precomputed
    # index files (NDVI/EVI/GNDVI/etc.) that AgriDataset ignores. Worth
    # confirming the class distribution isn't wildly different from your
    # training data before trusting the mIoU below.
    print("=" * 50)
    print("📊 2021 TEST SET -- class distribution")
    print("=" * 50)
    test_counts = compute_class_distribution(dataset.folders, num_classes=num_classes)
    print_class_distribution("Test ", test_counts, class_names)
    if (test_counts == 0).any():
        missing = [class_names[i] for i in range(num_classes) if test_counts[i] == 0]
        print(f"\n  ⚠️ Test set has ZERO pixels of: {missing} -- IoU/F1 for "
              f"{'this class' if len(missing) == 1 else 'these classes'} will be NaN below.")

    # Load modelweed
    model = IGMambaUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_ious = []
    all_f1s = []

    print("\n" + "=" * 50)
    print("🔍 Evaluating on 2021 Test set (never seen during training)...")
    print("=" * 50)
    with torch.no_grad():
        for imgs, masks, vi_pyr in tqdm.tqdm(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}

            outputs = model(imgs, vi_pyr)
            preds = torch.argmax(outputs, dim=1)

            for b in range(preds.shape[0]):
                iou, f1 = calculate_iou_and_f1(preds[b], masks[b], num_classes=num_classes, ignore_index=3)
                all_ious.append(iou)
                all_f1s.append(f1)

    all_ious = np.array(all_ious)
    all_f1s = np.array(all_f1s)

    print("\n" + "=" * 50)
    print("📊 FINAL 2021 TEST SET RESULTS (class 3 excluded as no-data)")
    print("=" * 50)

    for i, name in enumerate(class_names):
        if i == 3:
            print(f"🌿 Class {i} ({name:<15}): excluded (no-data, see identify_class3.py)")
            continue
        mean_iou = np.nanmean(all_ious[:, i]) * 100
        mean_f1 = np.nanmean(all_f1s[:, i]) * 100
        print(f"🌿 Class {i} ({name:<15}): IoU = {mean_iou:.2f}% | F1 = {mean_f1:.2f}%")

    overall_miou = np.nanmean(all_ious) * 100
    overall_mf1 = np.nanmean(all_f1s) * 100
    print("-" * 50)
    print(f"🏆 Overall mIoU (classes 0-2 only): {overall_miou:.2f}%")
    print(f"🏆 Overall Mean F1 (classes 0-2 only): {overall_mf1:.2f}%")
    print(f"   (measured on {len(dataset)} tiles from a separate held-out test folder)")
    print("=" * 50)


if __name__ == "__main__":
    test()