import torch
import numpy as np
from torch.utils.data import DataLoader
from model.unet_arch import IGMambaUNet
from utils.dataset import AgriDataset
from utils.splits import get_folder_split
from utils.metrics import (
    calculate_iou_and_f1,
    compute_class_distribution,
    print_class_distribution,
    suggest_class_weights,
)
import tqdm


def evaluate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    num_classes = 4
    val_ratio = 0.2
    split_seed = 42  # MUST match train.py -- same seed = same split = no leakage

    # 1. Load ONLY the held-out val split. Previously this loaded the full
    # dataset (same folders as training), so the reported mIoU/F1 was
    # training-set performance, not a real generalization measure.
    train_folders, val_folders = get_folder_split(root_dir, val_ratio=val_ratio, seed=split_seed)
    dataset = AgriDataset(root_dir, folder_list=val_folders)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    class_names = ["Background/Soil", "Crop", "Weed", "Class 3"]

    # 2. Class balance check on the val set actually being scored, plus the
    # train set for comparison -- lets you see at a glance whether the split
    # is representative and whether any class is rare enough to need
    # weighting in AgriLoss.
    print("=" * 50)
    print("📊 CLASS DISTRIBUTION (sanity check)")
    print("=" * 50)
    train_dataset_for_counts = AgriDataset(root_dir, folder_list=train_folders)
    train_counts = compute_class_distribution(train_dataset_for_counts.folders, num_classes=num_classes)
    val_counts = compute_class_distribution(dataset.folders, num_classes=num_classes)
    print_class_distribution("Train", train_counts, class_names)
    print_class_distribution("Val  ", val_counts, class_names)

    if (val_counts == 0).any():
        missing = [class_names[i] for i in range(num_classes) if val_counts[i] == 0]
        print(f"\n  ⚠️ Val split has ZERO pixels of: {missing} -- IoU/F1 for "
              f"{'this class' if len(missing) == 1 else 'these classes'} will be NaN below, "
              f"not a real 0%.")

    weights = suggest_class_weights(train_counts, ignore_index=3)
    print("\n  Suggested AgriLoss class weights (inverse frequency over classes 0-2, class 3 ignored):")
    for i, name in enumerate(class_names):
        print(f"    {name:<18}: {weights[i]:.3f}")
    print("  (Not applied automatically -- pass torch.tensor(weights) into "
          "AgriLoss(weight=...) if imbalance looks severe.)")

    # 3. Load Model
    model = IGMambaUNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    all_ious = []
    all_f1s = []

    print("\n" + "=" * 50)
    print("🔍 Evaluating Model on HELD-OUT VAL SPLIT...")
    print("=" * 50)
    with torch.no_grad():
        for imgs, masks, vi_pyr in tqdm.tqdm(val_loader):
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
    print("📊 FINAL BENCHMARK RESULTS (val split only, class 3 excluded as no-data)")
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
    print(f"   (measured on {len(dataset)} held-out folders never seen during training)")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()