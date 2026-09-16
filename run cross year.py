import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model.unet_arch import IGMambaUNet
from utils.dataset import AgriDataset
from utils.losses import AgriLoss
from utils.splits import get_cross_year_train_folders, get_folder_split
from utils.metrics import (
    compute_class_distribution, print_class_distribution,
    calculate_iou_and_f1,
)


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_cross_year():
    """
    CROSS-YEAR protocol: trains with row 3 COMPLETELY EXCLUDED from
    ImageData_Split, then evaluates on 2021 Test (which is row 3, cols
    1-10). This guarantees the modelweed has never seen that physical strip of
    field during training -- so any performance on 2021 Test reflects
    genuine cross-year generalization (same place, different flight/season),
    not memorization from an overlapping random split.

    Background: check_train_test_overlap.py found 9/10 of 2021 Test's
    folder names were also in the original random-split TRAIN set. This
    script is the fix -- excluding row 3 entirely, not just from one
    random split realization.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    test_2021_root = (
        r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\2021 Test\2021 Test"
    )
    num_classes = 4
    num_epochs = 20
    batch_size = 8
    lr = 1e-3
    checkpoint_path = "best_model_cross_year.pth"

    print("=" * 60)
    print(" AgriMamba -- CROSS-YEAR protocol (row 3 excluded from training)")
    print("=" * 60)
    print(f" Device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

    kept_folders, excluded_folders = get_cross_year_train_folders(root_dir, exclude_rows=(3,))
    print(f" Excluded {len(excluded_folders)} row-3 folders from ImageData_Split entirely.")
    print(f" Remaining pool for train/val: {len(kept_folders)} folders.")

    # Split the REMAINING (row-3-free) pool into train/val for checkpoint
    # selection -- reusing get_folder_split's random-shuffle logic, but note
    # it now operates over kept_folders only, so row 3 can never appear.
    import random
    rng = random.Random(42)
    shuffled = kept_folders[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.2))
    val_folders = shuffled[:n_val]
    train_folders = shuffled[n_val:]
    print("-" * 60)

    train_dataset = AgriDataset(root_dir, num_classes=num_classes, folder_list=train_folders)
    val_dataset = AgriDataset(root_dir, num_classes=num_classes, folder_list=val_folders)
    test_dataset = AgriDataset(test_2021_root, num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Class distribution:")
    train_counts = compute_class_distribution(train_dataset.folders, num_classes=num_classes)
    val_counts = compute_class_distribution(val_dataset.folders, num_classes=num_classes)
    test_counts = compute_class_distribution(test_dataset.folders, num_classes=num_classes)
    print_class_distribution("Train (row 3 excluded)", train_counts)
    print_class_distribution("Val   (row 3 excluded)", val_counts)
    print_class_distribution("Test  (2021 Test, row 3)", test_counts)
    print("=" * 60)

    model = IGMambaUNet(num_classes=num_classes).to(device)
    criterion = AgriLoss(ignore_index=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Model params: {n_params / 1e6:.2f}M")
    print(f" Train/val/test batches: {len(train_loader)}/{len(val_loader)}/{len(test_loader)}")
    print("=" * 60)

    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{num_epochs} [train]",
                    unit="batch", bar_format="{l_bar}{bar:30}{r_bar}", colour="green")

        for i, (imgs, masks, vi_pyr) in enumerate(pbar, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}

            optimizer.zero_grad()
            preds = model(imgs, vi_pyr)
            loss = criterion(preds, masks)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{running_loss/i:.4f}"})

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, masks, vi_pyr in tqdm(val_loader, desc=f"Epoch {epoch:>2}/{num_epochs} [val]  ",
                                             unit="batch", bar_format="{l_bar}{bar:30}{r_bar}", colour="cyan"):
                imgs, masks = imgs.to(device), masks.to(device)
                vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}
                preds = model(imgs, vi_pyr)
                loss = criterion(preds, masks)
                if torch.isfinite(loss):
                    val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        marker = " <- best (val)" if is_best else ""
        print(f"  Epoch {epoch:>2} done | train: {train_loss:.4f} | val: {val_loss:.4f} "
              f"| time: {format_time(epoch_time)}{marker}")

    print("=" * 60)
    print(f" Training complete. Best val loss: {best_val_loss:.4f}")
    print(f" Checkpoint saved to: {checkpoint_path}")
    print("=" * 60)

    # ---- Final evaluation on 2021 Test (genuinely unseen year AND location-in-training-set) ----
    print("\n" + "=" * 60)
    print(" FINAL CROSS-YEAR RESULTS (2021 Test, row 3 never seen in training)")
    print("=" * 60)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    class_names = ["Background/Soil", "Crop", "Weed", "Class 3"]
    all_ious, all_f1s = [], []
    with torch.no_grad():
        for imgs, masks, vi_pyr in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}
            preds = torch.argmax(model(imgs, vi_pyr), dim=1)
            for b in range(preds.shape[0]):
                iou, f1 = calculate_iou_and_f1(preds[b], masks[b], num_classes=num_classes, ignore_index=3)
                all_ious.append(iou)
                all_f1s.append(f1)

    all_ious, all_f1s = np.array(all_ious), np.array(all_f1s)
    for i, name in enumerate(class_names):
        if i == 3:
            print(f"🌿 Class {i} ({name:<15}): excluded (no-data)")
            continue
        print(f"🌿 Class {i} ({name:<15}): IoU = {np.nanmean(all_ious[:, i])*100:.2f}% "
              f"| F1 = {np.nanmean(all_f1s[:, i])*100:.2f}%")
    print("-" * 60)
    print(f"🏆 Cross-year Test mIoU (classes 0-2): {np.nanmean(all_ious)*100:.2f}%")
    print(f"🏆 Cross-year Test Mean F1: {np.nanmean(all_f1s)*100:.2f}%")
    print(f"   (row 3 was 100% excluded from training -- this is genuine")
    print(f"    cross-year generalization, not memorized ground)")
    print("=" * 60)


if __name__ == "__main__":
    run_cross_year()