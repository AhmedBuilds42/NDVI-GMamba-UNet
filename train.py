import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.unet_arch import IGMambaUNet
from utils.dataset import AgriDataset
from utils.losses import AgriLoss
from utils.splits import get_folder_split
from utils.metrics import compute_class_distribution, print_class_distribution


def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    num_classes = 4
    num_epochs = 20
    batch_size = 8
    lr = 1e-3
    val_ratio = 0.3
    split_seed = 42  # must match evaluate.py so train/val never overlap

    print("=" * 60)
    print(" AgriMamba -- IG-Mamba U-Net training")
    print("=" * 60)
    print(f" Device      : {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))
    print(f" Classes     : {num_classes}")
    print(f" Batch size  : {batch_size}")
    print(f" Epochs      : {num_epochs}")
    print(f" LR          : {lr}")
    print(f" Val ratio   : {val_ratio} (seed={split_seed})")
    print("-" * 60)

    train_folders, val_folders = get_folder_split(root_dir, val_ratio=val_ratio, seed=split_seed)

    train_dataset = AgriDataset(root_dir, num_classes=num_classes, folder_list=train_folders)
    val_dataset = AgriDataset(root_dir, num_classes=num_classes, folder_list=val_folders)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Class balance check -- also doubles as a split sanity check: if val is
    # missing a class entirely, that class's IoU during evaluation will be
    # all-NaN and misleading. Run once up front so you see it before training.
    print("-" * 60)
    print(" Class distribution (train vs. val split):")
    train_counts = compute_class_distribution(train_dataset.folders, num_classes=num_classes)
    val_counts = compute_class_distribution(val_dataset.folders, num_classes=num_classes)
    print_class_distribution("Train", train_counts)
    print_class_distribution("Val  ", val_counts)
    print("=" * 60)

    model = IGMambaUNet(num_classes=num_classes).to(device)
    criterion = AgriLoss(ignore_index=3)  # class 3 = no-data, see identify_class3.py
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f" Model params: {n_params / 1e6:.2f}M")
    print(f" Train batches/epoch: {len(train_loader)}  |  Val batches/epoch: {len(val_loader)}")
    print("=" * 60)

    best_val_loss = float('inf')
    history = []

    for epoch in range(1, num_epochs + 1):
        # ---- Train ----
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:>2}/{num_epochs} [train]",
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            colour="green",
        )

        for i, (imgs, masks, vi_pyr) in enumerate(pbar, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}

            optimizer.zero_grad()
            preds = model(imgs, vi_pyr)
            loss = criterion(preds, masks)

            if not torch.isfinite(loss):
                print(f"\n  ⚠️ Non-finite loss at epoch {epoch} batch {i} "
                      f"(loss={loss.item()}) -- skipping this batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg": f"{avg_loss:.4f}",
            })

        train_loss = running_loss / len(train_loader)

        # ---- Validate ----
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for imgs, masks, vi_pyr in tqdm(
                val_loader,
                desc=f"Epoch {epoch:>2}/{num_epochs} [val]  ",
                unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}",
                colour="cyan",
            ):
                imgs, masks = imgs.to(device), masks.to(device)
                vi_pyr = {k: v.to(device) for k, v in vi_pyr.items()}
                preds = model(imgs, vi_pyr)
                loss = criterion(preds, masks)
                if torch.isfinite(loss):
                    val_running_loss += loss.item()

        val_loss = val_running_loss / len(val_loader)
        epoch_time = time.time() - epoch_start
        history.append((train_loss, val_loss))

        # Best checkpoint now tracked by VAL loss, not train loss -- train
        # loss alone can't tell you if the modelweed is generalizing or just
        # memorizing the training folders.
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

        marker = " <- best (val)" if is_best else ""
        gap = train_loss - val_loss
        print(f"  Epoch {epoch:>2} done | train: {train_loss:.4f} | val: {val_loss:.4f} "
              f"(gap: {gap:+.4f}) | time: {format_time(epoch_time)}{marker}")

    print("=" * 60)
    print(f" Training complete. Best val loss: {best_val_loss:.4f}")
    print(f" Best checkpoint saved to: best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    train()