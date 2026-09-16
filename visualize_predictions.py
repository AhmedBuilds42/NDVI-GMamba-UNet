import torch
import matplotlib.pyplot as plt
from model.unet_arch import IGMambaUNet
from utils.dataset import AgriDataset


def visualize(sample_index=0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = r"C:\Abdullah\ahmed abdullah\AgriMamba\data\Multispectral Image Benchmark Dataset\ImageData_Split\ImageData_Split"
    dataset = AgriDataset(root_dir)

    img, mask, vi_pyr = dataset[sample_index]

    # Load Model
    model = IGMambaUNet(num_classes=4).to(device)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        img_input = img.unsqueeze(0).to(device)
        vi_input = {k: v.unsqueeze(0).to(device) for k, v in vi_pyr.items()}
        output = model(img_input, vi_input)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # False-color RGB (Red, Green, Blue from the 5 bands: idx 2, 1, 0)
    rgb = img[[2, 1, 0], :, :].permute(1, 2, 0).numpy()
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB Tile")
    axes[0].axis('off')

    axes[1].imshow(vi_pyr['vi_1'].squeeze(0).numpy(), cmap='RdYlGn')
    axes[1].set_title("Physics Guide (NDVI)")
    axes[1].axis('off')

    axes[2].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=3)
    axes[2].set_title("Ground Truth Mask")
    axes[2].axis('off')

    axes[3].imshow(pred, cmap='tab10', vmin=0, vmax=3)
    axes[3].set_title("IG-Mamba Prediction")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig("prediction_sample.png", dpi=300)
    print("✅ Saved visualization to prediction_sample.png")
    plt.show()


if __name__ == "__main__":
    visualize(5)  # Test on sample 5