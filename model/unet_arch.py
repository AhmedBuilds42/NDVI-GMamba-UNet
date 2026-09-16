import torch
import torch.nn as nn
from model.ig_mamba import IGMambaBlock

class IGMambaUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # STEM: 5 channels (Blue, Green, Red, RedEdge, NIR) -> 96 channels
        self.stem = nn.Conv2d(5, 96, kernel_size=4, stride=4)

        # ENCODER: 4 Stages
        self.stage1 = IGMambaBlock(96)
        self.merge1 = nn.Conv2d(96, 192, 2, stride=2)

        self.stage2 = IGMambaBlock(192)
        self.merge2 = nn.Conv2d(192, 384, 2, stride=2)

        self.stage3 = IGMambaBlock(384)
        self.merge3 = nn.Conv2d(384, 768, 2, stride=2)

        self.stage4 = IGMambaBlock(768)

        # DECODER
        self.up4 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.dec3 = nn.Conv2d(768, 384, 3, padding=1)

        self.up3 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.dec2 = nn.Conv2d(384, 192, 3, padding=1)

        self.up2 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.dec1 = nn.Conv2d(192, 96, 3, padding=1)

        self.up1 = nn.ConvTranspose2d(96, 64, 4, stride=4)
        self.seg_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, vi_pyr):
        # x: [B, 5, 512, 512]
        # vi_pyr: dict of [vi_1, vi_2, vi_3, vi_4] matching stage resolutions

        # Encoder
        # NOTE: s1/s2/s3 are now captured AFTER the VI-gated mamba stage,
        # not before it. Previously the skip tensors were cloned straight
        # off the stem/merge convs, before apply_mamba ran -- so the decoder
        # was reconstructing segmentation mostly from ungated features, and
        # the physics-informed VI gating only ever touched the bottleneck
        # (stage4). Since skip connections carry most of a U-Net's spatial
        # detail, that meant the modelweed's actual "IG-Mamba" contribution had
        # very little influence on the final output. Capturing skips after
        # gating means every resolution level -- not just the bottleneck --
        # is shaped by the VI-informed gate.
        x = self.stem(x)  # [96, 128, 128]
        x = self.apply_mamba(x, self.stage1, vi_pyr['vi_1'])
        s1 = x.clone()

        x = self.merge1(x)  # [192, 64, 64]
        x = self.apply_mamba(x, self.stage2, vi_pyr['vi_2'])
        s2 = x.clone()

        x = self.merge2(x)  # [384, 32, 32]
        x = self.apply_mamba(x, self.stage3, vi_pyr['vi_3'])
        s3 = x.clone()

        x = self.merge3(x)  # [768, 16, 16]
        x = self.apply_mamba(x, self.stage4, vi_pyr['vi_4'])

        # Decoder with Skips
        x = self.up4(x)
        x = torch.cat([x, s3], dim=1)
        x = self.dec3(x)

        x = self.up3(x)
        x = torch.cat([x, s2], dim=1)
        x = self.dec2(x)

        x = self.up2(x)
        x = torch.cat([x, s1], dim=1)
        x = self.dec1(x)

        x = self.up1(x)
        return self.seg_head(x)

    def apply_mamba(self, x, layer, vi):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        vi_flat = vi.flatten(2).transpose(1, 2)
        x = layer(x_flat, vi_flat)
        return x.transpose(1, 2).view(B, C, H, W)