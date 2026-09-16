import torch
import torch.nn as nn
import torch.nn.functional as F


class AgriLoss(nn.Module):
    """
    CrossEntropy + Dice, with class 3 treated as a void/no-data label rather
    than a real class to predict.

    Diagnosis (see identify_class3.py): 99.96% of class-3 pixels coincide
    with raw NaN band values from UAV orthomosaic gaps, and their spectral
    signature is ~0 across all 5 bands -- i.e. class 3 IS the dataset's own
    no-data marker, zero-filled by nan_to_num, not a real 4th land-cover
    class. Supervising the modelweed to predict it was letting it learn a
    trivial "predict void wherever input==0" shortcut, which was inflating
    IoU (99%+ on both val and test) without reflecting real segmentation
    skill.

    ignore_index=3 means:
      - CrossEntropyLoss: pixels labeled 3 contribute zero gradient.
      - Dice: those pixels are excluded from both prediction and target
        before computing intersection/union, and only classes 0-2 are scored.
    """

    def __init__(self, weight=None, ignore_index=3):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        # 1. Cross Entropy -- ignore_index handles class-3 exclusion natively.
        ce_loss = self.ce(inputs, targets)

        # 2. Dice Loss, excluding ignore_index pixels and classes.
        num_classes = inputs.shape[1]
        real_classes = [c for c in range(num_classes) if c != self.ignore_index]

        valid = (targets != self.ignore_index)  # [B, H, W]
        valid_f = valid.unsqueeze(1).float()    # [B, 1, H, W]

        probs = F.softmax(inputs, dim=1) * valid_f

        # Clamp ignored-pixel targets to a real class index before one-hot
        # (one_hot can't take ignore_index directly) -- doesn't matter which,
        # since valid_f zeroes those positions out right after.
        targets_clamped = targets.clone()
        targets_clamped[~valid] = real_classes[0]
        targets_one_hot = F.one_hot(targets_clamped, num_classes=num_classes) \
            .permute(0, 3, 1, 2).float() * valid_f

        smooth = 1.0
        dice_losses = []
        for c in real_classes:
            p = probs[:, c]
            t = targets_one_hot[:, c]
            intersection = (p * t).sum(dim=(1, 2))
            union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
            dice_losses.append(1 - ((2. * intersection + smooth) / (union + smooth)))

        dice_loss = torch.stack(dice_losses, dim=1).mean()

        return ce_loss + dice_loss