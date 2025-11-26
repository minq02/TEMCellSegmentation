import torch
import torch.nn.functional as F

def dice_loss(logits, targets, num_classes=5, eps=1e-6):
    """
    Multi-class Dice loss.

    logits: (B, C, H, W) raw scores from the network
    targets: (B, H, W) with class indices in [0, C-1]
    """
    # Convert logits to probabilities with softmax
    probs = F.softmax(logits, dim=1)              # (B, C, H, W)

    # One-hot encode the targets
    targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
    targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

    # Compute per-class Dice
    dims = (0, 2, 3)  # sum over batch + spatial dimensions
    intersection = torch.sum(probs * targets_one_hot, dims)
    cardinality  = torch.sum(probs + targets_one_hot, dims)

    dice = (2.0 * intersection + eps) / (cardinality + eps)  # (C,)
    dice_loss_value = 1.0 - dice.mean()  # average over classes

    return dice_loss_value
