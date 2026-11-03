"""
Module: losses.py
Description: Numerically stable combined Binary Cross-Entropy (BCE) + Dice Loss 
             and evaluation metrics (IoU, F1-Score, Precision, Recall) for class-imbalanced 
             marine oil spill segmentation.
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss operating directly on unnormalized network logits.
    Designed to prevent gradients from vanishing in extreme class imbalance
    (oil spills often occupy < 2% of pixels).
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits of shape (B, 1, H, W) or (B, H, W).
            targets: Binary ground truth masks of matching shape with values in {0, 1}.
        """
        probs = torch.sigmoid(logits)

        # Flatten batch items
        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        loss = 1.0 - dice

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BCEDiceLoss(nn.Module):
    """
    Compound loss combining Binary Cross-Entropy with Soft Dice Loss:
        L_total = (bce_weight * L_bce) + (dice_weight * L_dice)
    
    BCE penalizes pixel-level classification errors smoothly, while Dice loss 
    directly optimizes regional intersection to counterbalance severe background dominance.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tuple of (total_loss, bce_loss, dice_loss)
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
        return total_loss, bce_loss, dice_loss


class SegmentationMetrics:
    """
    Computes confusion-matrix based segmentation metrics:
    - Intersection over Union (IoU / Jaccard Index)
    - F1-Score (Dice Coefficient)
    - Precision
    - Recall
    """

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps

    def compute_batch(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Computes metric scores across a single batch.

        Args:
            logits: Raw logits (B, 1, H, W).
            targets: Binary ground truth masks (B, 1, H, W).
        """
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            targets = (targets > 0.5).float()

            preds_flat = preds.view(-1)
            targets_flat = targets.view(-1)

            # Confusion matrix counts
            tp = (preds_flat * targets_flat).sum().item()
            fp = (preds_flat * (1.0 - targets_flat)).sum().item()
            fn = ((1.0 - preds_flat) * targets_flat).sum().item()

            iou = (tp + self.eps) / (tp + fp + fn + self.eps)
            f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
            precision = (tp + self.eps) / (tp + fp + self.eps)
            recall = (tp + self.eps) / (tp + fn + self.eps)

            return {
                "iou": float(iou),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall)
            }


class MetricTracker:
    """Accumulates batch statistics to compute exact dataset-level metrics across an epoch."""

    def __init__(self, threshold: float = 0.5, eps: float = 1e-7):
        self.threshold = threshold
        self.eps = eps
        self.reset()

    def reset(self):
        self.total_loss = 0.0
        self.total_bce = 0.0
        self.total_dice = 0.0
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.total_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        total_loss: float,
        bce_loss: float,
        dice_loss: float
    ):
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()
            targets = (targets > 0.5).float()

            self.tp += (preds * targets).sum().item()
            self.fp += (preds * (1.0 - targets)).sum().item()
            self.fn += ((1.0 - preds) * targets).sum().item()

            self.total_loss += total_loss
            self.total_bce += bce_loss
            self.total_dice += dice_loss
            self.total_batches += 1

    def get_summary(self) -> Dict[str, float]:
        n = max(self.total_batches, 1)
        iou = (self.tp + self.eps) / (self.tp + self.fp + self.fn + self.eps)
        f1 = (2.0 * self.tp + self.eps) / (2.0 * self.tp + self.fp + self.fn + self.eps)
        precision = (self.tp + self.eps) / (self.tp + self.fp + self.eps)
        recall = (self.tp + self.eps) / (self.tp + self.fn + self.eps)

        return {
            "loss": self.total_loss / n,
            "bce": self.total_bce / n,
            "dice_loss": self.total_dice / n,
            "iou": float(iou),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall)
        }

