"""
Module: trainer.py
Description: Training and validation loop for marine oil spill semantic segmentation 
             with metric tracking (IoU, F1/Dice, BCE), checkpointing, and learning rate scheduling.
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import BCEDiceLoss, MetricTracker


class OilSpillTrainer:
    """
    Manages the training, validation, checkpointing, and evaluation loops for the U-Net model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "checkpoints",
        threshold: float = 0.5
    ):
        """
        Args:
            model: PyTorch U-Net module.
            train_loader: DataLoader for training patches.
            val_loader: DataLoader for validation patches.
            criterion: Loss function (defaults to BCEDiceLoss).
            optimizer: Optimizer (defaults to AdamW).
            scheduler: Optional learning rate scheduler.
            device: torch.device (defaults to CUDA if available, else CPU).
            checkpoint_dir: Directory where model weights will be saved.
            threshold: Probability threshold for positive classification (default: 0.5).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion or BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.threshold = threshold

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_iou": [],
            "train_f1": [],
            "val_loss": [],
            "val_iou": [],
            "val_f1": [],
            "val_precision": [],
            "val_recall": []
        }

    def train_epoch(self) -> Dict[str, float]:
        """Runs one training epoch over the training dataset."""
        self.model.train()
        tracker = MetricTracker(threshold=self.threshold)

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            logits = self.model(images)
            total_loss, bce_loss, dice_loss = self.criterion(logits, masks)

            total_loss.backward()
            # Gradient clipping to stabilize deep feature propagation
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            tracker.update(
                logits=logits,
                targets=masks,
                total_loss=total_loss.item(),
                bce_loss=bce_loss.item(),
                dice_loss=dice_loss.item()
            )

        return tracker.get_summary()

    def validate(self) -> Dict[str, float]:
        """Evaluates the model over the validation set."""
        self.model.eval()
        tracker = MetricTracker(threshold=self.threshold)

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                logits = self.model(images)
                total_loss, bce_loss, dice_loss = self.criterion(logits, masks)

                tracker.update(
                    logits=logits,
                    targets=masks,
                    total_loss=total_loss.item(),
                    bce_loss=bce_loss.item(),
                    dice_loss=dice_loss.item()
                )

        return tracker.get_summary()

    def fit(self, num_epochs: int, save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Executes the full training loop across the specified number of epochs.

        Args:
            num_epochs: Total training epochs.
            save_best_only: If True, only checkpoints the model when validation IoU improves.

        Returns:
            Dictionary containing historical epoch metrics.
        """
        best_val_iou = -1.0
        print(f"\n=======================================================")
        print(f"Starting Training on Device: {self.device}")
        print(f"Epochs: {num_epochs} | Batch Size: {self.train_loader.batch_size}")
        print(f"=======================================================\n")

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Learning rate scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            # Record metrics
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["train_f1"].append(train_metrics["f1"])

            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] "
                f"Train Loss: {train_metrics['loss']:.4f} (IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}) | "
                f"Val Loss: {val_metrics['loss']:.4f} (IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}, "
                f"Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}) | "
                f"LR: {current_lr:.1e}"
            )

            # Checkpoint best model
            if val_metrics["iou"] > best_val_iou:
                best_val_iou = val_metrics["iou"]
                best_ckpt_path = os.path.join(self.checkpoint_dir, "best_unet_oil_spill.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_iou": val_metrics["iou"],
                    "val_f1": val_metrics["f1"],
                    "val_loss": val_metrics["loss"]
                }, best_ckpt_path)
                print(f"  --> Saved new best checkpoint (Val IoU: {best_val_iou:.4f}) to {best_ckpt_path}")

        # Always save final model
        last_ckpt_path = os.path.join(self.checkpoint_dir, "last_unet_oil_spill.pth")
        torch.save(self.model.state_dict(), last_ckpt_path)
        print(f"\nTraining Complete! Best Validation IoU: {best_val_iou:.4f}")

        return self.history

