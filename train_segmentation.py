"""
train_segmentation.py
Main driver script for Phase 2: Oil Spill Semantic Segmentation.

Demonstrates:
  - Custom SAROilSpillDataset initialization with augmentations.
  - U-Net instantiation for single-channel SAR input.
  - Combined BCEDiceLoss formulation for severe class imbalance.
  - Training, validation, IoU / F1 tracking, and checkpointing.
"""

import os
import sys
import argparse
from typing import Tuple

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
from torch.utils.data import DataLoader

from src.segmentation.dataset import SAROilSpillDataset
from src.segmentation.unet import SAROilSpillUNet
from src.segmentation.losses import BCEDiceLoss
from src.segmentation.trainer import OilSpillTrainer
from generate_sample_patches import generate_synthetic_patches


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train U-Net for Marine Oil Spill Segmentation")
    parser.add_argument("--train-images", type=str, default=None, help="Directory containing training SAR patches.")
    parser.add_argument("--train-masks", type=str, default=None, help="Directory containing training binary masks.")
    parser.add_argument("--val-images", type=str, default=None, help="Directory containing validation SAR patches.")
    parser.add_argument("--val-masks", type=str, default=None, help="Directory containing validation binary masks.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 0.001).")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="Weight for BCE loss component (default: 0.5).")
    parser.add_argument("--dice-weight", type=float, default=0.5, help="Weight for Dice loss component (default: 0.5).")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save model weights.")
    parser.add_argument("--create-sample-patches", action="store_true", default=True, help="Auto-generate synthetic patches if missing.")

    args = parser.parse_args()

    # If directories not specified or don't exist, generate synthetic patches for demonstration
    if not args.train_images or not os.path.exists(args.train_images):
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = generate_synthetic_patches()
    else:
        train_img_dir, train_mask_dir = args.train_images, args.train_masks
        val_img_dir, val_mask_dir = args.val_images, args.val_masks

    # 1. Instantiate Datasets
    print("\n[1/4] Loading Datasets...")
    train_dataset = SAROilSpillDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        is_train=True
    )
    val_dataset = SAROilSpillDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print(f"  -> Training samples: {len(train_dataset)}")
    print(f"  -> Validation samples: {len(val_dataset)}")

    # 2. Instantiate Tailored Single-Channel U-Net
    print("\n[2/4] Initializing Single-Channel U-Net Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAROilSpillUNet(
        in_channels=1,
        out_channels=1,
        features=[32, 64, 128, 256, 512],
        dropout_p=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Total Trainable Parameters: {total_params:,}")
    print(f"  -> Execution Device: {device}")

    # 3. Setup Combined Loss & Optimizer
    print("\n[3/4] Configuring Combined BCEDiceLoss & AdamW Optimizer...")
    criterion = BCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        smooth=1e-6
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. Execute Training and Validation
    print("\n[4/4] Starting Model Training & Validation Loop...")
    trainer = OilSpillTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        threshold=0.5
    )

    history = trainer.fit(num_epochs=args.epochs)
    print("\n=== Phase 2 Pipeline Successfully Completed! ===")


if __name__ == "__main__":
    main()
