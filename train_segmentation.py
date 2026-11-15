"""
train_segmentation.py
Main driver script for Phase 2: Oil Spill Semantic Segmentation.

Demonstrates:
  - Custom SAROilSpillDataset initialization with augmentations.
  - U-Net instantiation for single-channel SAR input.
  - Combined BCEDiceLoss formulation for severe class imbalance.
  - Training, validation, IoU / F1 tracking, and checkpointing.
  - Graceful fallback for non-PyTorch environments with operational EMSA CFAR evaluation.
"""

import os
import sys
import argparse
from typing import Tuple

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from generate_sample_patches import generate_synthetic_patches
from src.segmentation.inference import OilSpillSegmenter

# Try importing torch
try:
    import torch
    from torch.utils.data import DataLoader
    from src.segmentation.dataset import SAROilSpillDataset
    from src.segmentation.unet import SAROilSpillUNet
    from src.segmentation.losses import BCEDiceLoss
    from src.segmentation.trainer import OilSpillTrainer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


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

    # Generate synthetic patches if needed
    if not args.train_images or not os.path.exists(args.train_images):
        train_img_dir, train_mask_dir, val_img_dir, val_mask_dir = generate_synthetic_patches()
    else:
        train_img_dir, train_mask_dir = args.train_images, args.train_masks
        val_img_dir, val_mask_dir = args.val_images, args.val_masks

    if not HAS_TORCH:
        print("\n" + "="*70)
        print("ℹ️  NOTE: PyTorch is not currently installed in this Python environment.")
        print("    Running Phase 2 Operational EMSA CleanSeaNet CFAR Segmentation Evaluator...")
        print("    To enable GPU/CPU Deep Learning U-Net training, run:")
        print("        pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("="*70 + "\n")

        # Evaluate validation set with CFAR detector
        segmenter = OilSpillSegmenter(detection_threshold=0.50)
        import tifffile
        val_files = [f for f in os.listdir(val_img_dir) if f.endswith(".npy") or f.endswith(".tif")][:5]
        print(f"[Phase 2 Validation] Evaluating {len(val_files)} SAR validation scenes with CFAR detector:")

        for vf in val_files:
            vpath = os.path.join(val_img_dir, vf)
            if vf.endswith(".npy"):
                import numpy as np
                img = np.load(vpath)
            else:
                import tifffile
                img = tifffile.imread(vpath)

            res = segmenter.segment_sar_image(img)
            print(f"  -> Scene: {vf[:25]} | Slick Area: {res['slick_area_km2']:.2f} km² | Damping Contrast: {res['damping_contrast_db']} dB | Conf: {res['mean_confidence']:.3f}")

        print("\n=== Phase 2 Validation Benchmark Completed Successfully! ===")
        return

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
