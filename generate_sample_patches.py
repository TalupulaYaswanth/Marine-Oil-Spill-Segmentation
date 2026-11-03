"""
generate_sample_patches.py
Utility script to generate synthetic single-channel SAR image patches and binary masks
for training and evaluating the U-Net segmentation model without external dependencies.
"""

import os
import sys
from typing import Tuple
import numpy as np

# Ensure UTF-8 output encoding on Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def generate_synthetic_patches(
    output_root: str = "data/patches",
    num_train: int = 30,
    num_val: int = 10,
    patch_size: int = 256,
    seed: int = 42
) -> Tuple[str, str, str, str]:
    """
    Synthesizes a realistic dataset of SAR image patches and corresponding binary masks:
      - Simulates marine clutter background using Gamma distribution.
      - Injects dark dampening oil slicks (curved ribbons / rotated ellipses).
      - Injects negative patches (calm ocean look-alikes without spills).
      - Adds multiplicative radar speckle noise.
    """
    np.random.seed(seed)

    for split, count in [("train", num_train), ("val", num_val)]:
        img_dir = os.path.join(output_root, split, "images")
        mask_dir = os.path.join(output_root, split, "masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for i in range(count):
            # 1. Background sea clutter (normalized [0.25, 0.80])
            clutter = np.random.gamma(shape=4.0, scale=0.1, size=(patch_size, patch_size)).astype(np.float32)
            clutter = (clutter - clutter.min()) / (clutter.max() - clutter.min() + 1e-6)
            clutter = 0.25 + 0.55 * clutter

            mask = np.zeros((patch_size, patch_size), dtype=np.float32)

            # 75% of patches contain spills, 25% are negative background/look-alike samples
            if np.random.rand() < 0.75:
                # Random center, semi-axes, and orientation
                cx = np.random.randint(60, patch_size - 60)
                cy = np.random.randint(60, patch_size - 60)
                rx = np.random.randint(25, 70)
                ry = np.random.randint(8, 24)
                angle = np.random.uniform(0, np.pi)

                y, x = np.ogrid[:patch_size, :patch_size]
                dx = x - cx
                dy = y - cy
                
                # Rotated coordinate ellipse simulating wind/current elongated slick
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                xr = dx * cos_a + dy * sin_a
                yr = -dx * sin_a + dy * cos_a

                ellipse = ((xr / rx) ** 2 + ((yr / ry) * 1.5) ** 2) <= 1.0
                mask[ellipse] = 1.0

                # Oil spill dampens capillary waves, drastically dropping backscatter
                clutter[ellipse] *= np.random.uniform(0.18, 0.32)

            # 2. Add realistic multiplicative radar speckle noise
            speckle = np.random.normal(1.0, 0.08, size=(patch_size, patch_size)).astype(np.float32)
            sar_patch = np.clip(clutter * speckle, 0.0, 1.0)

            # Save as standard float32 numpy arrays
            np.save(os.path.join(img_dir, f"patch_{i:04d}.npy"), sar_patch)
            np.save(os.path.join(mask_dir, f"patch_{i:04d}_mask.npy"), mask)

    print(f"Generated {num_train} train and {num_val} val synthetic SAR patches at: {output_root}")
    return (
        os.path.join(output_root, "train", "images"),
        os.path.join(output_root, "train", "masks"),
        os.path.join(output_root, "val", "images"),
        os.path.join(output_root, "val", "masks"),
    )


if __name__ == "__main__":
    generate_synthetic_patches()

