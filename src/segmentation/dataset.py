"""
Module: dataset.py
Description: PyTorch Dataset for loading single-channel Sentinel-1 SAR patches
             and corresponding binary oil spill segmentation masks with augmentations.
"""

import os
import glob
from typing import Tuple, List, Optional, Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SAROilSpillDataset(Dataset):
    """
    Dataset for single-channel SAR image patches (e.g., VV or VH polarizations)
    and binary ground-truth segmentation masks (0: Background/Ocean, 1: Oil Spill).
    
    Supports .npy numpy arrays, .pt torch tensors, and standard image formats (.png, .tif).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        patch_size: Optional[Tuple[int, int]] = (256, 256),
        transform: Optional[Callable] = None,
        is_train: bool = True
    ):
        """
        Args:
            images_dir: Directory containing SAR image patches.
            masks_dir: Directory containing corresponding binary mask patches.
            patch_size: Target (H, W) for safety resizing/validation.
            transform: Optional custom augmentation transform callable.
            is_train: Whether to apply default geospatial data augmentations (flips/rotations).
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.patch_size = patch_size
        self.transform = transform
        self.is_train = is_train

        # Supported file extensions
        supported_exts = ("*.npy", "*.npz", "*.pt", "*.png", "*.tif", "*.tiff")
        self.image_paths: List[str] = []
        for ext in supported_exts:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        self.image_paths.sort()

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No valid image patches found in directory: {images_dir}")

        # Pair images with masks by filename stem
        self.samples: List[Tuple[str, str]] = []
        for img_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # Look for matching mask file with any supported extension or with '_mask' suffix
            mask_candidates = [
                os.path.join(masks_dir, f"{base_name}{ext}")
                for ext in [".npy", ".pt", ".png", ".tif", ".tiff"]
            ] + [
                os.path.join(masks_dir, f"{base_name}_mask{ext}")
                for ext in [".npy", ".pt", ".png", ".tif", ".tiff"]
            ]

            matched_mask = None
            for cand in mask_candidates:
                if os.path.exists(cand):
                    matched_mask = cand
                    break

            if matched_mask:
                self.samples.append((img_path, matched_mask))
            else:
                raise FileNotFoundError(f"Corresponding mask not found for image patch: {img_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file(self, file_path: str) -> np.ndarray:
        """Loads data from .npy, .pt, or standard raster formats into a float32 numpy array."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path)
        elif ext == ".npz":
            data = np.load(file_path)
            arr = data[data.files[0]]
        elif ext == ".pt":
            tensor = torch.load(file_path, map_location="cpu")
            arr = tensor.numpy()
        else:
            # Load via PIL for standard image formats (.png, .tif)
            with Image.open(file_path) as img:
                arr = np.array(img)

        arr = arr.astype(np.float32)
        # Squeeze potential redundant channel dimensions: (H, W, 1) -> (H, W)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr.squeeze(-1)
        elif arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr.squeeze(0)
        return arr

    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies physically consistent geospatial augmentations for SAR imagery:
        - Random horizontal flip
        - Random vertical flip
        - Random orthogonal 90-degree rotations
        """
        # Random Horizontal Flip
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        # Random Vertical Flip
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        # Random 90-degree rotations (0, 90, 180, 270 deg)
        rot_k = np.random.choice([0, 1, 2, 3])
        if rot_k > 0:
            image = np.rot90(image, k=rot_k)
            mask = np.rot90(mask, k=rot_k)

        return image.copy(), mask.copy()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # 1. Load image patch and mask
        image = self._load_file(img_path)
        mask = self._load_file(mask_path)

        # 2. Ensure image values are normalized in [0, 1]
        img_min, img_max = image.min(), image.max()
        if img_max > 1.0 or img_min < 0.0:
            denom = (img_max - img_min) if (img_max - img_min) > 1e-6 else 1.0
            image = (image - img_min) / denom

        # 3. Ensure binary mask values (0.0 or 1.0)
        mask = (mask > 0.5).astype(np.float32)

        # 4. Apply training augmentations
        if self.is_train:
            if self.transform is not None:
                image, mask = self.transform(image, mask)
            else:
                image, mask = self._apply_augmentations(image, mask)

        # 5. Format to PyTorch Tensors: shape (C, H, W) -> (1, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor

