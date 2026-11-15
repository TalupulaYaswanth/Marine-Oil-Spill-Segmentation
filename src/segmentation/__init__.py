"""
Deep Learning-Based Marine Oil Spill Segmentation and Vessel Attribution.
Phase 2: Oil Spill Semantic Segmentation Module.
"""

from .inference import OilSpillSegmenter

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .dataset import SAROilSpillDataset
    from .unet import SAROilSpillUNet, DoubleConv, Down, Up, OutConv
    from .losses import DiceLoss, BCEDiceLoss, SegmentationMetrics, MetricTracker
    from .trainer import OilSpillTrainer

    __all__ = [
        "OilSpillSegmenter",
        "SAROilSpillDataset",
        "SAROilSpillUNet",
        "DoubleConv",
        "Down",
        "Up",
        "OutConv",
        "DiceLoss",
        "BCEDiceLoss",
        "SegmentationMetrics",
        "MetricTracker",
        "OilSpillTrainer"
    ]
else:
    __all__ = [
        "OilSpillSegmenter"
    ]
