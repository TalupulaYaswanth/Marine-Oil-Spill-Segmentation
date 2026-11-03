"""
Module: unet.py
Description: PyTorch U-Net architecture tailored for binary semantic segmentation 
             of marine oil spills from single-channel SAR imagery patches.
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    [Conv2d -> BatchNorm -> ReLU] * 2 block.
    Standard building block for both encoder and decoder stages.
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None, dropout_p: float = 0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, dropout_p=dropout_p)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling followed by skip-connection concatenation and DoubleConv.
    Supports either learnable Transposed Convolution or Bilinear interpolation.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False, dropout_p: float = 0.0):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout_p=dropout_p)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout_p=dropout_p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: Feature map from decoder path (to be upscaled).
            x2: Skip connection feature map from corresponding encoder path.
        """
        x1 = self.up(x1)

        # Handle potential odd dimension rounding mismatch via reflection/zero padding
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        if diff_y != 0 or diff_x != 0:
            x1 = F.pad(
                x1,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution mapping latent features to target output logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SAROilSpillUNet(nn.Module):
    """
    U-Net tailored for marine oil spill detection from single-channel SAR imagery.

    Default Features: [32, 64, 128, 256, 512] (balanced for fast training & memory efficiency)
    Outputs: Raw unnormalized logits (use Sigmoid for probabilities).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        bilinear: bool = False,
        dropout_p: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input SAR channels (default: 1 for single-channel VV or VH).
            out_channels: Number of output classes (default: 1 for binary segmentation logits).
            features: List of feature map depths across hierarchy levels.
            bilinear: Use bilinear upsampling instead of transposed convolutions.
            dropout_p: Dropout probability in bottleneck and deeper layers to curb speckle overfitting.
        """
        super().__init__()
        if features is None:
            features = [32, 64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        f = features
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, f[0], dropout_p=0.0)
        self.down1 = Down(f[0], f[1], dropout_p=0.0)
        self.down2 = Down(f[1], f[2], dropout_p=dropout_p * 0.5)
        self.down3 = Down(f[2], f[3], dropout_p=dropout_p)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = Down(f[3], f[4] // factor, dropout_p=dropout_p)

        # Decoder (Expanding Path)
        self.up1 = Up(f[4], f[3] // factor, bilinear=bilinear, dropout_p=dropout_p)
        self.up2 = Up(f[3], f[2] // factor, bilinear=bilinear, dropout_p=dropout_p * 0.5)
        self.up3 = Up(f[2], f[1] // factor, bilinear=bilinear, dropout_p=0.0)
        self.up4 = Up(f[1], f[0], bilinear=bilinear, dropout_p=0.0)

        # Final Classifier Projection (Logits)
        self.outc = OutConv(f[0], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning unnormalized logits.
        Input shape:  (B, 1, H, W)
        Output shape: (B, 1, H, W)
        """
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

