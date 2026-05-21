from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


NormName = Literal["instance", "group", "batch", "none"]


def _make_norm(num_channels: int, norm: NormName) -> nn.Module:
    """Create a 3D normalization layer suitable for small-batch MRI training."""
    if norm == "instance":
        # InstanceNorm is usually safer than BatchNorm for 3D MRI, where batch size is often 1--2.
        return nn.InstanceNorm3d(num_channels, affine=True, track_running_stats=False)
    if norm == "group":
        # Pick a valid group count no larger than 8.
        num_groups = min(8, num_channels)
        while num_channels % num_groups != 0:
            num_groups -= 1
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    if norm == "batch":
        return nn.BatchNorm3d(num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm='{norm}'. Expected one of: instance, group, batch, none.")


class DoubleConv(nn.Module):
    """
    Two 3D convolutions with normalization and LeakyReLU.

    If ``first_stride=2``, the block performs learned downsampling in the first
    convolution. This keeps the encoder close to the original implementation,
    which used strided convolutions rather than max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        first_stride: int = 1,
        norm: NormName = "instance",
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if first_stride not in (1, 2):
            raise ValueError(f"first_stride must be 1 or 2, got {first_stride}.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=first_stride, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Unet(nn.Module):
    """
    Lightweight 3D U-Net for SimCortex segmentation.

    Expected shapes
    ---------------
    Input:
        ``[B, c_in, D, H, W]``
    Output:
        raw logits ``[B, c_out, D, H, W]``

    Notes
    -----
    - ``c_out=9`` matches SimCortex seg labels: background 0 plus classes 1--8.
    - The total encoder downsampling factor is 16, so dataloader ``pad_mult=16`` is sufficient.
    - No softmax is applied here. Use raw logits with ``torch.nn.CrossEntropyLoss``.
    """

    def __init__(
        self,
        c_in: int = 1,
        c_out: int = 9,
        features: Sequence[int] = (16, 32, 64, 128, 128),
        norm: NormName = "instance",
        negative_slope: float = 0.2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(features) != 5:
            raise ValueError(f"features must contain 5 channel values, got {features}.")
        if c_in < 1:
            raise ValueError(f"c_in must be >= 1, got {c_in}.")
        if c_out < 2:
            raise ValueError(f"c_out must be >= 2, got {c_out}.")

        f1, f2, f3, f4, f5 = [int(v) for v in features]

        # Encoder: learned downsampling with stride-2 first conv at levels 2--5.
        self.enc1 = DoubleConv(c_in, f1, first_stride=1, norm=norm, negative_slope=negative_slope)
        self.enc2 = DoubleConv(f1, f2, first_stride=2, norm=norm, negative_slope=negative_slope)
        self.enc3 = DoubleConv(f2, f3, first_stride=2, norm=norm, negative_slope=negative_slope)
        self.enc4 = DoubleConv(f3, f4, first_stride=2, norm=norm, negative_slope=negative_slope)
        self.enc5 = DoubleConv(f4, f5, first_stride=2, norm=norm, negative_slope=negative_slope, dropout=dropout)

        # Decoder: trilinear upsample, skip concatenation, then double-conv refinement.
        # Upsample is stateless, so sharing this module across decoder levels is safe.
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec4 = DoubleConv(f5 + f4, f3, norm=norm, negative_slope=negative_slope)
        self.dec3 = DoubleConv(f3 + f3, f2, norm=norm, negative_slope=negative_slope)
        self.dec2 = DoubleConv(f2 + f2, f1, norm=norm, negative_slope=negative_slope)
        self.dec1 = DoubleConv(f1 + f1, f1, norm=norm, negative_slope=negative_slope)

        self.head = nn.Sequential(
            nn.Conv3d(f1, f1, kernel_size=3, padding=1, bias=False),
            _make_norm(f1, norm),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Conv3d(f1, c_out, kernel_size=1),
        )

    @staticmethod
    def _resize_to(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Resize decoder feature maps to exactly match the skip-connection shape."""
        if src.shape[2:] != tgt.shape[2:]:
            return F.interpolate(src, size=tgt.shape[2:], mode="trilinear", align_corners=False)
        return src

    @staticmethod
    def _check_input(x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"Expected input shape [B, C, D, H, W], got {tuple(x.shape)}.")
        if min(x.shape[2:]) < 16:
            raise ValueError(
                f"Input spatial shape {tuple(x.shape[2:])} is too small for 4 downsampling levels. "
                "Use spatial dimensions >= 16 or reduce the network depth."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)

        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        y = self.up(x5)
        y = self._resize_to(y, x4)
        y = self.dec4(torch.cat([y, x4], dim=1))

        y = self.up(y)
        y = self._resize_to(y, x3)
        y = self.dec3(torch.cat([y, x3], dim=1))

        y = self.up(y)
        y = self._resize_to(y, x2)
        y = self.dec2(torch.cat([y, x2], dim=1))

        y = self.up(y)
        y = self._resize_to(y, x1)
        y = self.dec1(torch.cat([y, x1], dim=1))

        return self.head(y)


# Optional alias for users who expect the conventional capitalization.
UNet = Unet


__all__ = ["Unet", "UNet", "DoubleConv"]
