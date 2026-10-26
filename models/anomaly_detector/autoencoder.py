"""
autoencoder.py
==============
Convolutional Autoencoder (ConvAutoencoder) for Unsupervised Tissue Baseline
Reconstruction and Anomaly Detection in Point-of-Care Medical Imaging.

Trained strictly on healthy/normal tissue scans. Pathological tissue lesions
produce high reconstruction error when subtracted from the input, generating
spatial pixel-wise anomaly heatmaps via the C++ AnomalyCalculus engine.
"""

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Continuous Convolutional Autoencoder for Normal Tissue Baseline Reconstruction.

    Args:
        in_channels: Number of input channels (1 for grayscale OCT/ultrasound).
    """
    def __init__(self, in_channels: int = 1):
        super(ConvAutoencoder, self).__init__()
        self.in_channels = in_channels

        # --- Encoder Layers ---
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),          # [B, 32, H/4, W/4]
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),          # [B, 64, H/4, W/4] bottleneck
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
        )

        # --- Decoder Layers ---
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 32, H/2, W/2]
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # [B, 16, H, W]
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward reconstruction pass.
        Input shape:  [Batch, Channels, Height, Width]
        Output shape: [Batch, Channels, Height, Width]
        """
        latent = self.enc(x)
        reconstruction = self.dec(latent)
        return reconstruction


# Backward-compatible alias for existing pipelines
TissueAutoencoder = ConvAutoencoder
