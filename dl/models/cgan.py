"""
Conditional Generative Adversarial Network (cDCGAN) for Wafer Map Generation
=============================================================================
Provides an alternative generative architecture using adversarial minimax training:
Generator generates synthetic wafer maps from noise conditioned on class labels,
while Discriminator learns to distinguish real wafers from fake ones given the label.
"""

import torch
import torch.nn as nn
from typing import Tuple


class CGANGenerator(nn.Module):
    """
    Conditional Generator: maps (latent noise z, class condition y) -> synthetic wafer map (1, 32, 32)
    """
    def __init__(self, latent_dim: int = 100, num_classes: int = 8, img_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.label_emb = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 64, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # Input: (B, 256, 4, 4)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),   # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid()  # Pixel range [0, 1]
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        y_feat = self.label_emb(y_onehot)
        zy = torch.cat([z, y_feat], dim=1)
        feat = self.fc(zy).view(-1, 256, 4, 4)
        return self.deconv(feat)


class CGANDiscriminator(nn.Module):
    """
    Conditional Discriminator: maps (wafer map, class condition y) -> probability of being real.
    """
    def __init__(self, img_channels: int = 1, num_classes: int = 8, img_size: int = 32):
        super().__init__()
        self.img_size = img_size

        self.label_proj = nn.Sequential(
            nn.Linear(num_classes, img_size * img_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv = nn.Sequential(
            # Input: (B, 2, 32, 32)
            nn.Conv2d(img_channels + 1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        y_spatial = self.label_proj(y_onehot).view(B, 1, self.img_size, self.img_size)
        xy = torch.cat([x, y_spatial], dim=1)
        return self.conv(xy)
