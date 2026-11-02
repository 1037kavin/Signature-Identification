"""
Conditional Variational Autoencoder (cVAE) for Wafer Map Defect Generation
===========================================================================
Implements the inverse/vice-versa paradigm:
Given a defect label (e.g. 'center', 'scratch', 'edge-ring') and a random latent vector z,
the model generates a realistic semiconductor wafer map displaying that spatial defect signature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class WaferEncoder(nn.Module):
    """
    Conditional Encoder: Encodes (Wafer Map, Failure Class) into latent distribution parameters (μ, log σ²).
    """
    def __init__(self, img_channels: int = 1, num_classes: int = 8, latent_dim: int = 64, img_size: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes

        # Spatially project class label into a 2D channel (B, 1, H, W)
        self.class_projection = nn.Sequential(
            nn.Linear(num_classes, img_size * img_size),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional backbone: takes (wafer + class channel) = 2 channels
        self.conv_net = nn.Sequential(
            # Input: (B, 2, 32, 32)
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1),  # -> (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (B, 128, 4, 4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()  # -> (B, 128 * 4 * 4 = 2048)
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        # Project class label to spatial feature map (B, 1, 32, 32)
        y_spatial = self.class_projection(y_onehot).view(B, 1, self.img_size, self.img_size)
        
        # Concatenate wafer map and conditioned class map
        xy = torch.cat([x, y_spatial], dim=1)  # (B, 2, 32, 32)
        features = self.conv_net(xy)

        mu = self.fc_mu(features)
        logvar = self.fc_logvar(features)
        return mu, logvar


class WaferDecoder(nn.Module):
    """
    Conditional Decoder (Generator): Decodes (Latent vector z, Failure Class) into a synthetic wafer map.
    """
    def __init__(self, latent_dim: int = 64, num_classes: int = 8, out_channels: int = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # Map combined (z + onehot) to spatial feature volume (B, 128, 4, 4)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * 4 * 4),
            nn.BatchNorm1d(128 * 4 * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Transposed convolutions to upsample back to 32x32
        self.deconv_net = nn.Sequential(
            # Input: (B, 128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # -> (B, 64, 8, 8)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # -> (B, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # -> (B, 16, 32, 32)
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, out_channels, kernel_size=3, stride=1, padding=1),   # -> (B, 1, 32, 32)
            nn.Sigmoid()  # Normalized range [0.0, 1.0]
        )

    def forward(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        zy = torch.cat([z, y_onehot], dim=1)
        feat = self.fc(zy).view(-1, 128, 4, 4)
        out = self.deconv_net(feat)
        return out


class ConditionalVAE(nn.Module):
    """
    Complete Conditional VAE Model for Wafer Maps.
    """
    def __init__(
        self,
        img_channels: int = 1,
        num_classes: int = 8,
        latent_dim: int = 64,
        img_size: int = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.encoder = WaferEncoder(
            img_channels=img_channels,
            num_classes=num_classes,
            latent_dim=latent_dim,
            img_size=img_size
        )
        self.decoder = WaferDecoder(
            latent_dim=latent_dim,
            num_classes=num_classes,
            out_channels=img_channels
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z ~ N(mu, std^2) using the reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x, y_onehot)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, y_onehot)
        return x_recon, mu, logvar

    def loss_function(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates ELBO Loss = Reconstruction Loss + beta * KL Divergence.
        Reconstruction uses BCE loss over pixel intensities.
        """
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        
        # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss

    @torch.no_grad()
    def sample(
        self,
        class_idx: int,
        n_samples: int = 1,
        device: str = "cpu",
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generates n_samples synthetic wafer maps for the specified failure class index.
        """
        self.eval()
        z = torch.randn(n_samples, self.latent_dim, device=device) * temperature
        y_onehot = torch.zeros(n_samples, self.num_classes, device=device)
        y_onehot[:, class_idx] = 1.0
        synthetic_wafers = self.decoder(z, y_onehot)
        return synthetic_wafers

    @torch.no_grad()
    def interpolate(
        self,
        class_a: int,
        class_b: int,
        steps: int = 10,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Morphs smoothly from class A to class B in latent + condition space.
        """
        self.eval()
        z = torch.randn(1, self.latent_dim, device=device).repeat(steps, 1)
        alphas = torch.linspace(0.0, 1.0, steps, device=device)

        y_onehot = torch.zeros(steps, self.num_classes, device=device)
        for i, alpha in enumerate(alphas):
            y_onehot[i, class_a] = 1.0 - alpha
            y_onehot[i, class_b] = alpha

        morph_wafers = self.decoder(z, y_onehot)
        return morph_wafers
