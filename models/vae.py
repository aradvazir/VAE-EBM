"""
Convolutional β-VAE for 64×64 anime faces.

Architecture:
  Encoder: Conv(64)→Conv(128)→Conv(256)→Conv(512) + ResBlocks → FC → (μ, log_var)
  Decoder: FC → ConvTranspose(512)→(256)→(128)→(64) + ResBlocks → Tanh

The encoder/decoder are symmetric.  GroupNorm + SiLU throughout for stability on MPS.
"""
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────── building blocks ─────────────────────────────────

class ResBlock(nn.Module):
    """Pre-activation residual block with GroupNorm + SiLU."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


def _norm_act(channels: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(nn.GroupNorm(groups, channels), nn.SiLU())


# ────────────────────────────── Encoder ──────────────────────────────────────

class Encoder(nn.Module):
    """Image → (μ, log_var).

    Spatial: 64 → 32 → 16 → 8 → 4  (four stride-2 convolutions)
    Channels: 3 → C → 2C → 4C → 8C
    Flatten → two linear heads: μ and log_var of shape (batch, latent_dim)
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        C = base_channels
        self.latent_dim = latent_dim

        self.net = nn.Sequential(
            # 64 → 32
            nn.Conv2d(in_channels, C, 4, stride=2, padding=1),
            nn.SiLU(),
            ResBlock(C, groups=min(8, C)),

            # 32 → 16
            nn.Conv2d(C, C * 2, 4, stride=2, padding=1),
            _norm_act(C * 2),
            ResBlock(C * 2),

            # 16 → 8
            nn.Conv2d(C * 2, C * 4, 4, stride=2, padding=1),
            _norm_act(C * 4),
            ResBlock(C * 4),

            # 8 → 4
            nn.Conv2d(C * 4, C * 8, 4, stride=2, padding=1),
            _norm_act(C * 8),
            ResBlock(C * 8),
        )

        # Spatial size after 4 stride-2 downsamples: image_size / 16
        flat_hw   = image_size // 16           # = 4 for 64×64
        flat_dim  = C * 8 * flat_hw * flat_hw  # = 512 * 4 * 4 = 8192

        self.fc_mu     = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


# ────────────────────────────── Decoder ──────────────────────────────────────

class Decoder(nn.Module):
    """z → image in [-1, 1].

    Spatial: 4 → 8 → 16 → 32 → 64  (four stride-2 transposed convolutions)
    Channels: 8C → 4C → 2C → C → 3
    """

    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 256,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        C  = base_channels
        hw = image_size // 16   # starting spatial size = 4

        self.fc      = nn.Linear(latent_dim, C * 8 * hw * hw)
        self.start_c = C * 8
        self.start_s = hw

        self.net = nn.Sequential(
            ResBlock(C * 8),

            # 4 → 8
            nn.ConvTranspose2d(C * 8, C * 4, 4, stride=2, padding=1),
            _norm_act(C * 4),
            ResBlock(C * 4),

            # 8 → 16
            nn.ConvTranspose2d(C * 4, C * 2, 4, stride=2, padding=1),
            _norm_act(C * 2),
            ResBlock(C * 2),

            # 16 → 32
            nn.ConvTranspose2d(C * 2, C, 4, stride=2, padding=1),
            _norm_act(C),
            ResBlock(C, groups=min(8, C)),

            # 32 → 64
            nn.ConvTranspose2d(C, out_channels, 4, stride=2, padding=1),
            nn.Tanh(),   # output ∈ [-1, 1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.start_c, self.start_s, self.start_s)
        return self.net(h)


# ─────────────────────────────── VAE ─────────────────────────────────────────

class VAE(nn.Module):
    """β-VAE combining Encoder + Decoder with the standard ELBO objective.

    Loss = E_q[log p(x|z)]  −  β · KL(q(z|x) ∥ p(z))
         = −MSE(x, x̂)       −  β · (-0.5 · Σ(1 + log_var − μ² − var))

    During training: z sampled via reparameterisation trick.
    During eval   : z = μ  (deterministic).
    """

    def __init__(
        self,
        in_channels: int   = 3,
        base_channels: int = 64,
        latent_dim: int    = 256,
        image_size: int    = 64,
        beta: float        = 1.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.beta       = beta

        self.encoder = Encoder(in_channels, base_channels, latent_dim, image_size)
        self.decoder = Decoder(in_channels, base_channels, latent_dim, image_size)

    # ── core methods ──────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            return mu + std * torch.randn_like(std)
        return mu  # no noise at eval time

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        x_recon     = self.decode(z)
        return x_recon, mu, log_var

    # ── loss ─────────────────────────────────────────────────────────

    def loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total, recon_loss, kl_loss)  — all per-sample means."""
        if beta is None:
            beta = self.beta

        B = x.size(0)

        # Reconstruction: sum over pixels, mean over batch
        recon = F.mse_loss(x_recon, x, reduction="sum") / B

        # KL divergence: -0.5 * Σ_j (1 + log_var_j - μ_j² - var_j)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(1).mean()

        total = recon + beta * kl
        return total, recon, kl

    # ── generation helpers ────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """Sample n images from the prior N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode (deterministic at eval)."""
        self.eval()
        mu, _ = self.encode(x)
        return self.decode(mu)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
