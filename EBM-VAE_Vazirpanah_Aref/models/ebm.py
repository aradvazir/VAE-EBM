"""
Energy-Based Model (EBM) operating in the VAE latent space.

Architecture:
  z ∈ ℝ^latent_dim  →  MLP with spectral norm  →  scalar energy E(z)

Design notes:
  - Spectral normalisation on every Linear layer keeps the Lipschitz constant
    bounded, which stabilises Langevin dynamics (the energy landscape does not
    become too jagged for SGLD to navigate).
  - Swish / SiLU activation: smooth, non-monotonic, works well for EBMs.
  - No BatchNorm — it interacts badly with SGLD because normalisation
    statistics depend on the batch, which includes fantasy particles.
  - LayerNorm is safe and helps prevent activation saturation.
  - Output is a scalar (no sigmoid / softmax) — raw energy E(z).
    Lower energy = more probable under p_θ(z) ∝ exp(−E_θ(z)) · p₀(z).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class EBM(nn.Module):
    """Small residual MLP that maps z → scalar energy.

    Args:
        latent_dim:  Dimensionality of VAE latent space (must match VAE).
        hidden_dim:  Width of hidden layers.
        num_layers:  Total number of linear layers (depth).
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "Need at least input + output layer"
        self.latent_dim = latent_dim

        dims = [latent_dim] + [hidden_dim] * (num_layers - 1) + [1]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            fc = spectral_norm(nn.Linear(dims[i], dims[i + 1]))
            layers.append(fc)
            if i < len(dims) - 2:            # no norm/activation after last layer
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

        # Skip connection: z → 1, stabilises gradient flow
        self.skip = spectral_norm(nn.Linear(latent_dim, 1, bias=False))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)
        Returns:
            energy: (B,)  scalar energy per sample
        """
        return (self.net(z) + self.skip(z)).squeeze(-1)

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        """Alias for forward — more readable in training code."""
        return self(z)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
