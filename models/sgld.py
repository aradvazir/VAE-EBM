"""
Stochastic Gradient Langevin Dynamics (SGLD) sampler for the EBM prior.

The update rule for each step t is:

    z_{t+1} = z_t  −  (η/2) · ∇_z E_θ(z_t)  +  √η · ε,    ε ~ N(0, I)

This is Langevin dynamics with step size η.  With small η and many steps it
converges to samples from p_θ(z) ∝ exp(−E_θ(z)).

Persistent Replay Buffer
────────────────────────
Initialising every MCMC chain from N(0, I) ("short-run" CD) is fast but the
chains rarely mix to the true distribution — they just drift a few steps from
the prior.  A replay buffer stores fantasy particles between training steps so
each chain can continue from where it left off.  This trades a small memory
cost for dramatically better sample quality and more informative gradients for
the EBM.

With probability `reinit_prob` a slot is re-initialised from N(0, I) to
prevent the buffer from collapsing to a single mode.

Usage:
    sampler = SGLDSampler(buffer_size=10_000, latent_dim=256, device=device)
    z_neg = sampler.sample(ebm, n_samples=64)  # returns (64, 256)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SGLDSampler:
    """Persistent-chain SGLD sampler.

    Args:
        buffer_size:   Number of persistent fantasy particles stored.
        latent_dim:    Latent space dimension (must match VAE + EBM).
        device:        Torch device.
        step_size:     SGLD step size η.
        noise_scale:   Noise multiplier σ (applied to √η · ε).
        n_steps:       Number of Langevin steps per call.
        reinit_prob:   Probability of re-initialising a slot from N(0,I).
        grad_clip:     Max norm of ∇_z E before clipping (prevents explosions).
    """

    def __init__(
        self,
        buffer_size: int   = 10_000,
        latent_dim: int    = 256,
        device: torch.device | str = "cpu",
        step_size: float   = 0.1,
        noise_scale: float = 0.005,
        n_steps: int       = 60,
        reinit_prob: float = 0.05,
        grad_clip: float   = 0.03,
    ) -> None:
        self.buffer_size  = buffer_size
        self.latent_dim   = latent_dim
        self.device       = torch.device(device)
        self.step_size    = step_size
        self.noise_scale  = noise_scale
        self.n_steps      = n_steps
        self.reinit_prob  = reinit_prob
        self.grad_clip    = grad_clip

        # Persistent buffer: stored on CPU to save GPU memory, moved on demand
        self.buffer = torch.randn(buffer_size, latent_dim)
        self._ptr   = 0     # next write position (circular)

    # ── public API ─────────────────────────────────────────────────────

    def sample(
        self,
        ebm: nn.Module,
        n_samples: int,
        training: bool = True,
    ) -> torch.Tensor:
        """Draw n_samples fantasy particles from the EBM prior.

        Returns detached tensor of shape (n_samples, latent_dim) on self.device.
        """
        # Sample indices from buffer (with replacement if n > buffer_size)
        idx = torch.randint(0, self.buffer_size, (n_samples,))

        # Optionally re-initialise some slots from N(0, I)
        reinit_mask = torch.rand(n_samples) < self.reinit_prob
        self.buffer[idx[reinit_mask]] = torch.randn(reinit_mask.sum(), self.latent_dim)

        z = self.buffer[idx].to(self.device).detach()

        # Run Langevin chain
        z = self._langevin(ebm, z)

        # Write updated particles back to buffer (on CPU)
        self.buffer[idx] = z.cpu().detach()

        return z.detach()

    # ── internals ──────────────────────────────────────────────────────

    def _langevin(self, ebm: nn.Module, z_init: torch.Tensor) -> torch.Tensor:
        """Run n_steps of SGLD starting from z_init.

        Uses torch.enable_grad() so that autograd.grad works correctly even
        when this method is called from inside a torch.no_grad() context
        (e.g. val_epoch).  The EBM parameters themselves never accumulate
        gradients here — only z does.
        """
        ebm.eval()
        z = z_init.clone()

        with torch.enable_grad():
            z.requires_grad_(True)
            for _ in range(self.n_steps):
                energy = ebm.energy(z).sum()
                grad   = torch.autograd.grad(energy, z)[0]

                # Clip per-sample gradient norm to stabilise dynamics
                grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale     = (self.grad_clip / grad_norm).clamp(max=1.0)
                grad      = grad * scale

                noise = torch.randn_like(z) * self.noise_scale
                z     = (z.detach() - (self.step_size / 2) * grad + noise)
                z.requires_grad_(True)

        ebm.train()
        return z.detach()

    def buffer_stats(self) -> dict:
        """Return mean / std of the replay buffer (useful for diagnostics)."""
        with torch.no_grad():
            return {
                "buf_mean": self.buffer.mean().item(),
                "buf_std":  self.buffer.std().item(),
            }