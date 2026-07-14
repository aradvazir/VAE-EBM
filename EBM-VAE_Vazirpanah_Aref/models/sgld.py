"""
Stochastic Gradient Langevin Dynamics (SGLD) sampler for the EBM prior.

The update rule for each step t is:

    z_{t+1} = z_t  −  (η/2) · ∇_z E_θ(z_t)  +  √η · ε,    ε ~ N(0, I)

Two modes
─────────
short_run=True  (high-dimensional latent spaces)
    Always starts from fresh N(0,I) noise.  No replay buffer.
    Fewer steps needed (20–30) but step_size should be larger (0.3–0.5).
    Avoids buffer collapse entirely — the most common failure mode when
    the VAE posterior is already close to N(0,I).

short_run=False  (persistent replay buffer)
    Stores fantasy particles between training steps.  Only beneficial
    when the EBM has a sharp, well-separated landscape.  With a well-
    trained VAE prior the buffer tends to collapse to data modes.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SGLDSampler:
    """SGLD sampler supporting both short-run and persistent-buffer modes.

    Args:
        buffer_size:   Number of persistent fantasy particles (ignored in short_run).
        latent_dim:    Latent space dimension.
        device:        Torch device.
        step_size:     SGLD step size η.
        noise_scale:   Noise multiplier σ (added each Langevin step).
        n_steps:       Number of Langevin steps per call.
        reinit_prob:   P(re-init a buffer slot from N(0,I)) — set 1.0 for short-run.
        grad_clip:     Max per-sample gradient norm before clipping.
        short_run:     If True, always start from fresh N(0,I); never use/update buffer.
    """

    def __init__(
        self,
        buffer_size: int   = 10_000,
        latent_dim: int    = 256,
        device             = "cpu",
        step_size: float   = 0.5,
        noise_scale: float = 0.1,
        n_steps: int       = 20,
        reinit_prob: float = 1.0,
        grad_clip: float   = 5.0,
        short_run: bool    = True,
    ) -> None:
        self.buffer_size  = buffer_size
        self.latent_dim   = latent_dim
        self.device       = torch.device(device)
        self.step_size    = step_size
        self.noise_scale  = noise_scale
        self.n_steps      = n_steps
        self.reinit_prob  = reinit_prob
        self.grad_clip    = grad_clip
        self.short_run    = short_run

        # Buffer only used when short_run=False
        self.buffer = torch.randn(buffer_size, latent_dim)

    # ── public API ─────────────────────────────────────────────────────

    def sample(
        self,
        ebm: nn.Module,
        n_samples: int,
        training: bool = True,
    ) -> torch.Tensor:
        """Draw n_samples fantasy particles from the EBM prior.

        Returns a detached tensor of shape (n_samples, latent_dim).
        When training=False the buffer is never written back (val-safe).
        """
        if self.short_run:
            # start from fresh noise
            z_init = torch.randn(n_samples, self.latent_dim, device=self.device)
            z_final = self._langevin(ebm, z_init)
            self._last_drift = (z_final - z_init).norm(dim=-1).mean().item()
            return z_final

        # ── persistent buffer path ──────────────────────────────────────
        idx = torch.randint(0, self.buffer_size, (n_samples,))

        # Stochastic re-init to prevent buffer collapse
        reinit_mask = torch.rand(n_samples) < self.reinit_prob
        if reinit_mask.any():
            self.buffer[idx[reinit_mask]] = torch.randn(
                int(reinit_mask.sum()), self.latent_dim
            )

        z_init_dev = self.buffer[idx].to(self.device).detach()
        z = self._langevin(ebm, z_init_dev)

        # Only write back during training
        if training:
            self.buffer[idx] = z.cpu().detach()

        self._last_drift = (z - z_init_dev).norm(dim=-1).mean().item()
        return z.detach()

    # ── internals ──────────────────────────────────────────────────────

    def _langevin(self, ebm: nn.Module, z_init: torch.Tensor) -> torch.Tensor:
        """Run n_steps of SGLD from z_init.

        Uses torch.enable_grad() so autograd.grad works inside no_grad contexts.
        EBM parameters never accumulate gradients here.
        """
        ebm.eval()
        z = z_init.clone().to(self.device)

        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            for _ in range(self.n_steps):
                energy = ebm.energy(z).sum()
                grad   = torch.autograd.grad(energy, z)[0]

                # Per-sample gradient clipping
                g_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                scale  = (self.grad_clip / g_norm).clamp(max=1.0)
                grad   = grad * scale

                noise = torch.randn_like(z) * self.noise_scale
                z     = (z.detach() - (self.step_size / 2) * grad + noise)
                z     = z.requires_grad_(True)

        ebm.train()
        return z.detach()

    def drift_stats(self) -> dict:
        """Return the average L2 drift from the last sample() call.

        For short_run mode: how far SGLD moved from fresh N(0,I) in n_steps.
        For buffer mode:    how far each chain moved from its buffer state.
        Higher drift = EBM gradient is pushing chains meaningfully.
        """
        return {"sgld_drift": getattr(self, "_last_drift", 0.0)}

    def buffer_stats(self) -> dict:
        d = self.drift_stats()
        return {"buf_mean": 0.0, "buf_std": d["sgld_drift"]}
