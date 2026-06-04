"""
Stochastic Gradient Langevin Dynamics (SGLD) for sampling from the EBM prior.

Target distribution:
    p_θ(z) ∝ exp(f_θ(z)) · p_0(z)   where p_0 = N(0, I)

The log-gradient of the target is:
    ∇_z log p_θ(z) = ∇_z f_θ(z)  +  ∇_z log p_0(z)
                   = ∇_z f_θ(z)  −  z

SGLD update (discretised Langevin):
    z_{t+1} = z_t  +  (η/2) · (∇_z f_θ(z_t) − z_t)  +  √η · ε_t
    ε_t ~ N(0, I)

Starting from z_0 ~ N(0, I) (the reference prior) and running K steps of
short-run MCMC gives approximate samples from p_θ.

Reference: Pang et al., NeurIPS 2020 — "Learning Latent Space EBM Prior Model"
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.enable_grad()   # ensure grad is on even when called inside torch.no_grad() contexts
def langevin_sample(
    ebm:        nn.Module,
    z_init:     torch.Tensor,
    steps:      int   = 60,
    step_size:  float = 0.2,
    noise_scale: float = 0.005,
    clip_val:   float | None = None,
) -> torch.Tensor:
    """
    Run short-run SGLD starting from z_init.

    Args:
        ebm:         EBM network f_θ.
        z_init:      Starting latent codes, shape (B, D).  Detached from graph.
        steps:       Number of Langevin steps.
        step_size:   Step size η.
        noise_scale: Magnitude of Gaussian noise injected each step.
                     Set lower than √η to reduce variance (biased but stable).
        clip_val:    Optional hard clamp on z after each step (e.g. 4.0).
                     Prevents extreme latents from destabilising training.

    Returns:
        z_k:  Approximate p_θ(z) samples, shape (B, D).  Detached, no grad.
    """
    ebm.eval()                                  # no dropout / BatchNorm side effects
    z = z_init.clone().detach().requires_grad_(True)

    for _ in range(steps):
        f_z  = ebm(z).sum()                     # scalar for autograd
        grad = torch.autograd.grad(f_z, z)[0]   # ∇_z f_θ(z)

        with torch.no_grad():
            noise = torch.randn_like(z) * noise_scale
            # SGLD: ascend log p_θ (= ascend f + descend ‖z‖²/2)
            z = z + 0.5 * step_size * (grad - z) + noise

            if clip_val is not None:
                z = z.clamp(-clip_val, clip_val)

        z = z.detach().requires_grad_(True)

    ebm.train()
    return z.detach()
