"""
Generate images using the trained EBM prior.

The training sampler uses short chains (40 steps) for speed.
For generation we need long chains (1000+ steps) to actually traverse the
energy landscape from E≈+31 (N(0,I) start) down to E≈-33 (posterior region).

Usage:
    python generate.py --vae_ckpt outputs/checkpoints/vae_best.pt \
                       --ebm_ckpt outputs/checkpoints/ebm_final_best.pt

Outputs in outputs/generated/:
    ebm_grid.png          — 8×8 grid from EBM prior (long SGLD chain)
    gaussian_grid.png     — 8×8 grid from N(0,I) prior  (baseline)
    side_by_side.png      — 4 rows: gaussian | ebm  (direct comparison)
    energy_trace.png      — energy vs SGLD step (shows chain converging)
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image, make_grid

from config import Config
from models.vae import VAE
from models.ebm import EBM


def load_models(vae_ckpt, ebm_ckpt, cfg, device):
    vae = VAE(in_channels=3, base_channels=cfg.base_channels,
              latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
    vae_state = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae.load_state_dict(vae_state.get("model_state_dict", vae_state))
    vae.eval()
    for p in vae.parameters(): p.requires_grad_(False)

    ebm = EBM(latent_dim=cfg.latent_dim, hidden_dim=512, num_layers=4).to(device)
    ebm_state = torch.load(ebm_ckpt, map_location=device, weights_only=False)
    ebm.load_state_dict(ebm_state.get("ebm_state_dict", ebm_state))
    ebm.eval()
    for p in ebm.parameters(): p.requires_grad_(False)

    print(f"VAE loaded from {vae_ckpt}")
    print(f"EBM loaded from {ebm_ckpt}")
    return vae, ebm


def long_chain_sgld(
    ebm: torch.nn.Module,
    n_samples: int,
    latent_dim: int,
    device: torch.device,
    n_steps: int = 3000,
    step_size: float = 0.1,
    noise_scale: float = 0.05,
    record_every: int = 50,
) -> tuple[torch.Tensor, list[float]]:
    """
    Long-chain SGLD starting from N(0,I).
    Returns final z and a list of mean energies sampled every record_every steps.
    Suitable for generation (not training — no replay buffer needed).
    """
    z = torch.randn(n_samples, latent_dim, device=device)
    energy_trace = []

    with torch.enable_grad():
        z.requires_grad_(True)
        for step in range(n_steps):
            energy = ebm.energy(z).sum()
            grad   = torch.autograd.grad(energy, z)[0]

            # No grad_clip here — let the chain follow the true gradient
            # (we're not doing training, so stability isn't a concern)
            noise = torch.randn_like(z) * noise_scale
            z = (z.detach() - (step_size / 2) * grad + noise)
            z.requires_grad_(True)

            if step % record_every == 0:
                with torch.no_grad():
                    e_mean = ebm.energy(z.detach()).mean().item()
                energy_trace.append(e_mean)
                if step % 200 == 0:
                    print(f"  SGLD step {step:4d}/{n_steps}  E={e_mean:.3f}")

    return z.detach(), energy_trace


def plot_energy_trace(energy_trace: list[float], record_every: int, out_path: Path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = [i * record_every for i in range(len(energy_trace))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, energy_trace, color="#4C72B0", lw=2)
    ax.set_xlabel("SGLD step")
    ax.set_ylabel("Mean energy E(z)")
    ax.set_title("Long-chain SGLD energy trace\n"
                 "(should decrease from +31 toward −33 as chain mixes)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Energy trace → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt", default="outputs/checkpoints/vae_best.pt")
    parser.add_argument("--ebm_ckpt", default="outputs/checkpoints/ebm_final_best.pt")
    parser.add_argument("--n_steps",  type=int, default=1000,
                        help="SGLD steps for generation (more = closer to prior)")
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--step_size", type=float, default=0.1)
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device: {device}\n")

    out_dir = Path(cfg.output_dir) / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    vae, ebm = load_models(args.vae_ckpt, args.ebm_ckpt, cfg, device)

    # ── Gaussian baseline ─────────────────────────────────────────────
    print("\nGenerating N(0,I) baseline samples...")
    with torch.no_grad():
        z_gauss = torch.randn(args.n_samples, cfg.latent_dim, device=device)
        imgs_gauss = (vae.decode(z_gauss) + 1) / 2
    save_image(imgs_gauss, out_dir / "gaussian_grid.png", nrow=8)
    print(f"  N(0,I) grid → {out_dir / 'gaussian_grid.png'}")

    # ── EBM prior samples — long chain ────────────────────────────────
    print(f"\nRunning long-chain SGLD ({args.n_steps} steps)...")
    z_ebm, energy_trace = long_chain_sgld(
        ebm, n_samples=args.n_samples, latent_dim=cfg.latent_dim,
        device=device, n_steps=args.n_steps,
        step_size=args.step_size, noise_scale=0.005
    )

    with torch.no_grad():
        imgs_ebm = (vae.decode(z_ebm) + 1) / 2

    save_image(imgs_ebm, out_dir / "ebm_grid.png", nrow=8)
    print(f"  EBM grid → {out_dir / 'ebm_grid.png'}")

    # ── Side-by-side comparison (first 32 of each) ───────────────────
    n_compare = min(32, args.n_samples)
    side = torch.cat([imgs_gauss[:n_compare], imgs_ebm[:n_compare]])
    save_image(side, out_dir / "side_by_side.png", nrow=n_compare // 4)
    print(f"  Side-by-side → {out_dir / 'side_by_side.png'}")

    # ── Energy trace ─────────────────────────────────────────────────
    plot_energy_trace(energy_trace, record_every=50,
                      out_path=out_dir / "energy_trace.png")

    # ── Log final energy stats ────────────────────────────────────────
    with torch.no_grad():
        e_gauss = ebm.energy(z_gauss).mean().item()
        e_ebm   = ebm.energy(z_ebm).mean().item()

    print(f"\nEnergy comparison:")
    print(f"  N(0,I) samples  E = {e_gauss:+.2f}")
    print(f"  EBM samples     E = {e_ebm:+.2f}")
    print(f"  Gap             = {e_gauss - e_ebm:+.2f}  "
          f"({'EBM found lower energy ✓' if e_ebm < e_gauss else 'EBM needs more steps ✗'})")
    print(f"\nIf EBM images still look blurry, try --n_steps 3000 --step_size 0.05")


if __name__ == "__main__":
    main()
