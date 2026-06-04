"""
Evaluation tools for the joint VAE+EBM.

    python evaluate.py --ckpt outputs/checkpoints/joint_best.pt

Produces in outputs/eval/:
  energy_landscape.png   — PCA 2D projection of val latents, coloured by energy
  interpolations.png     — linear walks between 8 pairs of val images
  fid_score.txt          — Fréchet Inception Distance (lower = better)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image, make_grid

from config import Config
from dataset import get_dataloaders
from models.vae import VAE
from models.ebm import EBM
from models.sgld import SGLDSampler


# ─────────────────────── load checkpoint ─────────────────────────────────────

def load_joint(ckpt_path, cfg, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    vae = VAE(in_channels=3, base_channels=cfg.base_channels,
              latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
    ebm = EBM(latent_dim=cfg.latent_dim, hidden_dim=512, num_layers=4).to(device)

    vae.load_state_dict(ckpt.get("vae_state_dict", ckpt))
    if "ebm_state_dict" in ckpt:
        ebm.load_state_dict(ckpt["ebm_state_dict"])

    vae.eval(); ebm.eval()
    print(f"Loaded joint checkpoint from '{ckpt_path}'")
    return vae, ebm


# ─────────────────────── 1. energy landscape ─────────────────────────────────

def plot_energy_landscape(vae, ebm, val_loader, device, out_path):
    """PCA-project val-set latents to 2D, colour by EBM energy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping energy landscape")
        return

    # Collect latent means for all val images
    mus, energies = [], []
    with torch.no_grad():
        for x in val_loader:
            x      = x.to(device)
            mu, _  = vae.encode(x)
            e      = ebm.energy(mu)
            mus.append(mu.cpu())
            energies.append(e.cpu())

    mus      = torch.cat(mus).numpy()       # (N, latent_dim)
    energies = torch.cat(energies).numpy()  # (N,)

    # PCA to 2D
    mean  = mus.mean(0)
    mus_c = mus - mean
    U, S, Vt = np.linalg.svd(mus_c, full_matrices=False)
    coords    = mus_c @ Vt[:2].T           # (N, 2)

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1],
                    c=energies, cmap="RdYlBu_r", s=6, alpha=0.7,
                    vmin=np.percentile(energies, 5),
                    vmax=np.percentile(energies, 95))
    plt.colorbar(sc, ax=ax, label="EBM energy  E(z)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Val-set latent space coloured by EBM energy\n"
                 "(blue = low energy / probable,  red = high energy / unlikely)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Energy landscape → {out_path}")


# ─────────────────────── 2. latent interpolation ─────────────────────────────

@torch.no_grad()
def plot_interpolations(vae, val_loader, device, out_path, n_pairs=8, steps=10):
    """
    Linear interpolation between n_pairs of val images in latent space.
    Each row: [img_a | step_1 ... step_{steps} | img_b]
    """
    x_batch = next(iter(val_loader))[: n_pairs * 2].to(device)

    mu, _ = vae.encode(x_batch)
    mu_a  = mu[:n_pairs]
    mu_b  = mu[n_pairs: n_pairs * 2]

    rows = []
    for a, b in zip(mu_a, mu_b):
        alphas = torch.linspace(0, 1, steps, device=device)
        z_interp = torch.stack([(1 - α) * a + α * b for α in alphas])   # (steps, D)
        imgs     = vae.decode(z_interp)                                   # (steps, 3, H, W)
        # prepend original images
        orig_a = vae.decode(a.unsqueeze(0))
        orig_b = vae.decode(b.unsqueeze(0))
        row    = torch.cat([orig_a, imgs, orig_b], dim=0)                 # (steps+2, ...)
        rows.append(row)

    grid = torch.cat(rows, dim=0)   # (n_pairs * (steps+2), 3, H, W)
    grid = (grid + 1) / 2
    save_image(grid, out_path, nrow=steps + 2)
    print(f"  Interpolations    → {out_path}")


# ─────────────────────── 3. FID ───────────────────────────────────────────────

def compute_fid(vae, sampler, ebm, val_loader, device, n_samples=2000):
    """
    Simplified FID using raw pixel statistics instead of Inception features.
    Not the standard FID but gives a comparable relative metric between runs.
    For true FID, install pytorch-fid.
    """
    # Real images — flatten to pixel vectors
    real_vecs = []
    with torch.no_grad():
        for x in val_loader:
            real_vecs.append(x.to(device).flatten(1))
            if sum(v.shape[0] for v in real_vecs) >= n_samples:
                break
    real_vecs = torch.cat(real_vecs)[:n_samples].float()  # (N, C*H*W)

    # Generated images
    gen_vecs = []
    batch_sz = 64
    with torch.no_grad():
        remaining = n_samples
        while remaining > 0:
            bs = min(batch_sz, remaining)
            z    = sampler.sample(ebm, n_samples=bs, training=False)
            imgs = vae.decode(z).flatten(1)
            gen_vecs.append(imgs)
            remaining -= bs
    gen_vecs = torch.cat(gen_vecs)[:n_samples].float()

    # Fréchet distance in pixel space (proxy metric)
    mu_r, mu_g   = real_vecs.mean(0), gen_vecs.mean(0)
    cov_r = torch.cov(real_vecs.T)
    cov_g = torch.cov(gen_vecs.T)

    diff = mu_r - mu_g
    # Simplified: use diagonal of covariance (full matrix is too expensive at high dim)
    var_r = cov_r.diag()
    var_g = cov_g.diag()
    fid   = (diff.pow(2).sum() + (var_r.sqrt() - var_g.sqrt()).pow(2).sum()).item()

    return fid


# ─────────────────────── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/checkpoints/joint_best.pt")
    parser.add_argument("--vae_only", action="store_true",
                        help="Load a plain VAE checkpoint (no EBM)")
    args = parser.parse_args()

    cfg    = Config()
    device = torch.device(cfg.device)

    out_dir  = Path(cfg.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, val_loader = get_dataloaders(cfg)

    vae, ebm = load_joint(args.ckpt, cfg, device)

    sampler = SGLDSampler(
        buffer_size=2000, latent_dim=cfg.latent_dim, device=device,
        step_size=0.1, noise_scale=0.01, n_steps=100,
        reinit_prob=0.05, grad_clip=0.3,
    )

    print("\nRunning evaluations…\n")

    # 1. Energy landscape
    plot_energy_landscape(vae, ebm, val_loader, device,
                          out_dir / "energy_landscape.png")

    # 2. Interpolations
    plot_interpolations(vae, val_loader, device,
                        out_dir / "interpolations.png")

    # 3. Pixel-FID
    print("  Computing pixel-FID (proxy)…", end=" ", flush=True)
    fid = compute_fid(vae, sampler, ebm, val_loader, device, n_samples=1000)
    print(f"pixel-FID = {fid:.1f}")
    (out_dir / "fid_score.txt").write_text(
        f"checkpoint : {args.ckpt}\npixel-FID  : {fid:.2f}\n"
    )

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
