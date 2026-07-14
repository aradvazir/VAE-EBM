"""
Joint VAE + EBM training.

This is the experiment version: the EBM actually shapes the VAE latent space.

Key differences from v3:
  - Clamp widened from [-5,5] → soft_clamp via tanh: gradient everywhere,
    no silent zero-gradient zones as energy drifts past ±5
  - Shaping weight α_max raised from 0.05 → 0.3 so the EBM term is
    actually meaningful relative to the recon loss
  - EBM reg: 1e-4 (same as v3 — small enough not to silence, enough to anchor)
  - SGLD params read from config (no hardcoded values)
  - Everything else from v3 kept: freeze/unfreeze, warm start β=1.0, lr=1e-5

Loss breakdown
──────────────
  VAE step (EBM params frozen):
    L_vae = recon
          + β  · KL(q || N(0,I))
          + α  · soft_clamp(E_θ(z⁺), scale=10)

    soft_clamp(x, scale) = scale · tanh(x / scale)
      → behaves like x near 0, saturates smoothly at ±scale
      → gradient is always non-zero (unlike hard clamp)
      → prevents runaway without killing signal

  EBM step (z⁺ detached from VAE):
    L_ebm = E_θ(z⁺) − E_θ(z⁻)
          + 1e-4 · (E_θ(z⁺)² + E_θ(z⁻)²)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from config import Config
from dataset import get_dataloaders
from models.vae import VAE
from models.ebm import EBM
from models.sgld import SGLDSampler
from plot_training import MetricsLogger


# ─────────────────────── helpers ─────────────────────────────────────────────

def soft_clamp(x: torch.Tensor, scale: float = 10.0) -> torch.Tensor:
    """Smooth clamp via tanh. Gradient is always non-zero. Saturates at ±scale."""
    return scale * torch.tanh(x / scale)


def get_beta(epoch: int, cfg, warm_start: bool) -> float:
    if warm_start:
        return cfg.beta_end
    if epoch >= cfg.beta_warmup_epochs:
        return cfg.beta_end
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * epoch / cfg.beta_warmup_epochs


def get_alpha(epoch: int, total_epochs: int, alpha_max: float = 1.0) -> float:
    """Ramp α from 0 → alpha_max over the first third of training."""
    ramp = min(epoch / (total_epochs // 3), 1.0)
    return alpha_max * ramp


def load_vae_weights(cfg, ckpt_path: str, device) -> VAE:
    model = VAE(in_channels=3, base_channels=cfg.base_channels,
                latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    print(f"VAE warm-started from '{ckpt_path}'  ({model.param_count():,} params)")
    return model


def load_joint_weights(cfg, ckpt_path: str, device):
    """Load both VAE and EBM from a joint checkpoint (e.g. joint_best.pt)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    vae  = VAE(in_channels=3, base_channels=cfg.base_channels,
               latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
    ebm  = EBM(latent_dim=cfg.latent_dim,
               hidden_dim=cfg.ebm_hidden_dim,
               num_layers=cfg.ebm_num_layers).to(device)
    vae.load_state_dict(ckpt["vae_state_dict"])
    ebm.load_state_dict(ckpt["ebm_state_dict"])
    print(f"Joint checkpoint resumed from '{ckpt_path}'")
    print(f"  VAE params : {vae.param_count():,}  |  EBM params : {ebm.param_count():,}")
    return vae, ebm


@torch.no_grad()
def save_samples(vae, sampler, ebm, val_loader, device, samp_dir, epoch):
    ebm.eval(); vae.eval()

    # Short-chain EBM samples (training quality)
    z_ebm = sampler.sample(ebm, n_samples=16, training=False)
    save_image((vae.decode(z_ebm) + 1) / 2,
               samp_dir / f"joint_ebm_epoch_{epoch:03d}.png", nrow=4)

    # N(0,I) baseline
    z_g = torch.randn(16, vae.latent_dim, device=device)
    save_image((vae.decode(z_g) + 1) / 2,
               samp_dir / f"joint_gaussian_epoch_{epoch:03d}.png", nrow=4)

    # Reconstructions
    x    = next(iter(val_loader))[:8].to(device)
    xhat = vae.reconstruct(x)
    comp = torch.cat([(x + 1) / 2, (xhat + 1) / 2])
    save_image(comp, samp_dir / f"joint_recon_epoch_{epoch:03d}.png", nrow=8)


# ─────────────────────── train / val ─────────────────────────────────────────

def train_epoch(vae, ebm, sampler, opt_vae, opt_ebm, loader,
                device, cfg, beta, alpha):
    vae.train(); ebm.train()
    stats = dict(recon=0.0, kl=0.0, vae_loss=0.0,
                 e_pos=0.0, e_neg=0.0, e_gap=0.0, ebm_loss=0.0)

    for i, x in enumerate(loader):
        x = x.to(device)
        B = x.size(0)

        # ── VAE step (EBM params frozen → grad only flows to encoder) ──
        x_recon, mu, log_var = vae(x)
        recon, _, kl         = vae.loss(x, x_recon, mu, log_var, beta=beta)
        z_pos_vae            = vae.reparameterize(mu, log_var)

        for p in ebm.parameters(): p.requires_grad_(False)
        e_sc = soft_clamp(ebm.energy(z_pos_vae), scale=10.0)   # gradient everywhere
        for p in ebm.parameters(): p.requires_grad_(True)

        vae_loss = recon + beta * kl + alpha * e_sc.mean()

        opt_vae.zero_grad()
        vae_loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
        opt_vae.step()

        # ── EBM step (CD on detached posterior mean) ──────────────────
        with torch.no_grad():
            mu2, lv2 = vae.encode(x)
            z_pos    = mu2   # deterministic posterior mean

        z_neg    = sampler.sample(ebm, n_samples=B)
        e_pos    = ebm.energy(z_pos)
        e_neg    = ebm.energy(z_neg)
        cd       = e_pos.mean() - e_neg.mean()
        reg      = 1e-4 * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        ebm_loss = cd + reg

        opt_ebm.zero_grad()
        ebm_loss.backward()
        nn.utils.clip_grad_norm_(ebm.parameters(), max_norm=1.0)
        opt_ebm.step()

        stats["recon"]    += recon.item()
        stats["kl"]       += kl.item()
        stats["vae_loss"] += vae_loss.item()
        stats["e_pos"]    += e_pos.mean().item()
        stats["e_neg"]    += e_neg.mean().item()
        stats["e_gap"]    += cd.item()
        stats["ebm_loss"] += ebm_loss.item()

        if i % cfg.log_interval == 0:
            buf = sampler.drift_stats()
            print(
                f"  [{i:4d}/{len(loader)}]  "
                f"recon {recon.item():.1f}  kl {kl.item():.1f}  "
                f"α·E {(alpha * e_sc.mean()).item():+.2f}  "
                f"E⁺ {e_pos.mean().item():+.2f}  E⁻ {e_neg.mean().item():+.2f}  "
                f"gap {cd.item():+.2f}  drift {buf['sgld_drift']:.2f}"
            )

    n = len(loader)
    return {k: v / n for k, v in stats.items()}


@torch.no_grad()
def val_epoch(vae, ebm, sampler, loader, device, beta, alpha):
    vae.eval(); ebm.eval()
    stats = dict(recon=0.0, kl=0.0, vae_loss=0.0,
                 e_pos=0.0, e_neg=0.0, e_gap=0.0)

    for x in loader:
        x = x.to(device)
        x_recon, mu, log_var = vae(x)
        recon, _, kl         = vae.loss(x, x_recon, mu, log_var, beta=beta)
        z_pos = mu
        e_pos = ebm.energy(z_pos)
        z_neg = sampler.sample(ebm, n_samples=x.size(0), training=False)
        e_neg = ebm.energy(z_neg)
        e_sc  = soft_clamp(e_pos, scale=10.0)

        stats["recon"]    += recon.item()
        stats["kl"]       += kl.item()
        stats["vae_loss"] += (recon + beta * kl + alpha * e_sc.mean()).item()
        stats["e_pos"]    += e_pos.mean().item()
        stats["e_neg"]    += e_neg.mean().item()
        stats["e_gap"]    += (e_pos.mean() - e_neg.mean()).item()

    n = len(loader)
    return {k: v / n for k, v in stats.items()}


# ─────────────────────── main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae_ckpt",   default=None,
                        help="Warm-start VAE only from this checkpoint")
    parser.add_argument("--joint_ckpt", default=None,
                        help="Resume full joint training (VAE+EBM) from this checkpoint")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--alpha_max",  type=float, default=1.0,
                        help="Max EBM shaping weight (default 1.0)")
    args = parser.parse_args()

    if args.joint_ckpt and args.vae_ckpt:
        raise ValueError("Specify --joint_ckpt OR --vae_ckpt, not both.")

    cfg        = Config()
    warm_start = (args.vae_ckpt is not None) or (args.joint_ckpt is not None)
    if args.epochs:
        cfg.num_epochs_ebm_vae = args.epochs
    device = torch.device(cfg.device)
    print(f"Device : {device}  |  warm_start={warm_start}\n")

    out_dir  = Path(cfg.output_dir)
    for d in ["checkpoints", "samples", "metrics"]:
        (out_dir / d).mkdir(parents=True, exist_ok=True)

    logger = MetricsLogger(out_dir / "metrics" / "joint_metrics.csv")
    train_loader, val_loader = get_dataloaders(cfg)

    if args.joint_ckpt:
        vae, ebm = load_joint_weights(cfg, args.joint_ckpt, device)
    elif args.vae_ckpt:
        vae = load_vae_weights(cfg, args.vae_ckpt, device)
        ebm = EBM(latent_dim=cfg.latent_dim,
                  hidden_dim=cfg.ebm_hidden_dim,
                  num_layers=cfg.ebm_num_layers).to(device)
        print(f"EBM params (fresh) : {ebm.param_count():,}\n")
    else:
        vae = VAE(in_channels=3, base_channels=cfg.base_channels,
                  latent_dim=cfg.latent_dim, image_size=cfg.image_size).to(device)
        ebm = EBM(latent_dim=cfg.latent_dim,
                  hidden_dim=cfg.ebm_hidden_dim,
                  num_layers=cfg.ebm_num_layers).to(device)
        print(f"Training from scratch.  EBM params : {ebm.param_count():,}\n")

    sampler = SGLDSampler(
        buffer_size=cfg.mcmc_buffer_size,
        latent_dim=cfg.latent_dim,
        device=device,
        step_size=cfg.mcmc_step_size,
        noise_scale=cfg.mcmc_noise,
        n_steps=cfg.mcmc_steps,
        reinit_prob=cfg.mcmc_reinit_prob,
        grad_clip=cfg.mcmc_grad_clip,
        short_run=True,
    )

    lr_vae  = 1e-5 if warm_start else cfg.lr_vae
    opt_vae = optim.AdamW(vae.parameters(), lr=lr_vae, weight_decay=1e-4)
    opt_ebm = optim.Adam( ebm.parameters(), lr=cfg.lr_ebm, betas=(0.9, 0.999))

    sched_vae = optim.lr_scheduler.CosineAnnealingLR(
        opt_vae, T_max=cfg.num_epochs_ebm_vae, eta_min=lr_vae * 0.1)
    sched_ebm = optim.lr_scheduler.StepLR(opt_ebm, step_size=20, gamma=0.5)

    best_recon = float("inf")
    best_gap   = float("inf")

    print("─" * 72)
    print(f"Joint VAE + EBM v4  |  VAE lr={lr_vae:.0e}  EBM lr={cfg.lr_ebm:.0e}")
    print(f"  warm_start={warm_start}  β={'1.0 fixed' if warm_start else 'ramping'}  "
          f"α_max={args.alpha_max}")
    print(f"  SGLD  steps={cfg.mcmc_steps}  η={cfg.mcmc_step_size}  "
          f"σ={cfg.mcmc_noise}  clip={cfg.mcmc_grad_clip}")
    print("─" * 72)
    print("What to watch:")
    print("  α·E  — should be non-zero throughout (soft_clamp gives gradient always)")
    print("  gap  — trends negative, should SLOW DOWN after epoch 20 (not grow forever)")
    print("  recon — stays near warm-start level or improves")
    print("  kl   — stable (no posterior collapse)")
    print("─" * 72 + "\n")

    for epoch in range(1, cfg.num_epochs_ebm_vae + 1):
        beta  = get_beta(epoch, cfg, warm_start)
        alpha = get_alpha(epoch, cfg.num_epochs_ebm_vae, args.alpha_max)
        t0    = time.time()

        tr = train_epoch(vae, ebm, sampler, opt_vae, opt_ebm,
                         train_loader, device, cfg, beta, alpha)
        vl = val_epoch(vae, ebm, sampler, val_loader, device, beta, alpha)
        sched_vae.step(); sched_ebm.step()

        sgld_drift = sampler.drift_stats()["sgld_drift"]
        logger.log(epoch, beta, alpha, tr, vl, sgld_drift)
        if epoch % cfg.sample_interval == 0:
            logger.plot(out_dir / "metrics")

        print(
            f"\nEpoch {epoch:3d}/{cfg.num_epochs_ebm_vae}  "
            f"({time.time()-t0:.0f}s)  β={beta:.2f}  α={alpha:.4f}\n"
            f"  Train  recon {tr['recon']:.1f}  kl {tr['kl']:.1f}  "
            f"E⁺ {tr['e_pos']:+.2f}  E⁻ {tr['e_neg']:+.2f}  gap {tr['e_gap']:+.2f}\n"
            f"  Val    recon {vl['recon']:.1f}  kl {vl['kl']:.1f}  "
            f"E⁺ {vl['e_pos']:+.2f}  E⁻ {vl['e_neg']:+.2f}  gap {vl['e_gap']:+.2f}"
            f"  drift {sgld_drift:.2f}"
        )

        if vl["recon"] < best_recon:
            best_recon = vl["recon"]
            torch.save({"epoch": epoch,
                        "vae_state_dict": vae.state_dict(),
                        "ebm_state_dict": ebm.state_dict(),
                        "val_recon": vl["recon"],
                        "val_e_gap": vl["e_gap"]},
                       out_dir / "checkpoints" / "joint_best.pt")
            print(f"  ✓ New best val recon: {best_recon:.1f}")

        if vl["e_gap"] < best_gap:
            best_gap = vl["e_gap"]
            print(f"  ✓ New best energy gap: {best_gap:+.2f}")

        if epoch % cfg.save_interval == 0:
            torch.save({"vae_state_dict": vae.state_dict(),
                        "ebm_state_dict": ebm.state_dict()},
                       out_dir / "checkpoints" / f"joint_epoch_{epoch:03d}.pt")

        if epoch % cfg.sample_interval == 0:
            vae.eval(); ebm.eval()
            save_samples(vae, sampler, ebm, val_loader, device,
                         out_dir / "samples", epoch)
            print(f"  Saved samples → {out_dir / 'samples'}")

    logger.close()
    logger.plot(out_dir / "metrics")
    print(f"\nDone.  Best val recon: {best_recon:.1f}  |  Best gap: {best_gap:+.2f}")
    print("Run: python generate.py --vae_ckpt outputs/checkpoints/vae_best.pt "
          "--ebm_ckpt outputs/checkpoints/joint_best.pt")


if __name__ == "__main__":
    main()
