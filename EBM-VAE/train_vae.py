"""
Train the vanilla β-VAE on anime faces.
"""
import time
from pathlib import Path

import torch
import torch.optim as optim
from torchvision.utils import save_image

from config import Config
from dataset import get_dataloaders
from models.vae import VAE
from plotting import plot_vae_training


# ─────────────────────── helpers ─────────────────────────────────────────────

def linear_beta(epoch: int, cfg: Config) -> float:
    """Linearly warm up β from beta_start → beta_end over beta_warmup_epochs."""
    if epoch >= cfg.beta_warmup_epochs:
        return cfg.beta_end
    return cfg.beta_start + (cfg.beta_end - cfg.beta_start) * epoch / cfg.beta_warmup_epochs


def save_samples(model: VAE, val_loader, device: torch.device, path: Path, n: int = 16) -> None:
    model.eval()
    # Random samples
    samples = model.sample(n, device)
    samples = (samples + 1) / 2   # [-1,1] → [0,1]
    save_image(samples, path.parent / f"samples_{path.stem}.png", nrow=4)

    # Reconstructions
    x_batch = next(iter(val_loader))[:n].to(device)
    with torch.no_grad():
        x_recon, _, _ = model(x_batch)
    comparison = torch.cat([x_batch[:n//2], x_recon[:n//2]])
    comparison = (comparison + 1) / 2
    save_image(comparison, path.parent / f"recon_{path.stem}.png", nrow=n//2)


# ─────────────────────── train / val loops ───────────────────────────────────

def train_epoch(model: VAE, loader, optimizer, device, beta: float, cfg: Config):
    model.train()
    totals = dict(loss=0.0, recon=0.0, kl=0.0)

    for i, x in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad()

        x_recon, mu, log_var = model(x)
        total, recon, kl     = model.loss(x, x_recon, mu, log_var, beta=beta)

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        totals["loss"]  += total.item()
        totals["recon"] += recon.item()
        totals["kl"]    += kl.item()

        if i % cfg.log_interval == 0:
            print(
                f"  [{i:4d}/{len(loader)}] "
                f"loss {total.item():.1f}  recon {recon.item():.1f}  "
                f"kl {kl.item():.2f}  β={beta:.3f}"
            )

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def val_epoch(model: VAE, loader, device, beta: float):
    model.eval()
    totals = dict(loss=0.0, recon=0.0, kl=0.0)
    for x in loader:
        x = x.to(device)
        x_recon, mu, log_var = model(x)
        total, recon, kl     = model.loss(x, x_recon, mu, log_var, beta=beta)
        totals["loss"]  += total.item()
        totals["recon"] += recon.item()
        totals["kl"]    += kl.item()
    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# ─────────────────────── main ────────────────────────────────────────────────

def main() -> None:
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"Device : {device}")

    # Directories
    out_dir   = Path(cfg.output_dir)
    ckpt_dir  = out_dir / "checkpoints"
    samp_dir  = out_dir / "samples"
    plot_dir  = out_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    samp_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = VAE(
        in_channels=3,
        base_channels=cfg.base_channels,
        latent_dim=cfg.latent_dim,
        image_size=cfg.image_size,
    ).to(device)
    print(f"VAE params: {model.param_count():,}")

    # Optimizer + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr_vae, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs_vae, eta_min=1e-6
    )

    # best_val_loss = float("inf")

    train_history: list[dict] = []
    val_history:   list[dict] = []
    beta_history:  list[float] = []

    for epoch in range(1, cfg.num_epochs_vae + 1):
        beta = linear_beta(epoch, cfg)
        t0   = time.time()

        train_stats = train_epoch(model, train_loader, optimizer, device, beta, cfg)
        val_stats   = val_epoch(model, val_loader, device, beta)
        scheduler.step()

        train_history.append(train_stats)
        val_history.append(val_stats)
        beta_history.append(beta)
        plot_vae_training(train_history, val_history, beta_history, plot_dir)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch:3d}/{cfg.num_epochs_vae}  ({elapsed:.0f}s)  β={beta:.3f}\n"
            f"  Train  loss {train_stats['loss']:.1f}  "
            f"recon {train_stats['recon']:.1f}  kl {train_stats['kl']:.2f}\n"
            f"  Val    loss {val_stats['loss']:.1f}  "
            f"recon {val_stats['recon']:.1f}  kl {val_stats['kl']:.2f}"
        )

        # Best checkpoint
        # if val_stats["loss"] < best_val_loss:
        #     best_val_loss = val_stats["loss"]
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_stats["loss"],
                "cfg": cfg,
            },
            ckpt_dir / "vae_best.pt",
        )
            # print(f"  ✓ New best val loss: {best_val_loss:.1f}")

        # Periodic checkpoint
        if epoch % cfg.save_interval == 0:
            torch.save(model.state_dict(), ckpt_dir / f"vae_epoch_{epoch:03d}.pt")

        # Sample images
        if epoch % cfg.sample_interval == 0:
            save_samples(model, val_loader, device, samp_dir / f"epoch_{epoch:03d}", n=16)
            print(f"  Saved samples → {samp_dir}")

    # print(f"\nTraining done.  Best val loss: {best_val_loss:.1f}")


if __name__ == "__main__":
    main()
