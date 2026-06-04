"""Central configuration for the Anime EBM-VAE project."""
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────
    data_dir: str = "data/animefacedataset"
    image_size: int = 64          # resize all images to 64×64
    num_workers: int = 0          # 0 is safer on MPS (no multiprocess fork issues)
    subset_size: int = 30_000     # use None for the full 63k dataset

    # ── Model ─────────────────────────────────────────────────────────
    latent_dim: int = 512
    base_channels: int = 64       # doubles at each downscale: 64→128→256→512

    # ── VAE Training ──────────────────────────────────────────────────
    batch_size: int = 64
    lr_vae: float = 1e-4
    num_epochs_vae: int = 20
    beta_start: float = 0.0       # KL warmup: start at 0, ramp to beta_end
    beta_end: float = 1.0
    beta_warmup_epochs: int = 20  # linearly ramp beta over first N epochs
    val_split: float = 0.1

    # ── EBM Training (Phase 3) ────────────────────────────────────────
    lr_ebm: float = 1e-4
    num_epochs_ebm_vae: int = 30
    # Short-run Langevin MCMC (SGLD) for EBM prior sampling
    mcmc_steps: int = 100          # steps per SGLD chain
    mcmc_step_size: float = 0.1   # step size η
    mcmc_noise: float = 0.005     # Gaussian noise magnitude σ

    # ── Device ────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ── Logging / Checkpointing ───────────────────────────────────────
    log_interval: int = 50        # print every N batches
    save_interval: int = 10       # save checkpoint every N epochs
    sample_interval: int = 5      # save sample images every N epochs
    output_dir: str = "outputs"
    num_sample_images: int = 8   # images to generate for visual eval
