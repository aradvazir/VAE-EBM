"""Central configuration for the Anime EBM-VAE project."""
from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # ── Data ──────────────────────────────────────────────────────────
    data_dir: str = "data/animefacedataset"
    image_size: int = 64          # resize all images to 64×64
    num_workers: int = 0 
    subset_size: int = 15_000

    # ── Model ─────────────────────────────────────────────────────────
    latent_dim: int = 256
    base_channels: int = 64       # doubles at each downscale: 64→128→256→512

    # ── VAE Training ──────────────────────────────────────────────────
    batch_size: int = 64
    lr_vae: float = 1e-4
    num_epochs_vae: int = 30
    beta_start: float = 0.0       # KL warmup: start at 0, ramp to beta_end
    beta_end: float = 1.0
    beta_warmup_epochs: int = 15  # linearly ramp beta over first N epochs
    val_split: float = 0.1

    # ── EBM Architecture ─────────────────────────────────────────────
    ebm_hidden_dim: int = 512
    ebm_num_layers: int = 4

    # ── EBM Training ─────────────────────────────────────────────────
    lr_ebm: float = 1e-4
    num_epochs_ebm_vae: int = 30

    # ── SGLD (short-run MCMC) ─────────────────
    # Always starts from N(0,I); avoids replay-buffer collapse.
    mcmc_buffer_size: int = 10_000
    mcmc_steps: int = 20
    mcmc_step_size: float = 0.5    # larger step compensates fewer steps
    mcmc_noise: float = 0.1        # more noise for better exploration
    mcmc_reinit_prob: float = 1.0
    mcmc_grad_clip: float = 5.0

    # ── Device ────────────────────────────────────────────────────────
    @property
    def device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    # ── Logging / Checkpointing ───────────────────────────────────────
    log_interval: int = 50
    save_interval: int = 10
    sample_interval: int = 5
    output_dir: str = "outputs"
    num_sample_images: int = 8
