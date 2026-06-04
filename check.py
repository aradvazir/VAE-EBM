"""
Sanity checks — run before starting a full training run.

    python check.py

Verifies:
  1. MPS / device is available
  2. Encoder produces correct mu / log_var shapes
  3. Decoder reconstructs to the right image shape
  4. VAE forward + loss runs without error
  5. Model parameter count looks reasonable
  6. Data directory exists (warns if missing, does not fail)
"""
import sys
import torch
from config import Config
from models.vae import VAE

def main():
    cfg    = Config()
    device = torch.device(cfg.device)
    print(f"[1/5] Device      : {device}")
    print(f"      MPS avail   : {torch.backends.mps.is_available()}")

    model = VAE(
        in_channels=3,
        base_channels=cfg.base_channels,
        latent_dim=cfg.latent_dim,
        image_size=cfg.image_size,
    ).to(device)
    print(f"[2/5] Parameters  : {model.param_count():,}")

    x = torch.randn(4, 3, cfg.image_size, cfg.image_size, device=device)
    print(f"[3/5] Input shape : {tuple(x.shape)}")

    mu, log_var = model.encode(x)
    print(f"      μ shape     : {tuple(mu.shape)}  (expect [4, {cfg.latent_dim}])")
    assert mu.shape == (4, cfg.latent_dim), "mu shape mismatch!"

    z      = model.reparameterize(mu, log_var)
    x_hat  = model.decode(z)
    print(f"      x̂ shape     : {tuple(x_hat.shape)}  (expect [4, 3, {cfg.image_size}, {cfg.image_size}])")
    assert x_hat.shape == x.shape, "decoder output shape mismatch!"

    x_recon, mu2, lv2 = model(x)
    total, recon, kl  = model.loss(x, x_recon, mu2, lv2)
    print(f"[4/5] Loss        : total={total.item():.2f}  recon={recon.item():.2f}  kl={kl.item():.2f}")
    assert torch.isfinite(total), "Loss is not finite!"

    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    print(f"[5/5] Grad check  : {len(grads)} tensors have gradients")
    assert grads, "No gradients computed!"

    from pathlib import Path
    data_ok = Path(cfg.data_dir).exists()
    print(f"\nData dir '{cfg.data_dir}' exists: {data_ok}")
    if not data_ok:
        print("  ⚠  Download the Kaggle dataset and set cfg.data_dir accordingly.")

    print("\n✓ All checks passed — ready to train!\n  Run: python train_vae.py")


if __name__ == "__main__":
    main()
