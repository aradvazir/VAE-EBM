"""
Training metrics logger + plot generator for joint VAE+EBM.

Two modes:

  1. Called automatically by train_joint.py after each epoch
     (train_joint imports MetricsLogger from here)

  2. Run standalone to regenerate all plots from a saved CSV:
     python plot_training.py
     python plot_training.py --csv outputs/metrics/joint_metrics.csv

Plots saved to outputs/metrics/:
  training_overview.png    — 2×3 panel: all key signals
  energy_dynamics.png      — energy E+/E- and gap in detail
  vae_health.png           — recon + KL + buf_std
"""
from __future__ import annotations

import csv
import argparse
from pathlib import Path
from typing import Any

# ─────────────────────── metrics logger ──────────────────────────────────────

class MetricsLogger:
    """Append one row per epoch to a CSV and regenerate plots on demand."""

    FIELDS = [
        "epoch", "beta", "alpha",
        "train_recon", "train_kl", "train_vae_loss",
        "train_e_pos", "train_e_neg", "train_e_gap", "train_ebm_loss",
        "val_recon",   "val_kl",   "val_vae_loss",
        "val_e_pos",   "val_e_neg",   "val_e_gap",
        "sgld_drift",
    ]

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict] = []

        # Load existing rows if resuming
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                self._rows = list(csv.DictReader(f))
            print(f"  Loaded {len(self._rows)} existing metric rows from {self.csv_path}")

        self._writer_f = open(self.csv_path, "a", newline="")
        self._writer   = csv.DictWriter(self._writer_f, fieldnames=self.FIELDS)
        if not self.csv_path.stat().st_size or len(self._rows) == 0:
            self._writer.writeheader()

    def log(self, epoch: int, beta: float, alpha: float,
            train: dict, val: dict, sgld_drift: float) -> None:
        row = {
            "epoch": epoch, "beta": round(beta, 4), "alpha": round(alpha, 5),
            "train_recon":    round(train["recon"],    3),
            "train_kl":       round(train["kl"],       3),
            "train_vae_loss": round(train["vae_loss"], 3),
            "train_e_pos":    round(train["e_pos"],    5),
            "train_e_neg":    round(train["e_neg"],    5),
            "train_e_gap":    round(train["e_gap"],    5),
            "train_ebm_loss": round(train.get("ebm_loss", 0), 5),
            "val_recon":      round(val["recon"],      3),
            "val_kl":         round(val["kl"],         3),
            "val_vae_loss":   round(val["vae_loss"],   3),
            "val_e_pos":      round(val["e_pos"],      5),
            "val_e_neg":      round(val["e_neg"],      5),
            "val_e_gap":      round(val["e_gap"],      5),
            "sgld_drift":     round(sgld_drift,         5),
        }
        self._rows.append(row)
        self._writer.writerow(row)
        self._writer_f.flush()

    def plot(self, out_dir: str | Path) -> None:
        if not self._rows:
            return
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _make_all_plots(self._rows, out_dir)

    def close(self) -> None:
        self._writer_f.close()


# ─────────────────────── plotting ────────────────────────────────────────────

def _col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows]


def _make_all_plots(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not installed — skipping plots  (pip install matplotlib)")
        return

    epochs = _col(rows, "epoch")
    C = {               # colour palette
        "recon":  "#4C72B0",
        "kl":     "#DD8452",
        "gap":    "#55A868",
        "e_pos":  "#C44E52",
        "e_neg":  "#8172B3",
        "buf":    "#937860",
        "alpha":  "#DA8BC3",
        "val":    "#64B5CD",
    }

    # ── 1. Training overview (2 × 3) ─────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Joint VAE + EBM — Training Overview", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.32)

    # [0,0] Reconstruction loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, _col(rows, "train_recon"), color=C["recon"], label="train")
    ax.plot(epochs, _col(rows, "val_recon"),   color=C["val"],   label="val", ls="--")
    ax.set_title("Reconstruction loss"); ax.set_xlabel("Epoch"); ax.legend()
    ax.grid(alpha=0.3)

    # [0,1] KL divergence
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, _col(rows, "train_kl"), color=C["kl"],  label="train")
    ax.plot(epochs, _col(rows, "val_kl"),   color=C["val"], label="val", ls="--")
    ax.set_title("KL divergence"); ax.set_xlabel("Epoch"); ax.legend()
    ax.grid(alpha=0.3)

    # [0,2] Energy gap  (most important EBM signal)
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, _col(rows, "train_e_gap"), color=C["gap"],  label="train")
    ax.plot(epochs, _col(rows, "val_e_gap"),   color=C["val"],  label="val", ls="--")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("Energy gap  E⁺ − E⁻\n(negative = EBM learning)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # [1,0] E+ and E- separately
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, _col(rows, "train_e_pos"), color=C["e_pos"], label="E⁺ (real)")
    ax.plot(epochs, _col(rows, "train_e_neg"), color=C["e_neg"], label="E⁻ (SGLD)")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("Energy values (train)\n(should stay bounded ±5)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    # [1,1] SGLD drift
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, _col(rows, "sgld_drift"), color=C["buf"])
    ax.set_title("SGLD drift (avg L2 from init)\n(higher = EBM pushing chains)")
    ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    # [1,2] Alpha ramp + beta ramp
    ax = fig.add_subplot(gs[1, 2])
    ax2 = ax.twinx()
    ax.plot(epochs,  _col(rows, "alpha"), color=C["alpha"], label="α (EBM shaping)")
    ax2.plot(epochs, _col(rows, "beta"),  color=C["kl"],    label="β (KL weight)", ls="--")
    ax.set_title("Schedule: α and β"); ax.set_xlabel("Epoch")
    ax.set_ylabel("α", color=C["alpha"]); ax2.set_ylabel("β", color=C["kl"])
    lines1, l1 = ax.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, l1+l2, fontsize=8)
    ax.grid(alpha=0.3)

    path = out_dir / "training_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")

    # ── 2. Energy dynamics (detail) ──────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("EBM Energy Dynamics", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, _col(rows, "train_e_pos"), color=C["e_pos"], label="E⁺ train")
    ax.plot(epochs, _col(rows, "val_e_pos"),   color=C["e_pos"], label="E⁺ val", ls="--", alpha=0.7)
    ax.plot(epochs, _col(rows, "train_e_neg"), color=C["e_neg"], label="E⁻ train")
    ax.plot(epochs, _col(rows, "val_e_neg"),   color=C["e_neg"], label="E⁻ val",  ls="--", alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("E⁺ (real posteriors) vs E⁻ (SGLD)")
    ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, _col(rows, "train_e_gap"), color=C["gap"],  label="train")
    ax.plot(epochs, _col(rows, "val_e_gap"),   color=C["val"],  label="val", ls="--")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.fill_between(epochs, _col(rows, "val_e_gap"), 0,
                    alpha=0.15, color=C["gap"],
                    where=[v < 0 for v in _col(rows, "val_e_gap")])
    ax.set_title("Energy gap  (shaded = EBM winning)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, _col(rows, "sgld_drift"), color=C["buf"], lw=2)
    ax.set_title("SGLD drift  avg ‖z_final − z_init‖₂\n(exploration indicator)")
    ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "energy_dynamics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")

    # ── 3. VAE health ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("VAE Health", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, _col(rows, "train_recon"), color=C["recon"], label="train recon")
    ax.plot(epochs, _col(rows, "val_recon"),   color=C["val"],   label="val recon", ls="--")
    ax.set_title("Reconstruction loss"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, _col(rows, "train_kl"), color=C["kl"],  label="train KL")
    ax.plot(epochs, _col(rows, "val_kl"),   color=C["val"], label="val KL", ls="--")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_title("KL divergence  (should stay > 0)"); ax.set_xlabel("Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, _col(rows, "train_vae_loss"), color=C["recon"], label="train total")
    ax.plot(epochs, _col(rows, "val_vae_loss"),   color=C["val"],   label="val total", ls="--")
    ax.set_title("Total VAE loss (recon + β·KL + α·E⁺)")
    ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    path = out_dir / "vae_health.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


# ─────────────────────── standalone ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="outputs/metrics/joint_metrics.csv")
    parser.add_argument("--out", default="outputs/metrics")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No CSV found at {csv_path}")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} epochs from {csv_path}")
    _make_all_plots(rows, Path(args.out))


if __name__ == "__main__":
    main()
