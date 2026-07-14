"""
Shared plotting utilities for VAE and EBM-VAE training.

All plots are saved to outputs/plots/ and updated every epoch so you can
watch training progress live (just refresh the image in Finder/Preview).

Each function is stateless — it takes the full history list and rewrites
the figure from scratch, so plots stay correct if training is resumed from
a checkpoint.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")           # no display needed, safe on all platforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── shared style ──────────────────────────────────────────────────────────────

STYLE = {
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4a",
    "axes.labelcolor":   "#c8ccd8",
    "axes.titlecolor":   "#e8eaf0",
    "xtick.color":       "#8890a0",
    "ytick.color":       "#8890a0",
    "grid.color":        "#2a2d3a",
    "grid.linewidth":    0.6,
    "grid.alpha":        0.7,
    "lines.linewidth":   1.8,
    "legend.facecolor":  "#1a1d27",
    "legend.edgecolor":  "#3a3d4a",
    "legend.labelcolor": "#c8ccd8",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
}

COLORS = {
    "total":  "#7c83f5",   # soft blue-violet
    "recon":  "#f5a623",   # amber
    "kl":     "#50e3c2",   # teal
    "beta":   "#e85d75",   # rose
    "e_pos":  "#50e3c2",   # teal  (real posteriors)
    "e_neg":  "#f5a623",   # amber (fantasy particles)
    "gap":    "#7c83f5",   # blue-violet
    "buf":    "#e85d75",   # rose
    "train":  "#7c83f5",
    "val":    "#f5a623",
}


def _ax(ax, title: str, xlabel: str = "Epoch", ylabel: str = "") -> None:
    ax.set_title(title, pad=6)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both")
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _epochs(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


# ── VAE plot ──────────────────────────────────────────────────────────────────

def plot_vae_training(
    train_history: list[dict],
    val_history:   list[dict],
    beta_history:  list[float],
    plot_dir: str | Path,
) -> None:
    """
    Four-panel VAE training dashboard saved to plot_dir/vae_training.png.

    Panels:
      1. Total loss (train + val)
      2. Reconstruction loss (train + val)
      3. KL divergence (train + val)
      4. β schedule
    """
    if not train_history:
        return

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 9))
        fig.suptitle("VAE Training Dashboard", fontsize=13, color="#e8eaf0", y=0.98)
        gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32,
                                left=0.07, right=0.97, top=0.92, bottom=0.07)

        eps   = _epochs(len(train_history))
        keys  = ["loss", "recon", "kl"]
        titles = ["Total ELBO loss", "Reconstruction loss (MSE)", "KL divergence"]
        ylabels = ["Loss", "Loss", "KL"]
        panels  = [(0, 0), (0, 1), (1, 0)]

        for (row, col), key, title, ylabel in zip(panels, keys, titles, ylabels):
            ax  = fig.add_subplot(gs[row, col])
            tr  = [h[key] for h in train_history]
            vl  = [h[key] for h in val_history]
            ax.plot(eps, tr, color=COLORS["train"], label="Train", alpha=0.9)
            ax.plot(eps, vl, color=COLORS["val"],   label="Val",   alpha=0.9)
            # best val marker
            best_ep  = int(np.argmin(vl)) + 1
            best_val = min(vl)
            ax.axvline(best_ep, color=COLORS["val"], lw=0.8, ls="--", alpha=0.5)
            ax.scatter([best_ep], [best_val], color=COLORS["val"], zorder=5, s=40)
            ax.annotate(f" best\n ep{best_ep}", xy=(best_ep, best_val),
                        color=COLORS["val"], fontsize=7.5, va="top")
            ax.legend(fontsize=8)
            _ax(ax, title, ylabel=ylabel)

        # β schedule
        ax_b = fig.add_subplot(gs[1, 1])
        ax_b.plot(eps, beta_history, color=COLORS["beta"], alpha=0.9)
        ax_b.fill_between(eps, 0, beta_history, color=COLORS["beta"], alpha=0.12)
        ax_b.set_ylim(0, max(beta_history) * 1.15)
        _ax(ax_b, "β schedule (KL weight)", ylabel="β")

        _save(fig, Path(plot_dir) / "vae_training.png")


# ── EBM-VAE plot ──────────────────────────────────────────────────────────────

def plot_ebm_training(
    train_history: list[dict],
    val_history:   list[dict],
    buf_std_history: list[float],
    plot_dir: str | Path,
) -> None:
    """
    Five-panel EBM training dashboard saved to plot_dir/ebm_training.png.

    Panels:
      1. Energy gap  (train + val)
      2. E⁺ and E⁻ together  (train)
      3. E⁺ and E⁻ together  (val)
      4. CD loss
      5. Replay buffer std
    """
    if not train_history:
        return

    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("EBM-VAE Training Dashboard", fontsize=13, color="#e8eaf0", y=0.98)
        gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.32,
                                left=0.06, right=0.97, top=0.92, bottom=0.07)

        eps = _epochs(len(train_history))

        # ── 1. Energy gap ──────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        t_gap = [h["e_gap"] for h in train_history]
        v_gap = [h["e_gap"] for h in val_history]
        ax1.plot(eps, t_gap, color=COLORS["train"], label="Train gap", alpha=0.9)
        ax1.plot(eps, v_gap, color=COLORS["val"],   label="Val gap",   alpha=0.9)
        ax1.fill_between(eps, t_gap, 0, color=COLORS["train"], alpha=0.08)
        ax1.axhline(0, color="#555", lw=0.8, ls="--")
        best_ep = int(np.argmin(v_gap)) + 1
        ax1.axvline(best_ep, color=COLORS["val"], lw=0.8, ls="--", alpha=0.5)
        ax1.legend(fontsize=8)
        _ax(ax1, "Energy gap  E⁺ − E⁻  (↓ = better)", ylabel="Gap")

        # ── 2. Train E⁺ / E⁻ ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(eps, [h["e_pos"] for h in train_history],
                 color=COLORS["e_pos"], label="E⁺ (real)", alpha=0.9)
        ax2.plot(eps, [h["e_neg"] for h in train_history],
                 color=COLORS["e_neg"], label="E⁻ (SGLD)", alpha=0.9)
        ax2.legend(fontsize=8)
        _ax(ax2, "Train energies", ylabel="Energy")

        # ── 3. Val E⁺ / E⁻ ───────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(eps, [h["e_pos"] for h in val_history],
                 color=COLORS["e_pos"], label="E⁺ (real)", alpha=0.9)
        ax3.plot(eps, [h["e_neg"] for h in val_history],
                 color=COLORS["e_neg"], label="E⁻ (SGLD)", alpha=0.9)
        ax3.legend(fontsize=8)
        _ax(ax3, "Val energies", ylabel="Energy")

        # ── 4. CD loss ────────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(eps, [h["loss"] for h in train_history],
                 color=COLORS["total"], alpha=0.9)
        ax4.axhline(0, color="#555", lw=0.8, ls="--")
        _ax(ax4, "Contrastive divergence loss (train)", ylabel="Loss")

        # ── 5. Buffer std ─────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(eps, buf_std_history, color=COLORS["buf"], alpha=0.9)
        ax5.fill_between(eps, 1.0, buf_std_history, color=COLORS["buf"], alpha=0.10)
        ax5.axhline(1.0, color="#555", lw=0.8, ls="--", label="Prior std=1")
        ax5.legend(fontsize=8)
        _ax(ax5, "Replay buffer std  (↑ = SGLD mixing)", ylabel="Std")

        # ── 6. Gap rate of change ─────────────────────────────────────
        ax6 = fig.add_subplot(gs[1, 2])
        if len(v_gap) > 1:
            delta = np.diff(v_gap)
            ax6.bar(eps[1:], delta, color=COLORS["gap"],
                    alpha=0.7, width=0.8)
            ax6.axhline(0, color="#555", lw=0.8, ls="--")
        _ax(ax6, "Val gap Δ per epoch  (→ 0 = converged)", ylabel="Δ Gap")

        _save(fig, Path(plot_dir) / "ebm_training.png")
