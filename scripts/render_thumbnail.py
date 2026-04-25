"""Render the 3-panel lead figure to thumbnail.png.

Produces the same visual output as the notebook's lead figure cell (without
the live red probe / marker), for use as the README + OpenGraph image.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    npz_path = REPO / "data" / "singular_gradient.npz"
    with np.load(npz_path) as z:
        t = z["t_axis"]
        raw = z["raw_grad_norm"]
        lam = z["lambda_bar_curves"]
        pre = z["preconditioned_grad_norm"]
        env_inv = z["envelope_inv_b_squared"]
        env_b2 = z["envelope_b_squared"]
        labels = list(z["probe_labels"])

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(len(labels) - 1, 1)) for i in range(len(labels))]

    rc = {
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharex=True)

        for i, lab in enumerate(labels):
            y = raw[i]
            mask = y > 0
            axes[0].loglog(t[mask], y[mask], color=colors[i],
                           linewidth=1.5, alpha=0.9)
        axes[0].loglog(t, env_inv, "k--", alpha=0.6, linewidth=1.2)
        axes[0].set_xlabel(r"noise level $t$")
        axes[0].set_ylabel(r"$\|(u - a D_t^*) / \sigma^2\|$")
        axes[0].set_title("raw gradient", fontsize=12, pad=8)
        axes[0].text(0.025, 0.965, "(a)", transform=axes[0].transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        axes[1].loglog(t, lam[0], color="#440154", linewidth=2.0)
        axes[1].loglog(t, env_b2, "k--", alpha=0.6, linewidth=1.2)
        axes[1].set_xlabel(r"noise level $t$")
        axes[1].set_ylabel(r"$\lambda(t)^2$")
        axes[1].set_title("conformal factor", fontsize=12, pad=8)
        axes[1].text(0.025, 0.965, "(b)", transform=axes[1].transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        for i, lab in enumerate(labels):
            y = pre[i]
            mask = y > 0
            axes[2].loglog(t[mask], y[mask], color=colors[i],
                           linewidth=1.5, alpha=0.9)
        axes[2].set_xlabel(r"noise level $t$")
        axes[2].set_ylabel(r"$\lambda(t)^2 \cdot \|(u - a D_t^*)/\sigma^2\|$")
        axes[2].set_title("bounded product", fontsize=12, pad=8)
        axes[2].text(0.025, 0.965, "(c)", transform=axes[2].transAxes,
                     fontsize=12, fontweight="bold", va="top", ha="left")

        handles = [Line2D([0], [0], color=colors[i], lw=2, label=lab)
                   for i, lab in enumerate(labels)]
        handles += [Line2D([0], [0], color="k", linestyle="--",
                           lw=1.2, label="analytic envelope")]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = REPO / "thumbnail.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
