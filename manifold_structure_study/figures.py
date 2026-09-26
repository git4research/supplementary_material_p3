from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.lines import Line2D

_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#393b79",
    "#637939",
    "#a55194",
]


def _assign_colors(labels):
    n = len(labels)
    if n <= len(_PALETTE):
        cols = _PALETTE[:n]
    else:
        hsv = colormaps["hsv"]
        cols = [hsv(i / n) for i in range(n)]
    return {lab: cols[i] for i, lab in enumerate(labels)}


def make_cumvar_figure(curves: dict, out_dir, filename="manifold_figure",
                       thresholds=(0.90, 0.95)):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(curves)
    colors = _assign_colors(labels)
    ids = [curves[lab]["twonn"] for lab in labels]

    xmax = 0
    for lab in labels:
        cum = curves[lab]["cumulative"]
        xmax = max(xmax, int(np.searchsorted(cum, 0.99) + 1))
    xmax = int(min(xmax * 1.05, max(len(curves[lab]["cumulative"]) for lab in labels)))

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    mark = []
    for lab in labels:
        col = colors[lab]
        cum = curves[lab]["cumulative"][:xmax]
        cum_sh = curves[lab]["cumulative_shuffle"][:xmax]
        ax.plot(np.arange(1, len(cum) + 1), cum, color=col, lw=1.3, label=lab)
        ax.plot(np.arange(1, len(cum_sh) + 1), cum_sh, color=col, lw=0.85,
                ls=(0, (4, 3)), alpha=0.6)
        idv = curves[lab]["twonn"]
        xi = int(round(idv))
        yi = float(curves[lab]["cumulative"][xi - 1])
        mark.append({"lab": lab, "id": idv, "xi": xi, "yi": yi, "col": col})

    label_x = max(m["xi"] for m in mark) + xmax * 0.035
    min_gap = 0.075
    y_prev = None
    for m in sorted(mark, key=lambda d: d["yi"]):
        ax.plot([m["xi"]], [m["yi"]], marker="o", ms=6, mfc="white",
                mec=m["col"], mew=1.5, zorder=5)
        ly = m["yi"] if y_prev is None else max(m["yi"], y_prev + min_gap)
        ly = min(ly, 0.98)
        y_prev = ly
        ax.plot([m["xi"], label_x], [m["yi"], ly], color=m["col"], lw=0.8,
                alpha=0.6, zorder=4)
        ax.text(label_x + xmax * 0.008, ly, f"ID {m['id']:.1f}", color=m["col"],
                va="center", ha="left", fontsize=10.5)

    for t in thresholds:
        ax.axhline(t, color="0.55", lw=0.8, ls=":")
        ax.text(xmax, t + 0.006, f"{int(t * 100)}%", color="0.4",
                ha="right", va="bottom", fontsize=10.5)

    model_leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0),
                          frameon=False, fontsize=11, title="Model",
                          title_fontsize=12, labelspacing=0.35,
                          handlelength=1.8, borderaxespad=0.2)
    ax.add_artist(model_leg)
    style_handles = [
        Line2D([0], [0], color="0.3", lw=1.6, label="real embeddings"),
        Line2D([0], [0], color="0.3", lw=1.0, ls=(0, (4, 3)),
               label="per-dimension shuffle"),
        Line2D([0], [0], marker="o", ms=6, mfc="white", mec="0.3", mew=1.5,
               ls="none", label="intrinsic dim (TwoNN)"),
    ]
    ax.legend(handles=style_handles, loc="lower right", bbox_to_anchor=(0.60, 0.0),
              frameon=False, fontsize=11, labelspacing=0.35, handlelength=2.0,
              borderaxespad=0.2)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Number of principal components", fontsize=12.5)
    ax.set_ylabel("Cumulative explained variance", fontsize=12.5)
    ax.set_title("Linear (PCA) vs. intrinsic dimension of the embeddings",
                 fontsize=13, pad=10)

    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{filename}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def make_decimation_figure(curves: dict, out_dir, filename="decimation_figure",
                           title="Sampling-scale sensitivity"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = list(curves)
    colors = _assign_colors(labels)
    est = curves[labels[0]]["estimator"] if labels else "twonn"
    est_label = "TwoNN" if est == "twonn" else est.upper()

    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    for lab in labels:
        c = colors[lab]
        d = curves[lab]
        s, m, sd = d["sizes"], d["mean"], d["std"]
        ax.fill_between(s, m - sd, m + sd, color=c, alpha=0.15, lw=0)
        ax.plot(s, m, color=c, lw=1.6, marker="o", ms=5, label=lab)

    ax.margins(x=0.03)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=11, title="Model", title_fontsize=12, labelspacing=0.35,
              handlelength=1.6)
    ax.set_xlabel("Number of sampled sentences", fontsize=12.5)
    ax.set_ylabel(f"Intrinsic dimension ({est_label})", fontsize=12.5)
    ax.set_title(title, fontsize=13, pad=10)

    fig.tight_layout()
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"{filename}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def make_figure_from_saved(results_dir, **kwargs):
    results_dir = Path(results_dir)
    curves = np.load(results_dir / "curves.npz", allow_pickle=True)["curves"].item()
    return make_cumvar_figure(curves, results_dir, **kwargs)
