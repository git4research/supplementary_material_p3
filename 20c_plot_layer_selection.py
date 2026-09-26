import os
import pandas as pd
import matplotlib.pyplot as plt

MODELS = [
    ("Qwen2.5-3B",   "outputs/qwen2.5_3b/layer_selection/layer_metrics.csv",                       36, 24, "#9ecae1", "o"),
    ("Qwen2.5-7B",   "hf_results/qwen_7b_L20/layer_selection/layer_metrics (1).csv",                28, 20, "#4292c6", "o"),
    ("Qwen2.5-14B",  "hf_results/qwen_14b_L24/layer_selection/layer_metrics_qwen_14b_L24.csv",      48, 24, "#08519c", "o"),
    ("Llama-3.1-8B", "hf_results/llama_8b_L16/layer_selection/layer_metrics.csv",                  32, 16, "#f16913", "s"),
    ("Gemma-2-9B",   "hf_results/gemma2_9b_L29/layer_selection/layer_metrics.csv",                  42, 29, "#74c476", "^"),
    ("Gemma-2-27B",  "hf_results/gemma2_7b_L28/layer_selection/layer_metrics_gemma2_27b_L28.csv",   46, 28, "#006d2c", "^"),
]
METRICS = [("ridge_r2", "Ridge $R^2$"), ("logreg_acc", "Logistic acc."), ("tts_jump01", "TTS jump-0/1 rate"),
           ("silhouette", "Silhouette"), ("composite", "Composite")]
OUT = "outputs/figures_multimodel/layer_selection_multimodel"

fig, axes = plt.subplots(1, len(METRICS), figsize=(7.2, 1.85), sharex=True)
for name, csv, nl, chosen, col, mk in MODELS:
    d = pd.read_csv(csv)
    depth = d["layer"] / nl * 100
    assert d.loc[d.composite.idxmax(), "layer"] == chosen, (name, "chosen layer is not argmax composite")
    for ax, (m, _) in zip(axes, METRICS):
        ax.plot(depth, d[m], color=col, marker=mk, ms=2.5, lw=1.0, label=name)
        r = d[d.layer == chosen]
        ax.plot(r["layer"] / nl * 100, r[m], marker="*", ms=8, color=col, mec="black", mew=0.4, ls="", zorder=5)
for ax, (_, title) in zip(axes, METRICS):
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.tick_params(labelsize=6, length=2, pad=1)
    ax.set_xticks([0, 50, 100]); ax.set_xlim(0, 104)
    ax.grid(alpha=0.25, lw=0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[3].axhline(0, color="0.5", lw=0.5)
axes[4].set_ylim(-0.05, 1.1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.03),
           handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 0.9), w_pad=0.6)
fig.text(0.5, 0.02, "relative depth (%)", ha="center", fontsize=7)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
print("wrote", OUT + ".pdf/.png")
