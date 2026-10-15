"""Draw one hard-coded trajectory on the 2-D UMAP projection (Appendix L figures).

Reads outputs/<MODEL>/knn_k<K>_euclidean/umap_coordinates.csv, overlays the numbered path in
PATH_ROWS (arrows, start/end markers) on the tier-colored scatter, and writes
umap_<MODEL>_k<K>_path.png and path_coordinates_<MODEL>_k<K>.csv to the same folder.
Usage: python 10_mark_path.py  (edit MODEL, K, and PATH_ROWS at the top to change the path)
"""
from __future__ import annotations

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import grid


MODEL = "qwen_embedding"
K = 15
GRAPH_METRIC_NAME = "euclidean"
OUTPUT_ROOT = "outputs"

TIER_COL = "tier"
INDEX_COL = "index"
PATH_INDEX_COL = "node_id"

UMAP_CSV = os.path.join(
    OUTPUT_ROOT,
    MODEL,
    f"knn_k{K}_{GRAPH_METRIC_NAME}",
    "umap_coordinates.csv",
)

OUT_PNG = os.path.join(
    OUTPUT_ROOT,
    MODEL,
    f"knn_k{K}_{GRAPH_METRIC_NAME}",
    f"umap_{MODEL}_k{K}_path.png",
)

TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]

TIER_COLORS = {
    "Collapse": "#ff3b3b",
    "Striving": "#ff8b3d",
    "Conflict": "#ffd93b",
    "Activation": "#34d399",
    "Growth": "#60a5fa",
    "Clarity": "#6366f1",
    "Unity": "#a855f7",
}


# Path to draw: (step, node index in the UMAP CSV, tier, score, sentence).
PATH_ROWS = [
    (0, 11, "Collapse", -4.75, "My voice feels too small to count."),
    (1, 244, "Conflict", -1.78, "Please, they're not even on my level."),
    (2, 250, "Conflict", -1.75, "I deserve more respect than any of them."),
    (3, 118, "Striving", -3.02, "I’m always checking where I stand compared to others."),
    (4, 120, "Striving", -3.02, "I keep improving myself so I won’t be left behind."),
    (5, 387, "Growth", 0.88, "I keep my word, even when it's hard."),
    (6, 401, "Growth", 0.96, "I stay honest, even when it feels vulnerable."),
    (7, 402, "Growth", 0.96, "I express my experience honestly, without pretending."),
    (8, 496, "Clarity", 2.68, "I can stay with the facts without getting lost in reactions."),
    (9, 357, "Activation", 0.45, "I’m learning to step back quickly before reactions take over."),
    (10, 445, "Clarity", 2.4, "I reason through emotion until insight appears."),
    (11, 477, "Clarity", 2.6, "Intelligence is the ability to pick out the details and understand how they connect."),
    (12, 474, "Clarity", 2.6, "Abstractions help me glimpse the unseen structure of things."),
    (13, 475, "Clarity", 2.6, "I can move between concrete and abstract with ease."),
    (14, 485, "Clarity", 2.63, "The concept is clear, I get it!"),
    (15, 464, "Clarity", 2.51, "What feels clearer now is the way the parts relate and make sense together."),
    (16, 489, "Clarity", 2.65, "Each part makes sense in relation to the whole."),
    (17, 551, "Unity", 3.8, "Everything is part of the same pulse."),
    (18, 609, "Unity", 4.48, "I am inseparable from the sacred wholeness that animates all."),
    (19, 543, "Unity", 3.6, "Everything feels sacred and astonishingly interconnected."),
    (20, 509, "Unity", 2.75, "Harmony is when every difference finds its place in the whole."),
    (21, 624, "Unity", 4.55, "The merging of knower and known into one living stillness."),
    (22, 589, "Unity", 4.3, "Beatitude is the quiet joy of knowing nothing is separate."),
    (23, 606, "Unity", 4.47, "Pure awareness, beyond subject and object."),
]


def main():
    df = pd.read_csv(UMAP_CSV, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if INDEX_COL not in df.columns:
        raise ValueError(
            f"Cannot find column '{INDEX_COL}' in {UMAP_CSV}. "
            f"Available columns: {list(df.columns)}"
        )

    path_df = pd.DataFrame(
        PATH_ROWS,
        columns=["step", PATH_INDEX_COL, "path_tier", "path_score", "path_sentence"],
    )

    # Attach UMAP coordinates to each path node by matching node_id to the index column.
    merged = path_df.merge(
        df[[INDEX_COL, "umap_x", "umap_y", TIER_COL]],
        left_on=PATH_INDEX_COL,
        right_on=INDEX_COL,
        how="left",
    )

    if merged[["umap_x", "umap_y"]].isna().any().any():
        missing = merged[merged["umap_x"].isna()][PATH_INDEX_COL].tolist()
        raise ValueError(f"Some path nodes were not found in UMAP CSV: {missing}")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all sentences colored by tier.
    for tier in TIER_ORDER:
        sub = df[df[TIER_COL] == tier]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=24,
            alpha=0.9,
            color=TIER_COLORS.get(tier),
            label=tier,
            edgecolors="none",
            zorder=1,
        )

    ax.plot(
        merged["umap_x"],
        merged["umap_y"],
        linewidth=2.2,
        alpha=0.9,
        color="black",
        zorder=3,
    )


    # Arrow for each consecutive step, then step numbers on top of each node.
    for i in range(len(merged) - 1):
        x1, y1 = merged.loc[i, ["umap_x", "umap_y"]]
        x2, y2 = merged.loc[i + 1, ["umap_x", "umap_y"]]

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                lw=1.6,
                alpha=0.85,
                shrinkA=4,
                shrinkB=4,
            ),
            zorder=5,
        )

    for _, row in merged.iterrows():
        ax.text(
            row["umap_x"],
            row["umap_y"],
            str(int(row["step"])),
            fontsize=8,
            color="black",
            ha="center",
            va="center",
            zorder=6,
        )

    start = merged.iloc[0]
    end = merged.iloc[-1]

    ax.scatter(
        start["umap_x"],
        start["umap_y"],
        s=180,
        marker="*",
        color="limegreen",
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
        label="Start",
    )

    ax.scatter(
        end["umap_x"],
        end["umap_y"],
        s=180,
        marker="X",
        color="red",
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
        label="End",
    )

    ax.text(
        start["umap_x"],
        start["umap_y"],
        " START",
        fontsize=15,
        fontweight="bold",
        color="green",
        zorder=8,
    )

    ax.text(
        end["umap_x"],
        end["umap_y"],
        " END",
        fontsize=15,
        fontweight="bold",
        color="red",
        zorder=8,
    )

    ax.legend(frameon=True, fontsize=12)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.get_frame().set_alpha(0.9)

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    merged.to_csv(
        os.path.join(
            OUTPUT_ROOT,
            MODEL,
            f"knn_k{K}_{GRAPH_METRIC_NAME}",
            f"path_coordinates_{MODEL}_k{K}.csv",
        ),
        index=False,
    )

    print(f"[OK] Saved path figure: {OUT_PNG}")


if __name__ == "__main__":
    main()
