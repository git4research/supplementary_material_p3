"""2-D UMAP plots of the sentence embeddings, coloured by tier, with kNN edges overlaid (Sections 2.2-2.3, 3.2).

For each encoder and k, runs UMAP (n_neighbors=k, cosine metric) on outputs/<model>/embeddings.npy and
draws the edges of the graph in outputs/<model>/knn_k<k>_euclidean/adjacency_csr.npz (from 3_build_knn.py).
Writes umap_coordinates.csv and umap_<model>_k<k>.png to that graph directory.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

try:
    import umap
except ImportError:
    raise ImportError("Please install umap-learn first: pip install umap-learn")


CSV_PATH = "full_dataset_with_scores_new.csv"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding"]

K_LIST = [10, 15, 20, 30]
GRAPH_FILENAME = "adjacency_csr.npz"
OUTPUT_ROOT = "outputs"
GRAPH_METRIC_NAME = "euclidean"

TIER_COL = "tier"

RANDOM_SEED = 42
MIN_DIST = 0.15
UMAP_METRIC = "cosine"
# True: UMAP on the embeddings; False: UMAP on dense adjacency rows.
USE_EMBEDDINGS_FOR_UMAP = True

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

# Plot-title labels; models not listed fall back to their key.
MODEL_LABELS = {
    "bge": "BGE-large",
    "mpnet": "MPNet",
    "gte": "GTE-large",
}


def run_single_k(df: pd.DataFrame, model: str, k: int) -> None:
    """Run UMAP for one (model, k) and save the coordinates and the tier-coloured plot."""
    model_dir = os.path.join(OUTPUT_ROOT, model)
    graph_dir = os.path.join(model_dir, f"knn_k{k}_{GRAPH_METRIC_NAME}")
    graph_path = os.path.join(graph_dir, GRAPH_FILENAME)
    embed_path = os.path.join(model_dir, "embeddings.npy")

    if not os.path.exists(graph_path):
        print(f"[Skip] Graph not found for model={model}, k={k}: {graph_path}")
        return

    os.makedirs(graph_dir, exist_ok=True)

    A = sparse.load_npz(graph_path)

    if len(df) != A.shape[0]:
        raise ValueError(
            f"CSV rows and graph nodes mismatch for model={model}, k={k}: "
            f"CSV={len(df)}, graph={A.shape[0]}"
        )

    if USE_EMBEDDINGS_FOR_UMAP:
        if not os.path.exists(embed_path):
            print(f"[Skip] Embeddings not found for model={model}: {embed_path}")
            return

        X = np.load(embed_path)
        if X.shape[0] != len(df):
            raise ValueError(
                f"Embeddings rows and CSV rows mismatch for model={model}: "
                f"embeddings={X.shape[0]}, CSV={len(df)}"
            )
    else:
        X = A.toarray()

    # UMAP neighbourhood size matches k; the drawn edges come from the saved kNN graph.
    reducer = umap.UMAP(
        n_neighbors=k,
        min_dist=MIN_DIST,
        n_components=2,
        metric=UMAP_METRIC,
        random_state=RANDOM_SEED,
    )
    Y = reducer.fit_transform(X)

    out_df = df.copy()
    out_df["umap_x"] = Y[:, 0]
    out_df["umap_y"] = Y[:, 1]

    out_csv = os.path.join(graph_dir, "umap_coordinates.csv")
    out_df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 7))

    Acoo = A.tocoo()
    # Draw each undirected edge once (i < j) as a faint line beneath the points.
    for i, j in zip(Acoo.row, Acoo.col):
        if i >= j:
            continue
        ax.plot(
            [Y[i, 0], Y[j, 0]],
            [Y[i, 1], Y[j, 1]],
            color="lightgray",
            linewidth=0.35,
            alpha=0.25,
            zorder=1,
        )

    for tier in TIER_ORDER:
        sub = out_df[out_df[TIER_COL] == tier]
        if len(sub) == 0:
            continue

        ax.scatter(
            sub["umap_x"],
            sub["umap_y"],
            s=28,
            alpha=0.9,
            color=TIER_COLORS.get(tier, None),
            label=tier,
            edgecolors="none",
            zorder=2,
        )

    input_name = "embeddings" if USE_EMBEDDINGS_FOR_UMAP else "adjacency rows"
    model_label = MODEL_LABELS.get(model, model)

    ax.set_title(
        f"{model_label} (k = {k})",
        fontsize=13,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(False)

    plt.tight_layout()

    out_png = os.path.join(graph_dir, f"umap_{model}_k{k}.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] model={model}, k={k}")
    print(f"     Saved coordinates: {out_csv}")
    print(f"     Saved figure:      {out_png}")


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if TIER_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TIER_COL}' column")

    for model in MODEL_LIST:
        print("\n" + "=" * 60)
        print(f"Processing model: {model}")
        print("=" * 60)

        for k in K_LIST:
            print("\n" + "-" * 50)
            print(f"Running UMAP plot for k = {k}")
            print("-" * 50)
            run_single_k(df, model, k)


if __name__ == "__main__":
    main()
