"""Geodesic-Euclidean Stretch (GES) on precomputed kNN graphs, with tier-pair heatmaps (Sections 2.2-2.3, 3.2).

For each model and k, samples 200 sentence pairs per ordered tier pair and takes the ratio of the weighted
shortest-path length on the kNN graph to the Euclidean distance between the embeddings. Reads
outputs/<model>/embeddings.npy and knn_k<k>_euclidean/adjacency_csr.npz; writes ges_tierpair_long.csv,
ges_heatmap.csv, ges_summary.csv and a heatmap PNG per k, plus outputs/<model>/ges_summary_all_k_euclidean.csv.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.csgraph import shortest_path


CSV_PATH = "full_dataset_with_scores_new.csv"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding",  "qwen2.5_3b_L24"]

K_LIST = [10, 15, 20, 30]
GRAPH_FILENAME = "adjacency_csr.npz"
OUTPUT_ROOT = "outputs"
GRAPH_METRIC_NAME = "euclidean"

TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]

# Random pairs sampled for each ordered (source tier, target tier) combination.
PAIRS_PER_TIERPAIR = 200
RANDOM_SEED = 123
EPS = 1e-12

HEATMAP_CMAP = "viridis"
HEATMAP_DPI = 300
ANNOT_DECIMALS = 2


def sample_pairs(rng: np.random.Generator, idx_a: np.ndarray, idx_b: np.ndarray, m: int):
    """Draw m (i, j) pairs with replacement, i from idx_a and j from idx_b; i != j when the groups coincide."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return []

    pairs = []
    same_group = np.array_equal(idx_a, idx_b)

    for _ in range(m):
        i = int(rng.choice(idx_a))
        j = int(rng.choice(idx_b))

        if same_group and len(idx_a) > 1:
            while j == i:
                j = int(rng.choice(idx_b))

        pairs.append((i, j))

    return pairs


def plot_heatmap_with_values(
    heatmap_df: pd.DataFrame,
    out_path: str,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = HEATMAP_CMAP,
    decimals: int = ANNOT_DECIMALS,
) -> None:
    """Plot a tier-pair GES heatmap with per-cell values on a fixed colour range."""
    values = heatmap_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(values, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_xticklabels(heatmap_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(heatmap_df.index)

    ax.set_xlabel("tier_dst")
    ax.set_ylabel("tier_src")
    ax.set_title(title)

    threshold = (vmin + vmax) / 2.0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                text = f"{v:.{decimals}f}"
                text_color = "white" if v > threshold else "black"
            else:
                text = "NA"
                text_color = "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean GES stretch")

    plt.tight_layout()
    plt.savefig(out_path, dpi=HEATMAP_DPI, bbox_inches="tight")
    plt.close(fig)


def compute_ges_for_k(
    df: pd.DataFrame,
    E: np.ndarray,
    model: str,
    k: int,
    base_seed: int,
):
    """GES for one (model, k): per-tier-pair and overall stretch statistics; None if the graph is missing."""
    graph_dir = os.path.join(OUTPUT_ROOT, model, f"knn_k{k}_{GRAPH_METRIC_NAME}")
    graph_path = os.path.join(graph_dir, GRAPH_FILENAME)

    if not os.path.exists(graph_path):
        print(f"[Skip] Graph not found for model={model}, k={k}: {graph_path}")
        return None

    # Separate, reproducible pair sample for each k.
    rng = np.random.default_rng(base_seed + k)

    A = sparse.load_npz(graph_path)

    if len(df) != E.shape[0] or len(df) != A.shape[0]:
        raise ValueError(
            f"CSV, embeddings, and graph size mismatch for model={model}, k={k}: "
            f"CSV={len(df)}, EMB={E.shape[0]}, GRAPH={A.shape[0]}"
        )

    print(f"Computing shortest paths for model={model}, k={k} ...")
    # All-pairs weighted geodesic distances; inf for pairs in different components.
    distG = shortest_path(A, directed=False, unweighted=False)

    tiers_as_str = df["tier"].astype(str)
    tiers_present = [t for t in TIER_ORDER if t in set(tiers_as_str)]
    idx_by_tier = {t: df.index[tiers_as_str == t].to_numpy() for t in tiers_present}

    rows = []
    all_stretches = []
    reachable_pairs = 0
    total_pairs = 0

    for ta in tiers_present:
        for tb in tiers_present:
            pairs = sample_pairs(rng, idx_by_tier[ta], idx_by_tier[tb], PAIRS_PER_TIERPAIR)
            stretches = []

            for i, j in pairs:
                total_pairs += 1
                g = distG[i, j]

                # Skip unreachable pairs and zero-distance pairs; reachable_ratio = pairs used / pairs sampled.
                if not np.isfinite(g):
                    continue

                # Euclidean distance on the embeddings as loaded; they are already unit-norm (script 1 saves L2-normalized vectors).
                e = float(np.linalg.norm(E[i] - E[j]))
                if e < EPS:
                    continue

                reachable_pairs += 1
                s = g / (e + EPS)
                stretches.append(s)
                all_stretches.append(s)

            rows.append(
                {
                    "model": model,
                    "tier_src": ta,
                    "tier_dst": tb,
                    "ges_mean": float(np.mean(stretches)) if stretches else np.nan,
                    "ges_median": float(np.median(stretches)) if stretches else np.nan,
                    "ges_std": float(np.std(stretches)) if stretches else np.nan,
                    "reachable_pairs": int(len(stretches)),
                    "sampled_pairs": int(len(pairs)),
                }
            )

    long_df = pd.DataFrame(rows)

    heatmap_df = long_df.pivot(index="tier_src", columns="tier_dst", values="ges_mean")
    heatmap_df = heatmap_df.reindex(index=tiers_present, columns=tiers_present)

    summary_df = pd.DataFrame(
        [
            {
                "model": model,
                "k": k,
                "reachable_ratio": reachable_pairs / total_pairs if total_pairs else np.nan,
                "ges_mean": float(np.mean(all_stretches)) if all_stretches else np.nan,
                "ges_median": float(np.median(all_stretches)) if all_stretches else np.nan,
                "ges_std": float(np.std(all_stretches)) if all_stretches else np.nan,
                "ges_pairs_used": int(len(all_stretches)),
                "total_pairs_sampled": int(total_pairs),
            }
        ]
    )

    return {
        "model": model,
        "k": k,
        "graph_dir": graph_dir,
        "long_df": long_df,
        "heatmap_df": heatmap_df,
        "summary_df": summary_df,
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "tier" not in df.columns:
        raise ValueError("CSV must contain 'tier' column")

    for model in MODEL_LIST:
        print("\n" + "=" * 70)
        print(f"Processing model: {model}")
        print("=" * 70)

        embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
        if not os.path.exists(embed_path):
            print(f"[Skip] Embeddings not found for model={model}: {embed_path}")
            continue

        E = np.load(embed_path)
        if E.ndim != 2:
            raise ValueError(f"Embeddings must be N x D, got {E.shape}")

        if len(df) != E.shape[0]:
            raise ValueError(
                f"CSV and embeddings size mismatch for model={model}: CSV={len(df)}, EMB={E.shape[0]}"
            )

        results = []

        for k in K_LIST:
            print("\n" + "-" * 60)
            print(f"Running GES for model={model}, k={k}")
            print("-" * 60)

            result = compute_ges_for_k(df, E, model, k, RANDOM_SEED)
            if result is not None:
                results.append(result)

        if not results:
            print(f"No valid k results were produced for model={model}.")
            continue

        # One colour scale shared across all k for this model, so its heatmaps are comparable.
        all_heatmap_values = []
        for result in results:
            vals = result["heatmap_df"].to_numpy(dtype=float)
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size > 0:
                all_heatmap_values.append(finite_vals)

        if not all_heatmap_values:
            raise ValueError(f"No finite heatmap values found across all k runs for model={model}.")

        all_heatmap_values = np.concatenate(all_heatmap_values)
        global_vmin = float(np.min(all_heatmap_values))
        global_vmax = float(np.max(all_heatmap_values))

        print("\n" + "-" * 60)
        print(f"Shared heatmap color scale across all k for model={model}")
        print("-" * 60)
        print(f"vmin = {global_vmin:.6f}")
        print(f"vmax = {global_vmax:.6f}")

        all_summary_rows = []

        for result in results:
            k = result["k"]
            graph_dir = result["graph_dir"]
            long_df = result["long_df"]
            heatmap_df = result["heatmap_df"]
            summary_df = result["summary_df"]

            os.makedirs(graph_dir, exist_ok=True)

            long_csv_path = os.path.join(graph_dir, "ges_tierpair_long.csv")
            heatmap_csv_path = os.path.join(graph_dir, "ges_heatmap.csv")
            summary_csv_path = os.path.join(graph_dir, "ges_summary.csv")
            heatmap_png_path = os.path.join(graph_dir, f"ges_heatmap_{model}_k{k}.png")

            long_df.to_csv(long_csv_path, index=False)
            heatmap_df.to_csv(heatmap_csv_path)
            summary_df.to_csv(summary_csv_path, index=False)

            plot_heatmap_with_values(
                heatmap_df=heatmap_df,
                out_path=heatmap_png_path,
                title=f"GES Heatmap ({model}, k={k})",
                vmin=global_vmin,
                vmax=global_vmax,
            )

            print(f"\n=== GES summary for model={model}, k={k} ===")
            print(summary_df.to_string(index=False))

            print(f"\n=== GES heatmap (mean stretch) for model={model}, k={k} ===")
            print(heatmap_df.round(3).to_string())

            print(f"\n[OK] Saved:")
            print(f"  - {long_csv_path}")
            print(f"  - {heatmap_csv_path}")
            print(f"  - {summary_csv_path}")
            print(f"  - {heatmap_png_path}")

            all_summary_rows.append(summary_df)

        if all_summary_rows:
            combined_summary = pd.concat(all_summary_rows, ignore_index=True)
            combined_path = os.path.join(OUTPUT_ROOT, model, f"ges_summary_all_k_{GRAPH_METRIC_NAME}.csv")
            combined_summary.to_csv(combined_path, index=False)

            print("\n" + "-" * 60)
            print(f"Combined GES summary across k for model={model}")
            print("-" * 60)
            print(combined_summary.to_string(index=False))
            print(f"\n[OK] Saved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()
