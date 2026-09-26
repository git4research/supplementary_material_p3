from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


CSV_PATH = "full_dataset_with_scores_new.csv"
TIER_COL = "tier"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding", "qwen2.5_3b_L24"]

K_LIST = [10, 15, 20, 30]
METRIC = "euclidean"
GRAPH_FILENAME = "adjacency_csr.npz"

OUTPUT_ROOT = "outputs"

PAIRS_PER_TIER = 400
MIN_TIER_SIZE = 5

SEEDS = list(range(2002, 2042))

SORT_BY_CONTAINMENT = True


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if TIER_COL not in df.columns:
        raise ValueError(f"CSV missing column '{TIER_COL}'. Found: {list(df.columns)}")
    return df


def load_graph(path: str) -> sparse.csr_matrix:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing graph file: {path} (run build_knn.py first)")
    A = sparse.load_npz(path).tocsr()
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square. Got {A.shape}")
    return A


def reconstruct_path(pred: np.ndarray, src: int, dst: int) -> List[int]:
    if src == dst:
        return [src]

    path: List[int] = []
    cur = int(dst)

    while cur != src and cur != -9999:
        path.append(cur)
        cur = int(pred[cur])

    if cur == -9999:
        return []

    path.append(int(src))
    path.reverse()
    return path


def sample_pairs(idx: np.ndarray, n_pairs: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    if idx.size < 2:
        return []

    pairs: List[Tuple[int, int]] = []
    for _ in range(n_pairs):
        a, b = rng.choice(idx, 2, replace=False)
        pairs.append((int(a), int(b)))
    return pairs


def compute_all_tier_containment_for_seed(
    df: pd.DataFrame,
    A: sparse.csr_matrix,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pred_cache: dict[int, np.ndarray] = {}

    summary_rows = []
    sample_rows = []

    tiers = sorted(df[TIER_COL].astype(str).unique().tolist())
    tier_values = df[TIER_COL].astype(str).values

    for tier in tiers:
        idx = np.where(tier_values == tier)[0].astype(int)

        if idx.size < max(2, MIN_TIER_SIZE):
            summary_rows.append(
                {
                    "seed": seed,
                    "tier": tier,
                    "n_points": int(idx.size),
                    "pairs": 0,
                    "containment_mean": np.nan,
                    "containment_median": np.nan,
                    "full_containment_rate": np.nan,
                }
            )
            continue

        pairs = sample_pairs(idx, PAIRS_PER_TIER, rng)
        S = set(idx.tolist())
        scores: List[float] = []

        for src, dst in pairs:
            if src not in pred_cache:
                _, pred = dijkstra(A, directed=False, indices=src, return_predecessors=True)
                pred_cache[src] = pred

            path = reconstruct_path(pred_cache[src], src, dst)

            if not path:
                cont = 0.0
                path_len = 0
                inside = 0
            else:
                inside = sum(1 for v in path if v in S)
                cont = inside / len(path)
                path_len = len(path)

            scores.append(cont)
            sample_rows.append(
                {
                    "seed": seed,
                    "tier": tier,
                    "src": src,
                    "dst": dst,
                    "path_len": path_len,
                    "inside_count": inside,
                    "containment": cont,
                }
            )

        scores_arr = np.array(scores, dtype=float)

        summary_rows.append(
            {
                "seed": seed,
                "tier": tier,
                "n_points": int(idx.size),
                "pairs": int(len(pairs)),
                "containment_mean": float(np.mean(scores_arr)) if scores_arr.size > 0 else np.nan,
                "containment_median": float(np.median(scores_arr)) if scores_arr.size > 0 else np.nan,
                "full_containment_rate": float(np.mean(np.isclose(scores_arr, 1.0))) if scores_arr.size > 0 else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    sample_df = pd.DataFrame(sample_rows)
    return summary_df, sample_df


def aggregate_across_seeds(per_seed_summary: pd.DataFrame) -> pd.DataFrame:
    grouped = per_seed_summary.groupby("tier", dropna=False)

    agg_df = grouped.agg(
        n_points=("n_points", "first"),
        n_seeds=("seed", "nunique"),
        pairs_per_seed=("pairs", "first"),
        containment_mean_mean=("containment_mean", "mean"),
        containment_mean_std=("containment_mean", "std"),
        containment_mean_median=("containment_mean", "median"),
        containment_median_mean=("containment_median", "mean"),
        containment_median_std=("containment_median", "std"),
        full_containment_rate_mean=("full_containment_rate", "mean"),
        full_containment_rate_std=("full_containment_rate", "std"),
    ).reset_index()

    if SORT_BY_CONTAINMENT:
        agg_df = agg_df.sort_values(
            by=["containment_mean_mean", "full_containment_rate_mean"],
            ascending=False,
        )
    else:
        agg_df = agg_df.sort_values("tier")

    return agg_df


def run_single_k(df: pd.DataFrame, model: str, k: int) -> pd.DataFrame | None:
    adj_path = os.path.join(OUTPUT_ROOT, model, f"knn_k{k}_{METRIC}", GRAPH_FILENAME)
    out_dir = os.path.join(OUTPUT_ROOT, model, f"convex_k{k}_{METRIC}")

    if not os.path.exists(adj_path):
        print(f"[Skip] Missing graph for model={model}, k={k}: {adj_path}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    A = load_graph(adj_path)

    if len(df) != A.shape[0]:
        raise ValueError(
            f"CSV rows ({len(df)}) != graph nodes ({A.shape[0]}) for model={model}, k={k}. Alignment issue."
        )

    all_seed_summaries = []
    all_seed_samples = []

    for seed in SEEDS:
        summary_df, sample_df = compute_all_tier_containment_for_seed(df, A, seed=seed)
        all_seed_summaries.append(summary_df)
        all_seed_samples.append(sample_df)

    per_seed_summary_df = pd.concat(all_seed_summaries, ignore_index=True)
    all_samples_df = pd.concat(all_seed_samples, ignore_index=True)
    agg_summary_df = aggregate_across_seeds(per_seed_summary_df)

    per_seed_summary_df.insert(0, "k", k)
    per_seed_summary_df.insert(0, "model", model)

    all_samples_df.insert(0, "k", k)
    all_samples_df.insert(0, "model", model)

    agg_summary_df.insert(0, "k", k)
    agg_summary_df.insert(0, "model", model)

    per_seed_summary_path = os.path.join(out_dir, "tier_geodesic_containment_by_seed.csv")
    agg_summary_path = os.path.join(out_dir, "tier_geodesic_containment_summary.csv")
    sample_path = os.path.join(out_dir, "tier_pair_containment_samples.csv")

    per_seed_summary_df.to_csv(per_seed_summary_path, index=False)
    agg_summary_df.to_csv(agg_summary_path, index=False)
    all_samples_df.to_csv(sample_path, index=False)

    print(f"\n[OK] Saved outputs for model={model}, k={k} to: {out_dir}")
    print(f"  - {os.path.basename(per_seed_summary_path)}")
    print(f"  - {os.path.basename(agg_summary_path)}")
    print(f"  - {os.path.basename(sample_path)}")

    print(f"\n=== Geodesic containment summary across seeds (model={model}, k={k}) ===")
    cols = [
        "tier",
        "n_points",
        "n_seeds",
        "pairs_per_seed",
        "containment_mean_mean",
        "containment_mean_std",
        "containment_mean_median",
        "full_containment_rate_mean",
        "full_containment_rate_std",
    ]
    print(agg_summary_df[cols].round(4).to_string(index=False))

    return agg_summary_df


def main() -> None:
    df = read_csv(CSV_PATH)

    for model in MODEL_LIST:
        print("\n" + "=" * 70)
        print(f"Processing model: {model}")
        print("=" * 70)

        combined_rows = []

        for k in K_LIST:
            print("\n" + "-" * 60)
            print(f"Running convexity for model={model}, k={k}")
            print("-" * 60)

            agg_summary_df = run_single_k(df, model, k)
            if agg_summary_df is not None:
                combined_rows.append(agg_summary_df)

        if combined_rows:
            combined_df = pd.concat(combined_rows, ignore_index=True)
            combined_path = os.path.join(OUTPUT_ROOT, model, f"convex_summary_all_k_{METRIC}.csv")
            combined_df.to_csv(combined_path, index=False)

            print("\n" + "-" * 60)
            print(f"Combined convexity summary across k for model={model}")
            print("-" * 60)
            print(
                combined_df[
                    [
                        "k",
                        "tier",
                        "containment_mean_mean",
                        "containment_mean_std",
                        "full_containment_rate_mean",
                        "full_containment_rate_std",
                    ]
                ]
                .round(4)
                .to_string(index=False)
            )
            print(f"\n[OK] Saved combined summary to: {combined_path}")


if __name__ == "__main__":
    main()
