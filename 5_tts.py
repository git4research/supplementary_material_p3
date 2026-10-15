"""Tier Transition Smoothness (TTS) on precomputed kNN graphs (Sections 2.2-2.3, 3.2).

An edge's jump is the absolute difference in ordinal tier index of its endpoints. Reports the
distribution of jumps over all edges and, per tier, the fraction of edges with jump 0, 1, 2, > 2.
Reads the dataset CSV and outputs/<model>/knn_k<k>_euclidean/adjacency_csr.npz (from 3_build_knn.py);
writes tts_overall.csv and tts_by_tier.csv to the same directory.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse


CSV_PATH = "full_dataset_with_scores_new.csv"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding", "qwen2.5_3b_L24"]

K_LIST = [10, 15, 20, 30]
METRIC = "euclidean"
GRAPH_FILENAME = "adjacency_csr.npz"

OUTPUT_ROOT = "outputs"

TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]
TIER_TO_IDX = {t: i for i, t in enumerate(TIER_ORDER)}


def compute_tts_for_graph(df: pd.DataFrame, graph_path: str):
    """Return (overall jump-size distribution, per-tier jump fractions) for one graph."""
    if "tier" not in df.columns:
        raise ValueError("CSV must contain 'tier' column")

    A = sparse.load_npz(graph_path)

    if len(df) != A.shape[0]:
        raise ValueError(
            f"CSV rows and graph nodes mismatch: CSV={len(df)}, graph={A.shape[0]}"
        )

    tiers = df["tier"].astype(str).to_numpy()
    tier_idx = np.array([TIER_TO_IDX.get(t, -1) for t in tiers], dtype=int)

    if np.any(tier_idx < 0):
        unknown = sorted(set(tiers[tier_idx < 0]))
        raise ValueError(f"Unknown tier labels: {unknown}")

    Acoo = A.tocoo()

    jump_counts = defaultdict(int)
    jump_counts_by_tier = {t: defaultdict(int) for t in TIER_ORDER}

    # The symmetric adjacency stores each edge twice; count it once (i < j).
    for i, j in zip(Acoo.row, Acoo.col):
        if i >= j:
            continue

        ti = tier_idx[i]
        tj = tier_idx[j]
        jump = abs(int(ti - tj))

        jump_counts[jump] += 1
        # Per-tier counts are edge incidents: an edge counts for both endpoints' tiers (twice if same tier).
        jump_counts_by_tier[tiers[i]][jump] += 1
        jump_counts_by_tier[tiers[j]][jump] += 1

    total_edges = sum(jump_counts.values())
    overall_rows = []
    for jump in sorted(jump_counts.keys()):
        count = jump_counts[jump]
        overall_rows.append(
            {
                "jump": jump,
                "count": count,
                "ratio": count / total_edges if total_edges else np.nan,
            }
        )
    overall_df = pd.DataFrame(overall_rows)

    tier_rows = []
    for t in TIER_ORDER:
        total = sum(jump_counts_by_tier[t].values())
        if total == 0:
            continue

        jump0 = jump_counts_by_tier[t].get(0, 0)
        jump1 = jump_counts_by_tier[t].get(1, 0)
        jump2 = jump_counts_by_tier[t].get(2, 0)
        jump3p = sum(v for k, v in jump_counts_by_tier[t].items() if k > 2)

        tier_rows.append(
            {
                "tier": t,
                "edge_incidents": total,
                "jump0_ratio": jump0 / total,
                "jump1_ratio": jump1 / total,
                "jump2_ratio": jump2 / total,
                "jump>2_ratio": jump3p / total,
            }
        )

    tier_df = pd.DataFrame(tier_rows)

    return overall_df, tier_df


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "tier" not in df.columns:
        raise ValueError("CSV must contain 'tier' column")

    for model in MODEL_LIST:
        print("\n" + "=" * 60)
        print(f"Processing model: {model}")
        print("=" * 60)

        for k in K_LIST:
            print("\n" + "-" * 50)
            print(f"Running TTS for k = {k}")
            print("-" * 50)

            graph_dir = os.path.join(
                OUTPUT_ROOT, model, f"knn_k{k}_{METRIC}"
            )
            graph_path = os.path.join(graph_dir, GRAPH_FILENAME)

            if not os.path.exists(graph_path):
                print(f"[Skip] model={model}, k={k}: graph not found")
                continue

            overall_df, tier_df = compute_tts_for_graph(df, graph_path)

            os.makedirs(graph_dir, exist_ok=True)

            overall_path = os.path.join(graph_dir, "tts_overall.csv")
            tier_path = os.path.join(graph_dir, "tts_by_tier.csv")

            overall_df.to_csv(overall_path, index=False)
            tier_df.to_csv(tier_path, index=False)

            print(f"[OK] model={model}, k={k}")
            print(f"     Saved: {overall_path}")
            print(f"     Saved: {tier_path}")

            print("\n=== TTS overall ===")
            print(overall_df.to_string(index=False))

            print("\n=== TTS by tier ===")
            print(tier_df.to_string(index=False))


if __name__ == "__main__":
    main()
