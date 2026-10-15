"""Geometry-only "transition corridor" baseline: no-revisit greedy walk (Section 3.3.1 / Appendix F.5-F.6).

From every Collapse/Striving/Conflict sentence, repeatedly step to the most cosine-similar
unvisited kNN neighbor (no score guidance), stopping at the first Unity node or a dead end.
Reports Hit@Unity and P(passed Activation/Growth | hit Unity). Reads outputs/<model>/embeddings.npy
and the kNN adjacency; writes per-run plots/CSVs and outputs/greedy_transition_corridor_all_models.csv.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from typing import Dict, List


CSV_PATH = "full_dataset_with_scores_new.csv"
OUTPUT_ROOT = "outputs"

MODEL_LIST = ["bge", "qwen_embedding", "qwen2.5_3b_L24"]
K_LIST = [15, 30]
GRAPH_METRIC_NAME = "euclidean"

LOW_START_TIERS = {"Collapse", "Striving", "Conflict"}
MAX_STEPS = 150
TIER_COL = "tier"

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

SHOW_PLOTS = False
SAVE_PLOTS = True


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def cosine_sim_vector(E: np.ndarray, i: int, nbrs: list[int] | np.ndarray) -> np.ndarray:
    # Cosine similarity of node i to each neighbor (assumes rows of E are L2-normalized).
    nbrs = np.asarray(nbrs, dtype=int)
    return E[nbrs] @ E[i]


def build_step_distribution(
    end_tiers_by_step: list[list[str]],
    tier_order: list[str],
) -> pd.DataFrame:
    """Tier proportions of all walkers' current nodes at each step (stopped walkers keep their last node)."""
    rows = []

    for step, end_tiers in enumerate(end_tiers_by_step, start=1):
        counts = pd.Series(end_tiers).value_counts(normalize=True)

        for tier in tier_order:
            rows.append({
                "step": step,
                "tier": tier,
                "proportion": float(counts.get(tier, 0.0)),
            })

    return pd.DataFrame(rows)


def plot_tier_dynamics(
    step_dist_df: pd.DataFrame,
    model: str,
    k: int,
    save_path: str | None = None,
) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
    })

    plt.figure(figsize=(6.5, 4))

    for tier in TIER_ORDER:
        sub = step_dist_df[step_dist_df["tier"] == tier]
        if len(sub) > 0:
            plt.plot(
                sub["step"],
                sub["proportion"],
                label=tier,
                color=TIER_COLORS[tier],
                linewidth=1.8,
                alpha=0.95,
            )

    plt.xlabel("Step")
    plt.ylabel("Proportion")

    plt.title(f"{model} (k={k})")

    plt.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    plt.legend(
        loc="upper left",
        frameon=False,
        ncol=2
    )

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if SHOW_PLOTS:
        plt.show()

    plt.close()

def run_single_experiment(df: pd.DataFrame, model: str, k: int) -> pd.DataFrame:
    embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
    graph_path = os.path.join(
        OUTPUT_ROOT, model, f"knn_k{k}_{GRAPH_METRIC_NAME}", "adjacency_csr.npz"
    )

    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    out_dir = os.path.join(OUTPUT_ROOT, model, f"greedy_walk_k{k}_{GRAPH_METRIC_NAME}")
    os.makedirs(out_dir, exist_ok=True)

    E = l2_normalize_rows(np.load(embed_path))
    A = sparse.load_npz(graph_path).tocsr()

    # Sanity-check printout of graph degree statistics and a few neighbor lists.
    print("\n--- Graph Debug Info ---")
    print("Graph path:", graph_path)
    print("Graph nnz:", A.nnz)

    degrees = np.diff(A.indptr)
    print("Average degree:", degrees.mean())
    print("Min degree:", degrees.min())
    print("Max degree:", degrees.max())

    for node in [0, 10, 50, 100]:
        nbrs = A.indices[A.indptr[node]:A.indptr[node + 1]]
        print(f"node={node}, degree={len(nbrs)}, nbrs[:10]={nbrs[:10]}")
    print("--- End Graph Debug ---\n")

    if len(df) != E.shape[0] or len(df) != A.shape[0]:
        raise ValueError(
            f"CSV / embeddings / graph size mismatch for model={model}, k={k}: "
            f"csv={len(df)}, embeddings={E.shape[0]}, graph={A.shape[0]}"
        )

    tiers = df[TIER_COL].astype(str).to_numpy()
    start_ids = [i for i, t in enumerate(tiers) if t in LOW_START_TIERS]

    if len(start_ids) == 0:
        raise ValueError(f"No starting nodes found for tiers: {sorted(LOW_START_TIERS)}")

    current_nodes = {sid: sid for sid in start_ids}
    visited = {sid: {sid} for sid in start_ids}
    paths = {sid: [sid] for sid in start_ids}
    stopped = {sid: False for sid in start_ids}

    for sid in start_ids:
        if tiers[sid] == "Unity":
            stopped[sid] = True

    end_tiers_by_step: list[list[str]] = []

    print("\n" + "=" * 72)
    print(f"Running greedy no-revisit walk for model={model}, k={k}")
    print("=" * 72)
    print(f"Found {len(start_ids)} starting nodes in {sorted(LOW_START_TIERS)}")
    print(f"MAX_STEPS={MAX_STEPS}")
    print("Stopping rule: stop on first Unity hit, or when no unvisited neighbor remains.")

    for step in range(1, MAX_STEPS + 1):
        new_nodes = {}

        for sid in start_ids:
            cur = current_nodes[sid]

            if stopped[sid]:
                new_nodes[sid] = cur
                continue

            # Candidate moves: graph neighbors this walker has not visited yet.
            nbrs = A.indices[A.indptr[cur]:A.indptr[cur + 1]]
            nbrs = [int(n) for n in nbrs if int(n) not in visited[sid]]

            if len(nbrs) == 0:
                stopped[sid] = True
                new_nodes[sid] = cur
                continue

            # Greedy step: most cosine-similar unvisited neighbor (score is never used).
            sims = cosine_sim_vector(E, cur, nbrs)
            nxt = int(nbrs[int(np.argmax(sims))])

            visited[sid].add(nxt)
            new_nodes[sid] = nxt
            paths[sid].append(nxt)

            if tiers[nxt] == "Unity":
                stopped[sid] = True

        current_nodes = new_nodes

        end_tiers = [tiers[current_nodes[sid]] for sid in start_ids]
        end_tiers_by_step.append(end_tiers)

        if all(stopped.values()):
            break

    step_dist_df = build_step_distribution(end_tiers_by_step, TIER_ORDER)

    path_summary_rows = []
    for sid, path in paths.items():
        path_tiers = [str(tiers[node]) for node in path]

        hit_unity = "Unity" in path_tiers
        passed_activation = "Activation" in path_tiers
        passed_growth = "Growth" in path_tiers

        path_summary_rows.append({
            "start_id": sid,
            "path_length": len(path),
            "end_tier": path_tiers[-1],
            "hit_unity": hit_unity,
            "passed_activation": passed_activation,
            "passed_growth": passed_growth,
        })

    path_summary_df = pd.DataFrame(path_summary_rows)

    hit_unity = float(path_summary_df["hit_unity"].mean())
    mean_length = float(path_summary_df["path_length"].mean())

    # Passage rates conditioned on paths that reached Unity (NaN if none did).
    unity_paths = path_summary_df[path_summary_df["hit_unity"]]
    if len(unity_paths) > 0:
        p_activation_given_unity = float(unity_paths["passed_activation"].mean())
        p_growth_given_unity = float(unity_paths["passed_growth"].mean())
        p_and_given_unity = float(
            (unity_paths["passed_activation"] & unity_paths["passed_growth"]).mean()
        )
        p_or_given_unity = float(
            (unity_paths["passed_activation"] | unity_paths["passed_growth"]).mean()
        )
    else:
        p_activation_given_unity = np.nan
        p_growth_given_unity = np.nan
        p_and_given_unity = np.nan
        p_or_given_unity = np.nan

    final_nodes = {sid: paths[sid][-1] for sid in start_ids}
    final_tiers = [tiers[final_nodes[sid]] for sid in start_ids]
    final_dist = (
        pd.Series(final_tiers)
        .value_counts(normalize=True)
        .reindex(TIER_ORDER, fill_value=0.0)
        .rename_axis("tier")
        .reset_index(name="proportion")
    )

    print("\n=== Summary statistics ===")
    print(f"Model: {model}")
    print(f"k: {k}")
    print(f"Hit@Unity: {hit_unity:.4f}")
    print(f"P(Passed Activation | Hit Unity): {p_activation_given_unity:.4f}")
    print(f"P(Passed Growth | Hit Unity): {p_growth_given_unity:.4f}")
    print(f"P(Passed Activation & Growth | Hit Unity): {p_and_given_unity:.4f}")
    print(f"P(Passed Activation or Growth | Hit Unity): {p_or_given_unity:.4f}")
    print(f"Mean path length: {mean_length:.4f}")

    print("\n=== Final tier distribution ===")
    print(final_dist.to_string(index=False))

    print("\n=== Example path lengths ===")
    print(path_summary_df["path_length"].describe().round(2).to_string())

    if SAVE_PLOTS:
        plot_path = os.path.join(out_dir, f"tier_dynamics_greedy_{model}_k{k}.png")
        plot_tier_dynamics(step_dist_df, model=model, k=k, save_path=plot_path)
        print(f"\nSaved plot: {plot_path}")

    summary_row = pd.DataFrame([{
        "model": model,
        "k": k,
        "Hit@Unity": hit_unity,
        "P(Passed Activation | Hit Unity)": p_activation_given_unity,
        "P(Passed Growth | Hit Unity)": p_growth_given_unity,
        "P(Passed Activation & Growth | Hit Unity)": p_and_given_unity,
        "P(Passed Activation or Growth | Hit Unity)": p_or_given_unity,
        "Length": mean_length,
        "n_start_paths": len(start_ids),
        "n_steps_executed": len(end_tiers_by_step),
    }])

    summary_csv = os.path.join(out_dir, f"greedy_transition_corridor_{model}_k{k}.csv")
    summary_row.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    return summary_row


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if TIER_COL not in df.columns:
        raise ValueError(f"CSV must contain column '{TIER_COL}'")

    all_rows = []

    for model in MODEL_LIST:
        for k in K_LIST:
            row = run_single_experiment(df, model, k)
            all_rows.append(row)

    combined = pd.concat(all_rows, ignore_index=True)

    combined_out = os.path.join(OUTPUT_ROOT, "greedy_transition_corridor_all_models.csv")
    combined.to_csv(combined_out, index=False)

    print("\n" + "=" * 72)
    print("Combined greedy transition corridor summary")
    print("=" * 72)
    print(combined.to_string(index=False))
    print(f"\nSaved combined summary: {combined_out}")


if __name__ == "__main__":
    main()
