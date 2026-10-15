"""Utility-guided stochastic trajectories on the kNN graph (Section 2.4 / Appendix F).

Starting from every Collapse/Striving/Conflict sentence, each walk moves only to strictly
uphill neighbors along a Ridge score direction, sampling by softmax of a utility that trades
off cosine similarity, score gain, and large jumps. Reports Unity hit rate, Activation/Growth
passage rates, and terminal tiers. Reads outputs/<model>/embeddings.npy and the kNN adjacency;
writes a tier-dynamics plot and sample-path CSV to outputs/<model>/gradient_paths_k<k>_euclidean/.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import Ridge
from typing import Dict, List, Any


CSV_PATH = "full_dataset_with_scores_new.csv"
OUTPUT_ROOT = "outputs"

MODEL_LIST = ["bge", "qwen_embedding", "qwen2.5_3b_L24"]
K_LIST = [15, 30]
GRAPH_METRIC_NAME = "euclidean"

TIER_COL = "tier"
TEXT_COL = "sentence"
SCORE_COL = "score"
INDEX_COL = "index"

LOW_START_TIERS = {"Collapse", "Striving", "Conflict"}

MAX_STEPS = 25
# Softmax temperature; EPSILON is the minimum score gain for a neighbor to count as uphill.
TAU = 0.1
RANDOM_SEED = 1
RIDGE_ALPHA = 1.0
EPSILON = 1e-6
N_SAMPLE_PATHS_TO_SAVE = 265

# Utility weights: lambda_sim (cosine), lambda_delta (score gain), lambda_jump (penalty on gain^2).
SIM_WEIGHT = 0.9
DELTA_WEIGHT = 0.8
JUMP_WEIGHT = 1

SHOW_PLOTS = False
SAVE_PLOTS = True

TIER_COLORS = {
    "Collapse": "#ff3b3b",
    "Striving": "#ff8b3d",
    "Conflict": "#ffd93b",
    "Activation": "#34d399",
    "Growth": "#60a5fa",
    "Clarity": "#6366f1",
    "Unity": "#a855f7",
}


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def fit_ridge_energy_direction(
    E: np.ndarray,
    scores: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit Ridge(scores ~ E) on all sentences; return weight vector w, intercept, and in-sample predictions."""
    model = Ridge(alpha=alpha)
    model.fit(E, scores)

    w = model.coef_.astype(float)
    b = float(model.intercept_)
    pred = model.predict(E).astype(float)

    return w, b, pred


def sample_next_node_gradient_guided(
    cur: int,
    E: np.ndarray,
    A: sparse.csr_matrix,
    w: np.ndarray,
    tau: float,
    eps: float,
    sim_weight: float = SIM_WEIGHT,
    delta_weight: float = DELTA_WEIGHT,
    jump_weight: float = JUMP_WEIGHT,
) -> int:
    """Sample one step among strictly uphill neighbors; returns `cur` itself if none exist (walk stops)."""
    nbrs = A.indices[A.indptr[cur]:A.indptr[cur + 1]]
    if nbrs.size == 0:
        return cur

    cur_vec = E[cur]
    nbr_vecs = E[nbrs]

    # Delta_ij = w . (x_j - x_i): projected score gain of each neighbor.
    deltas = (nbr_vecs - cur_vec) @ w

    valid_mask = deltas > eps
    if not np.any(valid_mask):
        return cur

    valid_nbrs = nbrs[valid_mask]
    valid_vecs = nbr_vecs[valid_mask]
    valid_deltas = deltas[valid_mask]

    # Rows of E are L2-normalized, so the dot product is cosine similarity.
    sims = valid_vecs @ cur_vec

    # U = lambda_sim*cos + lambda_delta*Delta - lambda_jump*Delta^2
    utility = (
        sim_weight * sims
        + delta_weight * valid_deltas
        - jump_weight * (valid_deltas ** 2)
    )

    # Numerically stable softmax(U / tau).
    logits = utility / tau
    logits = logits - np.max(logits)

    probs = np.exp(logits)
    probs = probs / probs.sum()

    nxt = int(np.random.choice(valid_nbrs, p=probs))
    return nxt


def summarize_paths(paths: Dict[int, List[int]], df: pd.DataFrame) -> pd.DataFrame:
    """One row per path: start/end tier and score, length, and whether Unity/Activation/Growth were visited."""
    rows = []
    has_index_col = INDEX_COL in df.columns
    has_score_col = SCORE_COL in df.columns

    for start_id, path in paths.items():
        start_row = df.iloc[start_id]
        end_id = path[-1]
        end_row = df.iloc[end_id]

        path_tiers = [str(df.iloc[node_id][TIER_COL]) for node_id in path]

        rows.append({
            "start_row_id": int(start_id),
            "start_index": start_row[INDEX_COL] if has_index_col else int(start_id),
            "start_tier": str(start_row[TIER_COL]),
            "start_score": float(start_row[SCORE_COL]) if has_score_col else np.nan,
            "path_length": int(len(path)),
            "end_row_id": int(end_id),
            "end_index": end_row[INDEX_COL] if has_index_col else int(end_id),
            "end_tier": str(end_row[TIER_COL]),
            "end_score": float(end_row[SCORE_COL]) if has_score_col else np.nan,
            "hit_unity": bool("Unity" in path_tiers),
            "passed_activation": bool("Activation" in path_tiers),
            "passed_growth": bool("Growth" in path_tiers),
        })

    return pd.DataFrame(rows)


def sample_paths_to_df(
    paths: Dict[int, List[int]],
    df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """Long-format table (one row per step) for a random subset of paths, with sentences, for inspection."""
    rng = np.random.default_rng(seed)

    start_ids = list(paths.keys())
    if len(start_ids) <= n_samples:
        sampled = start_ids
    else:
        sampled = list(rng.choice(start_ids, n_samples, replace=False))

    has_index_col = INDEX_COL in df.columns
    has_score_col = SCORE_COL in df.columns
    has_text_col = TEXT_COL in df.columns

    rows: List[Dict[str, Any]] = []

    for sample_id, sid in enumerate(sampled, 1):
        path = paths[sid]

        for step, node in enumerate(path):
            row = df.iloc[node]
            rows.append({
                "sample_path_id": sample_id,
                "start_row_id": int(sid),
                "step": int(step),
                "current_row_id": int(node),
                "current_index": row[INDEX_COL] if has_index_col else int(node),
                "tier": str(row[TIER_COL]),
                "score": float(row[SCORE_COL]) if has_score_col else np.nan,
                "sentence": str(row[TEXT_COL]) if has_text_col else "",
            })

        # Two blank rows separate consecutive paths in the CSV.
        rows.append({})
        rows.append({})

    return pd.DataFrame(rows)

def compute_step_dist_df(paths: Dict[int, List[int]], df: pd.DataFrame) -> pd.DataFrame:
    """Tier proportions at each step, over the paths still long enough to have that step."""
    rows = []
    for start_id, path in paths.items():
        for step, node_id in enumerate(path):
            row = df.iloc[node_id]
            rows.append({
                "start_row_id": int(start_id),
                "step": int(step),
                "tier": str(row[TIER_COL]),
            })

    path_steps_df = pd.DataFrame(rows)

    counts = (
        path_steps_df.groupby(["step", "tier"])
        .size()
        .reset_index(name="count")
    )

    total_per_step = counts.groupby("step")["count"].transform("sum")
    counts["proportion"] = counts["count"] / total_per_step

    return counts[["step", "tier", "proportion"]].copy()


def plot_tier_dynamics(
    step_dist_df: pd.DataFrame,
    model: str,
    k: int,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(6.5, 4))

    for tier, color in TIER_COLORS.items():
        sub = step_dist_df[step_dist_df["tier"] == tier]
        if len(sub) > 0:
            plt.plot(sub["step"], sub["proportion"], label=tier, color=color)

    plt.xlabel("Step")
    plt.ylabel("Tier proportion")
    plt.title(f"Tier Dynamics under Gradient-Guided Walk ({model}, k={k})")
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def run_single_experiment(df: pd.DataFrame, model: str, k: int) -> None:
    np.random.seed(RANDOM_SEED)

    embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
    graph_path = os.path.join(
        OUTPUT_ROOT, model, f"knn_k{k}_{GRAPH_METRIC_NAME}", "adjacency_csr.npz"
    )

    if not os.path.exists(embed_path):
        raise FileNotFoundError(f"Embedding file not found: {embed_path}")
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    out_dir = os.path.join(OUTPUT_ROOT, model, f"gradient_paths_k{k}_{GRAPH_METRIC_NAME}")
    os.makedirs(out_dir, exist_ok=True)

    E = np.load(embed_path)
    A = sparse.load_npz(graph_path).tocsr()

    if len(df) != E.shape[0] or len(df) != A.shape[0]:
        raise ValueError(
            f"CSV / embeddings / graph size mismatch for model={model}, k={k}: "
            f"csv={len(df)}, embeddings={E.shape[0]}, graph={A.shape[0]}"
        )

    E = l2_normalize_rows(E)

    tiers = df[TIER_COL].astype(str).to_numpy()
    scores = df[SCORE_COL].to_numpy(dtype=float)

    start_ids = [i for i, t in enumerate(tiers) if t in LOW_START_TIERS]
    if len(start_ids) == 0:
        raise ValueError(f"No starting points found for tiers: {sorted(LOW_START_TIERS)}")

    print("\n" + "=" * 72)
    print(f"Running gradient-guided paths for model={model}, k={k}")
    print("=" * 72)
    print(f"Found {len(start_ids)} starting points in tiers {sorted(LOW_START_TIERS)}")
    print(
        f"RIDGE_ALPHA={RIDGE_ALPHA}, TAU={TAU}, MAX_STEPS={MAX_STEPS}, "
        f"RANDOM_SEED={RANDOM_SEED}, EPSILON={EPSILON}, "
        f"SIM_WEIGHT={SIM_WEIGHT}, DELTA_WEIGHT={DELTA_WEIGHT}, JUMP_WEIGHT={JUMP_WEIGHT}"
    )

    w, b, pred = fit_ridge_energy_direction(E, scores, alpha=RIDGE_ALPHA)
    corr = float(np.corrcoef(pred, scores)[0, 1])
    print(f"Ridge score-vs-pred correlation: {corr:.4f}")

    # Advance all walks in lockstep; a walk stops permanently once it has no uphill neighbor.
    paths: Dict[int, List[int]] = {sid: [sid] for sid in start_ids}
    current_nodes: Dict[int, int] = {sid: sid for sid in start_ids}
    stopped: Dict[int, bool] = {sid: False for sid in start_ids}

    for _step in range(1, MAX_STEPS + 1):
        new_current_nodes: Dict[int, int] = {}

        for sid in start_ids:
            cur = current_nodes[sid]

            if stopped[sid]:
                new_current_nodes[sid] = cur
                continue

            nxt = sample_next_node_gradient_guided(
                cur=cur,
                E=E,
                A=A,
                w=w,
                tau=TAU,
                eps=EPSILON,
                sim_weight=SIM_WEIGHT,
                delta_weight=DELTA_WEIGHT,
                jump_weight=JUMP_WEIGHT,
            )

            if nxt == cur:
                stopped[sid] = True
                new_current_nodes[sid] = cur
                continue

            new_current_nodes[sid] = nxt
            paths[sid].append(nxt)

        current_nodes = new_current_nodes

    path_summary_df = summarize_paths(paths, df)
    step_dist_df = compute_step_dist_df(paths, df)
    sample_paths_df = sample_paths_to_df(
        paths=paths,
        df=df,
        n_samples=N_SAMPLE_PATHS_TO_SAVE,
        seed=RANDOM_SEED,
    )

    unity_hit_rate = float(path_summary_df["hit_unity"].mean())
    activation_passage_rate = float(path_summary_df["passed_activation"].mean())
    growth_passage_rate = float(path_summary_df["passed_growth"].mean())
    mean_path_length = float(path_summary_df["path_length"].mean())

    end_tier_distribution = (
        path_summary_df["end_tier"]
        .value_counts(normalize=True)
        .rename_axis("end_tier")
        .reset_index(name="proportion")
    )

    print("\n=== Summary statistics ===")
    print(f"Model: {model}")
    print(f"k: {k}")
    print(f"Unity hit rate: {unity_hit_rate:.4f}")
    print(f"Activation passage rate: {activation_passage_rate:.4f}")
    print(f"Growth passage rate: {growth_passage_rate:.4f}")
    print(f"Mean path length: {mean_path_length:.4f}")

    print("\n=== End-tier distribution ===")
    print(end_tier_distribution.to_string(index=False))

    if SAVE_PLOTS:
        plot_path = os.path.join(out_dir, f"tier_dynamics_{model}_k{k}.png")
        plot_tier_dynamics(
            step_dist_df=step_dist_df,
            model=model,
            k=k,
            save_path=plot_path,
        )
        print(f"\nSaved plot: {plot_path}")

    sample_csv_path = os.path.join(out_dir, f"sample_paths_{model}_k{k}.csv")
    sample_paths_df.to_csv(sample_csv_path, index=False)
    print(f"Saved sample paths CSV: {sample_csv_path}")


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    required = [TIER_COL, SCORE_COL]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV missing column '{c}'")

    for model in MODEL_LIST:
        for k in K_LIST:
            run_single_experiment(df, model, k)


if __name__ == "__main__":
    main()
