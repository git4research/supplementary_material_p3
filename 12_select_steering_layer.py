"""Select the steering layer per model from script 11 activations (Section 2.5.1, Appendix M).

Per candidate layer: 5-fold CV Ridge R^2 on scores, logistic-regression tier accuracy, kNN (k=30)
tier-jump 0/1 rate and tier silhouette; the chosen layer maximizes the weighted sum of min-max
normalized metrics. Outputs: layer_selection/layer_metrics.csv, chosen_layer.json, layer_metrics.png.
Example: python 12_select_steering_layer.py --model qwen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from model_config import PROJECT_ROOT, add_model_arguments, check_dataset_provenance, file_digest, get_model_spec

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, accuracy_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


CSV_PATH = "full_dataset_with_scores_new.csv"
TIER_COL = "tier"
SCORE_COL = "score"

LAYERS_OVERRIDE: List[int] | None = None

RIDGE_ALPHA = 1.0
LOGREG_C = 1.0
LOGREG_MAX_ITER = 2000
N_FOLDS = 5
SEED = 0

KNN_K = 30
KNN_METRIC = "euclidean"

SILHOUETTE_SAMPLE = None

TIER_ORDER = [
    "Collapse",
    "Striving",
    "Conflict",
    "Activation",
    "Growth",
    "Clarity",
    "Unity",
]

# Composite weights applied to min-max normalized metrics (sum to 1.0).
WEIGHTS = {
    "ridge_r2": 0.40,
    "logreg_acc": 0.25,
    "tts_jump01": 0.25,
    "silhouette": 0.10,
}

SAVE_PLOT = True


def discover_layers(model_dir: Path) -> List[int]:
    layers = []
    for child in sorted(model_dir.iterdir()):
        if child.is_dir() and child.name.startswith("layer_"):
            try:
                L = int(child.name.split("_")[1])
                if (child / "embeddings.npy").exists():
                    layers.append(L)
            except ValueError:
                continue
    return sorted(layers)


def ridge_r2_cv(E: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(E):
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(E[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(E[te])))
    return float(np.mean(scores))


def logreg_acc_cv(E: np.ndarray, labels: np.ndarray, n_folds: int, seed: int) -> float:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(E, labels):
        m = LogisticRegression(
            C=LOGREG_C,
            max_iter=LOGREG_MAX_ITER,
            solver="lbfgs",
            n_jobs=1,
        )
        m.fit(E[tr], labels[tr])
        accs.append(accuracy_score(labels[te], m.predict(E[te])))
    return float(np.mean(accs))


def tts_jump01_rate(E: np.ndarray, tier_idx: np.ndarray, k: int, metric: str) -> float:
    """Fraction of undirected kNN edges whose endpoints' tiers differ by at most one."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(E)
    _, idx = nn.kneighbors(E)
    # Drop the first neighbor (each point itself).
    idx = idx[:, 1:]

    n = E.shape[0]
    edges = set()
    for i in range(n):
        for j in idx[i]:
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges.add((a, b))

    if not edges:
        return 0.0

    jumps = np.array([abs(tier_idx[a] - tier_idx[b]) for a, b in edges])
    return float(np.mean(jumps <= 1))


def composite_rank(metrics_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted sum of per-metric min-max normalized values across layers (constant metric -> 0.5)."""
    score = pd.Series(0.0, index=metrics_df.index)
    for col, w in weights.items():
        v = metrics_df[col].astype(float)
        rng = v.max() - v.min()
        norm = (v - v.min()) / rng if rng > 1e-12 else pd.Series(0.5, index=v.index)
        score = score + w * norm
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, include_layer=False)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / CSV_PATH)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    spec = get_model_spec(args.model_id, args.model_key)
    model_dir = Path(args.output_root).expanduser().resolve() / spec.key
    activation_meta = json.loads((model_dir / "activation_meta.json").read_text())
    if activation_meta["model_id"] != spec.model_id:
        raise ValueError("Activation model does not match the requested model.")
    check_dataset_provenance(activation_meta, args.csv)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    for col in (TIER_COL, SCORE_COL):
        if col not in df.columns:
            raise ValueError(f"CSV missing column '{col}'")

    tiers_str = df[TIER_COL].astype(str).to_numpy()
    scores = df[SCORE_COL].to_numpy(dtype=float)

    tier_to_idx = {t: i for i, t in enumerate(TIER_ORDER)}
    tier_idx = np.array([tier_to_idx[t] for t in tiers_str], dtype=int)
    labels = tier_idx

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if LAYERS_OVERRIDE is not None:
        layers = LAYERS_OVERRIDE
    else:
        layers = activation_meta["layers_saved"]
    layers = [layer for layer in layers if layer in activation_meta["layers_saved"]]
    # Legacy (non-block-output) activations: keep only layers below n_hidden_layers.
    if activation_meta.get("activation_space") != "block_output":
        layers = [layer for layer in layers if layer < activation_meta["n_hidden_layers"]]
    layers = sorted(set(layers))
    missing_layers = sorted(set(layers) - set(discover_layers(model_dir)))
    if missing_layers:
        raise FileNotFoundError(f"Missing activations for candidate layers: {missing_layers}; rerun script 11")
    if not layers:
        raise FileNotFoundError(f"No layer_XX subdirs with embeddings.npy under {model_dir}")
    print(f"\nEvaluating {len(layers)} layers: {layers}")

    rows = []
    for L in layers:
        emb_path = model_dir / f"layer_{L:02d}" / "embeddings.npy"
        E = np.load(emb_path)
        if E.shape[0] != len(df):
            raise ValueError(
                f"Layer {L} has {E.shape[0]} rows, but CSV has {len(df)} rows."
            )

        print(f"\n--- layer {L:02d}  (dim={E.shape[1]}) ---")

        r2 = ridge_r2_cv(E, scores, n_folds=N_FOLDS, seed=SEED)
        acc = logreg_acc_cv(E, labels, n_folds=N_FOLDS, seed=SEED)
        tts = tts_jump01_rate(E, tier_idx, k=KNN_K, metric=KNN_METRIC)
        sil = float(silhouette_score(E, labels, metric=KNN_METRIC, sample_size=SILHOUETTE_SAMPLE, random_state=SEED))

        print(f"  ridge R^2     = {r2:.4f}")
        print(f"  logreg acc    = {acc:.4f}")
        print(f"  TTS jump-0+1  = {tts:.4f}  (k={KNN_K})")
        print(f"  silhouette    = {sil:.4f}")

        rows.append({
            "layer": L,
            "depth_percentage": 100 * L / activation_meta["n_hidden_layers"],
            "ridge_r2": r2,
            "logreg_acc": acc,
            "tts_jump01": tts,
            "silhouette": sil,
        })

    metrics_df = pd.DataFrame(rows).set_index("layer")

    metrics_df["composite"] = composite_rank(metrics_df, WEIGHTS)
    metrics_df = metrics_df.sort_index()
    recommended_layer = int(metrics_df["composite"].idxmax())

    print("\n" + "=" * 72)
    print("Per-layer metrics:")
    print("=" * 72)
    print(metrics_df.round(4).to_string())
    print()
    print(f"Recommended steering layer: {recommended_layer}")
    print(f"  (highest composite score under weights {WEIGHTS})")

    out_dir = model_dir / "layer_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "layer_metrics.csv"
    metrics_df.to_csv(csv_path)
    print(f"\nSaved metrics: {csv_path}")

    config = {
        "model_key": spec.key,
        "model_id": spec.model_id,
        "activation_space": activation_meta.get("activation_space", "legacy_hidden_states"),
        "recommended_layer": recommended_layer,
        "recommended_depth_percentage": 100 * recommended_layer / activation_meta["n_hidden_layers"],
        "n_hidden_layers": activation_meta["n_hidden_layers"],
        "layer_sweep": activation_meta.get("layer_sweep", {"strategy": "legacy_saved_layers"}),
        "dataset_sha256": file_digest(args.csv),
        "activation_meta_sha256": file_digest(model_dir / "activation_meta.json"),
        "composite_weights": WEIGHTS,
        "metrics_at_recommended": metrics_df.loc[recommended_layer].to_dict(),
        "knn_k_for_tts": KNN_K,
        "knn_metric": KNN_METRIC,
        "n_folds": N_FOLDS,
        "ridge_alpha": RIDGE_ALPHA,
        "all_layers_evaluated": layers,
    }
    cfg_path = out_dir / "chosen_layer.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved chosen-layer config: {cfg_path}")

    if SAVE_PLOT and not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharex=True)
        cols = ["ridge_r2", "logreg_acc", "tts_jump01", "silhouette"]
        titles = ["Ridge R²", "Logreg accuracy", "TTS jump-0+1 rate", "Silhouette"]
        for ax, col, title in zip(axes, cols, titles):
            tick_labels = [f"{layer}\n{metrics_df.loc[layer, 'depth_percentage']:.1f}%"
                           for layer in metrics_df.index]
            bars = ax.bar(tick_labels, metrics_df[col], color="#60a5fa")
            for x, L in enumerate(metrics_df.index):
                if L == recommended_layer:
                    bars[x].set_color("#a855f7")
            ax.set_title(title)
            ax.set_xlabel("block / actual depth")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            plt.setp(ax.get_xticklabels(), ha="right")
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"Layer selection across metrics ({spec.key})", y=1.02)
        fig.tight_layout()
        plot_path = out_dir / "layer_metrics.png"
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
