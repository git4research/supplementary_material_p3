"""Label-permutation tests for score R^2 and tier F1 (Appendix E.3).

For each representation, the observed metrics are means over N_SPLITS (20) random 80/20 splits
(Ridge for score R^2, logistic regression for weighted tier F1). Each null sample shuffles the
score or tier labels (N_PERM_SCORE / N_PERM_TIER = 200 permutations, shared across models) and
re-runs all splits. Reads the dataset CSV and all_embeddings/embeddings_*.npy next to this script;
writes per-model CSVs, null histograms and all_models_summary.csv to outputs/permutation_test/.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "full_dataset_with_scores_new.csv"
EMBEDDINGS_DIR = BASE_DIR / "all_embeddings"

REPRESENTATION_FILES: dict[str, list[Union[str, Path]]] = {
    "BAAI_bge_large_en_v1_5": [EMBEDDINGS_DIR / "embeddings_bge.npy"],
    "all-mpnet-base-v2": [EMBEDDINGS_DIR / "embeddings_mpnet.npy"],
    "all-MiniLM-L6-v2": [EMBEDDINGS_DIR / "embeddings_miniLM.npy"],
    "Qwen3_Embedding_0_6B": [EMBEDDINGS_DIR / "embeddings_qwen_embeddings.npy"],
    "Qwen2_5_3B_Instruct_L24": [EMBEDDINGS_DIR / "embeddings_qwen2.5_3B_L24.npy"],
}

OUT_DIR = BASE_DIR / "outputs" / "permutation_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 20
SPLIT_SEEDS = list(range(N_SPLITS))
TEST_SIZE = 0.2

N_PERM_SCORE = 200
N_PERM_TIER = 200

RIDGE_ALPHA = 1.0
LOGREG_MAX_ITER = 3000

PERM_RNG_SEED_BASE = 12345

TIER_ORDER = [
    "Collapse",
    "Striving",
    "Conflict",
    "Activation",
    "Growth",
    "Clarity",
    "Unity",
]

# Legacy tier names mapped onto the current label set.
TIER_ALIASES = {
    "Shadow": "Collapse",
    "Unconscious / Collapse": "Collapse",
    "Highly Contracted / Self-Destructive": "Collapse",
}


def stable_int_seed(s: str, mod: int = 1_000_000) -> int:
    # Deterministic integer seed derived from a string (MD5-based, stable across runs).
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def sanitize_model_name(model_name: str) -> str:
    return (
        model_name.replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
        .replace("-", "_")
    )


def resolve_existing_path(candidates: Iterable[Union[str, Path]]) -> Path:
    checked: list[str] = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(str(path))
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find embedding/activation file. Checked:\n  "
        + "\n  ".join(checked)
    )


def normalize_tier_name(tier: object) -> str:
    t = str(tier).strip()
    return TIER_ALIASES.get(t, t)


def encode_tiers(tiers: pd.Series) -> np.ndarray:
    """Map tier names to integer ids 0..6 in TIER_ORDER; raises on unknown names."""
    tier_to_idx = {tier: idx for idx, tier in enumerate(TIER_ORDER)}
    encoded = []
    unknown = []
    for raw in tiers:
        t = normalize_tier_name(raw)
        if t not in tier_to_idx:
            unknown.append(t)
        else:
            encoded.append(tier_to_idx[t])
    if unknown:
        raise ValueError(
            "Found tier names not in TIER_ORDER: "
            + ", ".join(sorted(set(unknown)))
            + f"\nExpected one of: {TIER_ORDER}"
        )
    return np.asarray(encoded, dtype=int)


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"sentence", "score", "tier"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["sentence", "score", "tier"]).reset_index(drop=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    df["tier"] = df["tier"].apply(normalize_tier_name)
    return df


def load_representation(path: Path, n_expected: int, model_name: str) -> np.ndarray:
    X = np.load(path)
    if X.shape[0] != n_expected:
        raise ValueError(
            f"Row mismatch for {model_name}: representation rows={X.shape[0]} "
            f"vs dataset rows={n_expected}. File: {path}"
        )
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array for {model_name}, got shape {X.shape}. File: {path}")
    return X


def eval_once_regression(X: np.ndarray, y_score: np.ndarray, seed: int) -> dict[str, float]:
    X_train, X_test, ys_train, ys_test = train_test_split(
        X, y_score, test_size=TEST_SIZE, random_state=seed
    )
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, ys_train)
    ys_pred = model.predict(X_test)
    return {
        "ridge_r2": float(r2_score(ys_test, ys_pred)),
        "ridge_mse": float(mean_squared_error(ys_test, ys_pred)),
    }


def eval_once_classification(X: np.ndarray, y_tier: np.ndarray, seed: int) -> dict[str, float]:
    # Stratified split when possible; falls back to an unstratified split if stratification fails.
    try:
        X_train, X_test, yt_train, yt_test = train_test_split(
            X, y_tier, test_size=TEST_SIZE, random_state=seed, stratify=y_tier
        )
    except ValueError:
        X_train, X_test, yt_train, yt_test = train_test_split(
            X, y_tier, test_size=TEST_SIZE, random_state=seed, stratify=None
        )

    model = LogisticRegression(max_iter=LOGREG_MAX_ITER)
    model.fit(X_train, yt_train)
    yt_pred = model.predict(X_test)
    return {
        "tier_acc": float(accuracy_score(yt_test, yt_pred)),
        "tier_f1": float(f1_score(yt_test, yt_pred, average="weighted")),
    }


def eval_repeated_splits(X: np.ndarray, y_score: np.ndarray, y_tier: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run regression and classification on every split seed; return per-split metrics and their mean/std."""
    rows = []
    for seed in SPLIT_SEEDS:
        m_reg = eval_once_regression(X, y_score, seed)
        m_clf = eval_once_classification(X, y_tier, seed)
        rows.append({"seed": seed, **m_reg, **m_clf})
    dfm = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = dfm[["ridge_r2", "ridge_mse", "tier_acc", "tier_f1"]].agg(["mean", "std"])
    return dfm, summary


def perm_p_value(null_vals: np.ndarray, observed: float) -> float:
    # One-sided permutation p-value with +1 correction: (#null >= observed + 1) / (N + 1).
    return (float(np.sum(null_vals >= observed)) + 1.0) / (len(null_vals) + 1.0)


def plot_hist(null_vals: np.ndarray, observed: float, title: str, xlabel: str, out_png: Path) -> None:
    plt.figure(figsize=(10, 7))
    plt.hist(null_vals, bins=35, alpha=0.9)
    plt.axvline(observed, linewidth=3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def score_permutation_test_with_indices(
    X: np.ndarray,
    y_score: np.ndarray,
    y_tier: np.ndarray,
    observed_mean_r2: float,
    perm_indices_score: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Null distribution of mean split R^2 with shuffled scores (tier labels unchanged), plus p-value."""
    null_r2 = []
    for k, idx in enumerate(perm_indices_score, start=1):
        ys = y_score[idx]
        dfm, _ = eval_repeated_splits(X, ys, y_tier)
        null_r2.append(float(dfm["ridge_r2"].mean()))
        if k % 25 == 0 or k == len(perm_indices_score):
            print(f"    score permutation {k}/{len(perm_indices_score)}")
    null_r2 = np.asarray(null_r2, dtype=float)
    p = perm_p_value(null_r2, observed_mean_r2)
    return null_r2, p


def tier_permutation_test_with_indices(
    X: np.ndarray,
    y_score: np.ndarray,
    y_tier: np.ndarray,
    observed_mean_f1: float,
    perm_indices_tier: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    """Null distribution of mean split weighted F1 with shuffled tiers (scores unchanged), plus p-value."""
    null_f1 = []
    for k, idx in enumerate(perm_indices_tier, start=1):
        yt = y_tier[idx]
        dfm, _ = eval_repeated_splits(X, y_score, yt)
        null_f1.append(float(dfm["tier_f1"].mean()))
        if k % 25 == 0 or k == len(perm_indices_tier):
            print(f"    tier permutation {k}/{len(perm_indices_tier)}")
    null_f1 = np.asarray(null_f1, dtype=float)
    p = perm_p_value(null_f1, observed_mean_f1)
    return null_f1, p


def run_for_model(
    model_name: str,
    X: np.ndarray,
    representation_path: Path,
    y_score: np.ndarray,
    y_tier: np.ndarray,
    perm_indices_score: list[np.ndarray],
    perm_indices_tier: list[np.ndarray],
) -> dict[str, object]:
    safe_name = sanitize_model_name(model_name)
    model_out = OUT_DIR / safe_name
    model_out.mkdir(parents=True, exist_ok=True)

    print("\n==============================")
    print(f"[MODEL] {model_name}")
    print(f"[FILE]  {representation_path}")
    print(f"[SHAPE] {X.shape}")
    print("==============================")

    per_split, summary = eval_repeated_splits(X, y_score, y_tier)
    obs_r2_mean = float(per_split["ridge_r2"].mean())
    obs_mse_mean = float(per_split["ridge_mse"].mean())
    obs_acc_mean = float(per_split["tier_acc"].mean())
    obs_f1_mean = float(per_split["tier_f1"].mean())

    per_split.to_csv(model_out / "observed_per_split.csv", index=False)
    summary.to_csv(model_out / "observed_summary_mean_std.csv")

    observed_row = pd.DataFrame([
        {
            "model": model_name,
            "representation_file": str(representation_path),
            "n_samples": int(X.shape[0]),
            "dimension": int(X.shape[1]),
            "n_splits": int(N_SPLITS),
            "test_size": float(TEST_SIZE),
            "ridge_r2_mean": obs_r2_mean,
            "ridge_mse_mean": obs_mse_mean,
            "tier_acc_mean": obs_acc_mean,
            "tier_f1_mean": obs_f1_mean,
        }
    ])
    observed_row.to_csv(model_out / "observed_metrics.csv", index=False)
    print(observed_row.to_string(index=False))

    print(f"[Score permutation test] N_PERM_SCORE={N_PERM_SCORE}")
    null_r2, p_r2 = score_permutation_test_with_indices(
        X, y_score, y_tier, obs_r2_mean, perm_indices_score
    )
    pd.DataFrame({"null_mean_r2": null_r2}).to_csv(model_out / "score_perm_null_r2.csv", index=False)
    pd.DataFrame([
        {
            "model": model_name,
            "perm_N": int(N_PERM_SCORE),
            "obs_r2_mean": obs_r2_mean,
            "perm_p_r2": float(p_r2),
            "perm_seed_base": int(PERM_RNG_SEED_BASE),
            "perm_indices_shared": True,
        }
    ]).to_csv(model_out / "score_perm_pvalues.csv", index=False)

    nested_score_png = model_out / "score_perm_r2_hist.png"
    top_score_png = OUT_DIR / f"{safe_name}_score_perm_r2_hist.png"
    plot_hist(
        null_r2,
        obs_r2_mean,
        title=f"{model_name} — Score Permutation Test (Mean Ridge R² across splits)",
        xlabel="Mean R² under shuffled score labels",
        out_png=nested_score_png,
    )
    # Also copy the histogram to the top-level output folder for easy access.
    shutil.copyfile(nested_score_png, top_score_png)

    print(f"[Tier permutation test] N_PERM_TIER={N_PERM_TIER}")
    null_f1, p_f1 = tier_permutation_test_with_indices(
        X, y_score, y_tier, obs_f1_mean, perm_indices_tier
    )
    pd.DataFrame({"null_mean_f1": null_f1}).to_csv(model_out / "tier_perm_null_f1.csv", index=False)
    pd.DataFrame([
        {
            "model": model_name,
            "perm_N": int(N_PERM_TIER),
            "obs_f1_mean": obs_f1_mean,
            "perm_p_f1": float(p_f1),
            "perm_seed_base": int(PERM_RNG_SEED_BASE),
            "perm_indices_shared": True,
        }
    ]).to_csv(model_out / "tier_perm_pvalues.csv", index=False)

    nested_tier_png = model_out / "tier_perm_f1_hist.png"
    top_tier_png = OUT_DIR / f"{safe_name}_tier_perm_f1_hist.png"
    plot_hist(
        null_f1,
        obs_f1_mean,
        title=f"{model_name} — Tier Permutation Test (Mean weighted F1 across splits)",
        xlabel="Mean weighted F1 under shuffled tier labels",
        out_png=nested_tier_png,
    )
    shutil.copyfile(nested_tier_png, top_tier_png)

    print("[Permutation p-values]")
    print(f"  score_perm_p_r2 = {p_r2:.6f}")
    print(f"  tier_perm_p_f1  = {p_f1:.6f}")
    print("[Saved paper-friendly PNGs]")
    print(f"  {top_score_png}")
    print(f"  {top_tier_png}")

    return {
        "model": model_name,
        "safe_model_name": safe_name,
        "representation_file": str(representation_path),
        "n_samples": int(X.shape[0]),
        "dimension": int(X.shape[1]),
        "ridge_r2_mean": obs_r2_mean,
        "ridge_mse_mean": obs_mse_mean,
        "tier_acc_mean": obs_acc_mean,
        "tier_f1_mean": obs_f1_mean,
        "perm_p_r2": float(p_r2),
        "perm_p_f1": float(p_f1),
        "score_perm_png": str(top_score_png),
        "tier_perm_png": str(top_tier_png),
        "out_dir": str(model_out),
    }


def main() -> None:
    df = load_dataframe(CSV_PATH)
    print(f"Loaded dataframe: shape = {df.shape} from {CSV_PATH}")

    y_score = df["score"].to_numpy(dtype=float)
    y_tier = encode_tiers(df["tier"])
    n = len(y_score)

    mapping_path = OUT_DIR / "tier_label_mapping.csv"
    pd.DataFrame({"label_id": np.arange(len(TIER_ORDER)), "tier_name": TIER_ORDER}).to_csv(
        mapping_path, index=False
    )
    print(f"[Saved tier label mapping] {mapping_path}")

    # Draw permutations once so every model is tested against the same shuffled labels.
    rng_score = np.random.default_rng(PERM_RNG_SEED_BASE + 1)
    rng_tier = np.random.default_rng(PERM_RNG_SEED_BASE + 2)

    perm_indices_score = [rng_score.permutation(n) for _ in range(N_PERM_SCORE)]
    perm_indices_tier = [rng_tier.permutation(n) for _ in range(N_PERM_TIER)]

    meta_path = OUT_DIR / "permutation_meta.csv"
    pd.DataFrame([
        {
            "perm_seed_base": PERM_RNG_SEED_BASE,
            "perm_seed_score": PERM_RNG_SEED_BASE + 1,
            "perm_seed_tier": PERM_RNG_SEED_BASE + 2,
            "N_PERM_SCORE": N_PERM_SCORE,
            "N_PERM_TIER": N_PERM_TIER,
            "shared_across_models": True,
            "n_samples": n,
            "n_splits": N_SPLITS,
            "test_size": TEST_SIZE,
            "csv_path": str(CSV_PATH),
        }
    ]).to_csv(meta_path, index=False)
    print(f"[Saved permutation meta] {meta_path}")

    all_rows = []
    for model_name, path_candidates in REPRESENTATION_FILES.items():
        representation_path = resolve_existing_path(path_candidates)
        X = load_representation(representation_path, n_expected=n, model_name=model_name)
        print(f"Loaded {model_name}: {X.shape} from {representation_path}")

        row = run_for_model(
            model_name=model_name,
            X=X,
            representation_path=representation_path,
            y_score=y_score,
            y_tier=y_tier,
            perm_indices_score=perm_indices_score,
            perm_indices_tier=perm_indices_tier,
        )
        all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    summary_path = OUT_DIR / "all_models_summary.csv"
    df_all.to_csv(summary_path, index=False)

    print(f"\n[Saved outputs] {OUT_DIR}/")
    print(f" - {mapping_path.name}")
    print(f" - {meta_path.name}")
    print(f" - {summary_path.name}")
    print(" - <model_name>/ observed_*.csv, score_perm_*, tier_perm_* and nested hist PNGs")
    print(" - top-level paper-friendly hist PNGs named <model_name>_score_perm_r2_hist.png and <model_name>_tier_perm_f1_hist.png")
    print("\n===== Summary across models =====")
    print(df_all.to_string(index=False))


if __name__ == "__main__":
    main()
