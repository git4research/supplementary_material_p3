"""Held-out score-direction ablation probe for the encoders and Qwen2.5-3B L24 (Sections 2.1, 3.1 / Appendix E).

Over 30 random 80/20 splits, fits a unit Ridge score direction on the training split, projects it out
of train and test embeddings, and compares Ridge/MLP test R^2 and MSE against the original embeddings
and a permuted-label direction control. Reads outputs/<model>/embeddings.npy and the dataset CSV;
writes outputs/ablation_probe/energy_ablation_probe_{summary,all_splits}.csv.
"""
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


CSV_PATH = "full_dataset_with_scores_new.csv"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding", "qwen2.5_3b_L24"]

OUT_ROOT = "outputs"
REPORT_DIR = "outputs/ablation_probe"
os.makedirs(REPORT_DIR, exist_ok=True)

RIDGE_ALPHA = 1.0
N_SPLITS = 30
TEST_SIZE = 0.2
BASE_SEED = 123

# If True, rows are re-normalized to unit length after the direction is removed.
RENORMALIZE_AFTER_ABLATION = False


def get_embedding_path(model_name: str) -> str:
    return os.path.join("outputs", model_name, "embeddings.npy")


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["sentence", "score", "tier"]).reset_index(drop=True)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    print(f"[LOAD] df shape: {df.shape}")
    return df


def unit_norm(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero-norm direction.")
    return v / n


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def fit_direction(X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Unit-norm Ridge coefficient vector, i.e. the linear score direction."""
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    return unit_norm(model.coef_)


def remove_direction(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Project each row of X onto the orthogonal complement of the unit vector w."""
    proj = X @ w
    X_abl = X - np.outer(proj, w)

    if RENORMALIZE_AFTER_ABLATION:
        X_abl = l2_normalize_rows(X_abl)

    return X_abl


def eval_ridge(X_train, X_test, y_train, y_test):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "r2": r2_score(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
    }


def eval_mlp(X_train, X_test, y_train, y_test, seed):
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=1000,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=20,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return {
        "r2": r2_score(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_for_model(df: pd.DataFrame, model_name: str) -> dict:
    """Run all splits and conditions for one model; return (summary dict, per-split DataFrame)."""
    print(f"\n==================== {model_name} ====================")

    X = np.load(get_embedding_path(model_name)).astype(np.float64)
    y = df["score"].values.astype(np.float64)

    if len(X) != len(y):
        raise ValueError(
            f"Embedding rows ({len(X)}) != dataframe rows ({len(y)}) for {model_name}"
        )

    records = []

    for split_id in range(N_SPLITS):
        seed = BASE_SEED + split_id

        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=TEST_SIZE,
            random_state=seed,
            shuffle=True,
        )

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Score ("energy") direction fit on the training split only (held-out protocol).
        w_energy = fit_direction(X_train, y_train)

        # Control: direction fit to permuted training scores, ablated the same way.
        rng = np.random.default_rng(seed)
        y_train_perm = rng.permutation(y_train)
        w_perm = fit_direction(X_train, y_train_perm)

        conditions = {
            "original": (X_train, X_test),
            "energy_ablated": (
                remove_direction(X_train, w_energy),
                remove_direction(X_test, w_energy),
            ),
            "perm_ablated": (
                remove_direction(X_train, w_perm),
                remove_direction(X_test, w_perm),
            ),
        }

        # Probes are retrained from scratch on each (possibly ablated) representation.
        for condition, (Xtr_cur, Xte_cur) in conditions.items():
            ridge_metrics = eval_ridge(Xtr_cur, Xte_cur, y_train, y_test)
            mlp_metrics = eval_mlp(Xtr_cur, Xte_cur, y_train, y_test, seed)

            records.append({
                "model": model_name,
                "split": split_id,
                "condition": condition,
                "ridge_r2": ridge_metrics["r2"],
                "ridge_mse": ridge_metrics["mse"],
                "mlp_r2": mlp_metrics["r2"],
                "mlp_mse": mlp_metrics["mse"],
            })

    df_long = pd.DataFrame(records)

    summary = {"model": model_name}

    for condition in ["original", "energy_ablated", "perm_ablated"]:
        sub = df_long[df_long["condition"] == condition]

        for metric in ["ridge_r2", "ridge_mse", "mlp_r2", "mlp_mse"]:
            mean, std = summarize(sub[metric].values)
            summary[f"{condition}_{metric}_mean"] = mean
            summary[f"{condition}_{metric}_std"] = std

    # Paired per-split deltas, original minus ablated (a positive R^2 delta means a drop).
    pivot = df_long.pivot(index="split", columns="condition")

    for ablated in ["energy_ablated", "perm_ablated"]:
        for metric in ["ridge_r2", "ridge_mse", "mlp_r2", "mlp_mse"]:
            delta_values = (
                pivot[metric]["original"].values
                - pivot[metric][ablated].values
            )
            mean, std = summarize(delta_values)
            prefix = "delta_energy" if ablated == "energy_ablated" else "delta_perm"
            summary[f"{prefix}_{metric}_mean"] = mean
            summary[f"{prefix}_{metric}_std"] = std

    return summary, df_long


def main():
    df = load_df(CSV_PATH)

    all_summaries = []
    all_long = []

    for model_name in MODEL_LIST:
        summary, df_long = run_for_model(df, model_name)
        all_summaries.append(summary)
        all_long.append(df_long)

    df_summary = pd.DataFrame(all_summaries)
    df_all_long = pd.concat(all_long, ignore_index=True)

    summary_path = os.path.join(REPORT_DIR, "energy_ablation_probe_summary.csv")
    long_path = os.path.join(REPORT_DIR, "energy_ablation_probe_all_splits.csv")

    df_summary.to_csv(summary_path, index=False)
    df_all_long.to_csv(long_path, index=False)

    print("\n===== FINAL SUMMARY =====")
    print(df_summary.to_string(index=False))

    print(f"\n[Saved summary] {summary_path}")
    print(f"[Saved all splits] {long_path}")


if __name__ == "__main__":
    main()
