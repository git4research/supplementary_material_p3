from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "full_dataset_with_scores_new.csv")
EMB_DIR = os.path.join(HERE, "all_embeddings")
DEFAULT_OUT = os.path.join(HERE, "outputs", "ablation_probe_llm")

MODELS = {
    "bge":             ("embeddings_bge.npy",              "BGE-large-en-v1.5",  "encoder"),
    "mpnet":           ("embeddings_mpnet.npy",            "all-mpnet-base-v2",  "encoder"),
    "minilm":          ("embeddings_miniLM.npy",           "all-MiniLM-L6-v2",   "encoder"),
    "qwen_embedding":  ("embeddings_qwen_embeddings.npy",  "Qwen3-Embedding-0.6B", "encoder"),
    "qwen2.5_3b_L24":  ("embeddings_qwen2.5_3B_L24.npy",   "Qwen2.5-3B", "llm"),
    "qwen2.5_7b_L20":  ("embeddings_qwen2.5_7b_L20.npy",   "Qwen2.5-7B", "llm"),
    "qwen2.5_14b_L24": ("embeddings_qwen2.5_14B_L24.npy",  "Qwen2.5-14B", "llm"),
    "llama3.1_8b_L16": ("embeddings_llama3.1_8B_L16.npy",  "Llama-3.1-8B", "llm"),
    "gemma2_9b_L29":   ("embeddings_gemma2_9B_L29.npy",    "Gemma-2-9B", "llm"),
    "gemma2_27b_L28":  ("embeddings_gemma2_27B_L28.npy",   "Gemma-2-27B", "llm"),
}

RIDGE_ALPHA = 1.0
N_SPLITS = 30
TEST_SIZE = 0.2
BASE_SEED = 123
RENORMALIZE_AFTER_ABLATION = False

CONDITIONS = ["original", "insample_ablated", "insample_perm", "heldout_ablated", "heldout_perm"]
METRICS = ["ridge_r2", "ridge_mse", "mlp_r2", "mlp_mse"]


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
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def fit_direction(X_train, y_train):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    return unit_norm(model.coef_)


def remove_direction(X, w):
    X_abl = X - np.outer(X @ w, w)
    return l2_normalize_rows(X_abl) if RENORMALIZE_AFTER_ABLATION else X_abl


def eval_ridge(X_train, X_test, y_train, y_test):
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"r2": r2_score(y_test, pred), "mse": mean_squared_error(y_test, pred)}


def eval_mlp(X_train, X_test, y_train, y_test, seed):
    model = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=seed,
                         early_stopping=True, n_iter_no_change=20)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"r2": r2_score(y_test, pred), "mse": mean_squared_error(y_test, pred)}


def summarize(values):
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), float(np.std(values, ddof=1))


def run_for_model(df, key):
    fname, name, group = MODELS[key]
    print(f"\n==================== {name} ({key}) ====================")
    X = np.load(os.path.join(EMB_DIR, fname)).astype(np.float64)
    y = df["score"].values.astype(np.float64)
    if len(X) != len(y):
        raise ValueError(f"Embedding rows ({len(X)}) != dataframe rows ({len(y)}) for {key}")
    print(f"[EMB] {fname}  shape={X.shape}")

    w_all = fit_direction(X, y)
    w_all_perm = fit_direction(X, np.random.default_rng(BASE_SEED).permutation(y))
    X_in, X_in_perm = remove_direction(X, w_all), remove_direction(X, w_all_perm)

    records = []
    for split_id in range(N_SPLITS):
        seed = BASE_SEED + split_id
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=seed, shuffle=True)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        w_energy = fit_direction(X_train, y_train)
        rng = np.random.default_rng(seed)
        w_perm = fit_direction(X_train, rng.permutation(y_train))

        conds = {
            "original": (X_train, X_test),
            "insample_ablated": (X_in[train_idx], X_in[test_idx]),
            "insample_perm": (X_in_perm[train_idx], X_in_perm[test_idx]),
            "heldout_ablated": (remove_direction(X_train, w_energy), remove_direction(X_test, w_energy)),
            "heldout_perm": (remove_direction(X_train, w_perm), remove_direction(X_test, w_perm)),
        }
        for condition, (Xtr, Xte) in conds.items():
            r = eval_ridge(Xtr, Xte, y_train, y_test)
            m = eval_mlp(Xtr, Xte, y_train, y_test, seed)
            records.append({"model": key, "split": split_id, "condition": condition,
                            "ridge_r2": r["r2"], "ridge_mse": r["mse"],
                            "mlp_r2": m["r2"], "mlp_mse": m["mse"]})
        if (split_id + 1) % 10 == 0:
            print(f"  split {split_id + 1}/{N_SPLITS}")

    df_long = pd.DataFrame(records)
    summary = {"model": key, "display_name": name, "group": group, "dim": X.shape[1]}
    for condition in CONDITIONS:
        sub = df_long[df_long["condition"] == condition]
        for metric in METRICS:
            mu, sd = summarize(sub[metric].values)
            summary[f"{condition}_{metric}_mean"] = mu
            summary[f"{condition}_{metric}_std"] = sd
    pivot = df_long.pivot(index="split", columns="condition")
    for ablated in CONDITIONS[1:]:
        prefix = f"delta_{ablated}"
        for metric in METRICS:
            d = pivot[metric]["original"].values - pivot[metric][ablated].values
            mu, sd = summarize(d)
            summary[f"{prefix}_{metric}_mean"] = mu
            summary[f"{prefix}_{metric}_std"] = sd
    return summary, df_long


def write_tex(df_summary: pd.DataFrame, path: str):
    SHORT = {"BGE-large-en-v1.5": "BGE-large", "all-mpnet-base-v2": "MPNet",
             "all-MiniLM-L6-v2": "MiniLM", "Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B"}
    COLS = ["original", "insample_ablated", "heldout_ablated", "heldout_perm"]

    def num(v):
        return f"{v:.3f}".replace("0.", ".", 1) if v >= 0 else "-" + f"{abs(v):.3f}".replace("0.", ".", 1)

    def half(rows):
        T = ["\\begin{tabular}[t]{@{}p{22mm}lcccc@{}}", "\\toprule",
             " & & & In-sample & Held-out & Perm. \\\\",
             "Model & Probe & Orig. & ablated & ablated & ctrl. \\\\", "\\midrule"]
        for k, (_, row) in enumerate(rows.iterrows()):
            name = SHORT.get(row["display_name"], row["display_name"])
            for j, (probe, metric) in enumerate([("Ridge", "ridge_r2"), ("MLP", "mlp_r2")]):
                first = f"\\multirow{{2}}{{*}}{{{name}}}" if j == 0 else ""
                T.append(f"{first} & {probe} & " +
                         " & ".join(num(row[f"{c}_{metric}_mean"]) for c in COLS) + " \\\\")
            if k < len(rows) - 1:
                nxt = rows.iloc[k + 1]["group"]
                T.append("\\midrule\\midrule" if nxt != row["group"] else "\\midrule")
        T += ["\\bottomrule", "\\end{tabular}"]
        return T

    df_summary = df_summary.reset_index(drop=True)
    perm_max = max(abs(df_summary[f"delta_{c}_{m}_mean"]).max()
                   for c in ("insample_perm", "heldout_perm") for m in ("ridge_r2", "mlp_r2"))
    mid = (len(df_summary) + 1) // 2
    left, right = df_summary.iloc[:mid], df_summary.iloc[mid:]
    L = ["% Generated by 2b_ablation_llm.py -- do not edit by hand",
         "% needs \\usepackage{booktabs,multirow}",
         "\\begin{table}[t]", "\\centering",
         f"\\caption{{Score-direction ablation. Mean test $R^2$ over {N_SPLITS} random 80/20 splits of "
         f"Ridge and MLP probes on the original embeddings (Orig.) and after projecting out the "
         f"Ridge-fitted score direction. \\emph{{In-sample}}: direction fit once on all sentences, as for "
         f"\\texttt{{global\\_ridge}} steering. \\emph{{Held-out}}: direction fit on each training split "
         f"only. Perm.\\ ctrl.: held-out ablation of a direction fit to permuted labels; permuted-label "
         f"controls change $R^2$ by at most {perm_max:.3f} under either protocol. Double rules separate "
         f"sentence encoders from the LLMs used for steering; standard deviations are in the "
         f"released results.}}",
         "\\label{tab:ablation_llm}",
         "\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
         "% both halves have the same natural width (fixed model column), so they scale identically",
         "\\resizebox{0.49\\linewidth}{!}{%"] + half(left) + ["}\\hfill\\resizebox{0.49\\linewidth}{!}{%"] + half(right) + ["}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    ap.add_argument("--group", choices=["encoder", "llm"], default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_df(CSV_PATH)

    summaries, longs = [], []
    keys = [k for k in MODELS
            if (args.models is None or k in args.models)
            and (args.group is None or MODELS[k][2] == args.group)]
    for key in keys:
        s, l = run_for_model(df, key)
        summaries.append(s)
        longs.append(l)

    df_summary = pd.DataFrame(summaries)
    df_long = pd.concat(longs, ignore_index=True)
    df_summary.to_csv(os.path.join(args.out_dir, "energy_ablation_probe_llm_summary.csv"), index=False)
    df_long.to_csv(os.path.join(args.out_dir, "energy_ablation_probe_llm_all_splits.csv"), index=False)
    write_tex(df_summary, os.path.join(args.out_dir, "energy_ablation_probe_llm_table.tex"))

    cols = ["display_name"] + [f"{c}_ridge_r2_mean" for c in CONDITIONS]
    print("\n===== SUMMARY (test R^2, mean over splits) =====")
    print(df_summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[Saved] {args.out_dir}")


if __name__ == "__main__":
    main()
