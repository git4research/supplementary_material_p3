from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

CSV_PATH = "full_dataset_with_scores_new.csv"
OUTPUT_ROOT = "outputs"
TEXT_COL = "sentence"
SCORE_COL = "score"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding"]

RIDGE_ALPHA = 1.0
N_HOLDOUT_SPLITS = 5
TEST_SIZE = 0.2
SEED = 0
N_RANDOM_TRIALS = 20

FIRST_PERSON_WORDS = {"i", "my", "me", "myself", "mine", "we", "us", "our", "ours", "ourselves"}
FIRST_PERSON_SUBJECT_STARTS = {"i", "i'm", "i've", "i'll", "i'd", "my", "we", "we're", "we've", "our"}


def _tokenize(s: str):
    return re.findall(r"[a-zA-Z']+", s.lower())


def compute_sent_length(sentences):
    return np.array([len(_tokenize(s)) for s in sentences], dtype=float)


def compute_fp_density(sentences):
    out = []
    for s in sentences:
        toks = _tokenize(s)
        n = max(len(toks), 1)
        out.append(sum(t in FIRST_PERSON_WORDS for t in toks) / n)
    return np.array(out, dtype=float)


def compute_fp_subject(sentences):
    out = []
    for s in sentences:
        toks = _tokenize(s)
        first = toks[0] if toks else ""
        out.append(1.0 if first in FIRST_PERSON_SUBJECT_STARTS else 0.0)
    return np.array(out, dtype=float)


def get_style_confound_scores(sentences):
    confounds = {
        "length": compute_sent_length(sentences),
        "fp_density": compute_fp_density(sentences),
        "fp_subject": compute_fp_subject(sentences),
    }
    for name, vals in confounds.items():
        if np.std(vals) < 1e-6:
            print(f"  !! WARNING: '{name}' has ~zero variance (std={np.std(vals):.2e}). "
                  f"Likely a data pipeline bug -- don't trust downstream numbers for it.")
    return confounds


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def fit_direction(E: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    model = Ridge(alpha=alpha)
    model.fit(E, y)
    w = model.coef_.astype(float)
    w_hat = w / np.linalg.norm(w)
    r2_in_sample = model.score(E, y)
    return w_hat, r2_in_sample


def holdout_r2(E: np.ndarray, y: np.ndarray, n_splits: int = N_HOLDOUT_SPLITS,
               test_size: float = TEST_SIZE, alpha: float = RIDGE_ALPHA, seed: int = SEED) -> float:
    scores = []
    for i in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=test_size, random_state=seed + i)
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        scores.append(m.score(Xte, yte))
    return float(np.mean(scores))


def orthogonalize(E: np.ndarray, directions: list[np.ndarray]) -> np.ndarray:
    D = np.stack(directions, axis=1)
    Q, _ = np.linalg.qr(D)
    E_c = E - E @ Q @ Q.T
    return l2_normalize_rows(E_c)


def random_direction_drop(E: np.ndarray, y: np.ndarray, w_score: float, r2_score: float,
                            n_directions: int, n_trials: int = N_RANDOM_TRIALS, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n_trials):
        rand_dirs = []
        for _ in range(n_directions):
            v = rng.normal(size=E.shape[1])
            v /= np.linalg.norm(v)
            rand_dirs.append(v)
        E_rand = orthogonalize(E, rand_dirs)
        _, r2_rand = fit_direction(E_rand, y)
        drops.append(r2_score - r2_rand)
    return float(np.mean(drops)), float(np.std(drops))


def run_for_model(model: str, sentences: list[str], y_score: np.ndarray,
                   confounds: dict[str, np.ndarray]) -> dict:
    import os
    embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
    if not os.path.exists(embed_path):
        print(f"  [skip] no embeddings found for model={model} at {embed_path}")
        return None

    E = l2_normalize_rows(np.load(embed_path))
    if E.shape[0] != len(y_score):
        print(f"  [skip] row mismatch for model={model}: E={E.shape[0]}, csv={len(y_score)}")
        return None

    w_score, r2_score_in = fit_direction(E, y_score)
    r2_score_holdout = holdout_r2(E, y_score)

    rows = []
    confound_dirs = {}
    for name, y_conf in confounds.items():
        w_conf, r2_conf_in = fit_direction(E, y_conf)
        r2_conf_holdout = holdout_r2(E, y_conf)
        confound_dirs[name] = w_conf

        cos_sim = float(np.dot(w_conf, w_score))

        E_orth = orthogonalize(E, [w_conf])
        _, r2_after = fit_direction(E_orth, y_score)
        drop = r2_score_in - r2_after

        rand_mean, rand_std = random_direction_drop(E, y_score, w_score, r2_score_in, n_directions=1)

        rows.append({
            "confound": name,
            "confound_R2_insample": round(r2_conf_in, 3),
            "confound_R2_holdout": round(r2_conf_holdout, 3),
            "cosine_with_score_dir": round(cos_sim, 3),
            "score_R2_after_removal": round(r2_after, 4),
            "score_R2_drop": round(drop, 4),
            "random_dir_drop_mean": round(rand_mean, 4),
            "random_dir_drop_std": round(rand_std, 4),
            "drop_vs_random_ratio": round(drop / rand_std, 1) if rand_std > 1e-8 else float("inf"),
        })

    individual_df = pd.DataFrame(rows)

    combined_result = None
    if len(confound_dirs) >= 1:
        all_dirs = list(confound_dirs.values())
        E_combined_orth = orthogonalize(E, all_dirs)
        _, r2_combined = fit_direction(E_combined_orth, y_score)
        combined_drop = r2_score_in - r2_combined
        rand_mean_m, rand_std_m = random_direction_drop(
            E, y_score, w_score, r2_score_in, n_directions=len(all_dirs)
        )
        combined_result = {
            "n_confounds_removed": len(all_dirs),
            "confounds": list(confound_dirs.keys()),
            "score_R2_before": round(r2_score_in, 4),
            "score_R2_after_combined_removal": round(r2_combined, 4),
            "combined_drop": round(combined_drop, 4),
            "matched_random_drop_mean": round(rand_mean_m, 4),
            "matched_random_drop_std": round(rand_std_m, 4),
            "drop_vs_random_ratio": round(combined_drop / rand_std_m, 1) if rand_std_m > 1e-8 else float("inf"),
        }

    return {
        "model": model,
        "score_R2_insample": round(r2_score_in, 3),
        "score_R2_holdout": round(r2_score_holdout, 3),
        "individual": individual_df,
        "combined": combined_result,
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    sentences = df[TEXT_COL].astype(str).tolist()
    y_score = df[SCORE_COL].to_numpy(dtype=float)

    print("=" * 72)
    print("STEP 1: computing writing-style confound scores (once, text-only)")
    print("=" * 72)
    confounds = get_style_confound_scores(sentences)
    print(f"\nConfounds available: {list(confounds.keys())}\n")

    all_results = []
    for model in MODEL_LIST:
        print("=" * 72)
        print(f"MODEL: {model}")
        print("=" * 72)
        result = run_for_model(model, sentences, y_score, confounds)
        if result is None:
            continue
        all_results.append(result)

        print(f"\nScore direction: R^2 (in-sample) = {result['score_R2_insample']}, "
              f"R^2 ({N_HOLDOUT_SPLITS}x held-out) = {result['score_R2_holdout']}")
        print("\n--- Individual confound removal ---")
        print(result["individual"].to_string(index=False))

        if result["combined"] is not None:
            c = result["combined"]
            print(f"\n--- Combined removal ({', '.join(c['confounds'])}, "
                  f"{c['n_confounds_removed']} directions) ---")
            print(f"  Score R^2 before removal:            {c['score_R2_before']}")
            print(f"  Score R^2 after combined removal:    {c['score_R2_after_combined_removal']}")
            print(f"  Drop:                                 {c['combined_drop']}")
            print(f"  Matched random-rank control (mean±std): "
                  f"{c['matched_random_drop_mean']} ± {c['matched_random_drop_std']}")
            print(f"  Drop vs. random-rank ratio (in std units): {c['drop_vs_random_ratio']}")
        print()


if __name__ == "__main__":
    main()
