"""Semantic-confound removal test (Appendix C.1).

Scores each sentence for valence (VADER, NRC-VAD), abstraction (Brysbaert concreteness) and
formality (RoBERTa formality ranker), fits a Ridge direction per confound in each embedding space,
projects those directions out, and measures how much in-sample score R^2 is lost, compared with
removing the same number of random directions. Reads the dataset CSV, outputs/<model>/embeddings.npy
and the lexicon files; writes outputs/confound_analysis_summary.csv.
"""
from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

CSV_PATH = "full_dataset_with_scores_new.csv"
OUTPUT_ROOT = "outputs"
TEXT_COL = "sentence"
SCORE_COL = "score"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding"]

RIDGE_ALPHA = 1.0
N_HOLDOUT_SPLITS = 5
TEST_SIZE = 0.2
SEED = 0

NRC_VAD_PATH = "nrc_vad_extracted/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
BRYSBAERT_PATH = "brysbaert_concreteness.csv"
USE_FORMALITY_CLASSIFIER = True


def compute_valence_vader(sentences):
    # VADER compound polarity in [-1, 1].
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(s)["compound"] for s in sentences])


def compute_valence_nrcvad(sentences, path):
    # Mean NRC-VAD valence over lexicon words in the sentence (0.0 if none match).
    lex = pd.read_csv(path, sep="\t", header=0, names=["word", "valence", "arousal", "dominance"])
    vad = dict(zip(lex["word"].str.lower(), lex["valence"]))
    out = []
    for s in sentences:
        words = [w.strip(".,!?;:\"'").lower() for w in s.split()]
        vals = [vad[w] for w in words if w in vad]
        out.append(np.mean(vals) if vals else 0.0)
    return np.array(out)


def compute_abstraction_brysbaert(sentences, path):
    # Abstraction = 5 - mean concreteness (Brysbaert 1-5 scale) of matched words; 2.5 if none match.
    lex = pd.read_csv(path)
    conc = dict(zip(lex["Word"].str.lower(), lex["Conc.M"]))
    scores = []
    for s in sentences:
        words = [w.strip(".,!?;:\"'").lower() for w in s.split()]
        vals = [conc[w] for w in words if w in conc]
        scores.append(5.0 - np.mean(vals) if vals else 2.5)
    return np.array(scores)


def compute_formality_classifier(sentences):
    # Classifier probability of the "formal" label for each sentence.
    from transformers import pipeline
    clf = pipeline("text-classification", model="s-nlp/roberta-base-formality-ranker",
                    top_k=None, truncation=True)
    scores = []
    for s in sentences:
        result = clf(s)[0]
        formal_score = next(r["score"] for r in result if r["label"].lower().startswith("formal"))
        scores.append(formal_score)
    return np.array(scores)


def get_confound_scores(sentences):
    """Compute every available confound score; any confound whose dependency or file is missing is skipped."""
    confounds = {}

    try:
        confounds["valence_vader"] = compute_valence_vader(sentences)
        print("[valence_vader] using VADER")
    except Exception as e:
        print(f"[valence_vader] SKIPPED -- {e}")

    try:
        if os.path.exists(NRC_VAD_PATH):
            confounds["valence_nrcvad"] = compute_valence_nrcvad(sentences, NRC_VAD_PATH)
            print("[valence_nrcvad] using NRC-VAD lexicon")
        else:
            print(f"[valence_nrcvad] SKIPPED -- {NRC_VAD_PATH} not found")
    except Exception as e:
        print(f"[valence_nrcvad] SKIPPED -- {e}")

    try:
        if os.path.exists(BRYSBAERT_PATH):
            confounds["abstraction"] = compute_abstraction_brysbaert(sentences, BRYSBAERT_PATH)
            print("[abstraction] using Brysbaert concreteness norms")
        else:
            print(f"[abstraction] SKIPPED -- {BRYSBAERT_PATH} not found")
    except Exception as e:
        print(f"[abstraction] SKIPPED -- {e}")

    if USE_FORMALITY_CLASSIFIER:
        try:
            confounds["formality"] = compute_formality_classifier(sentences)
            print("[formality] using s-nlp/roberta-base-formality-ranker")
        except Exception as e:
            print(f"[formality] SKIPPED -- {e}")

    for name, vals in confounds.items():
        if np.std(vals) < 1e-6:
            print(f"  !! WARNING: '{name}' has ~zero variance (std={np.std(vals):.2e}). "
                  f"Likely a data pipeline bug -- don't trust downstream numbers for it.")

    if "valence_vader" in confounds and "valence_nrcvad" in confounds:
        corr = np.corrcoef(confounds["valence_vader"], confounds["valence_nrcvad"])[0, 1]
        print(f"\n  Correlation between the two valence sources: {corr:.3f}")

    return confounds


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def fit_direction(E: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA):
    """Fit Ridge(y ~ E); return the unit-norm coefficient direction and in-sample R^2."""
    model = Ridge(alpha=alpha)
    model.fit(E, y)
    w = model.coef_.astype(float)
    w_hat = w / np.linalg.norm(w)
    r2_in_sample = model.score(E, y)
    return w_hat, r2_in_sample


def holdout_r2(E: np.ndarray, y: np.ndarray, n_splits: int = N_HOLDOUT_SPLITS,
               test_size: float = TEST_SIZE, alpha: float = RIDGE_ALPHA, seed: int = SEED) -> float:
    """Mean test R^2 of Ridge over n_splits random train/test splits."""
    scores = []
    for i in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=test_size, random_state=seed + i)
        m = Ridge(alpha=alpha).fit(Xtr, ytr)
        scores.append(m.score(Xte, yte))
    return float(np.mean(scores))


def orthogonalize(E: np.ndarray, directions: list[np.ndarray]) -> np.ndarray:
    """Project out the span of `directions` (via QR basis) from every row, then re-normalize rows."""
    D = np.stack(directions, axis=1)
    Q, _ = np.linalg.qr(D)
    E_c = E - E @ Q @ Q.T
    return l2_normalize_rows(E_c)


def random_direction_drop(E: np.ndarray, y: np.ndarray, w_score: float, r2_score: float,
                            n_directions: int, n_trials: int = 20, seed: int = SEED) -> tuple[float, float]:
    """Control: mean/std of the score-R^2 drop after removing n_directions random unit directions (n_trials draws)."""
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

        # Remove this confound's direction, refit the score regression, and record the R^2 drop.
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
            # Confound drop expressed in units of the random-control std.
            "drop_vs_random_ratio": round(drop / rand_std, 1) if rand_std > 1e-8 else float("inf"),
        })

    individual_df = pd.DataFrame(rows)

    # Remove all confound directions jointly; control with the same number of random directions.
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
    print("STEP 1: computing confound scores (once, text-only)")
    print("=" * 72)
    confounds = get_confound_scores(sentences)
    if not confounds:
        raise RuntimeError("No confound scores could be computed -- nothing to test.")
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

    if all_results:
        summary_rows = []
        for r in all_results:
            row = {
                "model": r["model"],
                "score_R2_holdout": r["score_R2_holdout"],
            }
            if r["combined"] is not None:
                row["n_confounds_removed"] = r["combined"]["n_confounds_removed"]
                row["combined_drop"] = r["combined"]["combined_drop"]
                row["random_rank_drop_mean"] = r["combined"]["matched_random_drop_mean"]
                row["drop_vs_random_ratio"] = r["combined"]["drop_vs_random_ratio"]
            summary_rows.append(row)
        summary_df = pd.DataFrame(summary_rows)

        print("=" * 72)
        print("CROSS-MODEL SUMMARY -- combined confound removal")
        print("=" * 72)
        print(summary_df.to_string(index=False))

        out_path = os.path.join(OUTPUT_ROOT, "confound_analysis_summary.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
