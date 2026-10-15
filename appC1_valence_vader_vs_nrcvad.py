"""Compare the two valence measures, VADER vs NRC-VAD (Appendix C.1).

For each embedding space, fits a Ridge direction for each valence score, reports their cosine,
in-sample and held-out R^2, and cross-removal (R^2 of one measure after projecting out the
other's direction). Reads the dataset CSV, the NRC-VAD lexicon and outputs/<model>/embeddings.npy;
writes outputs/valence_vader_vs_nrcvad_comparison.csv.
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

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding"]

NRC_VAD_PATH = "nrc_vad_extracted/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"

RIDGE_ALPHA = 1.0
N_HOLDOUT_SPLITS = 5
TEST_SIZE = 0.2
SEED = 0


def compute_valence_vader(sentences):
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


def remove_direction(E: np.ndarray, w_hat: np.ndarray) -> np.ndarray:
    """Project out unit direction w_hat from every row, then re-normalize rows."""
    proj = E @ w_hat
    E_abl = E - np.outer(proj, w_hat)
    return l2_normalize_rows(E_abl)


def run_for_model(model: str, E_raw: np.ndarray, y_vader: np.ndarray, y_nrcvad: np.ndarray) -> dict:
    E = l2_normalize_rows(E_raw)

    w_vader, r2_vader_in = fit_direction(E, y_vader)
    w_nrcvad, r2_nrcvad_in = fit_direction(E, y_nrcvad)

    r2_vader_holdout = holdout_r2(E, y_vader)
    r2_nrcvad_holdout = holdout_r2(E, y_nrcvad)

    cosine = float(np.dot(w_vader, w_nrcvad))

    # Cross-removal: decode each valence measure after removing the other's direction.
    E_minus_nrcvad = remove_direction(E, w_nrcvad)
    _, r2_vader_after_removing_nrcvad = fit_direction(E_minus_nrcvad, y_vader)

    E_minus_vader = remove_direction(E, w_vader)
    _, r2_nrcvad_after_removing_vader = fit_direction(E_minus_vader, y_nrcvad)

    return {
        "model": model,
        "cosine_vader_nrcvad": round(cosine, 3),
        "r2_vader_insample": round(r2_vader_in, 3),
        "r2_vader_holdout": round(r2_vader_holdout, 3),
        "r2_nrcvad_insample": round(r2_nrcvad_in, 3),
        "r2_nrcvad_holdout": round(r2_nrcvad_holdout, 3),
        "r2_vader_after_removing_nrcvad": round(r2_vader_after_removing_nrcvad, 3),
        "vader_drop_from_nrcvad_removal": round(r2_vader_in - r2_vader_after_removing_nrcvad, 4),
        "r2_nrcvad_after_removing_vader": round(r2_nrcvad_after_removing_vader, 3),
        "nrcvad_drop_from_vader_removal": round(r2_nrcvad_in - r2_nrcvad_after_removing_vader, 4),
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    sentences = df[TEXT_COL].astype(str).tolist()

    print("Computing valence scores (VADER + NRC-VAD)...")
    y_vader = compute_valence_vader(sentences)
    y_nrcvad = compute_valence_nrcvad(sentences, NRC_VAD_PATH)
    score_corr = float(np.corrcoef(y_vader, y_nrcvad)[0, 1])
    print(f"Raw score correlation (VADER vs NRC-VAD): {score_corr:.3f}\n")

    rows = []
    for model in MODEL_LIST:
        embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
        if not os.path.exists(embed_path):
            print(f"[skip] no embeddings for {model}")
            continue
        E_raw = np.load(embed_path)
        if E_raw.shape[0] != len(y_vader):
            print(f"[skip] row mismatch for {model}")
            continue

        print(f"=== {model} ===")
        result = run_for_model(model, E_raw, y_vader, y_nrcvad)
        rows.append(result)

        print(f"  cosine(w_vader, w_nrcvad):              {result['cosine_vader_nrcvad']}")
        print(f"  VADER R^2:   in-sample={result['r2_vader_insample']}, "
              f"holdout={result['r2_vader_holdout']}")
        print(f"  NRC-VAD R^2: in-sample={result['r2_nrcvad_insample']}, "
              f"holdout={result['r2_nrcvad_holdout']}")
        print(f"  VADER R^2 after removing NRC-VAD's direction:   "
              f"{result['r2_vader_after_removing_nrcvad']} "
              f"(drop={result['vader_drop_from_nrcvad_removal']})")
        print(f"  NRC-VAD R^2 after removing VADER's direction:   "
              f"{result['r2_nrcvad_after_removing_vader']} "
              f"(drop={result['nrcvad_drop_from_vader_removal']})\n")

    summary_df = pd.DataFrame(rows)
    print("=" * 100)
    print("SUMMARY -- VADER vs NRC-VAD valence: direction similarity + cross-removal decodability")
    print("=" * 100)
    print(summary_df.to_string(index=False))

    out_path = os.path.join(OUTPUT_ROOT, "valence_vader_vs_nrcvad_comparison.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print(f"""
How to read this:
  - cosine near 1.0: the two valence measures fit nearly the same direction
    in this model's embedding space, despite differing as raw scores
    (r={score_corr:.3f}). cosine near 0: they occupy largely different
    subspaces, i.e. the two tools are picking up on different aspects of
    "valence" as encoded in this model.
  - A LARGE drop when removing the other's direction means substantial
    shared representational content between the two valence sources. A
    SMALL drop (most of the R^2 surviving) means each captures something
    the other's direction does not.
""")


if __name__ == "__main__":
    main()
