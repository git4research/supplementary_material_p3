from __future__ import annotations

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

try:
    import umap  
    _UMAP_CLASS = umap.UMAP
except Exception:
    import umap.umap_ as umap
    _UMAP_CLASS = umap.UMAP

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize


CSV_PATH = "full_dataset_with_scores_new.csv"
OUT_DIR = os.path.join("outputs", "tfidf_baseline")

N_SPLITS = 20
TEST_SIZE = 0.2
RIDGE_ALPHA = 1.0
LOGREG_MAX_ITER = 2000
SEED0 = 0

NGRAM_MIN = 1
NGRAM_MAX = 2
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True

SAVE_IMAGE_2D = True
SAVE_IMAGE_3D = True
UMAP_DENSIFY = False

TIER_ORDER = [
    "Collapse",
    "Striving",
    "Conflict",
    "Activation",
    "Growth",
    "Clarity",
    "Unity",
]


def make_energetic_colormap():
    energetic_colors = [
        "#ff3b3b",
        "#ff8b3d",
        "#ffd93b",
        "#34d399",
        "#3b82f6",
        "#6366f1",
        "#a855f7",
    ]
    return LinearSegmentedColormap.from_list("chakra", energetic_colors)
energetic_cmap = make_energetic_colormap()


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sentence", "score", "tier"]).reset_index(drop=True)
    df["sentence"] = df["sentence"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"]).reset_index(drop=True)
    df["tier"] = df["tier"].astype(str)

    unknown = sorted(set(df["tier"].unique()) - set(TIER_ORDER))
    if unknown:
        print(f"[WARN] Found tiers not in TIER_ORDER: {unknown}")

    return df


def run_umap(X, n_components: int):
    reducer = _UMAP_CLASS(
        n_neighbors=20,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_components=n_components,
    )
    return reducer.fit_transform(X)


def save_tfidf_umap_2d(X_tfidf, energies: np.ndarray, out_path: str):
    norm = Normalize(vmin=float(energies.min()), vmax=float(energies.max()))
    emb_2d = run_umap(X_tfidf, n_components=2)

    fig, ax = plt.subplots(figsize=(7.5, 6), constrained_layout=True)

    sc = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=energies,
        cmap=energetic_cmap,
        norm=norm,
        s=16,
        alpha=0.95,
    )
    ax.set_title("TF-IDF — 2D UMAP (colored by energy score)", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = fig.colorbar(sc, ax=ax, location="right", fraction=0.045, pad=0.02)
    cbar.set_label("Energy Score", fontsize=12)

    if SAVE_IMAGE_2D:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"2D TF-IDF UMAP saved to: {out_path}")

    plt.close(fig)


def save_tfidf_umap_3d(X_tfidf, energies: np.ndarray, out_path: str):
    norm = Normalize(vmin=float(energies.min()), vmax=float(energies.max()))
    emb_3d = run_umap(X_tfidf, n_components=3)

    fig = plt.figure(figsize=(8, 6.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        emb_3d[:, 0],
        emb_3d[:, 1],
        emb_3d[:, 2],
        c=energies,
        cmap=energetic_cmap,
        norm=norm,
        s=12,
        alpha=0.9,
    )
    ax.set_title("TF-IDF — 3D UMAP (colored by energy score)", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    cbar = fig.colorbar(sc, ax=ax, location="right", fraction=0.045, pad=0.02)
    cbar.set_label("Energy Score", fontsize=12)

    if SAVE_IMAGE_3D:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"3D TF-IDF UMAP saved to: {out_path}")

    plt.close(fig)


def eval_once_tfidf_regression(
    texts: np.ndarray,
    y_score: np.ndarray,
    seed: int,
    vec_kwargs: dict,
) -> dict:

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y_score, test_size=TEST_SIZE, random_state=seed
    )

    vec = TfidfVectorizer(**vec_kwargs)
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "ridge_r2": r2_score(y_test, y_pred),
        "ridge_mse": mean_squared_error(y_test, y_pred),
    }


def save_tfidf_confusion_matrix(cm: np.ndarray, labels: list[str], out_path: str) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("TF-IDF — Confusion Matrix (Tier Classification)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def eval_once_tfidf_classification(
    texts: np.ndarray,
    y_tier: np.ndarray,
    seed: int,
    vec_kwargs: dict,
    save_artifacts: bool = False,
    out_dir: str | None = None,
) -> dict:

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, y_tier, test_size=TEST_SIZE, random_state=seed, stratify=y_tier
    )

    vec = TfidfVectorizer(**vec_kwargs)
    X_train = vec.fit_transform(X_train_txt)
    X_test = vec.transform(X_test_txt)

    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    if save_artifacts and out_dir is not None:
        labels = [t for t in TIER_ORDER if t in set(y_tier)]
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        save_tfidf_confusion_matrix(
            cm=cm,
            labels=labels,
            out_path=os.path.join(out_dir, "tfidf_confusion_matrix_seed0.png"),
        )

    return {
        "tier_acc": accuracy_score(y_test, y_pred),
        "tier_f1": f1_score(y_test, y_pred, average="weighted"),
    }


def main() -> None:

    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_dataframe(CSV_PATH)
    texts = df["sentence"].to_numpy(dtype=object)
    y_score = df["score"].to_numpy(dtype=float)
    y_tier = df["tier"].to_numpy(dtype=str)

    vec_kwargs = dict(
        ngram_range=(NGRAM_MIN, NGRAM_MAX),
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=SUBLINEAR_TF,
        lowercase=True,
    )

    vec_all = TfidfVectorizer(**vec_kwargs)
    X_all_tfidf = vec_all.fit_transform(texts)
    if UMAP_DENSIFY:
        X_all_tfidf_for_umap = X_all_tfidf.toarray()
    else:
        X_all_tfidf_for_umap = X_all_tfidf

    save_path_2d = os.path.join(OUT_DIR, "tfidf_umap_2d.png")
    save_path_3d = os.path.join(OUT_DIR, "tfidf_umap_3d.png")

    save_tfidf_umap_2d(X_all_tfidf_for_umap, y_score, save_path_2d)
    save_tfidf_umap_3d(X_all_tfidf_for_umap, y_score, save_path_3d)

    rows = []
    for seed in range(SEED0, SEED0 + N_SPLITS):

        m_reg = eval_once_tfidf_regression(texts, y_score, seed, vec_kwargs)
        m_clf = eval_once_tfidf_classification(
            texts,
            y_tier,
            seed,
            vec_kwargs,
            save_artifacts=(seed == SEED0),
            out_dir=OUT_DIR,
        )

        rows.append({"seed": seed, **m_reg, **m_clf})

    per_split = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    summary = per_split[["ridge_r2", "ridge_mse", "tier_acc", "tier_f1"]].agg(["mean", "std"])

    per_split.to_csv(os.path.join(OUT_DIR, "tfidf_observed_per_split.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "tfidf_observed_summary_mean_std.csv"))

    metrics = pd.DataFrame([{
        "model": f"TF-IDF ({NGRAM_MIN}-{NGRAM_MAX}gram)",
        "n_samples": int(len(df)),
        "n_splits": int(N_SPLITS),
        "test_size": float(TEST_SIZE),
        "ridge_r2_mean": float(per_split["ridge_r2"].mean()),
        "ridge_mse_mean": float(per_split["ridge_mse"].mean()),
        "tier_acc_mean": float(per_split["tier_acc"].mean()),
        "tier_f1_mean": float(per_split["tier_f1"].mean()),
        "tfidf_ngram": f"({NGRAM_MIN},{NGRAM_MAX})",
        "tfidf_min_df": int(MIN_DF),
        "tfidf_max_df": float(MAX_DF),
        "tfidf_sublinear_tf": bool(SUBLINEAR_TF),
    }])

    metrics.to_csv(os.path.join(OUT_DIR, "tfidf_metrics.csv"), index=False)

    print("\n[TF-IDF baseline]")
    print(metrics.to_string(index=False))
    print(f"\n[Saved outputs] {OUT_DIR}/")
    print(" - tfidf_umap_2d.png")
    print(" - tfidf_umap_3d.png")
    print(" - tfidf_confusion_matrix_seed0.png")
    print(" - tfidf_observed_per_split.csv")
    print(" - tfidf_observed_summary_mean_std.csv")
    print(" - tfidf_metrics.csv")


if __name__ == "__main__":
    main()
