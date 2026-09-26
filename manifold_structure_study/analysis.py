from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import config
import controls
import estimators


def analyze_one(es, rng, thresholds, twonn_discard, with_mle,
                covariance_repeats=config.COVARIANCE_REPEATS,
                covariance_rng=None, covariance_records=None):
    if not isinstance(covariance_repeats, (int, np.integer)) or covariance_repeats < 2:
        raise ValueError("covariance_repeats must be an integer >= 2")
    prof = estimators.pca_profile(es.X, thresholds)
    id_twonn = estimators.twonn(es.X, twonn_discard)

    row = {
        "model": es.label,
        "model_key": es.model_key,
        "layer": es.layer,
        "N": es.n,
        "ambient_dim": es.ambient,
        "dropped_duplicates": es.n_dropped,
        "sha16": es.sha16,
    }
    for t in thresholds:
        row[f"pca_n{int(round(t * 100))}"] = prof.n_components[t]
    row["participation_ratio"] = round(prof.participation_ratio, 2)
    row["twonn"] = round(id_twonn, 2)
    if with_mle:
        row["mle"] = round(estimators.mle(es.X), 2)

    Xsh = controls.shuffle_within_columns(es.X, rng)
    prof_sh = estimators.pca_profile(Xsh, thresholds)
    for t in (0.90, 0.95):
        if t in prof_sh.n_components:
            row[f"shuffle_pca_n{int(round(t * 100))}"] = prof_sh.n_components[t]
    row["shuffle_twonn"] = round(estimators.twonn(Xsh, twonn_discard), 2)

    Xr = controls.random_on_sphere(es.n, es.ambient, rng)
    row["random_twonn"] = round(estimators.twonn(Xr, twonn_discard), 2)

    cov_rng = rng if covariance_rng is None else covariance_rng
    normalize = bool(np.allclose(np.linalg.norm(es.X, axis=1), 1.0,
                                 rtol=1e-5, atol=1e-5))
    estimates = []
    for _ in range(covariance_repeats):
        Xcov = controls.covariance_matched_gaussian(es.X, cov_rng,
                                                  normalize=normalize)
        estimate = estimators.twonn(Xcov, twonn_discard)
        if not np.isfinite(estimate):
            raise ValueError(f"{es.label}: nonfinite covariance-control TwoNN")
        estimates.append(float(estimate))
    mean = float(np.mean(estimates))
    sd = float(np.std(estimates, ddof=1))
    row["covariance_twonn_mean"] = round(mean, 2)
    row["covariance_twonn_std"] = round(sd, 2)
    row["covariance_repeats"] = covariance_repeats
    row["covariance_l2_normalized"] = normalize
    if covariance_records is not None:
        covariance_records.append({
            "model": es.label, "source": str(es.path.resolve()), "sha16": es.sha16,
            "N": es.n, "ambient_dim": es.ambient,
            "l2_normalized_after_sampling": normalize,
            "twonn_discard_fraction": twonn_discard,
            "twonn_estimates": estimates, "mean": mean, "sample_std": sd,
        })

    return row, prof.cumulative, prof_sh.cumulative


def analyze_all(embedding_sets, results_dir, seed=0,
                thresholds=config.VARIANCE_THRESHOLDS,
                twonn_discard=config.TWONN_DISCARD_FRACTION,
                with_mle=config.WITH_MLE,
                covariance_repeats=config.COVARIANCE_REPEATS):
    if not isinstance(covariance_repeats, (int, np.integer)) or covariance_repeats < 2:
        raise ValueError("covariance_repeats must be an integer >= 2")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rows, curves, covariance_records = [], {}, []
    for es in embedding_sets:
        rng = np.random.default_rng(seed)
        print(f"Analyzing {es.label} (covariance control: {covariance_repeats} draws)...",
              flush=True)
        covariance_rng = np.random.default_rng(np.random.SeedSequence([seed, 1301]))
        row, cum, cum_sh = analyze_one(
            es, rng, thresholds, twonn_discard, with_mle,
            covariance_repeats=covariance_repeats, covariance_rng=covariance_rng,
            covariance_records=covariance_records)
        rows.append(row)
        curves[es.label] = {"cumulative": cum, "cumulative_shuffle": cum_sh,
                            "twonn": row["twonn"], "ambient": es.ambient}

    _save_csv(rows, results_dir / "manifold_results.csv")
    with open(results_dir / "manifold_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(results_dir / "covariance_controls.json", "w") as f:
        json.dump({
            "seed": seed, "rng_stream": [seed, 1301], "repeats": covariance_repeats,
            "moment_matching": "Empirical mean and covariance in expectation before optional L2 normalization",
            "normalization_policy": "Apply L2 after sampling only if all input row norms are approximately 1",
            "std_interpretation": "Sample SD across Gaussian draws; not a confidence interval",
            "models": covariance_records,
        }, f, indent=2)
    np.savez(results_dir / "curves.npz", curves=np.array(curves, dtype=object))

    return rows, curves


def _save_csv(rows, path):
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_table(rows):
    cols = [("model", "model", 22), ("ambient_dim", "amb", 7),
            ("pca_n90", "PCA90", 7), ("pca_n95", "PCA95", 7),
            ("shuffle_pca_n90", "shPCA90", 9), ("shuffle_pca_n95", "shPCA95", 9),
            ("participation_ratio", "PR", 7), ("twonn", "TwoNN", 8)]
    if any("mle" in r for r in rows):
        cols.append(("mle", "MLE", 8))
    cols += [("shuffle_twonn", "shufID", 9), ("random_twonn", "randID", 9)]
    if any("covariance_twonn_mean" in r for r in rows):
        cols += [("covariance_twonn_mean", "covID", 9),
                 ("covariance_twonn_std", "covSD", 8)]

    def fmt(v, w, left):
        s = "" if v is None else str(v)
        return f"{s:<{w}}" if left else f"{s:>{w}}"

    header = "".join(fmt(h, w, key == "model") for key, h, w in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("".join(fmt(r.get(key), w, key == "model") for key, _, w in cols))
    if any("covariance_twonn_mean" in r for r in rows):
        print("covID/covSD: TwoNN mean / sample SD across covariance-matched Gaussian draws.")
        print("Moments match in expectation before optional L2 normalization; SD is not a confidence interval.")
