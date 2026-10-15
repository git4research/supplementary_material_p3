"""Sampling-scale sensitivity of intrinsic dimension (Appendix D and G).

For each embedding set, estimates ID (TwoNN by default, or MLE) on repeated
random subsets at each fraction in DECIMATION_FRACTIONS (the full set once).
Writes decimation.csv and decimation.npz to the results directory.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import config
import estimators


def _estimator_fn(name, twonn_discard):
    if name == "twonn":
        return lambda A: estimators.twonn(A, twonn_discard)
    if name == "mle":
        return estimators.mle
    raise ValueError(f"unknown estimator '{name}' (use 'twonn' or 'mle')")


def default_sizes(n, fractions=config.DECIMATION_FRACTIONS):
    """Subsample sizes for the given fractions of n, clipped to [10, n], deduplicated and sorted."""
    return sorted({min(n, max(10, int(round(f * n)))) for f in fractions})


def decimation_curve(X, sizes, repeats, rng, estimator="twonn",
                     twonn_discard=config.TWONN_DISCARD_FRACTION):
    """Mean and SD of the ID estimate at each size; sizes >= n are estimated once on all points."""
    est = _estimator_fn(estimator, twonn_discard)
    n = X.shape[0]
    means, stds = [], []
    for s in sizes:
        if s >= n:
            vals = np.array([est(X)])
        else:
            vals = np.array([est(X[rng.choice(n, size=s, replace=False)])
                             for _ in range(repeats)])
        means.append(float(vals.mean()))
        stds.append(float(vals.std()))
    return np.asarray(means), np.asarray(stds)


def run_decimation(embedding_sets, results_dir, seed=config.SEED,
                   fractions=config.DECIMATION_FRACTIONS,
                   repeats=config.DECIMATION_REPEATS, estimator="twonn",
                   twonn_discard=config.TWONN_DISCARD_FRACTION):
    """Compute decimation curves for every embedding set and save them as CSV and NPZ."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    curves, rows = {}, []
    for es in embedding_sets:
        # Same seed for every model, so results do not depend on processing order.
        rng = np.random.default_rng(seed)
        sizes = default_sizes(es.n, fractions)
        means, stds = decimation_curve(es.X, sizes, repeats, rng, estimator, twonn_discard)
        curves[es.label] = {"sizes": np.asarray(sizes), "mean": means, "std": stds,
                            "estimator": estimator}
        for s, m, sd in zip(sizes, means, stds):
            rows.append({"model": es.label, "estimator": estimator, "n_sampled": int(s),
                         "id_mean": round(m, 3), "id_std": round(sd, 3),
                         "repeats": 1 if s >= es.n else repeats})

    with open(results_dir / "decimation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "estimator", "n_sampled",
                                          "id_mean", "id_std", "repeats"])
        w.writeheader()
        w.writerows(rows)
    np.savez(results_dir / "decimation.npz", curves=np.array(curves, dtype=object))
    return curves


def print_summary(curves):
    """Print one line per (model, subsample size)."""
    print(f"{'model':<22}{'n':>6}{'ID mean':>9}{'ID std':>8}")
    print("-" * 45)
    for lab, d in curves.items():
        for s, m, sd in zip(d["sizes"], d["mean"], d["std"]):
            print(f"{lab:<22}{int(s):>6}{m:>9.2f}{sd:>8.2f}")
