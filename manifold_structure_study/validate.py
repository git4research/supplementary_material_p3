"""Synthetic sanity checks for the intrinsic-dimension estimators (Appendix D and G).

Runs TwoNN (and optionally MLE) on point clouds with known dimension, prints a
table, asserts low-dimensional recovery, and can write validation.csv.
Usage: python validate.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.datasets import make_swiss_roll

import estimators


def _sphere(n, d, rng):
    """Uniform samples on the d-sphere S^d in R^(d+1)."""
    G = rng.standard_normal((n, d + 1))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def _hypercube(n, d, rng):
    """Uniform samples in the flat unit cube [0, 1]^d."""
    return rng.uniform(0.0, 1.0, (n, d))


def _linear_highdim(n, d, ambient, rng):
    """d-dim uniform latent mapped linearly into R^ambient, then projected onto the unit sphere."""
    z = rng.uniform(0.0, 1.0, (n, d))
    X = z @ rng.standard_normal((d, ambient))
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _curved_highdim(n, d, ambient, rng, scale=0.6):
    """d-dim uniform latent mapped through random sin/cos features into R^ambient, then unit-normalized."""
    z = rng.uniform(0.0, 1.0, (n, d))
    proj = z @ (rng.standard_normal((d, ambient // 2)) * scale)
    X = np.concatenate([np.sin(proj), np.cos(proj)], axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def build_cases(n=635, seed=0):
    """Return (name, points, true dimension) for each synthetic test case."""
    rng = np.random.default_rng(seed)
    return [
        ("Swiss roll",            make_swiss_roll(n, noise=0.0, random_state=seed)[0], 2),
        ("Sphere S^4 (R^5)",      _sphere(n, 4, rng),                                  4),
        ("Hypercube d=10 (flat)", _hypercube(n, 10, rng),                             10),
        ("Linear d=16 (R^512)",   _linear_highdim(n, 16, 512, rng),                   16),
        ("Curved d=16 (R^512)",   _curved_highdim(n, 16, 512, rng),                   16),
    ]


def run(n=635, seed=0, with_mle=True, results_dir=None, twonn_discard=0.10):
    """Estimate ID on each case, print a table, run sanity checks, and optionally write validation.csv."""
    cases = build_cases(n=n, seed=seed)
    rows = []
    for label, X, true_d in cases:
        row = {"manifold": label, "true_dim": true_d,
               "twonn": round(estimators.twonn(X, twonn_discard), 2)}
        if with_mle:
            try:
                row["mle"] = round(estimators.mle(X), 2)
            except ImportError:
                # scikit-dimension not installed: skip MLE for the remaining cases.
                with_mle = False
        rows.append(row)

    have_mle = any("mle" in r for r in rows)
    header = f"{'manifold':<24}{'true':>6}{'TwoNN':>8}" + (f"{'MLE':>8}" if have_mle else "")
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['manifold']:<24}{r['true_dim']:>6}{r['twonn']:>8.1f}"
        if have_mle:
            line += f"{r.get('mle', float('nan')):>8.1f}"
        print(line)
    print("\nRead: accurate at low dim (swiss roll -> ~2); under-estimates at higher\n"
          "dim and this N, so ID on real data is a conservative lower bound.")

    _sane(rows)

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        fields = ["manifold", "true_dim", "twonn"] + (["mle"] if have_mle else [])
        with open(results_dir / "validation.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return rows


def _sane(rows):
    """Raise if TwoNN misses the Swiss roll or 4-sphere dimension by more than loose bounds."""
    lookup = {r["manifold"]: r for r in rows}
    checks = [("Swiss roll", 1.3, 2.8), ("Sphere S^4 (R^5)", 3.0, 4.8)]
    for name, lo, hi in checks:
        v = lookup[name]["twonn"]
        if not (lo <= v <= hi):
            raise AssertionError(
                f"sanity check failed: TwoNN on {name} = {v}, expected in [{lo},{hi}]. "
                "Estimator or install likely broken."
            )
    print("sanity: low-dimensional recovery OK.")


if __name__ == "__main__":
    run(results_dir="results")
