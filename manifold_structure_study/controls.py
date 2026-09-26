from __future__ import annotations

import numpy as np


def shuffle_within_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Xs = X.copy()
    n = Xs.shape[0]
    for j in range(Xs.shape[1]):
        Xs[:, j] = Xs[rng.permutation(n), j]
    return Xs


def random_on_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal((n, d))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def covariance_matched_gaussian(
    X: np.ndarray, rng: np.random.Generator, *, normalize: bool = False
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 1:
        raise ValueError("covariance control needs an N x D array with N >= 3")
    if not np.isfinite(X).all():
        raise ValueError("covariance control requires finite values")
    mu = X.mean(axis=0, keepdims=True)
    centered = X - mu
    if not np.any(centered):
        raise ValueError("covariance control requires nonconstant data")
    n = len(X)
    sample = mu + rng.standard_normal((n, n)) @ centered / np.sqrt(n - 1)
    if normalize:
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("cannot normalize a zero covariance-control vector")
        sample = sample / norms
    return sample
