"""Reference distributions for the dimensionality diagnostics (Appendix D and G).

Each function returns a synthetic N x D array to compare against real embeddings:
coordinate-wise shuffled data, isotropic unit-sphere samples, or Gaussian samples
matched to the data's empirical mean and covariance.
"""
from __future__ import annotations

import numpy as np


def shuffle_within_columns(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute each column independently: keeps per-coordinate marginals, breaks joint structure."""
    Xs = X.copy()
    n = Xs.shape[0]
    for j in range(Xs.shape[1]):
        Xs[:, j] = Xs[rng.permutation(n), j]
    return Xs


def random_on_sphere(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n points uniformly on the unit sphere in R^d (normalized standard Gaussians)."""
    G = rng.standard_normal((n, d))
    return G / np.linalg.norm(G, axis=1, keepdims=True)


def covariance_matched_gaussian(
    X: np.ndarray, rng: np.random.Generator, *, normalize: bool = False
) -> np.ndarray:
    """Gaussian sample with the empirical mean and covariance of X; optionally L2-normalized after sampling."""
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
    # Random mixtures of the centered rows: covariance equals the sample covariance in expectation.
    sample = mu + rng.standard_normal((n, n)) @ centered / np.sqrt(n - 1)
    if normalize:
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(float).tiny):
            raise ValueError("cannot normalize a zero covariance-control vector")
        sample = sample / norms
    return sample
