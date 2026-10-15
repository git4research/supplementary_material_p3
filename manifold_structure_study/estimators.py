"""Dimensionality estimators for the manifold-structure study (Appendix D and G).

twonn(): TwoNN intrinsic dimension; mle(): MLE intrinsic dimension via
scikit-dimension; pca_profile(): number of PCA components needed for given
explained-variance thresholds, plus the participation ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def twonn(X: np.ndarray, discard_fraction: float = 0.10) -> float:
    """TwoNN estimate: slope of -log(1 - F) vs. log(r2/r1), fit through the origin after dropping the largest ratios."""
    if X.shape[0] < 3:
        raise ValueError("twonn needs at least 3 points")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    # Column 0 is the point itself; skip points whose nearest neighbor is at distance zero.
    mask = r1 > 0
    mu = np.sort(r2[mask] / r1[mask])
    n = mu.size
    keep = int(np.floor(n * (1.0 - discard_fraction)))
    mu = mu[:keep]
    # Empirical CDF of the kept ratios, computed with the full count n.
    F = np.arange(1, keep + 1) / n
    x, y = np.log(mu), -np.log(1.0 - F)
    # Least-squares slope of a line through the origin.
    return float(np.sum(x * y) / np.sum(x * x))


def mle(X: np.ndarray) -> float:
    """MLE intrinsic dimension from scikit-dimension with default settings."""
    try:
        import skdim
    except ImportError as e:
        raise ImportError(
            "mle() requires scikit-dimension (pip install scikit-dimension); "
            "run with --no-mle to skip it."
        ) from e
    return float(skdim.id.MLE().fit(X).dimension_)


@dataclass
class PCAProfile:
    """PCA summary: components per variance threshold, participation ratio, cumulative ratio, eigenvalues."""
    n_components: dict
    participation_ratio: float
    cumulative: np.ndarray
    eigenvalues: np.ndarray


def pca_profile(X: np.ndarray, thresholds=(0.80, 0.90, 0.95, 0.99)) -> PCAProfile:
    """Fit full PCA on mean-centered X and summarize its spectrum."""
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(svd_solver="full").fit(Xc)
    ev = pca.explained_variance_
    cum = np.cumsum(pca.explained_variance_ratio_)
    # Smallest number of components whose cumulative explained variance reaches each threshold.
    n_for = {t: int(np.searchsorted(cum, t) + 1) for t in thresholds}
    # Participation ratio: (sum of eigenvalues)^2 / (sum of squared eigenvalues).
    pr = float((ev.sum() ** 2) / np.square(ev).sum())
    return PCAProfile(n_components=n_for, participation_ratio=pr,
                      cumulative=cum, eigenvalues=ev)
