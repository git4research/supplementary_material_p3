from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def twonn(X: np.ndarray, discard_fraction: float = 0.10) -> float:
    if X.shape[0] < 3:
        raise ValueError("twonn needs at least 3 points")
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    dist, _ = nn.kneighbors(X)
    r1, r2 = dist[:, 1], dist[:, 2]
    mask = r1 > 0
    mu = np.sort(r2[mask] / r1[mask])
    n = mu.size
    keep = int(np.floor(n * (1.0 - discard_fraction)))
    mu = mu[:keep]
    F = np.arange(1, keep + 1) / n
    x, y = np.log(mu), -np.log(1.0 - F)
    return float(np.sum(x * y) / np.sum(x * x))


def mle(X: np.ndarray) -> float:
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
    n_components: dict
    participation_ratio: float
    cumulative: np.ndarray
    eigenvalues: np.ndarray


def pca_profile(X: np.ndarray, thresholds=(0.80, 0.90, 0.95, 0.99)) -> PCAProfile:
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(svd_solver="full").fit(Xc)
    ev = pca.explained_variance_
    cum = np.cumsum(pca.explained_variance_ratio_)
    n_for = {t: int(np.searchsorted(cum, t) + 1) for t in thresholds}
    pr = float((ev.sum() ** 2) / np.square(ev).sum())
    return PCAProfile(n_components=n_for, participation_ratio=pr,
                      cumulative=cum, eigenvalues=ev)
