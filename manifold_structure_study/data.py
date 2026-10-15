"""Load embedding matrices for the manifold-structure study (Appendix D and G).

Reads every *.npy file (an N x D array) in a directory, removes exact duplicate
rows, records a SHA-256 fingerprint of the raw array, and parses the model key
and layer tag from the file name. Returns a list of EmbeddingSet objects.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Matches a trailing layer tag such as '_L12' in the file stem.
_LAYER_RE = re.compile(r"_(L\d+)$")


@dataclass
class EmbeddingSet:
    """One deduplicated embedding matrix with its display label and provenance."""
    label: str
    model_key: str
    layer: str
    path: Path
    X: np.ndarray
    n_raw: int
    n_dropped: int
    sha16: str

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def ambient(self) -> int:
        return self.X.shape[1]


def parse_name(path: Path, overrides: dict | None = None):
    """Return (display label, model key, layer) from a name like embeddings_<model>_L<k>.npy."""
    stem = path.stem
    if stem.startswith("embeddings_"):
        stem = stem[len("embeddings_"):]
    layer = ""
    m = _LAYER_RE.search(stem)
    if m:
        layer = m.group(1)
        stem = stem[: m.start()]
    model_key = stem
    pretty = (overrides or {}).get(model_key, model_key)
    label = f"{pretty} ({layer})" if layer else pretty
    return label, model_key, layer


def load_one(path: Path, overrides: dict | None = None) -> EmbeddingSet:
    raw = np.load(path)
    if raw.ndim != 2:
        raise ValueError(f"{path.name}: expected a 2-D array, got shape {raw.shape}")
    # Fingerprint of the raw array as stored on disk (first 16 hex chars of SHA-256).
    sha16 = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()[:16]
    X = raw.astype(np.float64)
    # Drop exact duplicate vectors, keeping first occurrences in their original order.
    _, idx = np.unique(X, axis=0, return_index=True)
    X = X[np.sort(idx)]
    label, model_key, layer = parse_name(path, overrides)
    return EmbeddingSet(
        label=label, model_key=model_key, layer=layer, path=path, X=X,
        n_raw=raw.shape[0], n_dropped=raw.shape[0] - X.shape[0], sha16=sha16,
    )


def load_all(embeddings_dir, overrides: dict | None = None) -> list[EmbeddingSet]:
    """Load every *.npy file in a directory, in sorted file-name order."""
    embeddings_dir = Path(embeddings_dir)
    files = sorted(embeddings_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(
            f"No .npy files found in {embeddings_dir.resolve()}"
        )
    return [load_one(f, overrides) for f in files]
