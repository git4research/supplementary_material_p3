from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors


CSV_PATH = "full_dataset_with_scores_new.csv"
INDEX_COL = "index"

MODEL_LIST = ["bge", "mpnet", "minilm", "qwen_embedding", "qwen2.5_3b_L24"]

K_LIST = [10, 15, 20, 30]
METRIC: Literal["euclidean", "cosine"] = "euclidean"
NORMALIZE_EMBEDDINGS = True

OUTPUT_ROOT = "outputs"
RANDOM_SEED = 0


@dataclass
class KNNMeta:
    csv_path: str
    embed_path: str
    model: str
    n_nodes: int
    dim: int
    k: int
    metric: str
    normalize_embeddings: bool
    symmetrize_rule: str
    random_seed: int
    connected_components: int
    largest_component_size: int
    average_degree: float
    density: float


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def build_symmetric_knn_graph(X: np.ndarray, k: int, metric: str) -> sparse.csr_matrix:
    if k < 1:
        raise ValueError("k must be >= 1")

    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k must be smaller than number of nodes. Got k={k}, n={n}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X, return_distance=True)

    dists = dists[:, 1:]
    nbrs = nbrs[:, 1:]

    edge_dict: dict[tuple[int, int], float] = {}

    for i in range(n):
        for m in range(k):
            j = int(nbrs[i, m])
            w = float(dists[i, m])

            if i == j:
                continue

            a, b = (i, j) if i < j else (j, i)

            if (a, b) not in edge_dict:
                edge_dict[(a, b)] = w
            else:
                edge_dict[(a, b)] = min(edge_dict[(a, b)], w)

    rows, cols, data = [], [], []
    for (a, b), w in edge_dict.items():
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([w, w])

    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.tolil()
    A.setdiag(0.0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def main():
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if INDEX_COL not in df.columns:
        raise ValueError(f"CSV missing INDEX_COL='{INDEX_COL}'")

    for model in MODEL_LIST:
        print("\n" + "=" * 60)
        print(f"Processing model: {model}")
        print("=" * 60)

        embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")

        if not os.path.exists(embed_path):
            print(f"[Skip] embeddings not found: {embed_path}")
            continue

        X = np.load(embed_path)

        if X.ndim != 2:
            raise ValueError(f"Embeddings must be N x D, got {X.shape}")

        if len(df) != X.shape[0]:
            raise ValueError(
                f"Row mismatch: CSV has {len(df)} rows, embeddings have {X.shape[0]} rows"
            )

        if NORMALIZE_EMBEDDINGS:
            X = l2_normalize_rows(X)

        norms = np.linalg.norm(X, axis=1)
        print(f"Embedding norm mean = {norms.mean():.6f}")
        print(f"Embedding norm std  = {norms.std():.6f}")

        for k in K_LIST:
            print("\n" + "-" * 50)
            print(f"Building graph for k = {k}")
            print("-" * 50)

            A = build_symmetric_knn_graph(X, k=k, metric=METRIC)

            n_comp, labels = connected_components(A, directed=False)
            largest_cc = int(np.bincount(labels).max())
            avg_degree = float(A.getnnz(axis=1).mean())
            density = float(A.nnz / (A.shape[0] ** 2))

            print(f"Connected components: {n_comp}")
            print(f"Largest component: {largest_cc}/{A.shape[0]}")
            print(f"Average degree: {avg_degree:.2f}")
            print(f"Density: {density:.6f}")

            out_dir = os.path.join(OUTPUT_ROOT, model, f"knn_k{k}_{METRIC}")
            os.makedirs(out_dir, exist_ok=True)

            sparse.save_npz(os.path.join(out_dir, "adjacency_csr.npz"), A)

            meta = KNNMeta(
                csv_path=CSV_PATH,
                embed_path=embed_path,
                model=model,
                n_nodes=int(X.shape[0]),
                dim=int(X.shape[1]),
                k=k,
                metric=METRIC,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
                symmetrize_rule="undirected union with min edge weight",
                random_seed=RANDOM_SEED,
                connected_components=int(n_comp),
                largest_component_size=largest_cc,
                average_degree=avg_degree,
                density=density,
            )

            with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(asdict(meta), f, indent=2)

            print(f"[Saved] {out_dir}")


if __name__ == "__main__":
    main()
