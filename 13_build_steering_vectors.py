import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

from model_config import PROJECT_ROOT, add_model_arguments, check_dataset_provenance, experiment_from_args, file_digest


def fit_ridge_score_direction(embeddings, scores, alpha):
    probe = Ridge(alpha=alpha)
    probe.fit(embeddings, scores)
    direction = probe.coef_.astype(np.float32)
    if float(np.linalg.norm(direction)) < 1e-12:
        raise RuntimeError("Ridge produced a zero direction.")
    return direction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "full_dataset_with_scores_new.csv")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()
    config = experiment_from_args(args)
    meta = json.loads((config.model_dir / "activation_meta.json").read_text())
    if meta["model_id"] != config.spec.model_id:
        raise ValueError("Activation model differs from the selected checkpoint.")
    if config.layer not in meta["layers_saved"]:
        raise ValueError("The selected layer is absent from the current activation extraction.")
    if meta.get("activation_space") != "block_output":
        if config.spec.model_id != "Qwen/Qwen2.5-3B-Instruct" or config.layer >= meta["n_hidden_layers"]:
            raise ValueError("Re-extract with script 11 to align activations with the block-output hooks.")
    check_dataset_provenance(meta, args.csv)
    dataframe = pd.read_csv(args.csv, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.str.strip()
    scores = dataframe["score"].to_numpy(dtype=float)
    source = config.model_dir / f"layer_{config.layer:02d}" / "embeddings.npy"
    embeddings = np.load(source).astype(np.float32)
    if embeddings.shape != (len(scores), meta["hidden_dim"]):
        raise ValueError("Activation shape does not match dataset/model metadata.")
    if not np.isfinite(embeddings).all() or not np.isfinite(scores).all():
        raise ValueError("Activations and scores must be finite.")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
        raise ValueError("Expected L2-normalized activation rows from script 11.")

    module_spec = importlib.util.spec_from_file_location("knn_builder", PROJECT_ROOT / "3_build_knn.py")
    builder = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = builder
    module_spec.loader.exec_module(builder)
    graph = builder.build_symmetric_knn_graph(embeddings, k=30, metric="euclidean")
    direction = fit_ridge_score_direction(embeddings, scores, args.ridge_alpha)
    correlation = float(np.corrcoef(embeddings @ direction, scores)[0, 1])

    root = config.artifact_dir
    (root / "steering/directions").mkdir(parents=True, exist_ok=True)
    (root / "knn_k30_euclidean").mkdir(parents=True, exist_ok=True)
    np.save(root / "embeddings.npy", embeddings)
    np.save(root / "steering/directions/ridge_w.npy", direction)
    sparse.save_npz(root / "knn_k30_euclidean/adjacency_csr.npz", graph)
    provenance = {
        "model_id": config.spec.model_id, "model_key": config.artifact_key,
        "layer": config.layer, "activation_space": "block_output",
        "hidden_dim": int(embeddings.shape[1]), "n_samples": len(scores),
        "dataset_sha256": file_digest(args.csv), "source_embeddings": str(source),
        "source_sha256": file_digest(source), "source_activation_meta": meta,
        "knn_k": 30, "knn_metric": "euclidean",
    }
    (root / "artifact_meta.json").write_text(json.dumps(provenance, indent=2))
    steering_meta = {
        "model_id": config.spec.model_id, "model_key": config.artifact_key,
        "layer": config.layer, "embeddings_path": str(root / "embeddings.npy"),
        "n_samples": len(scores), "embedding_dim": int(embeddings.shape[1]),
        "ridge": {"alpha": args.ridge_alpha, "corr_pred_vs_score": correlation},
        "embedding_space_note": "Normalized block-output anchors; unnormalized ridge coefficients; match_hidden hook scaling.",
    }
    (root / "steering/meta.json").write_text(json.dumps(steering_meta, indent=2))
    (root / "knn_k30_euclidean/meta.json").write_text(json.dumps({
        "model": config.artifact_key, "n_nodes": len(scores), "dim": int(embeddings.shape[1]),
        "k": 30, "metric": "euclidean", "normalize_embeddings": True,
        "symmetrize_rule": "undirected union with min edge weight",
    }, indent=2))
    print(f"Prepared {root}; ridge corr(pred, score)={correlation:.4f}")


if __name__ == "__main__":
    main()
