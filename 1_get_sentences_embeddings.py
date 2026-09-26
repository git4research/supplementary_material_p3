import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


CSV_PATH = "full_dataset_with_scores_new.csv"
TEXT_COL = "sentence"

OUTPUT_ROOT = "outputs"

BATCH_SIZE = 32
NORMALIZE = True


MODEL_CONFIGS = {
    "bge": "BAAI/bge-large-en-v1.5",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen_embedding": "Qwen/Qwen3-Embedding-0.6B",
}


def generate_embeddings(model_name, model_id, sentences):
    print(f"\n==============================")
    print(f"Loading model: {model_id}")
    print(f"==============================")

    model = SentenceTransformer(model_id)

    embeddings = model.encode(
        sentences,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=NORMALIZE,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[OK] {model_name} shape: {embeddings.shape}")
    print(f"norm mean = {norms.mean():.6f}, std = {norms.std():.6f}")

    return embeddings


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Dataset shape: {df.shape}")

    if TEXT_COL not in df.columns:
        raise ValueError(f"Column '{TEXT_COL}' not found in CSV")

    sentences = df[TEXT_COL].astype(str).tolist()

    for model_key, model_id in MODEL_CONFIGS.items():
        model_dir = os.path.join(OUTPUT_ROOT, model_key)
        os.makedirs(model_dir, exist_ok=True)

        out_path = os.path.join(model_dir, "embeddings.npy")

        if os.path.exists(out_path):
            print(f"[Skip] {model_key}/embeddings.npy already exists")
            continue

        embeddings = generate_embeddings(model_key, model_id, sentences)

        np.save(out_path, embeddings)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
