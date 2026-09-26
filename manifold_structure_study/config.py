from pathlib import Path

EMBEDDINGS_DIR = Path("all_embeddings")

RESULTS_DIR = Path("results")

SEED = 0

VARIANCE_THRESHOLDS = (0.80, 0.90, 0.95, 0.99)

TWONN_DISCARD_FRACTION = 0.10

WITH_MLE = False

COVARIANCE_REPEATS = 20

DECIMATION_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DECIMATION_REPEATS = 20

LABEL_OVERRIDES = {
    "bge": "BGE",
    "gemma2_9B": "Gemma2-9B",
    "gemma2_27B": "Gemma2-27B",
    "llama3.1_8B": "Llama3.1-8B",
    "miniLM": "MiniLM",
    "mpnet": "MPNet",
    "qwen2.5_3B": "Qwen2.5-3B",
    "qwen2.5_7B": "Qwen2.5-7B",
    "qwen2.5_14B": "Qwen2.5-14B",
    "qwen_embeddings": "Qwen-Embedding",
}
