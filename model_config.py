"""Model registry, layer resolution and artifact paths for the steering experiments (Section 2.5).

Maps CLI aliases (qwen, llama, gemma) or Hugging Face IDs plus --model-key to artifact keys,
resolves the steering layer (explicit --layer, else script 12's chosen_layer.json, else the
registry default), and checks that script 13 artifacts exist and match the model/layer.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    key: str
    default_layer: int | None = None


# Registered aliases; other models are passed as a Hugging Face ID together with --model-key.
MODEL_REGISTRY = {
    "qwen": ModelSpec("Qwen/Qwen2.5-3B-Instruct", "qwen2.5_3b", 24),
    "llama": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama3.1_8b"),
    "gemma": ModelSpec("google/gemma-2-9b-it", "gemma2_9b"),
}


def get_model_spec(model_id="qwen", model_key=None):
    """Return the ModelSpec for an alias, HF ID or key; unregistered IDs need a unique model_key."""
    for alias, spec in MODEL_REGISTRY.items():
        if model_id in (alias, spec.model_id, spec.key):
            if model_key is not None and model_key != spec.key:
                if any(other.key == model_key for other in MODEL_REGISTRY.values()):
                    raise ValueError("That --model-key belongs to a different registered model.")
                return ModelSpec(spec.model_id, _validate_key(model_key))
            return spec
    if not model_key:
        raise ValueError("An unregistered --model requires a unique --model-key.")
    if any(spec.key == model_key for spec in MODEL_REGISTRY.values()):
        raise ValueError("That --model-key belongs to a different registered model.")
    return ModelSpec(model_id, _validate_key(model_key))


def _validate_key(key):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", key):
        raise ValueError("--model-key must be a directory name, without slashes.")
    return key


@dataclass(frozen=True)
class ExperimentConfig:
    """One (model, layer) experiment; steering artifacts live under OUTPUT_ROOT/<key>_L<layer>."""
    spec: ModelSpec
    layer: int
    output_root: Path

    @property
    def artifact_key(self):
        return f"{self.spec.key}_L{self.layer}"

    @property
    def model_dir(self):
        return self.output_root / self.spec.key

    @property
    def artifact_dir(self):
        return self.output_root / self.artifact_key

    @property
    def generations_dir(self):
        return self.artifact_dir / "generations"

    def hook_kwargs(self):
        return {
            "layer_index_1based": self.layer,
            "model_key": self.artifact_key,
            "output_root": str(self.output_root),
        }

    def validate_artifacts(self, require_graph=True):
        """Check script 13 artifacts and provenance; return SHA-256 digests for the run manifest."""
        required = [self.artifact_dir / "steering/directions/ridge_w.npy"]
        if require_graph:
            required.extend([
                self.artifact_dir / "embeddings.npy",
                self.artifact_dir / "knn_k30_euclidean/adjacency_csr.npz",
            ])
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}. Run script 13 for this model/layer.")
        manifest_path = self.artifact_dir / "artifact_meta.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            expected = {"model_id": self.spec.model_id, "layer": self.layer,
                        "activation_space": "block_output"}
            for field, value in expected.items():
                if manifest.get(field) != value:
                    raise ValueError(f"Artifact {field} mismatch in {manifest_path}")
            required.append(manifest_path)
        # Without artifact_meta.json, only the legacy Qwen2.5-3B layer-24 artifacts are accepted.
        elif self.spec.model_id != MODEL_REGISTRY["qwen"].model_id or self.artifact_key != "qwen2.5_3b_L24":
            raise FileNotFoundError(f"Missing provenance: {manifest_path}. Run script 13.")
        return {str(path.relative_to(self.artifact_dir)): file_digest(path) for path in required}


def resolve_experiment(model_id="qwen", layer=None, output_root=None, model_key=None):
    """Build an ExperimentConfig; the layer defaults to script 12's selection, then the registry default."""
    spec = get_model_spec(model_id, model_key)
    root = Path(output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    if layer is None:
        selection_path = root / spec.key / "layer_selection/chosen_layer.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection.get("model_key") != spec.key:
                raise ValueError(f"Model key mismatch in {selection_path}")
            if selection.get("model_id", spec.model_id) != spec.model_id:
                raise ValueError(f"Model identity mismatch in {selection_path}")
            layer = selection["recommended_layer"]
        else:
            layer = spec.default_layer
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 1:
        raise ValueError(f"Select a layer for {spec.key}: run script 12 or pass --layer.")
    return ExperimentConfig(spec, layer, root)


def add_model_arguments(parser, *, include_layer=True, runtime=False):
    parser.add_argument("--model", "--model-id", dest="model_id", default="qwen",
                        help="qwen, llama, gemma, or a Hugging Face model ID")
    parser.add_argument("--model-key", default=None,
                        help="Base artifact directory key for an unregistered model")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    if include_layer:
        parser.add_argument("--layer", type=int, default=None,
                            help="1-based block output; default: script 12 selection")
    if runtime:
        parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")


def experiment_from_args(args):
    return resolve_experiment(args.model_id, args.layer, args.output_root, args.model_key)


def generation_kwargs(args):
    return {"model_id": args.model_id, "model_key": args.model_key,
            "layer": args.layer, "output_root": args.output_root,
            "device": args.device, "dtype": args.dtype}


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dataset_provenance(meta, csv_path):
    """Raise if the CSV differs from the one hashed at activation extraction (when a hash was recorded)."""
    if meta.get("dataset_sha256") and meta["dataset_sha256"] != file_digest(csv_path):
        raise ValueError("Dataset content/order differs from the activation extraction dataset.")


def validate_single_experiment(rows):
    identities = {(row.get("model_id"), row.get("model_key"), row.get("layer")) for row in rows}
    if len(identities) > 1:
        raise ValueError("Mixed models/layers in one input. Score each experiment separately.")
