import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from layer_sweep import add_layer_sweep_arguments, make_layer_sweep, make_refinement_sweep
from model_config import PROJECT_ROOT, add_model_arguments, check_dataset_provenance, file_digest, get_model_spec
from model_runtime import decoder_layers, load_model_and_tokenizer, pick_device


def pool_hidden(hidden, attention_mask, mode):
    if mode == "last_token":
        return hidden[:, -1, :]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
    if mode == "last_nonpad":
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        indices = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(dim=1).values
        if torch.any(indices < 0):
            raise ValueError("Cannot pool an entirely padded sequence.")
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]
    raise ValueError(f"Unknown pooling mode: {mode}")


def collect_block_activations(model, encoding, layers_to_save, pooling):
    pooled_outputs = {}
    handles = []

    def make_capture(layer_number):
        def capture(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pooled_outputs[layer_number] = pool_hidden(
                hidden, encoding["attention_mask"], pooling
            ).detach().float().cpu().numpy().copy()
        return capture

    blocks = decoder_layers(model)
    try:
        for layer_number in layers_to_save:
            if not 1 <= layer_number <= len(blocks):
                raise ValueError(f"Layer {layer_number} outside 1..{len(blocks)}")
            handles.append(blocks[layer_number - 1].register_forward_hook(make_capture(layer_number)))
        with torch.no_grad():
            model(**encoding, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    if set(pooled_outputs) != set(layers_to_save):
        raise RuntimeError("Not all requested transformer blocks executed.")
    return pooled_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, include_layer=False, runtime=True)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "full_dataset_with_scores_new.csv")
    add_layer_sweep_arguments(parser, refinement_source=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--pooling", choices=["last_nonpad", "mean", "last_token"], default="last_nonpad")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        parser.error("--batch-size and --max-length must be positive")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    spec = get_model_spec(args.model_id, args.model_key)
    output_dir = Path(args.output_root).expanduser().resolve() / spec.key
    coarse_selection = None
    if args.refine_from:
        source_path = Path(args.refine_from).expanduser().resolve()
        source_model_dir = source_path.parent.parent
        if output_dir == source_model_dir:
            raise ValueError("Refinement needs a separate --output-root to preserve the coarse sweep.")
        source_meta_path = source_model_dir / "activation_meta.json"
        coarse_selection = json.loads(source_path.read_text())
        source_meta = json.loads(source_meta_path.read_text())
        for metadata in (coarse_selection, source_meta):
            if metadata.get("model_id") != spec.model_id or metadata.get("model_key") != spec.key:
                raise ValueError("Refinement source belongs to a different model.")
            check_dataset_provenance(metadata, args.csv)
        if coarse_selection.get("activation_meta_sha256") != file_digest(source_meta_path):
            raise ValueError("Coarse selection is stale or lacks provenance; rerun script 12 first.")
    dataframe = pd.read_csv(args.csv, encoding="utf-8-sig")
    dataframe.columns = dataframe.columns.str.strip()
    sentences = dataframe["sentence"].astype(str).tolist()
    if not sentences:
        raise ValueError("The input dataset is empty.")
    device = pick_device(args.device)
    model, tokenizer = load_model_and_tokenizer(spec.model_id, device, args.dtype)
    tokenizer.padding_side = "right"
    n_layers = len(decoder_layers(model))
    hidden_dim = model.config.hidden_size
    if coarse_selection is None:
        sweep = make_layer_sweep(n_layers, layers=args.layers, percentages=args.layer_percentages)
    else:
        expected = {"pooling": args.pooling, "max_length": args.max_length,
                    "seed": args.seed, "dtype": str(next(model.parameters()).dtype),
                    "activation_space": "block_output", "normalize_after_pool": True}
        for field, value in expected.items():
            if source_meta.get(field) != value:
                raise ValueError(f"Refinement extraction setting differs from coarse sweep: {field}")
        sweep = make_refinement_sweep(n_layers, coarse_selection)
        sweep.update({"source_selection_path": str(source_path),
                      "source_selection_sha256": file_digest(source_path),
                      "source_activation_meta_sha256": file_digest(source_meta_path)})
        print(f"Refinement adds {len(sweep['new_layers'])} unique layers: {sweep['new_layers']}")
    layers = sweep["layers"]
    print(f"Extracting {len(sentences)} sentences, layers={layers}, dim={hidden_dim}")
    print("Actual depths (%):", {layer: round(value, 3) for layer, value in sweep["actual_percentages"].items()})
    activations = {layer: np.zeros((len(sentences), hidden_dim), dtype=np.float32) for layer in layers}
    for start in tqdm(range(0, len(sentences), args.batch_size)):
        end = min(start + args.batch_size, len(sentences))
        encoding = tokenizer(sentences[start:end], return_tensors="pt", padding=True,
                             truncation=True, max_length=args.max_length).to(device)
        pooled = collect_block_activations(model, encoding, layers, args.pooling)
        for layer, values in pooled.items():
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite activations at layer {layer}; check dtype.")
            activations[layer][start:end] = values

    saved_paths = {}
    for layer, values in activations.items():
        values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        destination = output_dir / f"layer_{layer:02d}" / "embeddings.npy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, values)
        saved_paths[str(layer)] = str(destination)
        print(f"Saved {destination}: {values.shape}")
    meta = {
        "model_id": spec.model_id, "model_key": spec.key,
        "n_sentences": len(sentences), "hidden_dim": hidden_dim,
        "n_hidden_layers": n_layers, "layers_saved": layers,
        "layer_sweep": sweep,
        "pooling": args.pooling, "normalize_after_pool": True,
        "activation_space": "block_output", "max_length": args.max_length,
        "batch_size": args.batch_size, "dtype": str(next(model.parameters()).dtype),
        "device": device, "csv_path": str(args.csv.resolve()),
        "dataset_sha256": file_digest(args.csv), "saved_paths": saved_paths,
        "seed": args.seed,
    }
    (output_dir / "activation_meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
