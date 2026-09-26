from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from model_config import add_model_arguments, experiment_from_args, generation_kwargs, resolve_experiment
from model_runtime import get_layer_module, load_model_and_tokenizer, pick_device, resolve_dtype

import importlib.util

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("steering_hooks", _HERE / "14_steering_hooks.py")
_steering = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_steering)
ConstantDirectionSteer = _steering.ConstantDirectionSteer
LocalManifoldSteer = _steering.LocalManifoldSteer
RandomDirectionSteer = _steering.RandomDirectionSteer
DEFAULT_MODEL_ID = _steering.DEFAULT_MODEL_ID
DEFAULT_MODEL_KEY = _steering.DEFAULT_MODEL_KEY
DEFAULT_LAYER = _steering.DEFAULT_LAYER

SteeringFactory = Callable[[object, float, int], Optional[object]]


def _baseline(model, alpha, max_gen_tokens, config=None):
    return None


def _global_ridge(model, alpha, max_gen_tokens, config=None):
    config = config or resolve_experiment()
    return ConstantDirectionSteer(
        model,
        alpha=alpha,
        direction_name="ridge_w",
        name="global_ridge",
        **config.hook_kwargs(),
    )


def _local_manifold(model, alpha, max_gen_tokens, config=None):
    config = config or resolve_experiment()
    return LocalManifoldSteer(
        model,
        alpha=alpha,
        name="local_manifold",
        **config.hook_kwargs(),
    )


def _random_steer(model, alpha, max_gen_tokens, config=None):
    config = config or resolve_experiment()
    return RandomDirectionSteer(
        model,
        alpha=alpha,
        seed=0,
        name="random_steer",
        layer_index_1based=config.layer,
    )


STEERING_PRESETS: Dict[str, SteeringFactory] = {
    "baseline": _baseline,
    "global_ridge": _global_ridge,
    "local_manifold": _local_manifold,
}

ALL_CONDITIONS = list(STEERING_PRESETS.keys())

CONDITIONS_WITHOUT_ALPHA = {"baseline"}


def _resolve_factory(cond: str, config=None) -> SteeringFactory:
    factory = _random_steer if cond == "random_steer" else STEERING_PRESETS.get(cond)
    if factory is not None:
        return partial(factory, config=config) if config is not None else factory
    raise ValueError(
        f"Unknown condition: {cond!r}.  Known: {list(STEERING_PRESETS.keys()) + ['random_steer']}."
    )


def parse_conditions(value):
    names = ALL_CONDITIONS if value.strip().lower() == "all" else [name.strip() for name in value.split(",")]
    if not names or len(set(names)) != len(names):
        raise ValueError("Conditions must be non-empty and unique.")
    for name in names:
        _resolve_factory(name)
    return names


def _apply_chat_template(tokenizer, prompt: str, device: str) -> torch.Tensor:
    chat = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(inputs, torch.Tensor):
        inputs = inputs["input_ids"]
    return inputs.to(device)


def generate_one(
        model,
        tokenizer,
        prompt: str,
        steering,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        seed: int,
        device: str,
) -> Dict:
    if steering is not None and hasattr(steering, "reset"):
        steering.reset()

    inputs = _apply_chat_template(tokenizer, prompt, device)
    attention_mask = torch.ones_like(inputs)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    do_sample = temperature is not None and temperature > 0.0

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            disable_compile=True,
        )
    wall = time.time() - t0

    new_token_ids = out[0][inputs.shape[1]:]
    response = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    return {
        "response": response,
        "n_tokens": int(new_token_ids.shape[0]),
        "wall_seconds": round(float(wall), 3),
    }


def _row_key(row: Dict) -> tuple:
    return (
        row["eval_name"],
        row["prompt_id"],
        row["condition"],
        row["alpha"],
        row["sample_idx"],
    )


def _load_existing_keys(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    keys = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                keys.add(_row_key(r))
            except json.JSONDecodeError:
                continue
    return keys


def _append_jsonl(jsonl_path: Path, row: Dict):
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stable_seed(prompt_id, sample_idx, seed_base):
    prompt_seed = int.from_bytes(hashlib.sha256(str(prompt_id).encode()).digest()[:4], "big")
    return (seed_base + 1000 * sample_idx + prompt_seed) % (2 ** 32)


def _check_resume(out_path, manifest, resume):
    manifest_path = out_path.with_suffix(out_path.suffix + ".run.json")
    if resume and out_path.exists() and out_path.stat().st_size:
        if not manifest_path.exists():
            raise ValueError("Legacy output lacks a run manifest. Use a new --out path; --no-resume explicitly replaces the old output.")
        previous = json.loads(manifest_path.read_text())
        if previous != manifest:
            changed = sorted(key for key in set(previous) | set(manifest) if previous.get(key) != manifest.get(key))
            raise ValueError(f"Resume configuration mismatch: {changed}. Use a new --out path.")
    return manifest_path


def run_generation_grid(
        prompts: List[Dict],
        out_path: str,
        eval_name: str,
        conditions: List[str] = None,
        alphas: List[float] = None,
        n_samples: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        model_id: str = DEFAULT_MODEL_ID,
        layer: int = None,
        seed_base: int = 42,
        resume: bool = True,
        model=None,
        tokenizer=None,
        model_key=None,
        output_root=None,
        device="auto",
        dtype="auto",
) -> Path:
    if conditions is None:
        conditions = ALL_CONDITIONS
    if alphas is None:
        alphas = [0.1, 0.2, 0.3, 0.5]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError("Conditions must be non-empty and unique.")
    if any(condition != "baseline" for condition in conditions) and not alphas:
        raise ValueError("Steered conditions require at least one alpha.")
    if len(set(alphas)) != len(alphas) or not all(np.isfinite(alpha) for alpha in alphas):
        raise ValueError("Alpha values must be unique and finite.")

    for c in conditions:
        _resolve_factory(c)

    config = resolve_experiment(model_id, layer, output_root, model_key)
    model_id = config.spec.model_id
    layer = config.layer
    if not prompts or n_samples < 1 or max_new_tokens < 1:
        raise ValueError("Non-empty prompts and positive sample/token counts are required.")
    if len({prompt["id"] for prompt in prompts}) != len(prompts):
        raise ValueError("Prompt IDs must be unique within a run.")
    artifact_hashes = {}
    if any(condition in ("global_ridge", "local_manifold") for condition in conditions):
        artifact_hashes = config.validate_artifacts(require_graph="local_manifold" in conditions)
    device = str(next(model.parameters()).device) if model is not None else pick_device(device)
    actual_dtype = next(model.parameters()).dtype if model is not None else resolve_dtype(dtype, device, model_id)
    manifest = {
        "schema_version": 1, "model_id": model_id, "model_key": config.artifact_key,
        "layer": layer, "max_new_tokens": max_new_tokens, "temperature": temperature,
        "top_p": top_p, "seed_base": seed_base, "seed_scheme": "sha256-v1",
        "dtype": str(actual_dtype), "device": device, "eval_name": eval_name,
        "prompts_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "artifacts_sha256": artifact_hashes,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy")},
    }

    out_path = Path(out_path)
    manifest_path = _check_resume(out_path, manifest, resume)
    if (model is None) != (tokenizer is None):
        raise ValueError("Provide both model and tokenizer, or neither.")
    if model is None:
        model, tokenizer = load_model_and_tokenizer(model_id, device, dtype)
    loaded_id = getattr(model.config, "_name_or_path", "")
    if loaded_id and loaded_id != model_id:
        raise ValueError(f"Provided model {loaded_id} differs from requested {model_id}.")
    get_layer_module(model, layer)
    for condition in conditions:
        _resolve_factory(condition, config)(model, 0.0, max_new_tokens)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not resume and out_path.exists():
        print(f"[no-resume] Truncating existing JSONL at {out_path}")
        out_path.unlink()

    manifest_path.write_text(json.dumps(manifest, indent=2))

    existing_keys = _load_existing_keys(out_path) if resume else set()
    if existing_keys:
        print(f"[resume] Found {len(existing_keys)} existing rows in {out_path}; will skip them.")

    plan = []
    for cond in conditions:
        if cond in CONDITIONS_WITHOUT_ALPHA:
            plan.append((cond, None))
        else:
            for a in alphas:
                plan.append((cond, float(a)))
    total_gens = sum(
        1 for (cond, alpha) in plan
        for p in prompts
        for s in range(n_samples)
        if _row_key({
            "eval_name": eval_name,
            "prompt_id": p["id"],
            "condition": cond,
            "alpha": alpha,
            "sample_idx": s,
        }) not in existing_keys
    )
    print(f"\nGeneration plan: {len(plan)} (condition,alpha) tuples x {len(prompts)} prompts x {n_samples} samples")
    print(f"  total generations to run: {total_gens} (after resume filter)")
    print(f"  output JSONL (absolute): {out_path.resolve()}")

    done = 0
    failed = 0
    t_start = time.time()

    for (cond, alpha) in plan:
        factory = _resolve_factory(cond, config)
        steer = factory(model, alpha if alpha is not None else 0.0, max_new_tokens)
        if steer is not None:
            steer.attach()
        try:
            for p in prompts:
                pid = p["id"]
                prompt_text = p["prompt"]
                pmeta = p.get("metadata", None)
                for s_idx in range(n_samples):
                    row_key = (eval_name, pid, cond, alpha, s_idx)
                    if row_key in existing_keys:
                        continue
                    seed = stable_seed(pid, s_idx, seed_base)
                    try:
                        gen = generate_one(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt_text,
                            steering=steer,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            seed=seed,
                            device=device,
                        )
                    except Exception as e:
                        print(f"  [error] cond={cond} alpha={alpha} pid={pid} s={s_idx}: {e}")
                        failed += 1
                        continue

                    row = {
                        "eval_name": eval_name,
                        "prompt_id": pid,
                        "prompt": prompt_text,
                        "prompt_metadata": pmeta,
                        "condition": cond,
                        "alpha": alpha,
                        "sample_idx": s_idx,
                        "seed": seed,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "model_id": model_id,
                        "model_key": config.artifact_key,
                        "dtype": str(actual_dtype),
                        "layer": layer,
                        "response": gen["response"],
                        "n_tokens": gen["n_tokens"],
                        "wall_seconds": gen["wall_seconds"],
                    }
                    _append_jsonl(out_path, row)
                    done += 1

                    if done % 5 == 0 or done == total_gens:
                        elapsed = time.time() - t_start
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remaining = (total_gens - done) / rate if rate > 0 else float("inf")
                        print(
                            f"  [{done}/{total_gens}] cond={cond:<32s} alpha={str(alpha):<5s} "
                            f"pid={pid:<10s} s={s_idx} "
                            f"({gen['n_tokens']} tok, {gen['wall_seconds']:.1f}s)  "
                            f"avg {rate:.2f} gen/s, ETA {remaining / 60:.1f} min"
                        )
        finally:
            if steer is not None:
                steer.detach()

    elapsed = time.time() - t_start
    print(f"\nDone. Wrote {done} new generations in {elapsed / 60:.1f} min.")
    try:
        with open(out_path, "r") as f:
            actual_rows = sum(1 for line in f if line.strip())
    except FileNotFoundError:
        actual_rows = 0
    print(f"Total rows in {out_path}: {actual_rows}")
    if failed:
        raise RuntimeError(f"{failed} generations failed; successful rows are saved and can be resumed.")
    return out_path


def _smoke_prompts() -> List[Dict]:
    return [
        {
            "id": "SMOKE01",
            "prompt": "I feel like nothing I do makes a difference. What should I do?",
            "metadata": {"category": "futility"},
        },
        {
            "id": "SMOKE02",
            "prompt": "What is the capital of Australia?",
            "metadata": {"category": "factual"},
        },
    ]


def _smoke_test(args):
    config = experiment_from_args(args)
    out_path = Path(args.out) if args.out else config.generations_dir / "smoke_multimodel.jsonl"
    run_generation_grid(
        prompts=_smoke_prompts(),
        out_path=str(out_path),
        eval_name="smoke",
        conditions=parse_conditions(args.conditions),
        alphas=[0.3],
        n_samples=1,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=min(120, args.max_new_tokens),
        resume=not args.no_resume,
        **generation_kwargs(args),
    )

    print("\n" + "=" * 80)
    print("Smoke test responses:")
    print("=" * 80)
    with open(out_path, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for pid in sorted({r["prompt_id"] for r in rows}):
        prompt = next(r for r in rows if r["prompt_id"] == pid)["prompt"]
        print(f"\n--- {pid}: {prompt!r} ---")
        for r in rows:
            if r["prompt_id"] != pid:
                continue
            label = r["condition"]
            if r["alpha"] is not None:
                label += f" (α={r['alpha']})"
            print(f"\n  [{label}]\n  {r['response']}")


def _parse_cli():
    p = argparse.ArgumentParser()
    add_model_arguments(p, runtime=True)
    p.add_argument("--prompts", type=str, help="Path to prompts JSON (list of {id, prompt, metadata})")
    p.add_argument("--out", type=str, help="Output JSONL path")
    p.add_argument("--eval-name", type=str, default="custom",
                   help="Used as eval_name in JSONL rows (e.g. 'mental_health', 'truthfulqa')")
    p.add_argument("--conditions", type=str, default="all",
                   help="Comma-separated condition names, or 'all'")
    p.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5",
                   help="Comma-separated alpha values")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--no-resume", action="store_true", help="Overwrite existing JSONL")
    p.add_argument("--smoke", action="store_true", help="Run the built-in smoke test and exit")
    return p.parse_args()


def main():
    args = _parse_cli()

    if args.smoke:
        _smoke_test(args)
        return

    if not args.prompts or not args.out:
        print("Either pass --smoke or provide both --prompts and --out", file=sys.stderr)
        sys.exit(2)

    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("--prompts file must contain a non-empty JSON list")

    conditions = parse_conditions(args.conditions)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    run_generation_grid(
        prompts=prompts,
        out_path=args.out,
        eval_name=args.eval_name,
        conditions=conditions,
        alphas=alphas,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        **generation_kwargs(args),
        seed_base=args.seed_base,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
