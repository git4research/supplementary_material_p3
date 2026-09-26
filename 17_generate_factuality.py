from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, List
from model_config import add_model_arguments, experiment_from_args, generation_kwargs


_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("gen_iface", _HERE / "15_generate_with_steering.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


SIMPLEQA_URL = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
SIMPLEQA_CACHE = _HERE / "simpleqa_test_set.csv"

LONGFORM_PROMPTS = _HERE / "longform_factuality_prompts.json"

SIMPLEQA_N_FULL  = 100
SIMPLEQA_N_SMALL = 5
SIMPLEQA_SEED    = 0

LONGFORM_SMALL_IDS = ["LF001", "LF008", "LF013", "LF018", "LF024"]


def download_simpleqa(cache_path: Path = SIMPLEQA_CACHE) -> Path:
    if cache_path.exists():
        return cache_path
    print(f"Downloading SimpleQA from {SIMPLEQA_URL} ...")
    with urllib.request.urlopen(SIMPLEQA_URL) as resp:
        data = resp.read()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    print(f"  cached at: {cache_path} ({len(data)/1024:.1f} KB)")
    return cache_path


def load_simpleqa_subset(n: int, seed: int, cache_path: Path = SIMPLEQA_CACHE) -> List[Dict]:
    csv_path = download_simpleqa(cache_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  Loaded {len(rows)} SimpleQA rows from CSV")

    if not rows:
        raise RuntimeError("SimpleQA CSV is empty")
    cols = list(rows[0].keys())
    if "problem" not in cols or "answer" not in cols:
        raise RuntimeError(f"Unexpected SimpleQA columns: {cols}")

    rng = random.Random(seed)
    sampled = rng.sample(rows, k=min(n, len(rows)))

    prompts = []
    for i, r in enumerate(sampled):
        prompts.append({
            "id": f"SQA{i+1:03d}",
            "prompt": r["problem"],
            "metadata": {
                "ground_truth": r["answer"],
                "raw_metadata": r.get("metadata", ""),
                "source": "simpleqa",
            },
        })
    return prompts


def load_longform_prompts(small: bool) -> List[Dict]:
    if not LONGFORM_PROMPTS.exists():
        raise FileNotFoundError(f"Missing long-form prompts: {LONGFORM_PROMPTS}")
    with open(LONGFORM_PROMPTS) as f:
        all_prompts = json.load(f)
    if not small:
        return all_prompts
    sel = [p for p in all_prompts if p["id"] in LONGFORM_SMALL_IDS]
    missing = set(LONGFORM_SMALL_IDS) - {p["id"] for p in sel}
    if missing:
        raise RuntimeError(f"Long-form small-mode prompt ids not found: {missing}")
    return sel


def _print_small_summary(jsonl_path: Path, task: str, prompts: List[Dict]):
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta_by_id = {p["id"]: p.get("metadata", {}) for p in prompts}

    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    print("\n" + "=" * 88)
    print(f"SMALL-MODE RESPONSES  (task={task})")
    print("=" * 88)
    for pid in sorted(by_prompt.keys()):
        meta = meta_by_id.get(pid, {})
        prompt = by_prompt[pid][0]["prompt"]
        print(f"\n[{pid}]  {prompt!r}")
        if "ground_truth" in meta:
            print(f"  GROUND TRUTH: {meta['ground_truth']!r}")
        for r in by_prompt[pid]:
            label = r["condition"]
            if r["alpha"] is not None:
                label += f" (α={r['alpha']})"
            resp = r["response"].strip()
            if task == "longform" and len(resp) > 400:
                resp = resp[:400] + " ..."
            print(f"\n  [{label}]")
            for line in resp.splitlines():
                print(f"    {line}")
    print("\n" + "=" * 88)


def _parse_args():
    p = argparse.ArgumentParser()
    add_model_arguments(p, runtime=True)
    p.add_argument("--conditions", default="all")
    p.add_argument("--task", choices=["simpleqa", "longform"], required=True,
                   help="Which factuality task to run")
    p.add_argument("--small", action="store_true",
                   help="Preview: 5 prompts x alpha=0.3 x 1 sample (~5 min)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSONL path (default: derived from --task and --small)")
    p.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5",
                   help="Alpha values for full run (comma-separated)")
    p.add_argument("--n-samples", type=int, default=2,
                   help="Samples per (prompt, condition, alpha) cell (default 2 for factuality)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--n-simpleqa", type=int, default=SIMPLEQA_N_FULL,
                   help=f"How many SimpleQA questions in full mode (default {SIMPLEQA_N_FULL})")
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Override generation length cap (default 80 for simpleqa, 300 for longform)")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite existing JSONL instead of appending/resuming")
    return p.parse_args()


def main():
    args = _parse_args()
    config = experiment_from_args(args)
    conditions = _gen.parse_conditions(args.conditions)
    task = args.task

    if task == "simpleqa":
        n = SIMPLEQA_N_SMALL if args.small else args.n_simpleqa
        prompts = load_simpleqa_subset(n=n, seed=SIMPLEQA_SEED,
                                      cache_path=Path(args.output_root) / "datasets" / "simpleqa_test_set.csv")
        default_max_new = 80
        out_basename = "factuality_simpleqa"
        eval_name = "factuality_simpleqa"
    elif task == "longform":
        prompts = load_longform_prompts(small=args.small)
        default_max_new = 300
        out_basename = "factuality_longform"
        eval_name = "factuality_longform"
    else:
        raise ValueError(f"Unknown task: {task}")

    print(f"Task: {task}.  Loaded {len(prompts)} prompts.")

    if args.small:
        alphas = [0.3]
        n_samples = 1
        default_out = config.generations_dir / f"{out_basename}_small.jsonl"
    else:
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        n_samples = args.n_samples
        default_out = config.generations_dir / f"{out_basename}.jsonl"

    out_path = Path(args.out) if args.out else default_out
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_new

    n_steered = sum(condition != "baseline" for condition in conditions)
    n_combos = int("baseline" in conditions) + n_steered * len(alphas)
    n_gens = len(prompts) * n_combos * n_samples
    sec_per = 8 if task == "simpleqa" else 30
    eta_min = n_gens * sec_per / 60.0
    mode_label = "SMALL" if args.small else "FULL"
    print(f"\n[{mode_label} MODE -- {task}]")
    print(f"  prompts:        {len(prompts)}")
    print(f"  conditions:     {conditions}")
    print(f"  alphas:         {alphas}  ->  {n_combos} (condition,alpha) tuples")
    print(f"  samples:        {n_samples}")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"  total gens:     {n_gens}")
    print(f"  ETA:            ~{eta_min/60:.1f} hours (rough, at ~{sec_per}s/gen)")
    print(f"  output:         {out_path}")

    _gen.run_generation_grid(
        prompts=prompts,
        out_path=str(out_path),
        eval_name=eval_name,
        conditions=conditions,
        alphas=alphas,
        n_samples=n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_new_tokens=max_new_tokens,
        resume=not args.no_resume,
        **generation_kwargs(args),
    )

    if args.small:
        _print_small_summary(out_path, task=task, prompts=prompts)


if __name__ == "__main__":
    main()
