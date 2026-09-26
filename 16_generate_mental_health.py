from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from model_config import add_model_arguments, experiment_from_args, generation_kwargs


_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("gen_iface", _HERE / "15_generate_with_steering.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)


PROMPTS_PATH    = _HERE / "mental_health_prompts.json"


SMALL_PROMPT_IDS = ["MH001", "MH009", "MH016", "MH020", "MH027"]


def _parse_args():
    p = argparse.ArgumentParser(description="Mental-health response generation (script 16).  Judging is done by script 18.")
    add_model_arguments(p, runtime=True)
    p.add_argument("--prompts", type=Path, default=PROMPTS_PATH)
    p.add_argument("--conditions", default="all")
    p.add_argument("--small", action="store_true",
                   help="Preview mode: 5 representative prompts × α=0.3 × 1 sample (~5 min)")
    p.add_argument("--out", type=str, default=None,
                   help="Output JSONL path (default: mental_health.jsonl or mental_health_small.jsonl)")
    p.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5",
                   help="Alpha values for full run (comma-separated)")
    p.add_argument("--n-samples", type=int, default=3,
                   help="Samples per (prompt, condition, alpha) cell (default 3)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Generation length cap (default 200, ~150-180 actual)")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite existing JSONL instead of appending/resuming")
    p.add_argument("--print-responses", action="store_true",
                   help="After --small finishes, print all responses inline (recommended)")
    return p.parse_args()


def _print_small_summary(jsonl_path: Path, prompts_path: Path):
    with open(jsonl_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with open(prompts_path) as f:
        meta_by_id = {p["id"]: p.get("metadata", {}) for p in json.load(f)}

    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    print("\n" + "=" * 88)
    print("SMALL-MODE RESPONSES")
    print("=" * 88)
    for pid in sorted(by_prompt.keys()):
        meta = meta_by_id.get(pid, {})
        tier = meta.get("tier", "?")
        cat = meta.get("category", "?")
        prompt = by_prompt[pid][0]["prompt"]
        print(f"\n[{pid}]  tier={tier}  category={cat}")
        print(f"  PROMPT: {prompt!r}")
        for r in by_prompt[pid]:
            label = r["condition"]
            if r["alpha"] is not None:
                label += f" (α={r['alpha']})"
            print(f"\n  [{label}]")
            for line in r["response"].splitlines():
                print(f"    {line}")
    print("\n" + "=" * 88)


def main():
    args = _parse_args()
    config = experiment_from_args(args)
    conditions = _gen.parse_conditions(args.conditions)

    if not args.prompts.exists():
        raise FileNotFoundError(f"Prompts file not found: {args.prompts}. Select a dataset with --prompts (mh_diag_prompts.json is available).")
    with open(args.prompts) as f:
        all_prompts = json.load(f)
    print(f"Loaded {len(all_prompts)} prompts from {args.prompts.name}")

    n_steered = sum(condition != "baseline" for condition in conditions)

    if args.small:
        prompts = [p for p in all_prompts if p["id"] in SMALL_PROMPT_IDS]
        if len(prompts) != len(SMALL_PROMPT_IDS):
            missing = set(SMALL_PROMPT_IDS) - {p["id"] for p in prompts}
            raise RuntimeError(f"Small-mode prompt ids not found in prompts file: {missing}")
        alphas = [0.3]
        n_samples = 1
        out_path = Path(args.out) if args.out else config.generations_dir / "mental_health_small.jsonl"

        n_combos = int("baseline" in conditions) + n_steered
        n_gens = len(prompts) * n_combos * n_samples
        print(f"\n[SMALL MODE]")
        print(f"  prompts:    {len(prompts)} (one per tier band)")
        print(f"  conditions: {conditions}")
        print(f"  alphas:     {alphas}")
        print(f"  samples:    {n_samples}")
        print(f"  total gens: {n_gens}")
        print(f"  output:     {out_path}")

    else:
        prompts = all_prompts
        alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
        n_samples = args.n_samples
        out_path = Path(args.out) if args.out else config.generations_dir / "mental_health.jsonl"

        n_steered_combos = n_steered * len(alphas)
        n_combos = int("baseline" in conditions) + n_steered_combos
        n_gens = len(prompts) * n_combos * n_samples
        eta_min = n_gens * 20 / 60
        print(f"\n[FULL MODE]")
        print(f"  prompts:    {len(prompts)}")
        print(f"  conditions: {conditions}")
        print(f"  alphas:     {alphas}  -> {n_combos} (condition,alpha) tuples")
        print(f"  samples:    {n_samples}")
        print(f"  total gens: {n_gens}")
        print(f"  ETA:        ~{eta_min/60:.1f} hours (rough, at 20s/gen)")
        print(f"  output:     {out_path}")

    _gen.run_generation_grid(
        prompts=prompts,
        out_path=str(out_path),
        eval_name="mental_health",
        conditions=conditions,
        alphas=alphas,
        n_samples=n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_new_tokens=args.max_new_tokens,
        resume=not args.no_resume,
        **generation_kwargs(args),
    )

    if args.small and (args.print_responses or True):
        _print_small_summary(out_path, args.prompts)


if __name__ == "__main__":
    main()
