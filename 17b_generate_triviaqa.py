from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from model_config import add_model_arguments, experiment_from_args, generation_kwargs

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("gen_iface", _HERE / "15_generate_with_steering.py")
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

def main():
    p = argparse.ArgumentParser()
    add_model_arguments(p, runtime=True)
    p.add_argument("--conditions", default="all")
    p.add_argument("--prompts", type=str, required=True,
                   help="Prompts JSON from 17b_fetch_triviaqa.py")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--alphas", type=str, default="0.1,0.2,0.3,0.5")
    p.add_argument("--n-samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0.0 = greedy (cleaner for factual recall).")
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    config = experiment_from_args(args)
    conditions = _gen.parse_conditions(args.conditions)

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise SystemExit(f"Prompts file not found: {prompts_path}. Run 17b_fetch_triviaqa.py first.")
    prompts = json.loads(prompts_path.read_text())
    if not isinstance(prompts, list) or not prompts:
        raise SystemExit("Prompts file must be a non-empty JSON list.")
    print(f"Loaded {len(prompts)} prompts from {prompts_path.name}")
    print(f"  example: {prompts[0]['metadata'].get('question')!r} -> "
          f"answer={prompts[0]['metadata'].get('answer')!r}")

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    out_path = Path(args.out) if args.out else (config.generations_dir / "factuality_triviaqa.jsonl")

    n_steered = sum(condition != "baseline" for condition in conditions)
    n_combos = int("baseline" in conditions) + n_steered * len(alphas)
    n_gens = len(prompts) * n_combos * args.n_samples
    print(f"\n[TriviaQA run]")
    print(f"  conditions: {conditions}")
    print(f"  alphas:     {alphas}  -> {n_combos} (condition,alpha) tuples")
    print(f"  samples:    {args.n_samples}   max_new_tokens: {args.max_new_tokens}   temperature: {args.temperature}")
    print(f"  total gens: {n_gens}   (~{n_gens*8/60:.0f} min at ~8s/gen)")
    print(f"  output:     {out_path}")

    _gen.run_generation_grid(
        prompts=prompts,
        out_path=str(out_path),
        eval_name="triviaqa",
        conditions=conditions,
        alphas=alphas,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=0.95,
        max_new_tokens=args.max_new_tokens,
        resume=not args.no_resume,
        **generation_kwargs(args),
    )


if __name__ == "__main__":
    main()
