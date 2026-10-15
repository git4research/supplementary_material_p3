"""Run the steering pipeline for one model: scripts 11 -> 12 -> 13, then generation (Section 2.5).

--prepare runs activation extraction (11), layer selection (12; skipped if --layer is given) and
artifact building (13); --refine-layers repeats 11-13 on a finer grid under OUTPUT_ROOT/refinement.
Generation runs script 15 (--task smoke) or the task scripts 16, 17, or 17b + 17c.
Example: python run_steering_pipeline.py --model qwen --prepare --task smoke
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from layer_sweep import add_layer_sweep_arguments
from model_config import PROJECT_ROOT, add_model_arguments, experiment_from_args, get_model_spec


def run_script(filename, arguments):
    command = [sys.executable, "-B", str(PROJECT_ROOT / filename), *map(str, arguments)]
    print("Running:", filename, *map(str, arguments), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True,
                   env=dict(os.environ, PYTHONUNBUFFERED="1"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser, runtime=True)
    parser.add_argument("--prepare", action="store_true", help="Run extraction, layer selection and artifact preparation first")
    parser.add_argument("--prepare-only", action="store_true")
    add_layer_sweep_arguments(parser)
    parser.add_argument("--refine-layers", action="store_true",
                        help="After coarse preparation, add best-depth +/-5 percentage points in OUTPUT_ROOT/refinement")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--task", choices=["smoke", "triviaqa", "longform", "mental-health"], default="smoke")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--conditions", default="all")
    parser.add_argument("--alphas", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--n-samples", type=int)
    parser.add_argument("--temperature", type=float)
    args = parser.parse_args()
    has_candidates = args.layers is not None or args.layer_percentages is not None
    if args.layer is not None and (has_candidates or args.refine_layers):
        parser.error("--layer fixes one layer and cannot be combined with a candidate grid or --refine-layers")
    if (has_candidates or args.refine_layers) and not (args.prepare or args.prepare_only):
        parser.error("Candidate grids and --refine-layers require --prepare or --prepare-only")
    args.output_root = str(Path(args.output_root).expanduser().resolve())
    if args.prompts is not None:
        args.prompts = args.prompts.expanduser().resolve()
    if args.out is not None:
        args.out = args.out.expanduser().resolve()
    if args.task == "mental-health" and args.prompts is None and not args.prepare_only:
        parser.error("--task mental-health requires --prompts to select the intended prompt set")
    common = ["--model", args.model_id, "--output-root", args.output_root]
    if args.model_key:
        common.extend(["--model-key", args.model_key])
    runtime = ["--device", args.device, "--dtype", args.dtype]
    selected = ["--layer", args.layer] if args.layer is not None else []
    if args.prepare or args.prepare_only:
        extraction = [*common, *runtime, "--batch-size", args.batch_size]
        if args.layers is not None:
            extraction.extend(["--layers", ",".join(map(str, args.layers))])
        elif args.layer_percentages is not None:
            extraction.extend(["--layer-percentages", ",".join(map(str, args.layer_percentages))])
        elif args.layer is not None:
            extraction.extend(["--layers", args.layer])
        run_script("11_get_llm_activations.py", extraction)
        # With a fixed --layer, layer selection is skipped and that layer is used directly.
        if args.layer is None:
            run_script("12_select_steering_layer.py", common)
        run_script("13_build_steering_vectors.py", [*common, *selected])
        if args.refine_layers:
            # Refinement writes to a separate output root so the coarse sweep and its selection are preserved.
            spec = get_model_spec(args.model_id, args.model_key)
            source_selection = Path(args.output_root) / spec.key / "layer_selection/chosen_layer.json"
            args.output_root = str(Path(args.output_root) / "refinement")
            common = ["--model", args.model_id, "--output-root", args.output_root]
            if args.model_key:
                common.extend(["--model-key", args.model_key])
            print(f"Coarse results preserved. Refinement output root: {args.output_root}", flush=True)
            run_script("11_get_llm_activations.py", [*common, *runtime, "--batch-size", args.batch_size,
                                                   "--refine-from", source_selection])
            run_script("12_select_steering_layer.py", common)
            run_script("13_build_steering_vectors.py", common)
    if args.prepare_only:
        return
    # Resolve the steering layer: explicit --layer, else the script 12 selection under the current output root.
    config = experiment_from_args(args)
    generation = [*common, *runtime, "--layer", config.layer, "--conditions", args.conditions]
    for option in ("max_new_tokens", "n_samples", "temperature", "out"):
        value = getattr(args, option)
        if value is not None:
            generation.extend(["--" + option.replace("_", "-"), value])
    if args.task == "smoke":
        run_script("15_generate_with_steering.py", [*generation, "--smoke"])
    elif args.task == "triviaqa":
        prompts = args.prompts or PROJECT_ROOT / "triviaqa_prompts.json"
        run_script("17b_generate_triviaqa.py", [*generation, "--prompts", prompts, "--alphas", args.alphas])
        destination = args.out or config.generations_dir / "factuality_triviaqa.jsonl"
        run_script("17c_score_triviaqa.py", ["--in", destination,
                                           "--out-dir", config.artifact_dir / "figures"])
    elif args.task == "mental-health":
        run_script("16_generate_mental_health.py", [*generation, "--prompts", args.prompts, "--alphas", args.alphas])
    elif args.task == "longform":
        run_script("17_generate_factuality.py", [*generation, "--task", "longform", "--alphas", args.alphas])


if __name__ == "__main__":
    main()
