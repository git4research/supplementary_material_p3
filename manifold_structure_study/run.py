"""Command-line entry point for the manifold-structure study (Appendix D and G).

Loads all embedding sets, computes PCA / participation-ratio / intrinsic-dimension
diagnostics with reference controls, runs the sampling-scale (decimation) test and
optional synthetic validation, and writes tables and figures to the results directory.
Usage: python run.py --embeddings-dir all_embeddings --results-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
import data
import analysis
import figures
import validate


def _parse_args():
    """Parse command-line options; defaults come from config.py."""
    p = argparse.ArgumentParser(description="Manifold-structure study (PCA vs intrinsic dim).")
    p.add_argument("--embeddings-dir", type=Path, default=config.EMBEDDINGS_DIR,
                   help="directory of *.npy embedding files")
    p.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR,
                   help="output directory for tables and figures")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--covariance-repeats", type=int, default=config.COVARIANCE_REPEATS,
                   help="Gaussian covariance-control draws (>=2; default 20); table reports TwoNN mean/SD")
    mle = p.add_mutually_exclusive_group()
    mle.add_argument("--with-mle", dest="with_mle", action="store_true",
                     help="add the MLE intrinsic-dimension column (needs scikit-dimension)")
    mle.add_argument("--no-mle", dest="with_mle", action="store_false",
                     help="skip MLE; depend only on numpy + scikit-learn")
    p.set_defaults(with_mle=config.WITH_MLE)
    p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    dec = p.add_mutually_exclusive_group()
    dec.add_argument("--decimation", dest="decimation", action="store_true",
                     help="run the sampling-scale sensitivity experiment + figure (default on)")
    dec.add_argument("--no-decimation", dest="decimation", action="store_false",
                     help="skip the sampling-scale sensitivity experiment")
    p.set_defaults(decimation=True)
    p.add_argument("--decimation-repeats", type=int, default=config.DECIMATION_REPEATS,
                   help="random subsets drawn per subsample size")
    p.add_argument("--decimation-estimator", choices=("twonn", "mle"), default="twonn",
                   help="estimator for the decimation curves")
    p.add_argument("--validate", action="store_true", help="also run synthetic sanity checks")
    p.add_argument("--validate-only", action="store_true", help="run only synthetic checks")
    args = p.parse_args()
    if args.covariance_repeats < 2:
        p.error("--covariance-repeats must be >= 2")
    return args


def main():
    args = _parse_args()

    if args.validate_only:
        validate.run(seed=args.seed, with_mle=args.with_mle, results_dir=args.results_dir)
        return

    # Fall back to no MLE when scikit-dimension is not installed.
    with_mle = args.with_mle
    if with_mle:
        try:
            import skdim
        except ImportError:
            print("scikit-dimension not found -> skipping the MLE column "
                  "(pip install scikit-dimension to enable it).")
            with_mle = False

    sets = data.load_all(args.embeddings_dir, overrides=config.LABEL_OVERRIDES)
    print(f"Loaded {len(sets)} embedding set(s) from {Path(args.embeddings_dir).resolve()}:")
    for es in sets:
        note = f" (dropped {es.n_dropped} duplicate)" if es.n_dropped else ""
        print(f"  {es.label:<22} N={es.n} ambient={es.ambient} sha={es.sha16}{note}")
    print()

    rows, curves = analysis.analyze_all(
        sets, results_dir=args.results_dir, seed=args.seed, with_mle=with_mle,
        covariance_repeats=args.covariance_repeats)
    analysis.print_table(rows)
    print(f"\nTables -> {args.results_dir}/manifold_results.{{csv,json}}")
    print(f"Covariance draws and settings -> {args.results_dir}/covariance_controls.json")

    if not args.no_figures:
        paths = figures.make_cumvar_figure(curves, args.results_dir)
        print("Figure -> " + ", ".join(str(p) for p in paths))

    # Sampling-scale sensitivity: ID on random subsets of increasing size.
    if args.decimation:
        import decimation
        est = args.decimation_estimator
        if est == "mle" and not with_mle:
            print("MLE unavailable -> running decimation with TwoNN instead.")
            est = "twonn"
        print(f"\n--- decimation (sampling-scale sensitivity, {est}) ---")
        dcurves = decimation.run_decimation(
            sets, results_dir=args.results_dir, seed=args.seed,
            repeats=args.decimation_repeats, estimator=est)
        decimation.print_summary(dcurves)
        if not args.no_figures:
            dpaths = figures.make_decimation_figure(dcurves, args.results_dir)
            print("Figure -> " + ", ".join(str(p) for p in dpaths))
        print(f"Table -> {args.results_dir}/decimation.csv")

    if args.validate:
        print("\n--- synthetic validation (appendix) ---")
        validate.run(seed=args.seed, with_mle=with_mle, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
