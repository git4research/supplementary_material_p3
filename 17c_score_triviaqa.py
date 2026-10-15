"""Score TriviaQA generations by normalized alias matching (Section 2.5.3, short-form factuality).

A response is correct if a normalized gold alias equals or appears in the normalized response
(no LLM judge). Writes triviaqa_accuracy.csv per (condition, alpha) and an accuracy plot.
Usage: python 17c_score_triviaqa.py --model qwen --in factuality_triviaqa.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import defaultdict
from pathlib import Path
from model_config import add_model_arguments, experiment_from_args, validate_single_experiment

HERE = Path(__file__).parent
# Conditions that are scored; rows with any other condition are skipped.
KEEP = {"baseline", "global_ridge", "local_manifold"}
COND_ORDER = {"baseline": 0, "global_ridge": 1, "local_manifold": 2}
ALPHAS = [0.1, 0.2, 0.3, 0.5]
BASE_COLOR = {"baseline": "#7f7f7f", "global_ridge": "#d62728", "local_manifold": "#2ca02c"}

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")


def normalize(s: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles and punctuation, collapse whitespace."""
    s = (s or "").lower().replace("’", "'")
    s = _ARTICLES.sub(" ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return _WS.sub(" ", s).strip()


def em_hit(response: str, aliases) -> bool:
    """True if any normalized alias equals, or is a substring of, the normalized response."""
    r = normalize(response)
    if not r:
        return False
    for a in aliases:
        na = normalize(a)
        if na and (na == r or na in r):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    add_model_arguments(ap)
    ap.add_argument("--in", dest="infile", type=str,
                    default=None)
    ap.add_argument("--out-dir", type=str,
                    default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    infile = Path(args.infile) if args.infile else experiment_from_args(args).generations_dir / "factuality_triviaqa.jsonl"
    out_dir = Path(args.out_dir) if args.out_dir else infile.parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not infile.exists():
        raise SystemExit(f"Input not found: {infile}")

    # (condition, alpha) -> [hits, n]
    acc = defaultdict(lambda: [0, 0])
    with open(infile) as source:
        rows = [json.loads(line) for line in source if line.strip()]
    validate_single_experiment(rows)
    for r in rows:
        cond = r.get("condition")
        if cond not in KEEP:
            continue
        m = r.get("prompt_metadata") or {}
        key = (cond, r.get("alpha"))
        acc[key][1] += 1
        if em_hit(r.get("response", ""), m.get("aliases") or []):
            acc[key][0] += 1

    # Sort by condition (baseline, global_ridge, local_manifold), then alpha (baseline's None first).
    keys = sorted(acc, key=lambda k: (COND_ORDER.get(k[0], 9), k[0], -1 if k[1] is None else k[1]))
    print(f"\nTriviaQA EM accuracy  ({infile.name})")
    print(f"{'condition':<16}{'alpha':>6}{'hits':>6}{'n':>5}{'acc%':>7}")
    print("-" * 42)
    for k in keys:
        h, n = acc[k]
        astr = "base" if k[1] is None else f"{k[1]:.1f}"
        print(f"{k[0]:<16}{astr:>6}{h:>6}{n:>5}{100*h/n if n else 0:>6.1f}%")

    csv_path = out_dir / "triviaqa_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "alpha", "hits", "n", "accuracy_pct"])
        for k in keys:
            h, n = acc[k]
            w.writerow([k[0], ("" if k[1] is None else k[1]), h, n, (100 * h / n if n else 0)])
    print(f"\nwrote: {csv_path}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed; skipping plot)")
        return

    def rate(cond, alpha):
        h, n = acc.get((cond, alpha), [0, 0])
        return (100 * h / n) if n else None

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    by = rate("baseline", None)
    if by is not None:
        ax.axhline(by, color=BASE_COLOR["baseline"], ls="--", lw=1.6, label="baseline (no steering)")
        ax.text(0.535, by, f"baseline = {by:.0f}%", fontsize=8, color=BASE_COLOR["baseline"],
                va="center", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.2, alpha=0.85))
    for cond, marker, lw in (("global_ridge", "X", 2.0), ("local_manifold", "s", 2.8)):
        xs, ys = [], []
        for a in ALPHAS:
            v = rate(cond, a)
            if v is not None:
                xs.append(a); ys.append(v)
        if xs:
            ax.plot(xs, ys, color=BASE_COLOR[cond], marker=marker, lw=lw, markersize=8,
                    label=cond)
    ax.set_xlabel(r"$\alpha$ (steering strength)", fontsize=10)
    ax.set_ylabel("TriviaQA EM accuracy (%)", fontsize=10)
    ax.set_xticks(ALPHAS); ax.set_xlim(0.05, 0.56)
    # Upper y-limit: 1.6 x max(baseline accuracy, 10).
    ax.set_ylim(0, max([v for v in (rate(c, a) for c in ("baseline",) for a in [None]) if v] + [10]) * 1.6)
    ax.set_title("TriviaQA closed-book factuality (exact match)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    png = out_dir / "triviaqa_accuracy.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.savefig(out_dir / "triviaqa_accuracy.pdf", dpi=200, bbox_inches="tight")
    print(f"wrote: {png}")


if __name__ == "__main__":
    main()
