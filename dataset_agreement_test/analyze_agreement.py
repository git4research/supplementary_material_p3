"""Score the LLM judges' blind re-annotations against the original labels (Appendix B.5).

Parses replies in part1_tier/responses/ and part2_energy/responses/; computes tier agreement
(exact, within-one, kappa, Spearman) and pair concordance (overall, cross- vs same-tier, by score
gap) with bootstrap CIs and binomial tests; writes JSON, Markdown, LaTeX, CSVs and figures to results/.
Usage (from this directory): python analyze_agreement.py"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from common import TIER_IDX, TIER_ORDER, rubric_overlap

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# Bootstrap resamples and seed for 95% percentile CIs.
N_BOOT = 2000
BOOT_SEED = 0
# Bins (left-closed) for the absolute original-score gap within a pair.
GAP_BINS = [0, 1, 2, 4, 8, np.inf]
GAP_LABELS = ["<1", "1-2", "2-4", "4-8", ">=8"]
# Tolerant line patterns for 'S001,Tier' and 'P0001,A' answers (case-insensitive).
TIER_RE = re.compile(r"\b(S\d{3,4})\b\W{0,3}\s*(" + "|".join(TIER_ORDER) + r")\b", re.I)
PAIR_RE = re.compile(r"\b(P\d{4})\b\W{0,3}\s*([AB])\b", re.I)


def judge_name(path: Path) -> str:
    """Judge name from a response file name, e.g. tier_responses_<judge>.txt -> <judge>."""
    return re.sub(r"^(tier_|energy_)?responses?_", "", path.stem)


def parse(path: Path, regex, valid_ids, canon):
    """Extract ID -> answer; IDs given conflicting answers are dropped. Returns answers, counts, IDs to re-ask."""
    ans, conflicts, dup_same = {}, set(), 0
    text = path.read_text(encoding="utf-8", errors="replace")
    for m in regex.finditer(text):
        i, v = m.group(1).upper(), canon(m.group(2))
        if i not in valid_ids:
            continue
        if i in ans and ans[i] != v:
            conflicts.add(i)
        elif i in ans:
            dup_same += 1
        ans[i] = v
    for i in conflicts:
        ans.pop(i, None)
    missing = [i for i in valid_ids if i not in ans]
    return ans, {"parsed": len(ans), "missing": len(missing), "conflicting": len(conflicts),
                 "repeated_same": dup_same}, missing + sorted(conflicts)


def write_reask(part, judge, ids, table, fmt_item, rubric):
    """Write a follow-up prompt for missing or conflicting items; returns its file name, or None if none."""
    if not ids:
        return None
    out = RES / f"reask_{part}_{judge}.txt"
    items = [fmt_item(table.loc[i]) for i in ids if i in table.index]
    out.write_text(f"{rubric}\n\nThe following items were missing from your previous answer. "
                   f"Answer only these ({len(items)} items).\n\n" + "\n".join(items) + "\n", encoding="utf-8")
    return out.name


def boot_ci(stat_fn, n, rng):
    """95% percentile bootstrap CI of stat_fn over resampled row indices."""
    vals = [stat_fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def tier_stats(true_idx, pred_idx, score):
    """Agreement between original and judged tier ordinals, plus Spearman of judged tier vs. original score."""
    t, p = np.asarray(true_idx), np.asarray(pred_idx)
    return {
        "n": int(len(t)),
        "exact": float(np.mean(t == p)),
        "within_one": float(np.mean(np.abs(t - p) <= 1)),
        "kappa": float(cohen_kappa_score(t, p, labels=range(7))),
        "kappa_quadratic": float(cohen_kappa_score(t, p, labels=range(7), weights="quadratic")),
        "spearman_tier": float(spearmanr(t, p).statistic),
        "spearman_pred_vs_score": float(spearmanr(p, score).statistic),
    }


def analyze_tier(path, key, leak_mask):
    """Tier metrics for one judge on all sentences and excluding rubric examples, with CIs and a confusion matrix."""
    judge = judge_name(path)
    ans, comp, miss = parse(path, TIER_RE, set(key.index), lambda s: s.capitalize())
    rubric = (HERE / "part1_tier" / "rubric_tier.txt").read_text(encoding="utf-8").strip()
    comp["reask_file"] = write_reask("tier", judge, miss, key, lambda r: f"{r.name}\t{r.sentence}", rubric)
    d = key.loc[[i for i in key.index if i in ans]].copy()
    d["pred"] = d.index.map(ans)
    d["pred_idx"] = d["pred"].map(TIER_IDX)
    out = {"completeness": comp}
    rng = np.random.default_rng(BOOT_SEED)
    for name, sub in [("all", d), ("excl_rubric_examples", d[~leak_mask.reindex(d.index).values])]:
        s = tier_stats(sub.tier_idx, sub.pred_idx, sub.score)
        t, p, sc = sub.tier_idx.values, sub.pred_idx.values, sub.score.values
        s["ci95"] = {k: boot_ci(lambda ix, k=k: tier_stats(t[ix], p[ix], sc[ix])[k], len(sub), rng)
                     for k in ["exact", "within_one", "kappa_quadratic", "spearman_tier"]}
        out[name] = s
    cm = confusion_matrix(d.tier_idx, d.pred_idx, labels=range(7))
    out["confusion_matrix"] = cm.tolist()
    out["recall_by_tier"] = {TIER_ORDER[i]: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i in range(7)}
    # Most frequent tier confusions, counting both directions together.
    off = [(int(cm[i, j] + cm[j, i]), TIER_ORDER[i], TIER_ORDER[j]) for i in range(7) for j in range(i + 1, 7)]
    out["top_confusions"] = [{"pair": f"{a}/{b}", "count": c} for c, a, b in sorted(off, reverse=True)[:5]]
    return out, d[["pred"]].rename(columns={"pred": judge}), cm


def pair_stats(correct):
    """Concordance: fraction of pairs where the judge picked the higher-scored sentence."""
    c = np.asarray(correct, dtype=float)
    return float(c.mean()) if len(c) else float("nan")


def analyze_pairs(path, key, leak_ids):
    """Pair metrics for one judge: concordance overall, by tier relation and by score gap, with CIs and binomial tests."""
    judge = judge_name(path)
    ans, comp, miss = parse(path, PAIR_RE, set(key.index), lambda s: s.upper())
    rubric = (HERE / "part2_energy" / "rubric_energy.txt").read_text(encoding="utf-8").strip()
    comp["reask_file"] = write_reask("energy", judge, miss, key,
                                     lambda r: f"{r.name}\n  A: {r.A}\n  B: {r.B}\n", rubric)
    d = key.loc[[i for i in key.index if i in ans]].copy()
    d["choice"] = d.index.map(ans)
    out = {"completeness": comp, "rate_answer_A": float((d.choice == "A").mean()),
           "n_score_tied_excluded": int((d.higher == "tie").sum())}
    rng = np.random.default_rng(BOOT_SEED)
    # A pair is excluded from the second subset if either sentence is a rubric example.
    leak = d.A_index.isin(leak_ids) | d.B_index.isin(leak_ids)
    for name, sub in [("all", d), ("excl_rubric_examples", d[~leak])]:
        # Pairs with equal original scores have no correct answer and are left out.
        nt = sub[sub.higher != "tie"]
        corr = (nt.choice == nt.higher).values
        k, n = int(corr.sum()), int(len(corr))
        res = {"n_nontied": n, "n_concordant": k, "concordance": pair_stats(corr),
               "binom_p_two_sided": float(binomtest(k, n, 0.5).pvalue) if n else None,
               "ci95": boot_ci(lambda ix: pair_stats(corr[ix]), n, rng) if n else None}
        bins = pd.cut(nt.gap, GAP_BINS, labels=GAP_LABELS, right=False)
        res["by_gap"] = {str(b): {"n": int((bins == b).sum()),
                                  "concordance": pair_stats(corr[(bins == b).values]) if (bins == b).any() else None}
                         for b in GAP_LABELS}
        # Split by whether both sentences share an original tier.
        same = (nt.A_tier == nt.B_tier).values
        for part, mask in [("cross_tier", ~same), ("same_tier", same)]:
            c = corr[mask]
            res[part] = {"n": int(mask.sum()), "concordance": pair_stats(c),
                         "binom_p_two_sided": float(binomtest(int(c.sum()), len(c), 0.5).pvalue) if len(c) else None,
                         "ci95": boot_ci(lambda ix, c=c: pair_stats(c[ix]), len(c), rng) if len(c) else None}
        out[name] = res
    return out, d[["choice"]].rename(columns={"choice": judge})


def plot_confusion(cm, judge):
    """Save a heatmap of original vs. judged tier counts for one judge."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels(TIER_ORDER, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(TIER_ORDER, fontsize=8)
    ax.set_xlabel(f"{judge} label"); ax.set_ylabel("original label")
    fig.tight_layout(); fig.savefig(RES / f"confusion_{judge}.png", dpi=200); plt.close(fig)


def plot_gap(pair_results):
    """Plot concordance by score-gap bin for each judge (all pairs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for judge, r in pair_results.items():
        ys = [r["all"]["by_gap"][g]["concordance"] for g in GAP_LABELS]
        ax.plot(range(len(GAP_LABELS)), ys, marker="o", label=judge)
    ax.axhline(0.5, color="0.5", ls=":", lw=0.8)
    ax.set_xticks(range(len(GAP_LABELS))); ax.set_xticklabels(GAP_LABELS)
    ax.set_xlabel("original score gap"); ax.set_ylabel("concordance"); ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3, lw=0.4)
    fig.tight_layout(); fig.savefig(RES / "concordance_by_gap.png", dpi=200); plt.close(fig)


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def write_tex(summary):
    """Write booktabs LaTeX tables for tier agreement and pair concordance."""
    L = ["% Generated by analyze_agreement.py -- do not edit by hand", "% needs \\usepackage{booktabs}"]
    if summary["tier"]:
        L += ["\\begin{table}[htbp]", "\\centering", "\\small",
              "\\caption{Blind LLM re-annotation of tiers. Exact and within-one agreement (\\%), "
              "quadratic-weighted Cohen's $\\kappa$, and Spearman $\\rho$ between original and LLM tier "
              "ordinals; brackets give 95\\% bootstrap CIs.}", "\\label{tab:tier_agreement}",
              "\\begin{tabular}{lccccc}", "\\toprule",
              "Judge & $n$ & Exact & Within one & $\\kappa_w$ & $\\rho$ \\\\", "\\midrule"]
        for j, r in summary["tier"].items():
            a, c = r["all"], r["all"]["ci95"]
            L.append(f"{j.replace('_', ' ')} & {a['n']} & {pct(a['exact'])} [{pct(c['exact'][0])}, {pct(c['exact'][1])}] "
                     f"& {pct(a['within_one'])} & {a['kappa_quadratic']:.2f} [{c['kappa_quadratic'][0]:.2f}, "
                     f"{c['kappa_quadratic'][1]:.2f}] & {a['spearman_tier']:.2f} \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    if summary["energy"]:
        L += ["\\begin{table}[htbp]", "\\centering", "\\small",
              "\\caption{Blind pairwise energy judgments: concordance (\\%) with the original continuous "
              "scores on non-tied pairs: overall, split by whether the two sentences share a tier, and by score gap; "
              "brackets give 95\\% bootstrap CIs. Cross-tier pairs mainly test recovery of the tier order; "
              "same-tier pairs test the finer within-tier ordering.}", "\\label{tab:energy_agreement}",
              "\\begin{tabular}{lccc" + "c" * len(GAP_LABELS) + "}", "\\toprule",
              " & & & & \\multicolumn{" + str(len(GAP_LABELS)) + "}{c}{Score gap} \\\\",
              "\\cmidrule(lr){5-" + str(4 + len(GAP_LABELS)) + "}",
              "Judge & All & Cross-tier & Same-tier & " + " & ".join(g.replace(">=", "$\\ge$").replace("<", "$<$").replace("-", "--") for g in GAP_LABELS) + " \\\\",
              "\\midrule"]
        for j, r in summary["energy"].items():
            a = r["all"]
            ci = lambda r: f"{pct(r['concordance'])} [{pct(r['ci95'][0])}, {pct(r['ci95'][1])}]"
            L.append(f"{j.replace('_', ' ')} & {ci(a)} & {ci(a['cross_tier'])} & {ci(a['same_tier'])} & "
                     + " & ".join(pct(a["by_gap"][g]["concordance"]) for g in GAP_LABELS) + " \\\\")
        L.append("\\midrule")
        a0 = next(iter(summary["energy"].values()))["all"]
        L.append(f"$n$ & {a0['n_nontied']} & {a0['cross_tier']['n']} & {a0['same_tier']['n']} & " + " & ".join(
            str(next(iter(summary["energy"].values()))["all"]["by_gap"][g]["n"]) for g in GAP_LABELS) + " \\\\")
        L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (RES / "agreement_tables.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_report(summary):
    """Write a Markdown summary of all results to results/report.md."""
    L = ["# Dataset agreement test -- results", "",
         f"Dataset sentences: {summary['n_sentences']}; sentences quoted in a rubric: "
         f"{summary['n_rubric_example_sentences']} {summary['rubric_example_indices']}", ""]
    for j, r in summary["tier"].items():
        a, e, c = r["all"], r["excl_rubric_examples"], r["completeness"]
        L += [f"## Part 1 (tier) -- {j}",
              f"- completeness: {c}",
              f"- n={a['n']}: exact {pct(a['exact'])}%, within one {pct(a['within_one'])}%, "
              f"kappa {a['kappa']:.3f}, quadratic kappa {a['kappa_quadratic']:.3f}, "
              f"Spearman (tier) {a['spearman_tier']:.3f}, Spearman (judge tier vs score) {a['spearman_pred_vs_score']:.3f}",
              f"- 95% CIs: {a['ci95']}",
              f"- excluding rubric examples (n={e['n']}): exact {pct(e['exact'])}%, within one {pct(e['within_one'])}%, "
              f"quadratic kappa {e['kappa_quadratic']:.3f}",
              f"- recall by tier: " + ", ".join(f"{k} {pct(v)}" for k, v in r["recall_by_tier"].items()),
              f"- top confusions: " + ", ".join(f"{x['pair']} ({x['count']})" for x in r["top_confusions"]), ""]
    for j, r in summary["energy"].items():
        a, e, c = r["all"], r["excl_rubric_examples"], r["completeness"]
        L += [f"## Part 2 (energy) -- {j}",
              f"- completeness: {c}",
              f"- concordance {pct(a['concordance'])}% ({a['n_concordant']}/{a['n_nontied']} non-tied pairs; "
              f"{r['n_score_tied_excluded']} score-tied excluded), 95% CI {[pct(x) for x in a['ci95']]}, "
              f"binomial p = {a['binom_p_two_sided']:.3g}",
              f"- cross-tier pairs: {pct(a['cross_tier']['concordance'])}% (n={a['cross_tier']['n']}), 95% CI "
              f"{[pct(x) for x in a['cross_tier']['ci95']]}; same-tier pairs: {pct(a['same_tier']['concordance'])}% "
              f"(n={a['same_tier']['n']}), 95% CI {[pct(x) for x in a['same_tier']['ci95']]}, "
              f"binomial p = {a['same_tier']['binom_p_two_sided']:.3g}",
              f"- by gap: " + ", ".join(f"{g}: {pct(v['concordance'])}% (n={v['n']})" for g, v in a["by_gap"].items()),
              f"- excluding pairs with a rubric example: {pct(e['concordance'])}% (n={e['n_nontied']})",
              f"- answered 'A': {pct(r['rate_answer_A'])}% (true rate of A higher is "
              f"{pct(summary['true_rate_A_higher'])}%)", ""]
    if summary.get("consensus_disagreements"):
        c = summary["consensus_disagreements"]
        L += ["## Consensus disagreements (all judges agree with each other, not with the original tier)",
              f"- n = {c['n']} (listed in results/consensus_disagreements.csv; worth a manual review)",
              "- by original -> judged tier: " + ", ".join(f"{k} ({v})" for k, v in c["by_transition"].items()), ""]
    if summary.get("between_judges"):
        L += ["## Between judges", f"{summary['between_judges']}", ""]
    (RES / "report.md").write_text("\n".join(L), encoding="utf-8")


def main():
    """Load answer keys, flag rubric examples, analyze each judge reply, compare judges, and write outputs."""
    RES.mkdir(exist_ok=True)
    tkey = pd.read_csv(HERE / "key" / "tier_key.csv").set_index("id")
    tkey["tier_idx"] = tkey.tier.map(TIER_IDX)
    pkey = pd.read_csv(HERE / "key" / "pairs_key.csv").set_index("pair_id")
    rubrics = [HERE / "part1_tier" / "rubric_tier.txt", HERE / "part2_energy" / "rubric_energy.txt"]
    # Flag sentences quoted in a rubric so results can also be reported without them.
    leak = rubric_overlap(tkey.reset_index(), rubrics)
    leak.index = tkey.index
    leak_ids = set(tkey.loc[leak, "index"])

    summary = {"n_sentences": int(len(tkey)), "n_rubric_example_sentences": int(leak.sum()),
               "rubric_example_indices": sorted(int(x) for x in leak_ids),
               "true_rate_A_higher": float((pkey.higher == "A").mean()), "tier": {}, "energy": {}}
    tier_preds, pair_preds = [], []
    for f in sorted((HERE / "part1_tier" / "responses").glob("*.txt")):
        r, preds, cm = analyze_tier(f, tkey, leak)
        j = judge_name(f)
        summary["tier"][j] = r
        tier_preds.append(preds)
        plot_confusion(cm, j)
        # Per-judge list of sentences whose judged tier differs from the original.
        dis = tkey.join(preds, how="inner")
        dis = dis[dis[j] != dis.tier]
        dis.assign(tier_gap=(dis[j].map(TIER_IDX) - dis.tier_idx))[
            ["index", "sentence", "tier", j, "tier_gap", "score"]].sort_values(["tier", "score"]).to_csv(
            RES / f"disagreements_{j}.csv", index_label="id")
    pair_results = {}
    for f in sorted((HERE / "part2_energy" / "responses").glob("*.txt")):
        r, preds = analyze_pairs(f, pkey, leak_ids)
        j = judge_name(f)
        summary["energy"][j] = pair_results[j] = r
        pair_preds.append(preds)
    if pair_results:
        plot_gap(pair_results)

    # With 2+ judges: consensus disagreements with the original tier, and pairwise judge agreement.
    between = {}
    if len(tier_preds) > 1:
        T = pd.concat(tier_preds, axis=1, join="inner")
        judges = list(T.columns)
        same = T.nunique(axis=1) == 1
        C = tkey.join(T, how="inner")
        C = C[same.reindex(C.index).values & (C[judges[0]] != C.tier)]
        C.assign(judges_label=C[judges[0]])[["index", "sentence", "tier", "judges_label", "score"]].sort_values(
            ["tier", "score"]).to_csv(RES / "consensus_disagreements.csv", index_label="id")
        summary["consensus_disagreements"] = {
            "n": int(len(C)), "description": "all judges agree with each other but not with the original tier",
            "by_transition": {f"{a}->{b}": int(n) for (a, b), n in
                              C.groupby(["tier", judges[0]]).size().sort_values(ascending=False).items()}}
        for a, b in combinations(T.columns, 2):
            ia, ib = T[a].map(TIER_IDX), T[b].map(TIER_IDX)
            between[f"tier {a} vs {b}"] = {"n": int(len(T)), "exact": float((ia == ib).mean()),
                                           "kappa_quadratic": float(cohen_kappa_score(ia, ib, weights="quadratic"))}
    if len(pair_preds) > 1:
        Pp = pd.concat(pair_preds, axis=1, join="inner")
        for a, b in combinations(Pp.columns, 2):
            between[f"energy {a} vs {b}"] = {"n": int(len(Pp)), "same_choice": float((Pp[a] == Pp[b]).mean())}
    summary["between_judges"] = between

    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary)
    write_tex(summary)
    print((RES / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
