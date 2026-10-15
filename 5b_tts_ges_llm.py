"""kNN graphs, TTS and GES for all ten representations: four encoders and six LLMs (Sections 2.2-2.3, 3.2).

Self-contained counterpart of 3_build_knn.py, 5_tts.py and 6_ges_with_heatmap.py. Reads all_embeddings/*.npy
and the dataset CSV; writes graphs, TTS/GES CSVs and heatmaps to <out-dir>/<model>/knn_k<k>_euclidean/, and
tts_ges_summary.csv plus LaTeX tables to <out-dir> (default outputs/tts_ges_llm/).
Usage: python 5b_tts_ges_llm.py --group llm --table-k 15 --no-heatmaps
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.neighbors import NearestNeighbors


HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "full_dataset_with_scores_new.csv")
EMB_DIR = os.path.join(HERE, "all_embeddings")
DEFAULT_OUT = os.path.join(HERE, "outputs", "tts_ges_llm")

# key -> (embedding file in all_embeddings/, display name, group).
MODELS = {
    "bge":             ("embeddings_bge.npy",              "BGE-large-en-v1.5",    "encoder"),
    "mpnet":           ("embeddings_mpnet.npy",            "all-mpnet-base-v2",    "encoder"),
    "minilm":          ("embeddings_miniLM.npy",           "all-MiniLM-L6-v2",     "encoder"),
    "qwen_embedding":  ("embeddings_qwen_embeddings.npy",  "Qwen3-Embedding-0.6B", "encoder"),
    "qwen2.5_3b_L24":  ("embeddings_qwen2.5_3B_L24.npy",   "Qwen2.5-3B",           "llm"),
    "qwen2.5_7b_L20":  ("embeddings_qwen2.5_7b_L20.npy",   "Qwen2.5-7B",           "llm"),
    "qwen2.5_14b_L24": ("embeddings_qwen2.5_14B_L24.npy",  "Qwen2.5-14B",          "llm"),
    "llama3.1_8b_L16": ("embeddings_llama3.1_8B_L16.npy",  "Llama-3.1-8B",         "llm"),
    "gemma2_9b_L29":   ("embeddings_gemma2_9B_L29.npy",    "Gemma-2-9B",           "llm"),
    "gemma2_27b_L28":  ("embeddings_gemma2_27B_L28.npy",   "Gemma-2-27B",          "llm"),
}

K_LIST = [10, 15, 20, 30]
METRIC = "euclidean"
NORMALIZE_EMBEDDINGS = True
KNN_SEED = 0
TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]
TIER_TO_IDX = {t: i for i, t in enumerate(TIER_ORDER)}
# GES: random pairs sampled for each ordered (source tier, target tier) combination.
PAIRS_PER_TIERPAIR = 200
GES_SEED = 123
EPS = 1e-12


def l2_normalize_rows(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def build_symmetric_knn_graph(X, k, metric):
    """Undirected kNN graph (union of directed neighbours, Euclidean edge weights); same rule as 3_build_knn.py."""
    n = X.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"bad k={k} for n={n}")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X, return_distance=True)
    # Drop column 0, the point's match with itself.
    dists, nbrs = dists[:, 1:], nbrs[:, 1:]
    edge = {}
    for i in range(n):
        for m in range(k):
            j, w = int(nbrs[i, m]), float(dists[i, m])
            if i == j:
                continue
            a, b = (i, j) if i < j else (j, i)
            edge[(a, b)] = min(edge.get((a, b), w), w)
    rows, cols, data = [], [], []
    for (a, b), w in edge.items():
        rows += [a, b]; cols += [b, a]; data += [w, w]
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n)).tolil()
    A.setdiag(0.0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def compute_tts(tiers, A):
    """TTS: distribution of ordinal tier jumps over kNN edges, overall and per tier (as in 5_tts.py)."""
    tier_idx = np.array([TIER_TO_IDX[t] for t in tiers], dtype=int)
    Acoo = A.tocoo()
    jc = defaultdict(int)
    jc_t = {t: defaultdict(int) for t in TIER_ORDER}
    for i, j in zip(Acoo.row, Acoo.col):
        if i >= j:
            continue
        jump = abs(int(tier_idx[i] - tier_idx[j]))
        jc[jump] += 1
        jc_t[tiers[i]][jump] += 1
        jc_t[tiers[j]][jump] += 1
    total = sum(jc.values())
    overall = pd.DataFrame([{"jump": j, "count": jc[j], "ratio": jc[j] / total if total else np.nan}
                            for j in sorted(jc)])
    rows = []
    for t in TIER_ORDER:
        tot = sum(jc_t[t].values())
        if tot == 0:
            continue
        rows.append({"tier": t, "edge_incidents": tot,
                     "jump0_ratio": jc_t[t].get(0, 0) / tot,
                     "jump1_ratio": jc_t[t].get(1, 0) / tot,
                     "jump2_ratio": jc_t[t].get(2, 0) / tot,
                     "jump>2_ratio": sum(v for kk, v in jc_t[t].items() if kk > 2) / tot})
    return overall, pd.DataFrame(rows)


def sample_pairs(rng, idx_a, idx_b, m):
    """Draw m (i, j) pairs with replacement, i from idx_a and j from idx_b; i != j when the groups coincide."""
    if len(idx_a) == 0 or len(idx_b) == 0:
        return []
    pairs, same = [], np.array_equal(idx_a, idx_b)
    for _ in range(m):
        i = int(rng.choice(idx_a))
        j = int(rng.choice(idx_b))
        if same and len(idx_a) > 1:
            while j == i:
                j = int(rng.choice(idx_b))
        pairs.append((i, j))
    return pairs


def compute_ges(df, E, A, key, k):
    """GES: weighted shortest-path distance on A divided by Euclidean distance in E, per tier pair and overall."""
    rng = np.random.default_rng(GES_SEED + k)
    # All-pairs weighted geodesic distances; inf for pairs in different components.
    distG = shortest_path(A, directed=False, unweighted=False)
    tiers_s = df["tier"].astype(str)
    present = [t for t in TIER_ORDER if t in set(tiers_s)]
    idx_by = {t: df.index[tiers_s == t].to_numpy() for t in present}
    rows, all_s, reach, total = [], [], 0, 0
    for ta in present:
        for tb in present:
            st = []
            for i, j in sample_pairs(rng, idx_by[ta], idx_by[tb], PAIRS_PER_TIERPAIR):
                total += 1
                g = distG[i, j]
                # Skip unreachable pairs and zero-distance pairs; reachable_ratio = pairs used / pairs sampled.
                if not np.isfinite(g):
                    continue
                # Euclidean distance on E as loaded; the stored embeddings are already unit-norm, so this matches the graph's space.
                e = float(np.linalg.norm(E[i] - E[j]))
                if e < EPS:
                    continue
                reach += 1
                s = g / (e + EPS)
                st.append(s); all_s.append(s)
            rows.append({"model": key, "tier_src": ta, "tier_dst": tb,
                         "ges_mean": float(np.mean(st)) if st else np.nan,
                         "ges_median": float(np.median(st)) if st else np.nan,
                         "ges_std": float(np.std(st)) if st else np.nan,
                         "reachable_pairs": len(st), "sampled_pairs": PAIRS_PER_TIERPAIR})
    long_df = pd.DataFrame(rows)
    heat = long_df.pivot(index="tier_src", columns="tier_dst", values="ges_mean").reindex(index=present, columns=present)
    summ = {"reachable_ratio": reach / total if total else np.nan,
            "ges_mean": float(np.mean(all_s)) if all_s else np.nan,
            "ges_median": float(np.median(all_s)) if all_s else np.nan,
            "ges_std": float(np.std(all_s)) if all_s else np.nan,
            "ges_pairs_used": len(all_s), "total_pairs_sampled": total}
    return long_df, heat, summ


def plot_heatmap(heat, path, title, vmin, vmax):
    """Annotated tier-pair GES heatmap on a fixed colour range [vmin, vmax]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    v = heat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(v, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(heat.columns))); ax.set_yticks(range(len(heat.index)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right"); ax.set_yticklabels(heat.index)
    ax.set_xlabel("tier_dst"); ax.set_ylabel("tier_src"); ax.set_title(title)
    thr = (vmin + vmax) / 2.0
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ok = np.isfinite(v[i, j])
            ax.text(j, i, f"{v[i, j]:.2f}" if ok else "NA", ha="center", va="center", fontsize=9,
                    color="white" if ok and v[i, j] > thr else "black")
    fig.colorbar(im, ax=ax).set_label("Mean GES stretch")
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def run_for_model(df, key, out_root, heatmaps):
    fname, name, group = MODELS[key]
    print(f"\n==================== {name} ({key}) ====================")
    E = np.load(os.path.join(EMB_DIR, fname))
    if E.ndim != 2 or len(E) != len(df):
        raise ValueError(f"{key}: embeddings {E.shape} vs CSV rows {len(df)}")
    X = l2_normalize_rows(E) if NORMALIZE_EMBEDDINGS else E
    tiers = df["tier"].astype(str).to_numpy()
    unknown = sorted(set(tiers) - set(TIER_ORDER))
    if unknown:
        raise ValueError(f"unknown tiers {unknown}")

    np.random.seed(KNN_SEED)
    results = []
    for k in K_LIST:
        A = build_symmetric_knn_graph(X, k, METRIC)
        n_comp, labels = connected_components(A, directed=False)
        gdir = os.path.join(out_root, key, f"knn_k{k}_{METRIC}")
        os.makedirs(gdir, exist_ok=True)
        sparse.save_npz(os.path.join(gdir, "adjacency_csr.npz"), A)
        with open(os.path.join(gdir, "meta.json"), "w") as f:
            json.dump({"embed_path": os.path.join("all_embeddings", fname), "model": key, "n_nodes": int(X.shape[0]),
                       "dim": int(X.shape[1]), "k": k, "metric": METRIC, "normalize_embeddings": NORMALIZE_EMBEDDINGS,
                       "symmetrize_rule": "undirected union with min edge weight", "random_seed": KNN_SEED,
                       "connected_components": int(n_comp), "largest_component_size": int(np.bincount(labels).max()),
                       "average_degree": float(A.getnnz(axis=1).mean()),
                       "density": float(A.nnz / A.shape[0] ** 2)}, f, indent=2)

        tts_o, tts_t = compute_tts(tiers, A)
        tts_o.to_csv(os.path.join(gdir, "tts_overall.csv"), index=False)
        tts_t.to_csv(os.path.join(gdir, "tts_by_tier.csv"), index=False)

        long_df, heat, summ = compute_ges(df, E, A, key, k)
        long_df.to_csv(os.path.join(gdir, "ges_tierpair_long.csv"), index=False)
        heat.to_csv(os.path.join(gdir, "ges_heatmap.csv"))
        pd.DataFrame([{"model": key, "k": k, **summ}]).to_csv(os.path.join(gdir, "ges_summary.csv"), index=False)

        # tts_jump_le1 = fraction of edges joining the same or adjacent tiers.
        r = dict(tts_o.set_index("jump")["ratio"])
        results.append({"model": key, "display_name": name, "group": group, "k": k,
                        "connected_components": int(n_comp),
                        "tts_jump0": r.get(0, 0.0), "tts_jump1": r.get(1, 0.0), "tts_jump2": r.get(2, 0.0),
                        "tts_jump_gt2": sum(v for j, v in r.items() if j > 2),
                        "tts_jump_le1": r.get(0, 0.0) + r.get(1, 0.0),
                        **summ, "_heat": heat, "_gdir": gdir})
        print(f"  k={k}: components={n_comp}  TTS jump<=1={results[-1]['tts_jump_le1']:.3f}  "
              f"GES mean={summ['ges_mean']:.3f}")

    # One colour scale shared across all k for this model, so its heatmaps are comparable.
    if heatmaps:
        vals = np.concatenate([r["_heat"].to_numpy(float).ravel() for r in results])
        vals = vals[np.isfinite(vals)]
        for r in results:
            plot_heatmap(r["_heat"], os.path.join(r["_gdir"], f"ges_heatmap_{key}_k{r['k']}.png"),
                         f"GES Heatmap ({key}, k={r['k']})", float(vals.min()), float(vals.max()))
    return [{kk: v for kk, v in r.items() if not kk.startswith("_")} for r in results]


def write_tex(summary, path, table_k):
    """Compact LaTeX table: TTS at table_k and mean GES for every k."""
    SHORT = {"BGE-large-en-v1.5": "BGE-large", "all-mpnet-base-v2": "MPNet",
             "all-MiniLM-L6-v2": "MiniLM", "Qwen3-Embedding-0.6B": "Qwen3-Emb-0.6B"}
    ks = sorted(summary["k"].unique())
    tts = summary[summary["k"] == table_k].set_index("model")
    ges = summary.pivot(index="model", columns="k", values="ges_mean")
    models = [m for m in MODELS if m in tts.index]
    L = ["% Generated by 5b_tts_ges_llm.py -- do not edit by hand",
         "% needs \\usepackage{booktabs}",
         "\\begin{table}[t]", "\\centering", "\\small",
         f"\\caption{{Graph structure of the ten representations. TTS: fraction of $k{{=}}{table_k}$ nearest-"
         f"neighbour edges joining sentences in the same tier (0) or in tiers at most one step apart "
         f"($\\le$1). GES: mean ratio of graph geodesic to Euclidean distance over sampled sentence "
         f"pairs, for each $k$. The rule separates sentence encoders from the LLMs used for steering.}}",
         "\\label{tab:tts_ges}",
         "\\begin{tabular}{l" + "cc" + "c" * len(ks) + "}", "\\toprule",
         f" & \\multicolumn{{2}}{{c}}{{TTS ($k{{=}}{table_k}$)}} & \\multicolumn{{{len(ks)}}}{{c}}{{GES mean}} \\\\",
         f"\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-{3 + len(ks)}}}",
         "Model & 0 & $\\le$1 & " + " & ".join(f"$k{{=}}{k}$" for k in ks) + " \\\\", "\\midrule"]
    for i, m in enumerate(models):
        row = tts.loc[m]
        name = SHORT.get(row["display_name"], row["display_name"])
        L.append(f"{name} & {row['tts_jump0']:.3f} & {row['tts_jump_le1']:.3f} & " +
                 " & ".join(f"{ges.loc[m, k]:.2f}" for k in ks) + " \\\\")
        if i < len(models) - 1 and MODELS[models[i + 1]][2] != MODELS[m][2]:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


PAPER_NAME = {"bge": "BGE", "mpnet": "MPNet", "minilm": "MiniLM", "qwen_embedding": "Qwen-Emb",
              "qwen2.5_3b_L24": "Qwen2.5-3B", "qwen2.5_7b_L20": "Qwen2.5-7B",
              "qwen2.5_14b_L24": "Qwen2.5-14B", "llama3.1_8b_L16": "Llama-3.1-8B",
              "gemma2_9b_L29": "Gemma-2-9B", "gemma2_27b_L28": "Gemma-2-27B"}
LLM_NOTE = ("LLM rows are residual-stream activations of the instruction-tuned models at the steering "
            "layer (Qwen2.5-3B L24, Qwen2.5-7B L20, Qwen2.5-14B L24, Llama-3.1-8B L16, Gemma-2-9B L29, "
            "Gemma-2-27B L28).")


def _side_by_side(left, right):
    return left + ["\\hspace{2em}"] + right


def write_full_tex(summary, out_dir):
    """Write the full TTS table (all jump sizes, all k) and full GES table (mean/median/std, all k)."""
    models = [m for m in MODELS if m in set(summary["model"])]
    ks = sorted(summary["k"].unique())
    S = summary.set_index(["model", "k"])

    cols = [("tts_jump0", True), ("tts_jump1", True), ("tts_jump2", False), ("tts_jump_gt2", False)]

    def tts_half(klist):
        T = ["\\begin{tabular}[t]{@{}p{19mm}ccccc@{}}", "\\toprule",
             "Model & $k$ & jump0 & jump1 & jump2 & jump$>$2 \\\\", "\\midrule"]
        for bi, k in enumerate(klist):
            # Bold the best value within each k: highest for jump0/jump1, lowest for jump2/jump>2.
            best = {c: (max if hi else min)(S.loc[(m, k), c] for m in models) for c, hi in cols}
            for i, m in enumerate(models):
                cells = []
                for c, _ in cols:
                    v = S.loc[(m, k), c]
                    cells.append(f"\\textbf{{{v:.3f}}}" if abs(v - best[c]) < 5e-4 else f"{v:.3f}")
                T.append(f"{PAPER_NAME[m]} & {k} & " + " & ".join(cells) + " \\\\")
                if i < len(models) - 1 and MODELS[models[i + 1]][2] != MODELS[m][2]:
                    T.append("\\cmidrule(l){1-6}")
            if bi < len(klist) - 1:
                T.append("\\midrule")
        T += ["\\bottomrule", "\\end{tabular}"]
        return T

    half = (len(ks) + 1) // 2
    L = ["% Generated by 5b_tts_ges_llm.py -- do not edit by hand",
         "% needs \\usepackage{booktabs,graphicx}",
         "\\begin{table}[htbp!]", "\\centering", "\\scriptsize",
         "\\caption{Overall Tier Transition Smoothness (TTS) across representations and neighborhood sizes. "
         "Values report the proportion of graph edges with ordinal jump magnitude $0$, $1$, $2$, or greater "
         "than $2$. Bold values indicate the best result within each $k$ (higher is better for jump0 and "
         "jump1; lower is better for jump2 and jump$>$2). Within each $k$, a rule separates sentence "
         "encoders from LLMs. " + LLM_NOTE + "}",
         "\\label{tab:tts_overall}", "\\setlength{\\tabcolsep}{3pt}"]
    L += _side_by_side(tts_half(ks[:half]), tts_half(ks[half:])) + ["\\end{table}"]
    with open(os.path.join(out_dir, "tts_full_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")

    def ges_half(mlist):
        T = ["\\begin{tabular}[t]{@{}p{19mm}cccc@{}}", "\\toprule",
             "Model & $k$ & Mean & Median & Std \\\\", "\\midrule"]
        for i, m in enumerate(mlist):
            for k in ks:
                r = S.loc[(m, k)]
                T.append(f"{PAPER_NAME[m]} & {k} & {r['ges_mean']:.3f} & {r['ges_median']:.3f} & {r['ges_std']:.3f} \\\\")
            if i < len(mlist) - 1:
                T.append("\\midrule")
        T += ["\\bottomrule", "\\end{tabular}"]
        return T

    mh = (len(models) + 1) // 2
    L = ["% Generated by 5b_tts_ges_llm.py -- do not edit by hand",
         "% needs \\usepackage{booktabs,graphicx}",
         "\\begin{table}[htbp!]", "\\centering", "\\scriptsize",
         "\\caption{Overall Geodesic--Euclidean Stretch (GES) across representations and neighborhood sizes. "
         "Values greater than 1 indicate that intrinsic graph-based distances exceed ambient Euclidean "
         "distances. " + LLM_NOTE + "}",
         "\\label{tab:ges_overall}", "\\setlength{\\tabcolsep}{3pt}"]
    L += _side_by_side(ges_half(models[:mh]), ges_half(models[mh:])) + ["\\end{table}"]
    with open(os.path.join(out_dir, "ges_full_table.tex"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    ap.add_argument("--group", choices=["encoder", "llm"], default=None)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--table-k", type=int, default=15, choices=K_LIST)
    ap.add_argument("--no-heatmaps", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    keys = [k for k in MODELS if (args.models is None or k in args.models)
            and (args.group is None or MODELS[k][2] == args.group)]
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for key in keys:
        rows += run_for_model(df, key, args.out_dir, not args.no_heatmaps)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "tts_ges_summary.csv"), index=False)
    write_tex(summary, os.path.join(args.out_dir, "tts_ges_table.tex"), args.table_k)
    write_full_tex(summary, args.out_dir)

    print("\n===== SUMMARY =====")
    print(summary[["display_name", "k", "connected_components", "tts_jump0", "tts_jump_le1",
                   "reachable_ratio", "ges_mean"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n[Saved] {args.out_dir}")


if __name__ == "__main__":
    main()
