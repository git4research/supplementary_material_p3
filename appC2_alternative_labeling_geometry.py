from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import shortest_path

warnings.filterwarnings("ignore")

CSV_PATH = "full_dataset_with_scores_new.csv"
OUTPUT_ROOT = "outputs"
TEXT_COL = "sentence"
TIER_COL = "tier"

MODEL_LIST = ["bge", "qwen_embedding"]
K_LIST = [30]
GRAPH_METRIC_NAME = "euclidean"

REAL_TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]
POSITION_LABELS = [f"pos{i+1}" for i in range(7)]

BRYSBAERT_PATH = "brysbaert_concreteness.csv"
USE_FORMALITY_CLASSIFIER = True

PAIRS_PER_POSITION = 150
GREEDY_MAX_STEPS = 150
SEED = 0


def compute_valence_vader(sentences):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    return np.array([sia.polarity_scores(s)["compound"] for s in sentences])


def compute_abstraction_brysbaert(sentences, path):
    lex = pd.read_csv(path)
    conc = dict(zip(lex["Word"].str.lower(), lex["Conc.M"]))
    out = []
    for s in sentences:
        words = [w.strip(".,!?;:\"'").lower() for w in s.split()]
        vals = [conc[w] for w in words if w in conc]
        out.append(5.0 - np.mean(vals) if vals else 2.5)
    return np.array(out)


def compute_formality_classifier(sentences):
    from transformers import pipeline
    clf = pipeline("text-classification", model="s-nlp/roberta-base-formality-ranker",
                    top_k=None, truncation=True)
    out = []
    for s in sentences:
        result = clf(s)[0]
        out.append(next(r["score"] for r in result if r["label"].lower().startswith("formal")))
    return np.array(out)


def get_confound_scores(sentences):
    confounds = {}
    try:
        confounds["valence"] = compute_valence_vader(sentences)
    except Exception as e:
        print(f"[valence] SKIPPED -- {e}")

    try:
        if os.path.exists(BRYSBAERT_PATH):
            confounds["abstraction"] = compute_abstraction_brysbaert(sentences, BRYSBAERT_PATH)
        else:
            print(f"[abstraction] SKIPPED -- {BRYSBAERT_PATH} not found")
    except Exception as e:
        print(f"[abstraction] SKIPPED -- {e}")

    if USE_FORMALITY_CLASSIFIER:
        try:
            confounds["formality"] = compute_formality_classifier(sentences)
        except Exception as e:
            print(f"[formality] SKIPPED -- {e}")

    for name, vals in confounds.items():
        if np.std(vals) < 1e-6:
            print(f"  !! WARNING: '{name}' has ~zero variance -- likely a pipeline bug.")
    return confounds


def bin_by_matched_sizes(continuous_scores, real_tiers, real_tier_order):
    tier_sizes = {t: int((real_tiers == t).sum()) for t in real_tier_order}
    n = len(continuous_scores)
    order = np.argsort(continuous_scores, kind="stable")
    positions = np.empty(n, dtype=object)
    cursor = 0
    for i, t in enumerate(real_tier_order):
        size = tier_sizes[t]
        positions[order[cursor:cursor + size]] = POSITION_LABELS[i]
        cursor += size
    return positions


def real_tiers_to_positions(real_tiers, real_tier_order):
    name_to_pos = {t: POSITION_LABELS[i] for i, t in enumerate(real_tier_order)}
    return np.array([name_to_pos[t] for t in real_tiers], dtype=object)


def l2_normalize_rows(x, eps=1e-12):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def load_model_graph(model, k):
    embed_path = os.path.join(OUTPUT_ROOT, model, "embeddings.npy")
    graph_path = os.path.join(OUTPUT_ROOT, model, f"knn_k{k}_{GRAPH_METRIC_NAME}", "adjacency_csr.npz")
    E = l2_normalize_rows(np.load(embed_path))
    A = sparse.load_npz(graph_path).tocsr()
    return E, A


def compute_full_shortest_paths(A_sym):
    dist_matrix, predecessors = shortest_path(
        A_sym, method="D", directed=False, unweighted=False, return_predecessors=True
    )
    return dist_matrix, predecessors


def reconstruct_path(predecessors, src, dst):
    if src == dst:
        return [src]
    path = [dst]
    cur = dst
    while cur != src and cur != -9999:
        cur = predecessors[src, cur]
        path.append(cur)
    if cur == -9999:
        return []
    path.reverse()
    return path


def metric_tts(A, positions, position_labels):
    pos_idx = {p: i for i, p in enumerate(position_labels)}
    idx_arr = np.array([pos_idx[p] for p in positions])
    Acoo = A.tocoo()
    jump0 = jump1 = total = 0
    for i, j in zip(Acoo.row, Acoo.col):
        if i >= j:
            continue
        jump = abs(int(idx_arr[i] - idx_arr[j]))
        total += 1
        if jump == 0:
            jump0 += 1
        if jump <= 1:
            jump1 += 1
    return {
        "tts_jump0_ratio": jump0 / total if total else np.nan,
        "tts_jump<=1_ratio": jump1 / total if total else np.nan,
    }


def metric_ges(dist_matrix, E, positions, position_labels, n_pairs, seed):
    rng = np.random.default_rng(seed)
    stretches = []
    reachable, total = 0, 0
    for p in position_labels:
        idxs = np.where(positions == p)[0]
        if len(idxs) < 2:
            continue
        n_try = min(n_pairs, len(idxs) * (len(idxs) - 1) // 2)
        for _ in range(n_try):
            i, j = rng.choice(idxs, size=2, replace=False)
            total += 1
            g = dist_matrix[i, j]
            if not np.isfinite(g):
                continue
            e = float(np.linalg.norm(E[i] - E[j]))
            if e < 1e-12:
                continue
            reachable += 1
            stretches.append(g / e)
    return {
        "ges_mean_stretch": float(np.mean(stretches)) if stretches else np.nan,
        "ges_reachable_ratio": reachable / total if total else np.nan,
    }


def metric_containment(dist_matrix, predecessors, positions, position_labels, n_pairs, seed):
    rng = np.random.default_rng(seed)
    per_position = {}
    for p in position_labels:
        idxs = np.where(positions == p)[0]
        if len(idxs) < 2:
            per_position[p] = np.nan
            continue
        n_try = min(n_pairs, len(idxs) * (len(idxs) - 1) // 2)
        scores = []
        for _ in range(n_try):
            i, j = rng.choice(idxs, size=2, replace=False)
            if not np.isfinite(dist_matrix[i, j]):
                continue
            path = reconstruct_path(predecessors, i, j)
            if not path:
                continue
            scores.append(np.mean([positions[node] == p for node in path]))
        per_position[p] = float(np.mean(scores)) if scores else np.nan

    result = {f"containment_{p}": v for p, v in per_position.items()}
    result["containment_mean_all"] = float(np.nanmean(list(per_position.values())))
    result["containment_mean_endpoints"] = float(np.nanmean(
        [per_position[position_labels[0]], per_position[position_labels[-1]]]
    ))
    result["containment_mean_middle"] = float(np.nanmean(
        [per_position[p] for p in position_labels[1:-1]]
    ))
    return result


def metric_greedy_walk(A, E, positions, position_labels, max_steps, seed=0):
    start_positions = set(position_labels[:3])
    target_position = position_labels[-1]
    corridor_positions = set(position_labels[3:5])

    start_ids = [i for i, p in enumerate(positions) if p in start_positions]
    if not start_ids:
        return {"walk_hit_rate": np.nan, "walk_mean_length": np.nan, "walk_corridor_pass_rate": np.nan}

    current = {sid: sid for sid in start_ids}
    visited = {sid: {sid} for sid in start_ids}
    paths = {sid: [sid] for sid in start_ids}
    stopped = {sid: (positions[sid] == target_position) for sid in start_ids}

    for _ in range(max_steps):
        new_current = {}
        for sid in start_ids:
            cur = current[sid]
            if stopped[sid]:
                new_current[sid] = cur
                continue
            nbrs = A.indices[A.indptr[cur]:A.indptr[cur + 1]]
            nbrs = [int(n) for n in nbrs if int(n) not in visited[sid]]
            if not nbrs:
                stopped[sid] = True
                new_current[sid] = cur
                continue
            sims = E[nbrs] @ E[cur]
            nxt = int(nbrs[int(np.argmax(sims))])
            visited[sid].add(nxt)
            new_current[sid] = nxt
            paths[sid].append(nxt)
            if positions[nxt] == target_position:
                stopped[sid] = True
        current = new_current
        if all(stopped.values()):
            break

    hits, lengths, corridor_hits = [], [], []
    for sid, path in paths.items():
        path_positions = [positions[n] for n in path]
        hit = target_position in path_positions
        hits.append(hit)
        lengths.append(len(path))
        if hit:
            corridor_hits.append(any(p in corridor_positions for p in path_positions))

    return {
        "walk_hit_rate": float(np.mean(hits)),
        "walk_mean_length": float(np.mean(lengths)),
        "walk_corridor_pass_rate": float(np.mean(corridor_hits)) if corridor_hits else np.nan,
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    sentences = df[TEXT_COL].astype(str).tolist()
    real_tiers = df[TIER_COL].astype(str).to_numpy()

    print("Computing confound scores...")
    confounds = get_confound_scores(sentences)
    print(f"Confounds available: {list(confounds.keys())}\n")

    all_rows = []

    for model in MODEL_LIST:
        for k in K_LIST:
            print(f"=== model={model}, k={k} ===")
            E, A = load_model_graph(model, k)
            if len(df) != E.shape[0]:
                print(f"  [skip] size mismatch")
                continue
            A_sym = A.maximum(A.T)

            print("  computing all-pairs shortest paths (shared across labelings)...")
            dist_matrix, predecessors = compute_full_shortest_paths(A_sym)

            labelings = {"real_tiers": real_tiers_to_positions(real_tiers, REAL_TIER_ORDER)}
            for name, scores in confounds.items():
                labelings[name] = bin_by_matched_sizes(scores, real_tiers, REAL_TIER_ORDER)

            for label_name, positions in labelings.items():
                row = {"model": model, "k": k, "labeling": label_name}
                row.update(metric_tts(A, positions, POSITION_LABELS))
                row.update(metric_ges(dist_matrix, E, positions, POSITION_LABELS, PAIRS_PER_POSITION, SEED))
                row.update(metric_containment(dist_matrix, predecessors, positions, POSITION_LABELS,
                                                PAIRS_PER_POSITION, SEED))
                row.update(metric_greedy_walk(A, E, positions, POSITION_LABELS, GREEDY_MAX_STEPS, SEED))
                all_rows.append(row)
                print(f"  [done] {label_name}")

    result_df = pd.DataFrame(all_rows)

    summary_cols = [
        "model", "k", "labeling",
        "tts_jump0_ratio", "tts_jump<=1_ratio",
        "ges_mean_stretch", "ges_reachable_ratio",
        "containment_mean_all", "containment_mean_endpoints", "containment_mean_middle",
        "walk_hit_rate", "walk_mean_length", "walk_corridor_pass_rate",
    ]
    print("\n" + "=" * 100)
    print("CONSOLIDATED COMPARISON: real tiers vs. alternative labelings")
    print("=" * 100)
    print(result_df[summary_cols].round(4).to_string(index=False))

    out_path = os.path.join(OUTPUT_ROOT, "alternative_labeling_all_metrics.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved full results (including per-position containment): {out_path}")


if __name__ == "__main__":
    main()
