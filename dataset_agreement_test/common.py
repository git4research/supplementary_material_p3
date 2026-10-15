"""Shared settings and helpers for the dataset agreement test (Appendix B.5).

Defines the dataset path, ordered tier labels, random seeds and pair count, a
loader that cleans and deduplicates the labeled sentences, and a fuzzy-match
check that flags sentences quoted as examples in the judge rubrics.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET = HERE / "key" / "full_dataset_with_scores_new.csv"
TIER_ORDER = ["Collapse", "Striving", "Conflict", "Activation", "Growth", "Clarity", "Unity"]
TIER_IDX = {t: i for i, t in enumerate(TIER_ORDER)}

# Seeds for the sentence shuffle (part 1) and pair sampling (part 2).
SEED_TIER = 20260923
SEED_PAIRS = 20260924
N_PAIRS = 1000
# Minimum difflib similarity for a sentence to count as a rubric example.
RUBRIC_MATCH_RATIO = 0.85


def norm_text(s: str) -> str:
    """Normalize for fuzzy matching: straighten quotes, collapse whitespace, lowercase, drop punctuation except apostrophes."""
    s = str(s).replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s).strip().lower()
    return re.sub(r"[^\w' ]", "", s).strip()


def load_dataset() -> pd.DataFrame:
    """Load labeled sentences, drop incomplete rows, validate tiers, drop duplicate sentences, add tier_idx."""
    df = pd.read_csv(DATASET, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["sentence", "tier", "score"]).copy()
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["score"])
    bad = sorted(set(df["tier"]) - set(TIER_ORDER))
    if bad:
        raise ValueError(f"unknown tiers: {bad}")
    n0 = len(df)
    # Exact duplicate sentences are dropped (first kept); counts are recorded in df.attrs.
    df = df.drop_duplicates(subset="sentence", keep="first").reset_index(drop=True)
    df.attrs["n_rows"], df.attrs["n_dropped_duplicates"] = n0, n0 - len(df)
    df["tier_idx"] = df["tier"].map(TIER_IDX)
    return df


def rubric_overlap(df: pd.DataFrame, rubric_paths) -> pd.Series:
    """True where a sentence fuzzy-matches a quoted string (8+ chars) in any rubric file."""
    quoted = []
    for p in rubric_paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            quoted += [q for q in re.findall(r'"([^"]+)"', line) if len(q) >= 8]
    qn = [norm_text(q) for q in quoted]
    def hit(s):
        sn = norm_text(s)
        return any(difflib.SequenceMatcher(None, sn, q).ratio() >= RUBRIC_MATCH_RATIO for q in qn)
    return df["sentence"].apply(hit)
