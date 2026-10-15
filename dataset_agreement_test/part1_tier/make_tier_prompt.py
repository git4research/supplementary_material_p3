"""Build the blind tier-assignment prompt for part 1 of the agreement test (Appendix B.5).

Shuffles the deduplicated sentences with a fixed seed, assigns new IDs (S001, ...),
and writes prompts/tier_prompt_full.txt, prompts/tier_sentences.json and the key ../key/tier_key.csv.
Usage: python make_tier_prompt.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Make the shared common.py importable.
sys.path.insert(0, str(HERE.parent))
from common import SEED_TIER, load_dataset

df = load_dataset()
# Shuffle and re-ID so neither order nor IDs reveal the original tier or index.
order = np.random.default_rng(SEED_TIER).permutation(len(df))
df = df.iloc[order].reset_index(drop=True)
width = max(3, len(str(len(df))))
df.insert(0, "id", [f"S{i + 1:0{width}d}" for i in range(len(df))])

(HERE / "prompts").mkdir(exist_ok=True)
rubric = (HERE / "rubric_tier.txt").read_text(encoding="utf-8").strip()
lines = [f"{r.id}\t{r.sentence}" for r in df.itertuples()]
reminder = (f"END OF SENTENCES.\n\nNow output your answer: exactly {len(df)} lines, one per sentence, "
            f"in the order given ({df.id.iloc[0]} to {df.id.iloc[-1]}), each line `ID,Tier` using only "
            "Collapse, Striving, Conflict, Activation, Growth, Clarity, Unity. Judge every item yourself (no code or tools). Plain text inside a single code block (```), no JSON, no explanations, headers, "
            "or other text. If you cannot fit all lines in one reply, stop at the end of a complete line; "
            "I will reply \"continue\" and you resume from the next ID in a new code block.")
# Prompt = rubric + tab-separated ID/sentence lines + answer-format reminder.
prompt = (f"{rubric}\n\nSENTENCES ({len(df)} total; output exactly {len(df)} lines)\n\n"
          + "\n".join(lines) + "\n\n" + reminder + "\n")
(HERE / "prompts" / "tier_prompt_full.txt").write_text(prompt, encoding="utf-8")
(HERE / "prompts" / "tier_sentences.json").write_text(
    json.dumps([{"id": r.id, "sentence": r.sentence} for r in df.itertuples()], ensure_ascii=False, indent=1),
    encoding="utf-8")
# Answer key mapping new IDs back to original index, tier and score.
df[["id", "index", "tier", "score", "sentence"]].to_csv(HERE.parent / "key" / "tier_key.csv", index=False)

print(f"rows in dataset: {df.attrs.get('n_rows', len(df))}, duplicates dropped: "
      f"{df.attrs.get('n_dropped_duplicates', 0)}, sentences in prompt: {len(df)}")
print(f"wrote prompts/tier_prompt_full.txt ({len(prompt):,} chars) and ../key/tier_key.csv")
