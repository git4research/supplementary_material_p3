import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import N_PAIRS, SEED_PAIRS, load_dataset

df = load_dataset()
n = len(df)
rng = np.random.default_rng(SEED_PAIRS)
seen, pairs = set(), []
while len(pairs) < N_PAIRS:
    i, j = (int(x) for x in rng.choice(n, size=2, replace=False))
    key = (min(i, j), max(i, j))
    if key in seen:
        continue
    seen.add(key)
    pairs.append((i, j) if rng.random() < 0.5 else (j, i))

rows = []
for k, (a, b) in enumerate(pairs, 1):
    ra, rb = df.iloc[a], df.iloc[b]
    higher = "A" if ra.score > rb.score else "B" if rb.score > ra.score else "tie"
    rows.append({"pair_id": f"P{k:04d}", "A": ra.sentence, "B": rb.sentence,
                 "A_index": int(ra["index"]), "B_index": int(rb["index"]),
                 "A_tier": ra.tier, "B_tier": rb.tier, "A_score": float(ra.score), "B_score": float(rb.score),
                 "higher": higher, "gap": round(abs(float(ra.score) - float(rb.score)), 4)})
P = pd.DataFrame(rows)

(HERE / "prompts").mkdir(exist_ok=True)
rubric = (HERE / "rubric_energy.txt").read_text(encoding="utf-8").strip()
blocks = [f"{r.pair_id}\n  A: {r.A}\n  B: {r.B}" for r in P.itertuples()]
reminder = (f"END OF PAIRS.\n\nNow output your answer: exactly {len(P)} lines, one per pair, in the order "
            f"given ({P.pair_id.iloc[0]} to {P.pair_id.iloc[-1]}), each line `ID,A` or `ID,B` naming the "
            "sentence with HIGHER energy. Judge every item yourself (no code or tools). Plain text inside a single code block (```), no JSON, no explanations, headers, or other text. If you cannot fit all "
            "lines in one reply, stop at the end of a complete line; I will reply \"continue\" and you "
            "resume from the next ID in a new code block.")
prompt = (f"{rubric}\n\nPAIRS ({len(P)} total; output exactly {len(P)} lines)\n\n" + "\n\n".join(blocks)
          + "\n\n" + reminder + "\n")
(HERE / "prompts" / "energy_prompt_full.txt").write_text(prompt, encoding="utf-8")
(HERE / "prompts" / "energy_pairs.json").write_text(
    json.dumps([{"id": r.pair_id, "A": r.A, "B": r.B} for r in P.itertuples()], ensure_ascii=False, indent=1),
    encoding="utf-8")
P[["pair_id", "A", "B"]].assign(answer="").to_excel(HERE / "prompts" / "energy_pairs.xlsx", index=False)
P.to_csv(HERE.parent / "key" / "pairs_key.csv", index=False)

print(f"sentences: {n}, pairs: {len(P)}, score-tied pairs: {(P.higher == 'tie').sum()}, "
      f"A shown as higher: {(P.higher == 'A').mean():.3f}")
print(f"gap quantiles: {P.gap.quantile([.1, .25, .5, .75, .9]).round(2).to_dict()}")
print(f"wrote prompts/energy_prompt_full.txt ({len(prompt):,} chars), energy_pairs.json/.xlsx, ../key/pairs_key.csv")
