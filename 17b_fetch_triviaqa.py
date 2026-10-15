"""Download and subsample TriviaQA closed-book questions (Section 2.5.3, short-form factuality).

Streams the rc.nocontext validation split from Hugging Face, shuffles it with a fixed seed, and
writes prompt dicts (question, gold answer, alias list) to a JSON file.
Usage: python 17b_fetch_triviaqa.py --n 500 --seed 0 --out triviaqa_prompts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).parent

# Instruction wrapped around each question to elicit a short answer.
PROMPT_TEMPLATE = ("Answer the following trivia question with just the answer, "
                   "as briefly as possible (a few words, no explanation).\n\n"
                   "Question: {q}")


def load_triviaqa(n: int, seed: int, config: str) -> List[Dict]:
    """Return up to n shuffled validation questions that have a non-empty question and alias set."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("pip install datasets --break-system-packages") from e

    # Retry with trust_remote_code=True if the plain load fails (older dataset loaders).
    try:
        ds = load_dataset("mandarjoshi/trivia_qa", config, split="validation", streaming=True)
    except Exception:
        ds = load_dataset("mandarjoshi/trivia_qa", config, split="validation",
                          streaming=True, trust_remote_code=True)
    # Streaming shuffle: approximate, within a 2000-example buffer.
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    prompts = []
    for row in ds:
        if len(prompts) >= n:
            break
        q = (row.get("question") or "").strip()
        ans = row.get("answer") or {}
        value = ans.get("value") or ""
        # Gold aliases: canonical value + aliases + normalized aliases, deduplicated and sorted.
        alias_set = sorted({a for a in ([value]
                                        + list(ans.get("aliases") or [])
                                        + list(ans.get("normalized_aliases") or []))
                            if a})
        if not q or not alias_set:
            continue
        prompts.append({
            "id": f"TQA{len(prompts)+1:04d}",
            "prompt": PROMPT_TEMPLATE.format(q=q),
            "metadata": {"question": q, "answer": value, "aliases": alias_set,
                         "source": "triviaqa"},
        })
    return prompts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--config", type=str, default="rc.nocontext")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="triviaqa_prompts.json")
    args = p.parse_args()

    prompts = load_triviaqa(args.n, args.seed, args.config)
    out = Path(args.out)
    out.write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
    print(f"Wrote {len(prompts)} TriviaQA prompts -> {out}")
    if prompts:
        m = prompts[0]["metadata"]
        print(f"  example: {m['question']!r} -> answer={m['answer']!r} "
              f"(#aliases={len(m['aliases'])})")


if __name__ == "__main__":
    main()
