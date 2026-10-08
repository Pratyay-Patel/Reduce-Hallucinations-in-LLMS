"""
mac_pipeline/dev_cot_test.py
----------------------------
Small comparison of the two scoring protocols on math multiple choice:
  letter   zero-shot option likelihood (the pipeline's default)
  cot      reason first (greedy, bounded), then score the letters after a fixed stem

Runs the 1.5B generalist and the 1.5B math specialist on N AQuA-RAT test questions.

  python mac_pipeline/dev_cot_test.py --n 60 --max-new-tokens 256 --out <dir>
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd
from datasets import load_dataset

from common.data import LETTERS
from common.scorer import OptionScorer

MODELS = {"generalist_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "math_1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct"}


def load_aqua(n, seed=0):
    import re
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    rx = re.compile(r"^\s*([A-E])\s*[\)\.:]\s*")
    rows = []
    for i, r in enumerate(ds):
        opts = [rx.sub("", o).strip() for o in r["options"]]
        if len(opts) != 5:
            continue
        rows.append(dict(id=f"aqua:test:{i}", question=r["question"].strip(), choices=opts,
                         gold=LETTERS.index(r["correct"].strip().upper())))
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), n, replace=False).tolist())
    return [rows[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    qs = load_aqua(args.n)
    dev = config.pick_device()
    print(f"{len(qs)} AQuA questions, device {dev}", flush=True)
    rows = []
    for role in args.models:
        sc = OptionScorer(MODELS[role], dev, chat_template=False)
        for proto in ("letter", "cot"):
            t0 = time.time()
            for k, q in enumerate(qs):
                if proto == "letter":
                    s = sc.score(q["question"], q["choices"], q["gold"])
                    s.update(reasoning_tokens=0, truncated=0, stated_letter=None, stated_matches_scored=None, reasoning="")
                else:
                    s = sc.score_cot(q["question"], q["choices"], q["gold"], max_new_tokens=args.max_new_tokens)
                s.pop("logprobs", None); s.pop("rendered_prompt", None)
                rows.append(dict(id=q["id"], model=role, protocol=proto, gold=q["gold"], **s))
                if proto == "cot":   # save after every reasoning question: a stopped run keeps its rows
                    pd.DataFrame(rows).to_csv(os.path.join(args.out, "cot_test.csv"), index=False)
                if (k + 1) % 20 == 0:
                    sub = [r for r in rows if r["model"] == role and r["protocol"] == proto]
                    print(f"  {role} {proto}: {k + 1}/{len(qs)} acc {np.mean([r['correct'] for r in sub]):.3f} "
                          f"{np.mean([r['seconds'] for r in sub]):.1f} s/q", flush=True)
            print(f"{role} {proto}: done in {time.time() - t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "cot_test.csv"), index=False)
        sc.unload()
    df = pd.DataFrame(rows)
    print("\n=== summary ===")
    g = df.groupby(["model", "protocol"]).agg(n=("correct", "size"), acc=("correct", "mean"),
                                              s_per_q=("seconds", "mean"), reasoning_tokens=("reasoning_tokens", "mean"),
                                              truncated=("truncated", "mean"), conf=("graded", "mean"))
    print(g.round(3).to_string())
    cot = df[df.protocol == "cot"]
    for role, d in cot.groupby("model"):
        st = d.dropna(subset=["stated_matches_scored"])
        top = d.pred_letter.value_counts(normalize=True)
        print(f"{role}: stated an 'Answer: X' line in {len(st)}/{len(d)}; stated==scored in {st.stated_matches_scored.mean() if len(st) else float('nan'):.2f}; "
              f"top letter {top.index[0]} {top.iloc[0]:.0%}")
    with open(os.path.join(args.out, "cot_examples.txt"), "w", encoding="utf-8") as f:
        for _, r in cot.groupby("model").head(2).iterrows():
            f.write(f"### {r.model} id={r.id} gold={LETTERS[r.gold]} scored={r.pred_letter} correct={r.correct}\n{r.reasoning}\n\n")


if __name__ == "__main__":
    main()
