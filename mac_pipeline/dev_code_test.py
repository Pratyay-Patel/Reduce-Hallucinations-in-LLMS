"""
mac_pipeline/dev_code_test.py
-----------------------------
Small comparison on the CodeMMLU questions in the bank: 1.5B generalist vs 1.5B coder vs 3B
(8-bit on the 4 GB dev GPU; fp16 on the Mac), letter-scored, raw and chat formats.

  python mac_pipeline/dev_code_test.py --out <dir> [--limit N] [--eight-bit-3b]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd

from common.data import load_questions

MODELS = {"generalist_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
          "coder_1.5b": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
          "qwen3b": "Qwen/Qwen2.5-3B-Instruct"}


def make_scorer(mid, dev, chat, eight_bit):
    if eight_bit:
        # the PoC scorer has the bitsandbytes int8 path (CUDA only)
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "poc"))
        import score_options
        score_options.LETTERS = "ABCDE"          # the PoC scorer knows A-D; CodeMMLU has 5-option items
        return score_options.OptionScorer(mid, eight_bit=True, chat_template=chat, max_length=1024)
    from common.scorer import OptionScorer
    return OptionScorer(mid, dev, chat_template=chat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--eight-bit-3b", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    q = load_questions()
    q = q[q.dataset == "codemmlu"].reset_index(drop=True)
    if args.limit:
        q = q.sample(args.limit, random_state=config.SEED).sort_index()
    dev = config.pick_device()
    print(f"{len(q)} CodeMMLU questions, device {dev}", flush=True)
    rows = []
    for role in args.models:
        for chat in (False, True):
            eb = args.eight_bit_3b and role == "qwen3b"
            sc = make_scorer(MODELS[role], dev, chat, eb)
            t0 = time.time()
            for k, r in q.iterrows():
                s = sc.score(r["question"], r["choices"], int(r["gold"]))
                rows.append(dict(id=r["id"], subject=r["subject"], model=role, chat=int(chat), eight_bit=int(eb),
                                 gold=r["gold"], pred=s["pred"], correct=s["correct"], graded=s["graded"],
                                 seconds=s["seconds"]))
            sub = [x for x in rows if x["model"] == role and x["chat"] == int(chat)]
            print(f"{role:<16} {'chat' if chat else 'raw ':<5} acc {np.mean([x['correct'] for x in sub]):.3f}  "
                  f"{np.mean([x['seconds'] for x in sub]):.2f} s/q  ({time.time() - t0:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(args.out, "code_test.csv"), index=False)
            sc.unload()
    df = pd.DataFrame(rows)
    print("\n=== accuracy by model x format ===")
    print(df.pivot_table(index="model", columns="chat", values="correct", aggfunc="mean").round(3).to_string())
    best = df.groupby(["model", "chat"]).correct.mean().reset_index().sort_values("correct", ascending=False)
    best = best.groupby("model").head(1)
    print("\n=== best format per model, by subject ===")
    sel = df.merge(best[["model", "chat"]], on=["model", "chat"])
    print(sel.pivot_table(index="subject", columns="model", values="correct", aggfunc="mean").round(3).to_string())
    print("\nletters predicted (best format):")
    for m, g in sel.groupby("model"):
        vc = pd.Series(g.pred).map(lambda i: "ABCDE"[int(i)]).value_counts(normalize=True).round(2)
        print(f"  {m}: {vc.to_dict()}")


if __name__ == "__main__":
    main()
