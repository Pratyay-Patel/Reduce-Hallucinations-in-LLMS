"""
mac_pipeline/01_build_questions.py
----------------------------------
Build the question bank and the train/test split, once.

  mmlu  20 subjects x 50 (seeded sample of the test split)        4 options
  aqua  AQuA-RAT test + validation                                  5 options
  arc   ARC-Challenge test, seeded sample of 500                    3-5 options

Writes questions/questions.csv and questions/split.csv (70/30, stratified on dataset x subject).

Run:  python mac_pipeline/01_build_questions.py
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402  (sets HF_HOME on Windows before datasets is imported)

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common.data import LETTERS, format_prompt

rng = np.random.default_rng(config.SEED)


def _sample(n_total, n):
    if n_total <= n:
        return list(range(n_total))
    return sorted(rng.choice(n_total, n, replace=False).tolist())


def build_mmlu():
    rows = []
    for group, subjects in config.MMLU_SUBJECTS.items():
        for s in subjects:
            ds = load_dataset("cais/mmlu", s, split="test")
            for i in _sample(len(ds), config.MMLU_PER_SUBJECT):
                r = ds[i]
                rows.append(dict(id=f"mmlu:{s}:{i}", dataset="mmlu", subject=s, group=group,
                                 source_split="test", question=r["question"].strip(),
                                 choices=[str(c).strip() for c in r["choices"]], gold=int(r["answer"])))
    return rows


_AQUA_OPT = re.compile(r"^\s*([A-E])\s*[\)\.:]\s*")


def build_aqua():
    rows = []
    for split in config.AQUA_SPLITS:
        ds = load_dataset("deepmind/aqua_rat", "raw", split=split)
        for i, r in enumerate(ds):
            opts = [_AQUA_OPT.sub("", o).strip() for o in r["options"]]
            gold = LETTERS.index(r["correct"].strip().upper())
            if len(opts) != 5 or not (0 <= gold < 5):
                continue
            rows.append(dict(id=f"aqua:{split}:{i}", dataset="aqua", subject="aqua", group="math",
                             source_split=split, question=r["question"].strip(),
                             choices=opts, gold=gold))
    return rows


def build_arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i in _sample(len(ds), config.ARC_N):
        r = ds[i]
        labels = [str(x) for x in r["choices"]["label"]]
        texts = [str(x).strip() for x in r["choices"]["text"]]
        key = str(r["answerKey"])
        if key not in labels or not (3 <= len(texts) <= 5):
            continue
        rows.append(dict(id=f"arc:test:{i}", dataset="arc", subject="arc_challenge", group="science",
                         source_split="test", question=r["question"].strip(),
                         choices=texts, gold=labels.index(key)))
    return rows


def main():
    rows = build_mmlu() + build_aqua() + build_arc()
    q = pd.DataFrame(rows)
    before = len(q)
    q = q.drop_duplicates("question").reset_index(drop=True)
    q["n_options"] = q["choices"].apply(len)
    q["gold_letter"] = q["gold"].apply(lambda g: LETTERS[g])
    q["prompt_text"] = [format_prompt(a, b) for a, b in zip(q["question"], q["choices"])]
    q["choices"] = q["choices"].apply(json.dumps)

    strat = q["dataset"] + "/" + q["subject"]
    tr, te = train_test_split(q.index, test_size=config.TEST_FRAC, random_state=config.SEED,
                              stratify=strat)
    split = pd.DataFrame({"id": q["id"], "fold": "train"})
    split.loc[te, "fold"] = "test"

    cols = ["id", "dataset", "subject", "group", "source_split", "question", "choices",
            "n_options", "gold", "gold_letter", "prompt_text"]
    q[cols].to_csv(config.QUESTIONS_CSV, index=False)
    split.to_csv(config.SPLIT_CSV, index=False)

    print(f"questions: {len(q)} (dropped {before - len(q)} duplicate texts)")
    m = q.merge(split, on="id")
    print(m.groupby(["dataset", "fold"]).size().unstack(fill_value=0))
    print("\noptions per dataset:")
    print(m.groupby("dataset")["n_options"].value_counts().unstack(fill_value=0))
    print(f"\nwrote {config.QUESTIONS_CSV}\n      {config.SPLIT_CSV}")


if __name__ == "__main__":
    main()
