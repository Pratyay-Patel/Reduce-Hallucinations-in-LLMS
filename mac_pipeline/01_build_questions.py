"""
mac_pipeline/01_build_questions.py
----------------------------------
Build the question bank and the train/test split, once.

  mmlu      20 subjects x 50 (seeded sample of the test split)         4 options
  csqa      CommonsenseQA validation, seeded sample of 500              5 options
  arc       ARC-Challenge test, seeded sample of 500                     3-5 options
  codemmlu  CodeMMLU knowledge subsets, 100 each (5 subsets)            3-4 options
  codegen   HumanEval (164) + MBPP sanitized test (257)                  generation + unit tests

Only the datasets listed in config.DATASETS are built.
Writes questions/questions.csv and questions/split.csv (70/30, stratified on dataset x subject).
If a split.csv already exists, questions already in it keep their fold; only new ids are
assigned, so swapping one dataset never disturbs what has already been scored.

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


def build_csqa():
    # own generator: build_arc() draws from the shared `rng` right after build_mmlu(), and
    # that sequence must stay identical so the ARC sample (already scored) does not change
    own = np.random.default_rng(config.SEED + 1)
    ds = load_dataset("tau/commonsense_qa", split="validation")
    idx = sorted(own.choice(len(ds), min(config.CSQA_N, len(ds)), replace=False).tolist())
    rows = []
    for i in idx:
        r = ds[i]
        labels = [str(x) for x in r["choices"]["label"]]
        texts = [str(x).strip() for x in r["choices"]["text"]]
        key = str(r["answerKey"]).strip()
        if key not in labels or len(texts) != 5:
            continue
        rows.append(dict(id=f"csqa:validation:{i}", dataset="csqa", subject="commonsense", group="commonsense",
                         source_split="validation", question=r["question"].strip(),
                         choices=texts, gold=labels.index(key)))
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


def build_codemmlu():
    # own generator (seed + 2): must not disturb the MMLU/ARC draws from the shared `rng`
    own = np.random.default_rng(config.SEED + 2)
    rows = []
    for sub in config.CODEMMLU_SUBSETS:
        ds = load_dataset("Fsoft-AIC/CodeMMLU", sub, split="test")
        idx = sorted(own.choice(len(ds), min(config.CODEMMLU_PER_SUBSET, len(ds)), replace=False).tolist())
        for i in idx:
            r = ds[i]
            texts = [str(c).strip() for c in r["choices"]]
            key = str(r["answer"]).strip().upper()
            if key not in LETTERS[:len(texts)] or not (3 <= len(texts) <= 5):
                continue
            rows.append(dict(id=f"codemmlu:{sub}:{i}", dataset="codemmlu", subject=sub, group="code",
                             source_split="test", question=r["question"].strip(),
                             choices=texts, gold=LETTERS.index(key)))
    return rows


def build_codegen():
    """Code generation problems. `question` is the full text the model is asked to solve (also
    the router's embedding text); `test_code` holds what the executor needs. choices is empty,
    n_options 0 (chance accuracy 0), gold -1: these rows are scored by running the tests."""
    own = np.random.default_rng(config.SEED + 3)
    rows = []
    ds = load_dataset("openai/openai_humaneval", split="test")
    idx = sorted(own.choice(len(ds), min(config.HUMANEVAL_N, len(ds)), replace=False).tolist())
    for i in idx:
        r = ds[i]
        prompt = r["prompt"].rstrip() + "\n"
        question = "Complete the following Python function.\n\n```python\n" + prompt + "```"
        rows.append(dict(id=f"codegen:humaneval:{r['task_id'].split('/')[-1]}", dataset="codegen",
                         subject="humaneval", group="code", source_split="test", question=question,
                         choices=[], gold=-1,
                         test_code=json.dumps(dict(kind="humaneval", prompt=prompt,
                                                   entry_point=r["entry_point"], test=r["test"]))))
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    idx = sorted(own.choice(len(ds), min(config.MBPP_N, len(ds)), replace=False).tolist())
    for i in idx:
        r = ds[i]
        asserts = [str(a) for a in r["test_list"]]
        imports = [str(a) for a in (r["test_imports"] or [])]
        question = (r["prompt"].strip() + "\nYour code should pass these tests:\n\n```python\n"
                    + "\n".join(asserts) + "\n```")
        rows.append(dict(id=f"codegen:mbpp:{r['task_id']}", dataset="codegen", subject="mbpp",
                         group="code", source_split="test", question=question, choices=[], gold=-1,
                         test_code=json.dumps(dict(kind="mbpp", imports=imports, asserts=asserts))))
    return rows


BUILDERS = {"mmlu": build_mmlu, "csqa": build_csqa, "arc": build_arc,
            "codemmlu": build_codemmlu, "codegen": build_codegen}


def main():
    # build order is fixed (mmlu, csqa, arc share/seed generators in that order), not DATASETS order
    rows = []
    for name, fn in BUILDERS.items():
        if name in config.DATASETS:
            rows += fn()
    q = pd.DataFrame(rows)
    if "test_code" not in q.columns:
        q["test_code"] = ""
    before = len(q)
    q = q.drop_duplicates("question").reset_index(drop=True)
    q["n_options"] = q["choices"].apply(len)
    q["gold_letter"] = q["gold"].apply(lambda g: LETTERS[g] if g >= 0 else "")
    q["prompt_text"] = [format_prompt(a, b) if b else a for a, b in zip(q["question"], q["choices"])]
    q["choices"] = q["choices"].apply(json.dumps)
    q["test_code"] = q["test_code"].where(q["test_code"].notna(), "")

    # keep existing fold assignments; split only ids not seen before
    old = pd.read_csv(config.SPLIT_CSV) if config.SPLIT_CSV.exists() else pd.DataFrame(columns=["id", "fold"])
    known = dict(zip(old["id"], old["fold"]))
    new_idx = q.index[~q["id"].isin(known)]
    split = pd.DataFrame({"id": q["id"], "fold": q["id"].map(known)})
    if len(new_idx):
        strat = (q.loc[new_idx, "dataset"] + "/" + q.loc[new_idx, "subject"])
        _, te = train_test_split(new_idx, test_size=config.TEST_FRAC, random_state=config.SEED, stratify=strat)
        split.loc[new_idx, "fold"] = "train"
        split.loc[te, "fold"] = "test"
    print(f"kept folds for {len(q) - len(new_idx)} existing ids; assigned {len(new_idx)} new ids")

    cols = ["id", "dataset", "subject", "group", "source_split", "question", "choices",
            "n_options", "gold", "gold_letter", "prompt_text", "test_code"]
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
