"""
mac_pipeline/common/data.py
---------------------------
Question bank I/O and the one prompt format every model sees.
"""

import json

import numpy as np
import pandas as pd

import config

LETTERS = "ABCDE"


def format_prompt(question: str, choices: list) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


def chance_of(n_options):
    """Per-question chance accuracy: 1/k for k options, 0 for generation tasks (n_options = 0)."""
    n = np.asarray(n_options, dtype=float)
    return np.where(n > 0, 1.0 / np.where(n > 0, n, 1.0), 0.0)


def is_codegen(row) -> bool:
    return row["dataset"] == "codegen"


def score_row(scorer, row) -> dict:
    """Score one question-bank row with the right method for its dataset."""
    if is_codegen(row):
        t = row["test_code"]
        test = json.loads(t) if isinstance(t, str) else t
        return scorer.score_code(row["question"], test, max_new_tokens=config.CODEGEN_MAX_NEW_TOKENS,
                                 timeout=config.CODEGEN_EXEC_TIMEOUT)
    if row["dataset"] in config.COT_DATASETS:
        return scorer.score_cot(row["question"], row["choices"], int(row["gold"]),
                                max_new_tokens=config.COT_MAX_NEW_TOKENS)
    return scorer.score(row["question"], row["choices"], int(row["gold"]))


def load_questions(with_split: bool = True) -> pd.DataFrame:
    q = pd.read_csv(config.QUESTIONS_CSV)
    q["choices"] = q["choices"].apply(json.loads)
    if "test_code" not in q.columns:
        q["test_code"] = None
    if with_split and config.SPLIT_CSV.exists():
        s = pd.read_csv(config.SPLIT_CSV)
        q = q.merge(s, on="id", how="left")
    return q


def subset(q: pd.DataFrame, n: int) -> pd.DataFrame:
    """Seeded sample of n rows spread across subjects (the questions are stored grouped by
    subject, so the first n rows would all be one or two hard STEM subjects)."""
    if n <= 0 or n >= len(q):
        return q
    # not config.SEED: the fold split used that seed on the same rows, and pandas.sample with
    # the same seed draws the same permutation, so a --limit sample came out all-train
    return q.sample(n, random_state=config.SEED + 1000).sort_index()


def load_scores(roles=None, repeat: int = 0) -> pd.DataFrame:
    """All score rows for the given roles (default: every role with a file)."""
    roles = roles or config.ALL_ROLES
    frames = []
    for r in roles:
        p = config.SCORES_DIR / (f"{r}.csv" if repeat == 0 else f"{r}_r{repeat}.csv")
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(f"no score files for {roles} in {config.SCORES_DIR}")
    return pd.concat(frames, ignore_index=True)


def pivot_scores(scores: pd.DataFrame, roles, fold=None):
    """Wide table indexed by question id: correct/graded/seconds/energy per role.
    Keeps only questions scored by every role."""
    s = scores[scores["model"].isin(roles)]
    if fold is not None:
        s = s[s["fold"] == fold]
    # dropna=False: a --no-energy dev run has energy_kwh_attr all NaN, which pivot_table would
    # otherwise drop silently and break every downstream column lookup
    p = s.pivot_table(index="id", columns="model",
                      values=["correct", "graded", "seconds", "energy_kwh_attr", "n_options"], dropna=False)
    full = p.dropna(subset=[("correct", r) for r in roles])
    if len(full) < len(p):
        # a role is missing some questions (e.g. one dataset never scored): everything
        # downstream silently shrinks to the intersection, so say so loudly
        cov = s.groupby(["model", "dataset"]).size().unstack(fill_value=0)
        print(f"WARNING: only {len(full)} of {len(p)} questions"
              f"{' in fold ' + fold if fold else ''} are scored by every role; the rest are dropped.\n"
              f"coverage (rows per model x dataset):\n{cov.to_string()}\n")
    return full
