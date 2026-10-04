"""
mac_pipeline/common/data.py
---------------------------
Question bank I/O and the one prompt format every model sees.
"""

import json

import pandas as pd

import config

LETTERS = "ABCDE"


def format_prompt(question: str, choices: list) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"Question: {question}\n{opts}\nAnswer:"


def load_questions(with_split: bool = True) -> pd.DataFrame:
    q = pd.read_csv(config.QUESTIONS_CSV)
    q["choices"] = q["choices"].apply(json.loads)
    if with_split and config.SPLIT_CSV.exists():
        s = pd.read_csv(config.SPLIT_CSV)
        q = q.merge(s, on="id", how="left")
    return q


def subset(q: pd.DataFrame, n: int) -> pd.DataFrame:
    """Seeded sample of n rows spread across subjects (the questions are stored grouped by
    subject, so the first n rows would all be one or two hard STEM subjects)."""
    if n <= 0 or n >= len(q):
        return q
    return q.sample(n, random_state=config.SEED).sort_index()


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
    p = s.pivot_table(index="id", columns="model",
                      values=["correct", "graded", "seconds", "energy_kwh_attr", "n_options"])
    full = p.dropna(subset=[("correct", r) for r in roles])
    if len(full) < len(p):
        # a role is missing some questions (e.g. one dataset never scored): everything
        # downstream silently shrinks to the intersection, so say so loudly
        cov = s.groupby(["model", "dataset"]).size().unstack(fill_value=0)
        print(f"WARNING: only {len(full)} of {len(p)} questions"
              f"{' in fold ' + fold if fold else ''} are scored by every role; the rest are dropped.\n"
              f"coverage (rows per model x dataset):\n{cov.to_string()}\n")
    return full
