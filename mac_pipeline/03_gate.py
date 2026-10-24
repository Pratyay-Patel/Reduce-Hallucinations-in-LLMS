"""
mac_pipeline/03_gate.py
-----------------------
The complementarity gate, for every ordered pair of models, per dataset and overall.

  On the questions model A got WRONG, how often is model B right, and is that better than
  guessing (each question's own 1/k)?  One-sided test, both directions.
    PASS both ways  -> complementarity: routing between A and B can beat either alone
    one way only    -> dominance: just use the better one
    neither         -> nothing to route

The decision uses the TRAIN split (so no test peeking); the test split is reported too.

  python mac_pipeline/03_gate.py
Outputs results/gate.csv, results/gate.md
"""

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common.data import chance_of, load_scores, pivot_scores

ALPHA = 0.05
MC = 20000


def chance_test(hits: int, ks: np.ndarray, seed: int = 0):
    """P(B right on >= hits of A's errors | B guesses), with per-question chance 1/k."""
    n = len(ks)
    if n == 0:
        return np.nan, np.nan
    p = chance_of(ks)                      # 0 for generation questions (n_options = 0)
    expected = float(p.sum())
    if expected == 0.0:                    # guessing never passes a unit test
        return 0.0, (0.0 if hits > 0 else 1.0)
    if np.all(ks == ks[0]):
        return expected, binomtest(hits, n, float(p[0]), alternative="greater").pvalue
    rng = np.random.default_rng(seed)
    draws = (rng.random((MC, n)) < p).sum(axis=1)
    return expected, float((draws >= hits).mean())


def gate_rows(piv, roles, scope, fold):
    C = {r: piv[("correct", r)].values.astype(int) for r in roles}
    K = piv[("n_options", roles[0])].values.astype(int)
    rows = []
    for a, b in itertools.permutations(roles, 2):
        wrong = C[a] == 0
        n = int(wrong.sum())
        hits = int(C[b][wrong].sum())
        exp, p = chance_test(hits, K[wrong])
        rows.append(dict(scope=scope, fold=fold, A=a, B=b, n_A_wrong=n, B_right_on_A_wrong=hits,
                         rate=hits / n if n else np.nan, chance=exp / n if n else np.nan,
                         headroom_pts=100 * (hits - exp) / len(piv) if n else np.nan,
                         p_value=p, passes=bool(p < ALPHA) if n else False,
                         acc_A=C[a].mean(), acc_B=C[b].mean(),
                         oracle_AB=np.maximum(C[a], C[b]).mean(), n_questions=len(piv)))
    return rows


def main():
    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    if len(roles) < 2:
        sys.exit(f"need at least two scored models, have {roles}")
    print(f"models: {roles}")
    out = []
    for fold in ("train", "test"):
        piv = pivot_scores(scores, roles, fold=fold)
        out += gate_rows(piv, roles, "all", fold)
        for ds in config.DATASETS:
            ids = scores[(scores["dataset"] == ds)]["id"].unique()
            sub = piv[piv.index.isin(ids)]
            if len(sub):
                out += gate_rows(sub, roles, ds, fold)
    g = pd.DataFrame(out)
    g.to_csv(config.RESULTS_DIR / "gate.csv", index=False)

    # pair verdicts from TRAIN, overall
    lines = ["# Complementarity gate", "",
             "Decision from the train split; test shown for reference. "
             "`rate` = B's accuracy on A's errors; `chance` = mean 1/k there; "
             "`headroom` = chance-corrected points B could add to A on the whole set.", ""]
    for fold in ("train", "test"):
        t = g[(g.scope == "all") & (g.fold == fold)]
        lines += [f"## Overall, {fold} (n={int(t.n_questions.iloc[0])})", "",
                  "| A | B | A wrong | B right there | rate | chance | headroom (pts) | p | passes |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for _, r in t.iterrows():
            lines.append(f"| {r.A} | {r.B} | {r.n_A_wrong} | {r.B_right_on_A_wrong} | {r.rate:.3f} | "
                         f"{r.chance:.3f} | {r.headroom_pts:+.1f} | {r.p_value:.3f} | "
                         f"{'PASS' if r.passes else '-'} |")
        lines.append("")
    tr = g[(g.scope == "all") & (g.fold == "train")].set_index(["A", "B"])
    lines += ["## Pair verdicts (train, overall)", "", "| pair | verdict |", "|---|---|"]
    verdicts = {}
    for a, b in itertools.combinations(roles, 2):
        ab, ba = tr.loc[(a, b), "passes"], tr.loc[(b, a), "passes"]
        v = ("COMPLEMENTARY: both pass -> routable" if ab and ba else
             f"DOMINANCE: only {b} fixes {a}'s errors -> use {b}" if ab and not ba else
             f"DOMINANCE: only {a} fixes {b}'s errors -> use {a}" if ba and not ab else
             "NOTHING TO ROUTE: neither passes")
        verdicts[(a, b)] = v
        lines.append(f"| {a} / {b} | {v} |")
    pool_pairs = [(a, b) for a, b in itertools.combinations(config.POOL, 2) if a in roles and b in roles]
    complementary = [p for p in pool_pairs if verdicts.get(p, "").startswith("COMPLEMENTARY")]
    lines += ["", "## Decision", ""]
    if complementary:
        lines.append(f"At least one pool pair is complementary ({complementary}). "
                     "Train the hybrid router (04) and compare it to the plain cascade.")
    else:
        lines.append("No pool pair is complementary. Report the plain cascade only; "
                     "the hybrid step is skipped (dominance regime).")
    # per-dataset table, train
    lines += ["", "## Per dataset (train)", "",
              "| dataset | A | B | rate | chance | headroom | p | passes |", "|---|---|---|---:|---:|---:|---:|---|"]
    for _, r in g[(g.scope != "all") & (g.fold == "train")].iterrows():
        lines.append(f"| {r.scope} | {r.A} | {r.B} | {r.rate:.3f} | {r.chance:.3f} | {r.headroom_pts:+.1f} | "
                     f"{r.p_value:.3f} | {'PASS' if r.passes else '-'} |")
    # base at chance anywhere?
    lines += ["", "## Base model above chance per dataset (train)", ""]
    for ds in config.DATASETS:
        s = scores[(scores.model == config.BASE) & (scores.fold == "train") & (scores.dataset == ds)]
        if len(s):
            acc, ch = s.correct.mean(), float(chance_of(s.n_options).mean())
            flag = "OK" if acc > ch + 0.05 else "AT CHANCE -- confidence there is noise"
            lines.append(f"- {ds}: base {acc:.3f} vs chance {ch:.3f}  {flag}")
    (config.RESULTS_DIR / "gate.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[: 12 + 2 * len(roles) * (len(roles) - 1)]))
    print(f"\nwrote {config.RESULTS_DIR / 'gate.csv'} and gate.md")


if __name__ == "__main__":
    main()
