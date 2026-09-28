"""
mac_pipeline/05_simulate.py
---------------------------
Test split, stored scores, attributed energy. Accuracy, energy (kWh) and emissions (gCO2e)
for every configuration:

  singles           each model alone
  oracle            best of pool / best of base+pool per question
  cascade base->X   base answers; confidence < tau -> X            (tau grid)
  hybrid            base answers; confidence < tau -> router-picked pool member
  random            escalation at matched rate, and the cost-matched random line
  matched-accuracy  cheapest cascade / hybrid point that matches each single model

Router overhead (embedding energy per question) is charged on every escalated question in
the hybrid. Per-question energy for a model is its attributed share of that model's run.

  python mac_pipeline/05_simulate.py
Outputs results/simulated.csv, results/simulated.md, results/frontier.png
"""

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd

from common import energy
from common.data import load_questions, load_scores, pivot_scores

# gCO2e per kWh used to convert attributed kWh into emissions, taken from the base run
def carbon_factor(scores_roles):
    for r in scores_roles:
        for ds in config.DATASETS:
            s = energy.load_summary(f"{r}_{ds}_r0")
            if s and s.get("carbon_intensity_g_per_kwh"):
                return s["carbon_intensity_g_per_kwh"]
    return np.nan


def upper_hull(singles):
    """Upper concave envelope of (acc, cost) points: the best accuracy any random mixture of
    two single models can reach at a given cost. A weak mid-cost model is skipped over."""
    pts = sorted(((c, a) for a, c in singles))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:   # hull[-1] is below chord
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def random_at_cost(cost, singles):
    """Accuracy of cost-matched random routing between single models (upper hull)."""
    h = upper_hull(singles)
    if cost <= h[0][0]:
        return h[0][1]
    if cost >= h[-1][0]:
        return h[-1][1]
    for (c1, a1), (c2, a2) in zip(h, h[1:]):
        if c1 <= cost <= c2:
            return a1 + ((cost - c1) / (c2 - c1) if c2 > c1 else 0) * (a2 - a1)
    return h[-1][1]


def main():
    q = load_questions()
    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    pool = [r for r in config.POOL if r in roles]
    base = config.BASE
    piv = pivot_scores(scores, roles, fold="test")
    ids = piv.index.tolist()
    n = len(ids)
    ds_of = q.set_index("id").loc[ids, "dataset"].values
    C = {r: piv[("correct", r)].values.astype(int) for r in roles}
    Ekwh = {r: piv[("energy_kwh_attr", r)].values.astype(float) for r in roles}
    conf = piv[("graded", base)].values
    gfac = carbon_factor(roles)
    rt = energy.load_summary("router_embed")
    router_kwh_q = rt["kwh_per_question"] if rt else 0.0
    if not rt:
        print("WARNING: no router_embed energy summary; router overhead charged as 0")

    # router predictions on test
    with open(config.ROUTER_DIR / "router.pkl", "rb") as f:
        router = pickle.load(f)
    E_all = np.load(config.ROUTER_DIR / "embeddings.npy")
    eid = pd.read_csv(config.ROUTER_DIR / "embedding_ids.csv")["id"].tolist()
    id2row = {i: k for k, i in enumerate(eid)}
    E = E_all[[id2row[i] for i in ids]]
    P = np.vstack([router["models"][r]["lr"].predict_proba(
        router["models"][r]["scaler"].transform(
            np.hstack([conf.reshape(-1, 1), router["models"][r]["pca"].transform(E)])))[:, 1]
        for r in pool])
    choice = P.argmax(axis=0)
    stack = np.vstack([C[r] for r in pool])
    stackE = np.vstack([Ekwh[r] for r in pool])
    thr = json.loads((config.ROUTER_DIR / "thresholds.json").read_text())
    tau_star = thr["tau_star"]

    rows = []

    def add(name, kind, acc_vec, kwh_vec, tau=None, esc=None, extra=None):
        d = dict(config=name, kind=kind, tau=tau, escalated=None if esc is None else float(esc.mean()),
                 acc=float(acc_vec.mean()), kwh_per_q=float(kwh_vec.mean()),
                 gco2e_per_q=float(kwh_vec.mean() * gfac) if gfac == gfac else np.nan,
                 wh_per_1000=float(kwh_vec.mean() * 1e6), n=n)
        for ds in config.DATASETS:
            m = ds_of == ds
            d[f"acc_{ds}"] = float(acc_vec[m].mean()) if m.any() else np.nan
        if extra:
            d.update(extra)
        rows.append(d)
        return d

    for r in roles:
        add(f"always-{r}", "single", C[r], Ekwh[r])
    add("oracle-pool", "oracle", stack.max(axis=0), stackE.mean(axis=0))
    add("oracle-all", "oracle", np.maximum(stack.max(axis=0), C[base]), Ekwh[base])
    singles = [(C[r].mean(), Ekwh[r].mean()) for r in roles]
    best_single = max(roles, key=lambda r: C[r].mean())

    for tau in config.TAU_GRID:
        esc = conf < tau
        f = esc.mean()
        for x in pool:
            acc = np.where(esc, C[x], C[base])
            kwh = Ekwh[base] + np.where(esc, Ekwh[x], 0.0)
            rnd = (1 - f) * C[base].mean() + f * C[x].mean()          # random escalation, same rate
            add(f"cascade base->{x}", "cascade", acc, kwh, tau, esc,
                dict(target=x, acc_random_rate=rnd, acc_random_cost=random_at_cost(kwh.mean(), singles)))
        acc = np.where(esc, stack[choice, np.arange(n)], C[base])
        kwh = Ekwh[base] + np.where(esc, stackE[choice, np.arange(n)] + router_kwh_q, 0.0)
        best_pool = max(pool, key=lambda r: C[r].mean())          # plain cascade target: best pool member
        plain = np.where(esc, C[best_pool], C[base]).mean()
        orc = np.where(esc, stack.max(axis=0), C[base]).mean()
        rnd = (1 - f) * C[base].mean() + f * stack.mean()             # random pool member at same rate
        add("hybrid", "hybrid", acc, kwh, tau, esc,
            dict(acc_plain_best=plain, acc_oracle_router=orc, acc_random_rate=rnd,
                 acc_random_cost=random_at_cost(kwh.mean(), singles),
                 split=json.dumps({pool[j]: int(((choice == j) & esc).sum()) for j in range(len(pool))})))

    R = pd.DataFrame(rows)
    R.to_csv(config.RESULTS_DIR / "simulated.csv", index=False)

    # ---- markdown
    L = ["# Simulated results (test split, attributed energy)", "",
         f"n = {n} test questions, models = {roles}, carbon factor = {gfac:.0f} gCO2e/kWh, "
         f"router overhead = {router_kwh_q * 3.6e6:.3f} J per escalated question", "",
         "## Single models and oracles", "",
         "| config | acc | " + " | ".join(f"acc {d}" for d in config.DATASETS) + " | Wh / 1000 q | gCO2e / q |",
         "|---|---:|" + "---:|" * len(config.DATASETS) + "---:|---:|"]
    for _, r in R[R.kind.isin(["single", "oracle"])].iterrows():
        L.append(f"| {r.config} | {r.acc:.3f} | " + " | ".join(f"{r[f'acc_{d}']:.3f}" for d in config.DATASETS)
                 + f" | {r.wh_per_1000:.2f} | {r.gco2e_per_q:.4f} |")
    L += ["", f"## Cascade and hybrid at tau* = {tau_star} (chosen on train)", "",
          "| config | escalated | acc | vs best single | vs plain->best | vs random (rate) | vs random (cost) | Wh / 1000 q | vs best single energy |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    bs = R[R.config == f"always-{best_single}"].iloc[0]
    for _, r in R[(R.kind.isin(["cascade", "hybrid"])) & (np.isclose(R.tau, tau_star))].iterrows():
        plain = r.get("acc_plain_best", np.nan)
        L.append(f"| {r.config} | {r.escalated:.1%} | {r.acc:.3f} | {r.acc - bs.acc:+.3f} | "
                 f"{'' if plain != plain else f'{r.acc - plain:+.3f}'} | {r.acc - r.acc_random_rate:+.3f} | "
                 f"{r.acc - r.acc_random_cost:+.3f} | {r.wh_per_1000:.2f} | "
                 f"{(r.kwh_per_q - bs.kwh_per_q) / bs.kwh_per_q:+.0%} |")
    L += ["", "## Full tau sweep", "", "| config | tau | escalated | acc | vs random (cost) | Wh / 1000 q |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in R[R.kind.isin(["cascade", "hybrid"])].sort_values(["config", "tau"]).iterrows():
        L.append(f"| {r.config} | {r.tau:.2f} | {r.escalated:.1%} | {r.acc:.3f} | "
                 f"{r.acc - r.acc_random_cost:+.3f} | {r.wh_per_1000:.2f} |")
    L += ["", "## Cheapest routed point matching each single model's accuracy", ""]
    routed = R[R.kind.isin(["cascade", "hybrid"])]
    for r0 in roles:
        tgt = C[r0].mean()
        ok = routed[(routed.acc >= tgt - 1e-9) & (routed.escalated > 0) & (routed.escalated < 1)].sort_values("kwh_per_q")
        if len(ok):
            r = ok.iloc[0]
            d = (r.kwh_per_q - Ekwh[r0].mean()) / Ekwh[r0].mean()
            verdict = "SAVES" if d < -0.005 else ("costs MORE" if d > 0.005 else "same")
            L.append(f"- match always-{r0} ({tgt:.3f}): {r.config} tau={r.tau:.2f} acc {r.acc:.3f} at "
                     f"{d:+.0%} energy [{verdict}], {r.escalated:.0%} escalated")
        else:
            L.append(f"- match always-{r0} ({tgt:.3f}): no non-degenerate routed point reaches it")
    (config.RESULTS_DIR / "simulated.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))

    # ---- frontier figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for _, r in R[R.kind == "single"].iterrows():
            ax.scatter(r.wh_per_1000, r.acc, marker="s", s=60, zorder=3)
            ax.annotate(r.config.replace("always-", ""), (r.wh_per_1000, r.acc), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        xs = np.linspace(R[R.kind == "single"].wh_per_1000.min(), R[R.kind == "single"].wh_per_1000.max(), 100)
        ax.plot(xs, [random_at_cost(x / 1e6, singles) for x in xs], "--", color="grey", label="cost-matched random")
        for x in pool:
            c = R[R.config == f"cascade base->{x}"].sort_values("wh_per_1000")
            ax.plot(c.wh_per_1000, c.acc, "-o", ms=3, label=f"cascade base->{x}")
        h = R[R.kind == "hybrid"].sort_values("wh_per_1000")
        ax.plot(h.wh_per_1000, h.acc, "-o", ms=4, lw=2, color="black", label="hybrid")
        ax.set_xlabel("Wh per 1,000 questions (attributed)")
        ax.set_ylabel("accuracy (test)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "frontier.png", dpi=150)
    except Exception as e:
        print("figure skipped:", e)
    print(f"\nwrote results/simulated.csv, simulated.md, frontier.png")


if __name__ == "__main__":
    main()
