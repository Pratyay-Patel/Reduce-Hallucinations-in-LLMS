"""
mac_pipeline/05_simulate.py
---------------------------
Test split, stored scores, attributed energy. Accuracy, energy (kWh) and emissions (gCO2e)
for every configuration:

  singles            each model alone
  oracle             best of the routed tier / best of all models, per question
  route-only         router picks a routed model; no escalation
  cascade X->ESC     X answers; confidence < tau -> escalation model      (tau grid)
  route-cascade      router picks; confidence < tau -> escalation model  (tau grid, and per-route tau*)
  random             cost-matched random mixing of single models (upper hull)
  matched-accuracy   cheapest routed point that matches each single model

Router overhead (embedding energy per question) is charged on every question that is
routed. Per-question energy for a model is its attributed share of that model's run.

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


def carbon_factor(scores_roles):
    """gCO2e per kWh CodeCarbon applied for the configured country, from any run summary."""
    for r in scores_roles:
        for ds in config.DATASETS:
            s = energy.load_summary(f"{r}_{ds}_r0")
            f = energy.factor_of(s) if s else None
            if f:
                return f
    return np.nan


def upper_hull(singles):
    """Upper concave envelope of (acc, cost) points: the best accuracy any random mixture of
    two single models can reach at a given cost. A weak mid-cost model is skipped over."""
    pts = sorted(((c, a) for a, c in singles))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def random_at_cost(cost, singles):
    h = upper_hull(singles)
    if cost <= h[0][0]:
        return h[0][1]
    if cost >= h[-1][0]:
        return h[-1][1]
    for (c1, a1), (c2, a2) in zip(h, h[1:]):
        if c1 <= cost <= c2:
            return a1 + ((cost - c1) / (c2 - c1) if c2 > c1 else 0) * (a2 - a1)
    return h[-1][1]


def router_pick(router, E, routed):
    P = np.vstack([router["models"][r]["clf"].predict_proba(
        router["models"][r]["scaler"].transform(router["models"][r]["pca"].transform(E)))[:, 1]
        for r in routed])
    return P, P.argmax(axis=0)


def main():
    q = load_questions()
    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    routed = [r for r in config.ROUTED if r in roles]
    esc_role = config.ESCALATION
    base = config.BASE
    piv = pivot_scores(scores, roles, fold="test")
    ids = piv.index.tolist()
    n = len(ids)
    ds_of = q.set_index("id").loc[ids, "dataset"].values
    C = {r: piv[("correct", r)].values.astype(int) for r in roles}
    Ekwh = {r: piv[("energy_kwh_attr", r)].values.astype(float) for r in roles}
    CONF = {r: piv[("graded", r)].values for r in roles}
    gfac = carbon_factor(roles)
    rt = energy.load_summary("router_embed")
    router_kwh_q = (energy.record_kwh(rt) / rt["n"]) if rt and rt.get("n") else 0.0
    if not rt:
        print("WARNING: no router_embed energy summary; router overhead charged as 0")

    with open(config.ROUTER_DIR / "router.pkl", "rb") as f:
        router = pickle.load(f)
    E_all = np.load(config.ROUTER_DIR / "embeddings.npy")
    eid = pd.read_csv(config.ROUTER_DIR / "embedding_ids.csv")["id"].tolist()
    id2row = {i: k for k, i in enumerate(eid)}
    E = E_all[[id2row[i] for i in ids]]
    P, pick = router_pick(router, E, routed)
    thr = json.loads((config.ROUTER_DIR / "thresholds.json").read_text())
    tau_global, tau_route, tau_cascade = thr["tau_star"], thr["tau_per_route"], thr["tau_cascade"]

    stack = np.vstack([C[r] for r in routed])
    stackE = np.vstack([Ekwh[r] for r in routed])
    stackConf = np.vstack([CONF[r] for r in routed])
    ar = np.arange(n)
    c_pick, e_pick, conf_pick = stack[pick, ar], stackE[pick, ar], stackConf[pick, ar]

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
    add("oracle-routed", "oracle", stack.max(axis=0), stackE.mean(axis=0))
    add("oracle-all", "oracle", np.maximum(stack.max(axis=0), C[esc_role]), Ekwh[base])
    singles = [(C[r].mean(), Ekwh[r].mean()) for r in roles]
    best_single = max(roles, key=lambda r: C[r].mean())
    split = json.dumps({routed[j]: int((pick == j).sum()) for j in range(len(routed))})
    add("route-only", "route", c_pick, e_pick + router_kwh_q, tau=0.0, esc=np.zeros(n, bool),
        extra=dict(split=split, acc_random_cost=random_at_cost((e_pick + router_kwh_q).mean(), singles)))

    for tau in config.TAU_GRID:
        for x in routed:                                   # plain cascades, no router
            esc = CONF[x] < tau
            acc = np.where(esc, C[esc_role], C[x])
            kwh = Ekwh[x] + np.where(esc, Ekwh[esc_role], 0.0)
            f = esc.mean()
            add(f"cascade {x}->{esc_role}", "cascade", acc, kwh, tau, esc,
                dict(first=x, acc_random_rate=(1 - f) * C[x].mean() + f * C[esc_role].mean(),
                     acc_random_cost=random_at_cost(kwh.mean(), singles)))
        esc = conf_pick < tau                              # route then cascade, one global tau
        acc = np.where(esc, C[esc_role], c_pick)
        kwh = router_kwh_q + e_pick + np.where(esc, Ekwh[esc_role], 0.0)
        f = esc.mean()
        add("route-cascade", "route-cascade", acc, kwh, tau, esc,
            dict(acc_random_rate=(1 - f) * c_pick.mean() + f * C[esc_role].mean(),
                 acc_random_cost=random_at_cost(kwh.mean(), singles), split=split))
    # route then cascade with the per-route tau chosen on train
    tau_vec = np.array([tau_route[routed[j]] for j in pick])
    esc = conf_pick < tau_vec
    acc = np.where(esc, C[esc_role], c_pick)
    kwh = router_kwh_q + e_pick + np.where(esc, Ekwh[esc_role], 0.0)
    f = esc.mean()
    add("route-cascade*", "route-cascade*", acc, kwh, None, esc,
        dict(acc_random_rate=(1 - f) * c_pick.mean() + f * C[esc_role].mean(),
             acc_random_cost=random_at_cost(kwh.mean(), singles), split=split,
             tau_per_route=json.dumps(tau_route)))

    R = pd.DataFrame(rows)
    R.to_csv(config.RESULTS_DIR / "simulated.csv", index=False)

    # ---- markdown
    bs = R[R.config == f"always-{best_single}"].iloc[0]
    L = ["# Simulated results (test split, attributed energy)", "",
         f"n = {n} test questions, models = {roles}, routed tier = {routed}, escalation = {esc_role}, "
         f"carbon factor = {gfac:.0f} gCO2e/kWh, router overhead = {router_kwh_q * 3.6e6:.3f} J per question", "",
         "## Single models and oracles", "",
         "| config | acc | " + " | ".join(f"acc {d}" for d in config.DATASETS) + " | Wh / 1000 q | gCO2e / q |",
         "|---|---:|" + "---:|" * len(config.DATASETS) + "---:|---:|"]
    for _, r in R[R.kind.isin(["single", "oracle", "route"])].iterrows():
        L.append(f"| {r.config} | {r.acc:.3f} | " + " | ".join(f"{r[f'acc_{d}']:.3f}" for d in config.DATASETS)
                 + f" | {r.wh_per_1000:.2f} | {r.gco2e_per_q:.4f} |")
    L += ["", f"Router picks on test: {split}", "",
          f"## Headline configurations (tau chosen on train: per-route {tau_route}, global {tau_global}, "
          f"plain cascade {tau_cascade})", "",
          "| config | tau | escalated | acc | " + " | ".join(f"acc {d}" for d in config.DATASETS) +
          " | vs best single | vs random (cost) | Wh / 1000 q | vs best single energy |",
          "|---|---:|---:|---:|" + "---:|" * len(config.DATASETS) + "---:|---:|---:|---:|"]
    head = pd.concat([
        R[R.kind == "route-cascade*"],
        R[(R.kind == "route-cascade") & np.isclose(R.tau.astype(float), tau_global)],
        R[(R.config == f"cascade {base}->{esc_role}") & np.isclose(R.tau.astype(float), tau_cascade)],
        R[R.kind == "route"]])
    for _, r in head.iterrows():
        tau_s = "per-route" if r.kind == "route-cascade*" else f"{r.tau:.2f}"
        L.append(f"| {r.config} | {tau_s} | {r.escalated:.1%} | {r.acc:.3f} | " +
                 " | ".join(f"{r[f'acc_{d}']:.3f}" for d in config.DATASETS) +
                 f" | {r.acc - bs.acc:+.3f} | {r.acc - r.acc_random_cost:+.3f} | {r.wh_per_1000:.2f} | "
                 f"{(r.kwh_per_q - bs.kwh_per_q) / bs.kwh_per_q:+.0%} |")
    L += ["", "## Full tau sweep", "", "| config | tau | escalated | acc | vs random (cost) | Wh / 1000 q |",
          "|---|---:|---:|---:|---:|---:|"]
    for _, r in R[R.kind.isin(["cascade", "route-cascade"])].sort_values(["config", "tau"]).iterrows():
        L.append(f"| {r.config} | {r.tau:.2f} | {r.escalated:.1%} | {r.acc:.3f} | "
                 f"{r.acc - r.acc_random_cost:+.3f} | {r.wh_per_1000:.2f} |")
    L += ["", "## Cheapest routed point matching each single model's accuracy", ""]
    routed_rows = R[R.kind.isin(["cascade", "route-cascade", "route-cascade*"])]
    for r0 in roles:
        tgt = C[r0].mean()
        ok = routed_rows[(routed_rows.acc >= tgt - 1e-9) & (routed_rows.escalated > 0) & (routed_rows.escalated < 1)].sort_values("kwh_per_q")
        if len(ok):
            r = ok.iloc[0]
            d = (r.kwh_per_q - Ekwh[r0].mean()) / Ekwh[r0].mean()
            verdict = "SAVES" if d < -0.005 else ("costs MORE" if d > 0.005 else "same")
            tau_s = "per-route" if r.kind == "route-cascade*" else f"tau={r.tau:.2f}"
            L.append(f"- match always-{r0} ({tgt:.3f}): {r.config} {tau_s} acc {r.acc:.3f} at "
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
        for _, r in R[R.kind.isin(["single", "route"])].iterrows():
            ax.scatter(r.wh_per_1000, r.acc, marker="s", s=60, zorder=3)
            ax.annotate(r.config.replace("always-", ""), (r.wh_per_1000, r.acc), textcoords="offset points",
                        xytext=(4, 4), fontsize=8)
        sg = R[R.kind == "single"]
        xs = np.linspace(sg.wh_per_1000.min(), sg.wh_per_1000.max(), 100)
        ax.plot(xs, [random_at_cost(x / 1e6, singles) for x in xs], "--", color="grey", label="cost-matched random")
        for x in routed:
            c = R[R.config == f"cascade {x}->{esc_role}"].sort_values("wh_per_1000")
            ax.plot(c.wh_per_1000, c.acc, "-o", ms=3, label=f"cascade {x}->{esc_role}")
        h = R[R.kind == "route-cascade"].sort_values("wh_per_1000")
        ax.plot(h.wh_per_1000, h.acc, "-o", ms=4, lw=2, color="black", label="route-cascade (global tau)")
        s = R[R.kind == "route-cascade*"].iloc[0]
        ax.scatter(s.wh_per_1000, s.acc, marker="*", s=160, color="red", zorder=4, label="route-cascade (per-route tau)")
        ax.set_xlabel("Wh per 1,000 questions (attributed)")
        ax.set_ylabel("accuracy (test)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "frontier.png", dpi=150)
    except Exception as e:
        print("figure skipped:", e)
    print("\nwrote results/simulated.csv, simulated.md, frontier.png")


if __name__ == "__main__":
    main()
