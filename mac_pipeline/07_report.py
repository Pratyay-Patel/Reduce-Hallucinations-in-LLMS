"""
mac_pipeline/07_report.py
-------------------------
Assemble the paper tables and figures from results/, scores/ and energy/.

  Table 1  per-model accuracy per dataset; Wh, gCO2e, seconds per 1,000 questions (+- over repeats)
  Table 2  gate matrix (ordered pairs, train)
  Table 3  end-to-end measured: base, singles, plain cascade, route, route-cascade
  Table 4  simulated vs measured energy for the same configurations
  Fig 1    accuracy-energy frontier (from 05)      Fig 2  per-dataset accuracy by configuration

Writes results/report.md, results/table_*.csv, results/results.xlsx, results/fig2_per_dataset.png

  python mac_pipeline/07_report.py
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd

from common import energy
from common.data import load_scores


def table1():
    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    test = scores[scores["fold"] == "test"]
    rows = []
    for r in roles:
        s = test[test.model == r]
        d = dict(model=r, hf_id=config.MODELS[r]["id"], n_test=len(s), acc=s.correct.mean())
        for ds in config.DATASETS:
            d[f"acc_{ds}"] = s[s.dataset == ds].correct.mean()
        # energy from the full (train+test) runs, per 1,000 questions
        tot_kwh, tot_cc, tot_g, tot_s, tot_n, pm, src = 0.0, 0.0, 0.0, 0.0, 0, None, set()
        for ds in config.DATASETS:
            e = energy.load_summary(f"{r}_{ds}_r0")
            if e:
                tot_kwh += energy.record_kwh(e); tot_cc += e.get("total_kwh") or 0.0
                tot_g += energy.emissions_g(e) or 0.0; tot_s += e["duration_s"]; tot_n += e["n"]
                pm = e.get("powermetrics")
                src.add("powermetrics SoC" if e.get("soc_kwh") is not None else "CodeCarbon model")
        if tot_n:
            d.update(wh_per_1000=tot_kwh / tot_n * 1e6, codecarbon_wh_per_1000=tot_cc / tot_n * 1e6,
                     gco2e_per_1000=tot_g / tot_n * 1000,
                     s_per_1000=tot_s / tot_n * 1000, mean_w=tot_kwh * 3.6e6 / tot_s if tot_s else np.nan,
                     energy_source="/".join(sorted(src)), powermetrics=pm)
        # +- from repeats on the sanity subset
        reps = [energy.load_summary(os.path.basename(p)[:-5])
                for p in glob.glob(str(config.ENERGY_DIR / f"{r}_mmlu_r[1-9].json"))]
        if reps:
            v = [energy.record_kwh(x) / x["n"] * 1e6 for x in reps]
            d.update(wh_per_1000_repeat_mean=np.mean(v), wh_per_1000_repeat_sd=np.std(v, ddof=1) if len(v) > 1 else np.nan,
                     n_repeats=len(v))
        d["peak_mem_gb"] = s.peak_mem_gb.max()
        rows.append(d)
    return pd.DataFrame(rows)


def table2():
    p = config.RESULTS_DIR / "gate.csv"
    if not p.exists():
        return pd.DataFrame()
    g = pd.read_csv(p)
    return g[(g.scope == "all") & (g.fold == "train")][
        ["A", "B", "n_A_wrong", "B_right_on_A_wrong", "rate", "chance", "headroom_pts", "p_value", "passes"]]


def global_factor():
    """Carbon factor from any scoring-run summary (old end-to-end JSONs lack the fields)."""
    for r in config.ALL_ROLES:
        for ds in config.DATASETS:
            e = energy.load_summary(f"{r}_{ds}_r0")
            f = energy.factor_of(e) if e else None
            if f:
                return f
    return np.nan


def table3():
    rows = []
    gf = global_factor()
    for p in sorted(glob.glob(str(config.RESULTS_DIR / "end_to_end_*.json"))):
        j = json.loads(open(p).read())
        # recompute energy of record and emissions from the passes, so old JSONs (written
        # before the emissions fix) are reported consistently with new ones
        passes = j.get("per_pass", {})
        kwh = sum((v.get("record_kwh") if v.get("record_kwh") is not None else energy.record_kwh(v))
                  for v in passes.values() if v) if passes else j["total_kwh"]
        d = dict(config=j["config"], tau=j["tau"] if j.get("tau") is not None else np.nan, n=j["n"], acc=j["accuracy"],
                 escalated=j["escalation_rate"], wh_per_1000=kwh / j["n"] * 1e6,
                 gco2e_per_q=(kwh * gf / j["n"]) if gf == gf else np.nan,
                 total_kwh=kwh, wall_min=j["wall_s"] / 60)
        for ds, v in j["per_dataset"].items():
            d[f"acc_{ds}"] = v["acc"]
        rows.append(d)
    T = pd.DataFrame(rows)
    if len(T):
        best = T[T.config.str.startswith("single:") | (T.config == "base")]
        if len(best):
            b = best.sort_values("acc").iloc[-1]
            T["d_acc_vs_best_single"] = T.acc - b.acc
            T["d_energy_vs_best_single"] = (T.wh_per_1000 - b.wh_per_1000) / b.wh_per_1000
    return T


def table4(T3):
    p = config.RESULTS_DIR / "simulated.csv"
    if not p.exists() or not len(T3):
        return pd.DataFrame()
    S = pd.read_csv(p)
    rows = []
    for _, r in T3.iterrows():
        cfg = r.config
        tau_s = S.tau.astype(float)
        if cfg == "route-cascade":
            m = S[S.kind == "route-cascade*"] if r.tau != r.tau else S[(S.kind == "route-cascade") & np.isclose(tau_s, r.tau)]
        elif cfg == "route":
            m = S[S.config == "route-only"]
        elif cfg.startswith("cascade:"):
            m = S[(S.config == f"cascade {config.BASE}->{cfg.split(':')[1]}") & np.isclose(tau_s, r.tau)]
        elif cfg == "base":
            m = S[S.config == f"always-{config.BASE}"]
        elif cfg.startswith("single:"):
            m = S[S.config == f"always-{cfg.split(':')[1]}"]
        else:
            m = S.iloc[0:0]
        if len(m):
            s = m.iloc[0]
            rows.append(dict(config=cfg, tau=r.tau, acc_measured=r.acc, acc_simulated=s.acc,
                             wh_measured=r.wh_per_1000, wh_simulated=s.wh_per_1000,
                             energy_gap=(r.wh_per_1000 - s.wh_per_1000) / s.wh_per_1000 if s.wh_per_1000 else np.nan))
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols=None, fmt=None):
    if df is None or not len(df):
        return "_(not available yet)_"
    cols = cols or list(df.columns)
    fmt = fmt or {}
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                cells.append(fmt.get(c, "{:.3f}").format(v) if v == v else "")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def fig2(T3):
    if not len(T3):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = [c for c in T3.columns if c.startswith("acc_")]
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.8 / len(T3)
        x = np.arange(len(cols))
        for i, (_, r) in enumerate(T3.iterrows()):
            ax.bar(x + i * w, [r[c] for c in cols], w, label=f"{r.config}" + (f" τ={r.tau:.2f}" if r.tau == r.tau else ""))
        ax.set_xticks(x + w * (len(T3) - 1) / 2)
        ax.set_xticklabels([c[4:] for c in cols])
        ax.set_ylabel("accuracy (test)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(config.RESULTS_DIR / "fig2_per_dataset.png", dpi=150)
    except Exception as e:
        print("fig2 skipped:", e)


def main():
    T1, T2, T3 = table1(), table2(), table3()
    T4 = table4(T3)
    for name, t in (("table1_models", T1), ("table2_gate", T2), ("table3_end_to_end", T3), ("table4_sim_vs_measured", T4)):
        t.to_csv(config.RESULTS_DIR / f"{name}.csv", index=False)
    fig2(T3)

    pct = "{:+.1%}"
    L = ["# Results", "",
         "## Table 1 - single models (test accuracy; energy per 1,000 questions from the full scoring runs)", "",
         md_table(T1, ["model", "acc"] + [f"acc_{d}" for d in config.DATASETS] +
                  ["wh_per_1000", "gco2e_per_1000", "s_per_1000", "mean_w", "wh_per_1000_repeat_sd", "peak_mem_gb", "powermetrics"],
                  {"wh_per_1000": "{:.2f}", "gco2e_per_1000": "{:.2f}", "s_per_1000": "{:.0f}", "mean_w": "{:.1f}",
                   "wh_per_1000_repeat_sd": "{:.2f}", "peak_mem_gb": "{:.2f}"}),
         "", "## Table 2 - gate (train split, overall)", "",
         md_table(T2, fmt={"p_value": "{:.3f}", "headroom_pts": "{:+.1f}"}),
         "", "## Table 3 - end-to-end, measured on the test split", "",
         md_table(T3, ["config", "tau", "acc"] + [f"acc_{d}" for d in config.DATASETS] +
                  ["escalated", "wh_per_1000", "gco2e_per_q", "d_acc_vs_best_single", "d_energy_vs_best_single"],
                  {"tau": "{:.2f}", "escalated": "{:.1%}", "wh_per_1000": "{:.2f}", "gco2e_per_q": "{:.4f}",
                   "d_acc_vs_best_single": "{:+.3f}", "d_energy_vs_best_single": pct}),
         "", "## Table 4 - simulated (attributed) vs measured energy", "",
         md_table(T4, fmt={"tau": "{:.2f}", "wh_measured": "{:.2f}", "wh_simulated": "{:.2f}", "energy_gap": pct}),
         "", "Figures: `frontier.png` (from 05_simulate), `fig2_per_dataset.png`.", ""]
    if len(T4) and (T4.energy_gap.abs() > 0.10).any():
        L.append("**Note:** at least one configuration's measured energy differs from the attributed "
                 "estimate by more than 10 %. On the fanless Air the same model's power drifts 30-44 % "
                 "between runs hours apart (Results_Run2), so this gap mixes attribution error with "
                 "power drift; report it as the between-run spread, not as a validation failure.")
    (config.RESULTS_DIR / "report.md").write_text("\n".join(L), encoding="utf-8")

    try:
        with pd.ExcelWriter(config.RESULTS_DIR / "results.xlsx", engine="openpyxl") as xw:
            T1.to_excel(xw, sheet_name="T1 models", index=False)
            T2.to_excel(xw, sheet_name="T2 gate", index=False)
            T3.to_excel(xw, sheet_name="T3 end-to-end", index=False)
            T4.to_excel(xw, sheet_name="T4 sim vs measured", index=False)
            p = config.RESULTS_DIR / "simulated.csv"
            if p.exists():
                pd.read_csv(p).to_excel(xw, sheet_name="simulated sweep", index=False)
        print("wrote results/results.xlsx")
    except Exception as e:
        print("xlsx skipped:", e)
    print("\n".join(L))
    print("\nwrote results/report.md and table_*.csv")


if __name__ == "__main__":
    main()
