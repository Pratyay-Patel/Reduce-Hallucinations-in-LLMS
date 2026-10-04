"""
mac_pipeline/check_outputs.py
-----------------------------
Verify a step's output files exist and are complete. run_all.sh calls this after every step;
a non-zero exit means the step's outputs are missing or short.

  python mac_pipeline/check_outputs.py questions
  python mac_pipeline/check_outputs.py scores --role base [--datasets mmlu csqa arc] [--repeat 1 --limit 200]
  python mac_pipeline/check_outputs.py gate | router | simulate | report
  python mac_pipeline/check_outputs.py e2e --name hybrid_0.60
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import pandas as pd

from common import energy

problems = []


def need(path, what=""):
    if not path.exists() or path.stat().st_size == 0:
        problems.append(f"missing or empty: {path} {what}")
        return False
    return True


def check_questions():
    if need(config.QUESTIONS_CSV) and need(config.SPLIT_CSV):
        q = pd.read_csv(config.QUESTIONS_CSV)
        s = pd.read_csv(config.SPLIT_CSV)
        if set(q["id"]) != set(s["id"]):
            problems.append("questions.csv and split.csv list different ids")
        for ds in config.DATASETS:
            n = int((q["dataset"] == ds).sum())
            if n == 0:
                problems.append(f"no questions for dataset {ds}")
            print(f"  {ds}: {n} questions")


def check_scores(role, datasets, repeat, limit):
    path = config.SCORES_DIR / (f"{role}.csv" if repeat == 0 else f"{role}_r{repeat}.csv")
    if not need(path):
        return
    s = pd.read_csv(path)
    q = pd.read_csv(config.QUESTIONS_CSV)
    for ds in datasets:
        rows = s[s["dataset"] == ds]
        expect = int((q["dataset"] == ds).sum()) if not limit else min(limit, int((q["dataset"] == ds).sum()))
        if len(rows) != expect:
            problems.append(f"{path.name}: {ds} has {len(rows)} rows, expected {expect}")
        if rows["correct"].isna().any():
            problems.append(f"{path.name}: {ds} has null `correct`")
        if len(rows) and rows["energy_kwh_attr"].isna().any():
            problems.append(f"{path.name}: {ds} has null attributed energy")
        if len(rows):
            top = rows["pred_letter"].value_counts(normalize=True)
            if top.iloc[0] > 0.45:
                problems.append(f"{path.name}: {ds} predicts '{top.index[0]}' {top.iloc[0]:.0%} of the time (position habit)")
            chance = (1.0 / rows["n_options"]).mean()
            if rows["correct"].mean() < chance + 0.05:
                problems.append(f"{path.name}: {ds} accuracy {rows['correct'].mean():.3f} is at chance ({chance:.3f})")
        e = energy.load_summary(f"{role}_{ds}_r{repeat}")
        if e is None:
            problems.append(f"energy summary {role}_{ds}_r{repeat}.json missing")
        else:
            if energy.IS_MAC and (e.get("soc_kwh") is None or e["soc_kwh"] <= 0):
                problems.append(f"energy summary {role}_{ds}_r{repeat}.json has no powermetrics SoC energy")
            if (e.get("n") or 0) != len(rows):
                problems.append(f"energy summary {role}_{ds}_r{repeat}.json n={e.get('n')} but {len(rows)} score rows")
        print(f"  {path.name} {ds}: {len(rows)} rows, acc {rows['correct'].mean():.3f}"
              + (f", {energy.record_kwh(e) * 1e6 / max(len(rows), 1):.2f} Wh/1000" if e else ""))


def check_gate():
    need(config.RESULTS_DIR / "gate.csv")
    need(config.RESULTS_DIR / "gate.md")


def check_router():
    need(config.ROUTER_DIR / "router.pkl")
    need(config.ROUTER_DIR / "embeddings.npy")
    need(config.ROUTER_DIR / "embedding_ids.csv")
    if need(config.ROUTER_DIR / "thresholds.json"):
        t = json.loads((config.ROUTER_DIR / "thresholds.json").read_text())
        if "tau_star" not in t:
            problems.append("thresholds.json has no tau_star")
        else:
            print(f"  tau* = {t['tau_star']}   head = {t.get('head', '?')}")
    need(config.RESULTS_DIR / "router_train.md")


def check_simulate():
    need(config.RESULTS_DIR / "simulated.csv")
    need(config.RESULTS_DIR / "simulated.md")
    need(config.RESULTS_DIR / "frontier.png")


def check_e2e(name):
    j = config.RESULTS_DIR / f"end_to_end_{name}.json"
    log = config.RESULTS_DIR / f"end_to_end_{name}_log.csv"
    if need(j) and need(log):
        d = json.loads(j.read_text())
        L = pd.read_csv(log)
        if len(L) != d.get("n"):
            problems.append(f"{log.name}: {len(L)} rows but json says n={d.get('n')}")
        if L["final_correct"].isna().any():
            problems.append(f"{log.name}: null final_correct")
        if energy.IS_MAC:
            for k, v in d.get("per_pass", {}).items():
                if v and (v.get("soc_kwh") is None or v["soc_kwh"] <= 0):
                    problems.append(f"{j.name}: pass '{k}' has no powermetrics SoC energy")
        print(f"  {name}: acc {d.get('accuracy'):.3f}, {d.get('wh_per_1000'):.2f} Wh/1000, escalated {d.get('escalation_rate'):.1%}")


def check_report():
    for f in ("report.md", "results.xlsx", "table1_models.csv", "table2_gate.csv",
              "table3_end_to_end.csv", "table4_sim_vs_measured.csv"):
        need(config.RESULTS_DIR / f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["questions", "scores", "gate", "router", "simulate", "e2e", "report"])
    ap.add_argument("--role")
    ap.add_argument("--datasets", nargs="*", default=config.DATASETS)
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name")
    a = ap.parse_args()
    print(f"check {a.stage}" + (f" {a.role}" if a.role else "") + (f" {a.name}" if a.name else ""))
    if a.stage == "questions":
        check_questions()
    elif a.stage == "scores":
        check_scores(a.role, a.datasets, a.repeat, a.limit)
    elif a.stage == "gate":
        check_gate()
    elif a.stage == "router":
        check_router()
    elif a.stage == "simulate":
        check_simulate()
    elif a.stage == "e2e":
        check_e2e(a.name)
    elif a.stage == "report":
        check_report()
    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print("  ok")


if __name__ == "__main__":
    main()
