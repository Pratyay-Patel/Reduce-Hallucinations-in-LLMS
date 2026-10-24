"""
mac_pipeline/06_run_end_to_end.py
---------------------------------
The live measurement: run the deployed decision rule on the TEST split with energy tracked.

  router pass   embed every test question, pick a routed 1.5B model        (route* configs)
  pass 1        each routed model on the questions assigned to it
  pass 2        the escalation model on the questions whose first answer was below tau

End-to-end energy = sum of the tracked passes. Questions are grouped by model rather than
interleaved per query because every model resident at once would not fit; the per-question
computation is identical.

  python mac_pipeline/06_run_end_to_end.py --config route-cascade        # per-route tau from train
  python mac_pipeline/06_run_end_to_end.py --config route-cascade --tau 0.9   # one global tau
  python mac_pipeline/06_run_end_to_end.py --config route                 # router only, no escalation
  python mac_pipeline/06_run_end_to_end.py --config cascade:qwen3b        # generalist -> qwen3b, no router
  python mac_pipeline/06_run_end_to_end.py --config base                  # generalist alone
  python mac_pipeline/06_run_end_to_end.py --config single:qwen3b         # always-X anchor

Outputs results/end_to_end_<config>.json and ..._log.csv  (':' in the config becomes '-')
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd

from common import energy
from common.data import load_questions, score_row
from common.scorer import OptionScorer


def run_model(role, sub: pd.DataFrame, tag: str, device, no_energy: bool, idle_secs: int):
    """Score `sub` with one model inside one tracked run. Returns (rows df, summary or None)."""
    m = config.MODELS[role]
    sc = OptionScorer(m["id"], device, chat_template=m["chat_template"], adapter=m.get("adapter"))
    score_row(sc, sub.iloc[0])                                        # warm-up, untracked
    idle = {} if no_energy else energy.idle_baseline(idle_secs, tag)
    rows = []
    if no_energy:
        from contextlib import nullcontext
        ctx = nullcontext({"data": {}})
    else:
        ctx = energy.tracked(tag)
    with ctx as run:
        for _, r in sub.iterrows():
            s = score_row(sc, r)
            rows.append(dict(id=r["id"], model=role, pred=s["pred"], correct=s["correct"],
                             graded=s["graded"], seconds=s["seconds"]))
    sc.unload()
    df = pd.DataFrame(rows)
    summ = None
    if not no_energy:
        summ = energy.summarize_run(run, role=role, model_id=m["id"], dataset="e2e", repeat=0,
                                    n=len(df), idle=idle, extra=dict(tag=tag))
        energy.write_summary(summ, tag)
        df["kwh_attr"] = energy.attribute(energy.record_kwh(summ), df["seconds"])
    else:
        df["kwh_attr"] = 0.0
    time.sleep(3)
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="route-cascade",
                    help="route-cascade | route | cascade:<role> | base | single:<role>")
    ap.add_argument("--tau", type=float, default=None,
                    help="override: one global tau (default: per-route tau from thresholds.json for "
                         "route-cascade, tau_cascade for cascade:X)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--idle-secs", type=int, default=config.IDLE_SECS)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or config.pick_device()
    q = load_questions()
    test = q[(q["fold"] == "test") & q["dataset"].isin(config.DATASETS)].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)
    n = len(test)
    cfg = args.config
    kind, _, arg = cfg.partition(":")
    routed, esc_role = config.ROUTED, config.ESCALATION
    thr_path = config.ROUTER_DIR / "thresholds.json"
    thr = json.loads(thr_path.read_text()) if thr_path.exists() else {}

    tau_per_route = None
    if kind == "route-cascade":
        if args.tau is not None:
            tau_per_route = {r: args.tau for r in routed}
        else:
            tau_per_route = thr["tau_per_route"]
        tau = args.tau
    elif kind == "route":
        tau = 0.0
    elif kind == "cascade":
        assert arg in config.MODELS, f"cascade target must be a role, got {arg}"
        tau = args.tau if args.tau is not None else thr["tau_cascade"]
    elif kind == "base":
        tau = 0.0
    elif kind == "single":
        assert arg in config.MODELS
        tau = 1.01
    else:
        sys.exit(f"unknown config {cfg}")
    name = cfg.replace(":", "-")
    print(f"config={cfg}  tau={tau if tau is not None else tau_per_route}  n={n}  device={device}")

    log = test[["id", "dataset", "subject", "n_options", "gold"]].copy()
    log["first_model"] = config.BASE
    log["first_pred"], log["first_correct"], log["first_conf"] = np.nan, np.nan, np.nan
    log["tau_used"] = tau if tau is not None else np.nan
    log["escalated"] = False
    log["router_kwh_attr"], log["first_kwh_attr"], log["pool_kwh_attr"] = 0.0, 0.0, 0.0
    for r in routed:
        log[f"router_p_{r}"] = np.nan
    passes = {}
    t0 = time.time()

    if kind == "single":
        df, summ = run_model(arg, test, f"e2e_{name}_{arg}", device, args.no_energy, args.idle_secs)
        passes[arg] = summ
        log["first_model"] = arg
        log["escalated"] = True
        log["chosen_model"] = arg
        log["final_pred"], log["final_correct"] = df["pred"].values, df["correct"].values
        log["pool_kwh_attr"] = df["kwh_attr"].values
    else:
        # ---- router pass (route* configs): ex-ante, from the question text
        if kind in ("route", "route-cascade"):
            with open(config.ROUTER_DIR / "router.pkl", "rb") as f:
                router = pickle.load(f)
            from sentence_transformers import SentenceTransformer
            texts = test["prompt_text"].tolist()
            if args.no_energy:
                from contextlib import nullcontext
                ctx = nullcontext({"data": {}})
            else:
                ctx = energy.tracked(f"e2e_{name}_router")
            with ctx as run:
                E = SentenceTransformer(config.ROUTER_EMBEDDER).encode(
                    texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                P = np.vstack([router["models"][r]["clf"].predict_proba(
                    router["models"][r]["scaler"].transform(router["models"][r]["pca"].transform(E)))[:, 1]
                    for r in routed])
            if not args.no_energy:
                rs = energy.summarize_run(run, role="router", model_id=config.ROUTER_EMBEDDER,
                                          dataset="e2e", repeat=0, n=n, idle={})
                energy.write_summary(rs, f"e2e_{name}_router")
                passes["router"] = rs
                log["router_kwh_attr"] = energy.record_kwh(rs) / n
            for j, r in enumerate(routed):
                log[f"router_p_{r}"] = P[j]
            log["first_model"] = [routed[k] for k in P.argmax(axis=0)]
            if tau_per_route is not None:
                log["tau_used"] = log["first_model"].map(tau_per_route)
            print("router picks:", log["first_model"].value_counts().to_dict())

        # ---- pass 1: each routed model on its questions (base/cascade: the generalist on all)
        for r in sorted(log["first_model"].unique()):
            idx = log.index[log["first_model"] == r]
            df, summ = run_model(r, test.loc[idx], f"e2e_{name}_{r}", device, args.no_energy, args.idle_secs)
            passes[r] = summ
            log.loc[idx, "first_pred"], log.loc[idx, "first_correct"] = df["pred"].values, df["correct"].values
            log.loc[idx, "first_conf"], log.loc[idx, "first_kwh_attr"] = df["graded"].values, df["kwh_attr"].values
            print(f"  {r}: {len(idx)} questions, acc there {df['correct'].mean():.3f}")

        # ---- escalation decision
        if kind in ("route-cascade", "cascade"):
            log["escalated"] = log["first_conf"] < log["tau_used"]
        log["chosen_model"] = np.where(log["escalated"], esc_role, log["first_model"])
        log["final_pred"], log["final_correct"] = log["first_pred"], log["first_correct"]
        esc = log.index[log["escalated"]]
        print(f"first-pass acc {log['first_correct'].mean():.3f}   escalating {len(esc)}/{n} ({len(esc) / n:.1%})")

        # ---- pass 2: the escalation model on the escalated questions
        if len(esc):
            target = esc_role if kind != "cascade" else arg
            df, summ = run_model(target, test.loc[esc], f"e2e_{name}_{target}", device, args.no_energy, args.idle_secs)
            passes[target] = summ
            log.loc[esc, "final_pred"], log.loc[esc, "final_correct"] = df["pred"].values, df["correct"].values
            log.loc[esc, "pool_kwh_attr"] = df["kwh_attr"].values
            log.loc[esc, "chosen_model"] = target
            print(f"  {target}: {len(esc)} questions, acc there {df['correct'].mean():.3f}")

    log["total_kwh_attr"] = log["router_kwh_attr"] + log["first_kwh_attr"] + log["pool_kwh_attr"]
    log.to_csv(config.RESULTS_DIR / f"end_to_end_{name}_log.csv", index=False)

    total_kwh = float(sum(energy.record_kwh(p) for p in passes.values() if p))
    gco2 = float(sum((energy.emissions_g(p) or 0.0) for p in passes.values() if p))
    out = dict(config=cfg, tau=tau, tau_per_route=tau_per_route, n=n,
               accuracy=float(log["final_correct"].mean()),
               first_pass_accuracy=float(log["first_correct"].mean()) if log["first_correct"].notna().any() else None,
               escalation_rate=float(log["escalated"].mean()),
               total_kwh=total_kwh, wh_per_1000=total_kwh / n * 1e6, gco2e=gco2, gco2e_per_q=gco2 / n,
               wall_s=time.time() - t0,
               per_pass={k: {**{kk: (v or {}).get(kk) for kk in
                                 ("soc_kwh", "total_kwh", "gco2e", "duration_s", "n", "mean_w", "idle_w",
                                  "carbon_factor_g_per_kwh", "cpu_kwh", "ram_kwh")},
                             "record_kwh": energy.record_kwh(v) if v else None}
                         for k, v in passes.items()},
               per_dataset={ds: dict(acc=float(g["final_correct"].mean()), n=int(len(g)),
                                     escalated=float(g["escalated"].mean()),
                                     routed=g["first_model"].value_counts().to_dict())
                            for ds, g in log.groupby("dataset")},
               routed_split=log["first_model"].value_counts().to_dict(),
               chosen_split=log["chosen_model"].value_counts().to_dict(),
               timestamp=datetime.now(timezone.utc).isoformat())
    (config.RESULTS_DIR / f"end_to_end_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{cfg}: acc {out['accuracy']:.3f}  escalated {out['escalation_rate']:.1%}  "
          f"{out['wh_per_1000']:.2f} Wh/1000q  {gco2:.2f} gCO2e  ({out['wall_s'] / 60:.1f} min)")
    print(f"wrote results/end_to_end_{name}.json and _log.csv")


if __name__ == "__main__":
    main()
