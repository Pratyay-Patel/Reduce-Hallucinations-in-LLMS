"""
mac_pipeline/06_run_end_to_end.py
---------------------------------
The live measurement: run the deployed decision rule on the TEST split with CodeCarbon on.

  pass 1  base model on every test question            -> confidence, escalation set
  router  embed the escalated questions, pick a pool member (hybrid only)
  pass 2  each pool model on the questions assigned to it, one model at a time

End-to-end energy = sum of the tracked passes. Questions are grouped by model rather than
interleaved per query because all four models resident would need ~23 GB; the per-question
computation is identical.

  python mac_pipeline/06_run_end_to_end.py --config hybrid                 # tau* from train
  python mac_pipeline/06_run_end_to_end.py --config hybrid --tau 0.7
  python mac_pipeline/06_run_end_to_end.py --config cascade:qwen3b
  python mac_pipeline/06_run_end_to_end.py --config base                   # tau = 0 anchor
  python mac_pipeline/06_run_end_to_end.py --config single:phi4mini        # always-X anchor

Outputs results/end_to_end_<config>_<tau>.json and ..._log.csv
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
from common.data import load_questions
from common.scorer import OptionScorer


def run_model(role, sub: pd.DataFrame, tag: str, device, no_energy: bool, idle_secs: int):
    """Score `sub` with one model inside one tracked run. Returns (rows df, summary or None)."""
    m = config.MODELS[role]
    sc = OptionScorer(m["id"], device, chat_template=m["chat_template"])
    r0 = sub.iloc[0]
    sc.score(r0["question"], r0["choices"], int(r0["gold"]))          # warm-up, untracked
    idle = {} if no_energy else energy.idle_baseline(idle_secs, tag)
    rows = []
    if no_energy:
        from contextlib import nullcontext
        ctx = nullcontext({"data": {}})
    else:
        ctx = energy.tracked(tag)
    with ctx as run:
        for _, r in sub.iterrows():
            s = sc.score(r["question"], r["choices"], int(r["gold"]))
            rows.append(dict(id=r["id"], model=role, pred=s["pred"], correct=s["correct"],
                             graded=s["graded"], seconds=s["seconds"]))
    sc.unload()
    df = pd.DataFrame(rows)
    summ = None
    if not no_energy:
        summ = energy.summarize_run(run, role=role, model_id=m["id"], dataset="e2e", repeat=0,
                                    n=len(df), idle=idle, extra=dict(tag=tag))
        energy.write_summary(summ, tag)
        df["kwh_attr"] = energy.attribute(summ["total_kwh"], df["seconds"])
    else:
        df["kwh_attr"] = 0.0
    time.sleep(3)
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="hybrid",
                    help="hybrid | cascade:<pool role> | base | single:<role>")
    ap.add_argument("--tau", type=float, default=None, help="default: tau* from router/thresholds.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--idle-secs", type=int, default=config.IDLE_SECS)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or config.pick_device()
    q = load_questions()
    test = q[q["fold"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)
    n = len(test)
    cfg = args.config
    kind, _, arg = cfg.partition(":")
    pool = config.POOL

    if kind == "hybrid":
        tau = args.tau if args.tau is not None else json.loads(
            (config.ROUTER_DIR / "thresholds.json").read_text())["tau_star"]
    elif kind == "cascade":
        tau = args.tau if args.tau is not None else json.loads(
            (config.ROUTER_DIR / "thresholds.json").read_text())["tau_star"]
        assert arg in pool, f"cascade target must be one of {pool}"
    elif kind == "base":
        tau = 0.0
    elif kind == "single":
        tau = 1.01
        assert arg in config.MODELS
    else:
        sys.exit(f"unknown config {cfg}")
    name = f"{cfg.replace(':', '-')}_{tau:.2f}"
    print(f"config={cfg}  tau={tau}  n={n}  device={device}")

    log = test[["id", "dataset", "subject", "n_options", "gold"]].copy()
    passes = {}
    t0 = time.time()

    # ---- pass 1: base (or the single model)
    if kind == "single":
        df, summ = run_model(arg, test, f"e2e_{name}_{arg}", device, args.no_energy, args.idle_secs)
        passes[arg] = summ
        log["base_pred"], log["base_correct"], log["base_conf"] = np.nan, np.nan, np.nan
        log["escalated"] = True
        log["chosen_model"] = arg
        log["final_pred"], log["final_correct"] = df["pred"].values, df["correct"].values
        log["base_kwh_attr"], log["router_kwh_attr"], log["pool_kwh_attr"] = 0.0, 0.0, df["kwh_attr"].values
    else:
        df, summ = run_model(config.BASE, test, f"e2e_{name}_base", device, args.no_energy, args.idle_secs)
        passes[config.BASE] = summ
        log["base_pred"], log["base_correct"], log["base_conf"] = df["pred"].values, df["correct"].values, df["graded"].values
        log["base_kwh_attr"] = df["kwh_attr"].values
        log["escalated"] = log["base_conf"] < tau
        log["router_kwh_attr"], log["pool_kwh_attr"] = 0.0, 0.0
        for r in pool:
            log[f"router_p_{r}"] = np.nan
        esc = log[log["escalated"]]
        print(f"base acc {log['base_correct'].mean():.3f}   escalating {len(esc)}/{n} ({len(esc) / n:.1%})")

        # ---- router pass (hybrid only)
        if kind == "hybrid" and len(esc):
            with open(config.ROUTER_DIR / "router.pkl", "rb") as f:
                router = pickle.load(f)
            from sentence_transformers import SentenceTransformer
            texts = test.loc[esc.index, "prompt_text"].tolist()
            conf = esc["base_conf"].values.reshape(-1, 1)
            if args.no_energy:
                from contextlib import nullcontext
                ctx = nullcontext({"data": {}})
            else:
                ctx = energy.tracked(f"e2e_{name}_router")
            with ctx as run:
                E = SentenceTransformer(config.ROUTER_EMBEDDER).encode(
                    texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
                P = np.vstack([router["models"][r]["lr"].predict_proba(
                    router["models"][r]["scaler"].transform(
                        np.hstack([conf, router["models"][r]["pca"].transform(E)])))[:, 1]
                    for r in pool])
            if not args.no_energy:
                rs = energy.summarize_run(run, role="router", model_id=config.ROUTER_EMBEDDER,
                                          dataset="e2e", repeat=0, n=len(esc), idle={})
                energy.write_summary(rs, f"e2e_{name}_router")
                passes["router"] = rs
                log.loc[esc.index, "router_kwh_attr"] = rs["total_kwh"] / len(esc)
            for j, r in enumerate(pool):
                log.loc[esc.index, f"router_p_{r}"] = P[j]
            log.loc[esc.index, "chosen_model"] = [pool[k] for k in P.argmax(axis=0)]
        elif kind == "cascade":
            log.loc[esc.index, "chosen_model"] = arg
        log.loc[~log["escalated"], "chosen_model"] = config.BASE
        log["final_pred"], log["final_correct"] = log["base_pred"], log["base_correct"]

        # ---- pass 2: each pool model on its assigned questions
        for r in pool:
            idx = log.index[log["chosen_model"] == r]
            if len(idx) == 0:
                continue
            sub = test.loc[idx]
            df, summ = run_model(r, sub, f"e2e_{name}_{r}", device, args.no_energy, args.idle_secs)
            passes[r] = summ
            log.loc[idx, "final_pred"] = df["pred"].values
            log.loc[idx, "final_correct"] = df["correct"].values
            log.loc[idx, "pool_kwh_attr"] = df["kwh_attr"].values
            print(f"  {r}: {len(idx)} questions, acc there {df['correct'].mean():.3f}")

    log["total_kwh_attr"] = log["base_kwh_attr"] + log["router_kwh_attr"] + log["pool_kwh_attr"]
    log.to_csv(config.RESULTS_DIR / f"end_to_end_{name}_log.csv", index=False)

    total_kwh = float(sum((p or {}).get("total_kwh", 0.0) for p in passes.values()))
    gco2 = float(sum((p or {}).get("gco2e", 0.0) for p in passes.values()))
    out = dict(config=cfg, tau=tau, n=n, accuracy=float(log["final_correct"].mean()),
               escalation_rate=float(log["escalated"].mean()),
               total_kwh=total_kwh, wh_per_1000=total_kwh / n * 1e6, gco2e=gco2, gco2e_per_q=gco2 / n,
               wall_s=time.time() - t0,
               per_pass={k: {kk: (v or {}).get(kk) for kk in ("total_kwh", "gco2e", "duration_s", "n", "mean_w", "idle_w")}
                         for k, v in passes.items()},
               per_dataset={ds: dict(acc=float(g["final_correct"].mean()), n=int(len(g)),
                                     escalated=float(g["escalated"].mean()))
                            for ds, g in log.groupby("dataset")},
               chosen_split=log["chosen_model"].value_counts().to_dict(),
               timestamp=datetime.now(timezone.utc).isoformat())
    (config.RESULTS_DIR / f"end_to_end_{name}.json").write_text(json.dumps(out, indent=2))
    print(f"\n{cfg} tau={tau}: acc {out['accuracy']:.3f}  escalated {out['escalation_rate']:.1%}  "
          f"{out['wh_per_1000']:.2f} Wh/1000q  {gco2:.2f} gCO2e  ({out['wall_s'] / 60:.1f} min)")
    print(f"wrote results/end_to_end_{name}.json and _log.csv")


if __name__ == "__main__":
    main()
