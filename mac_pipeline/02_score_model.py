"""
mac_pipeline/02_score_model.py
------------------------------
Score one model on every question (train and test) with option-likelihood scoring, one
CodeCarbon-tracked run per dataset, energy attributed per question by share of wall time.

  python mac_pipeline/02_score_model.py --role base
  python mac_pipeline/02_score_model.py --role phi4mini --dataset mmlu
  python mac_pipeline/02_score_model.py --role base --limit 200 --repeat 1     # energy repeat

Outputs
  scores/<role>.csv            (repeat 0)   or   scores/<role>_r<k>.csv   (repeat k)
  energy/<role>_<dataset>_r<k>.json,  energy/idle_<role>_<dataset>_r<k>.json
Re-running replaces that role's rows for the datasets scored.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd

from common import energy
from common.data import load_questions, subset
from common.scorer import OptionScorer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=list(config.MODELS))
    ap.add_argument("--dataset", nargs="*", default=config.DATASETS)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per dataset (0 = all)")
    ap.add_argument("--repeat", type=int, default=0, help="0 = main run; k>0 = energy repeat k")
    ap.add_argument("--idle-secs", type=int, default=config.IDLE_SECS)
    ap.add_argument("--no-energy", action="store_true", help="skip CodeCarbon (dev smoke test only)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--chat-template", choices=["auto", "on", "off"], default="auto")
    args = ap.parse_args()

    role, m = args.role, config.MODELS[args.role]
    device = args.device or config.pick_device()
    ct = m["chat_template"] if args.chat_template == "auto" else (args.chat_template == "on")
    q = load_questions()
    out_csv = config.SCORES_DIR / (f"{role}.csv" if args.repeat == 0 else f"{role}_r{args.repeat}.csv")

    print(f"role={role}  model={m['id']}  device={device}  chat_template={ct}  repeat={args.repeat}")
    t0 = time.time()
    scorer = OptionScorer(m["id"], device, chat_template=ct)
    print(f"loaded in {time.time() - t0:.0f}s   peak mem {scorer.peak_mem_gb():.2f} GB")

    new_rows = []
    for ds in args.dataset:
        sub = q[q["dataset"] == ds]
        if args.limit:
            sub = subset(sub, args.limit)          # seeded, spread across subjects
        if len(sub) == 0:
            continue
        tag = f"{role}_{ds}_r{args.repeat}"

        # warm-up (not tracked): first call pays kernel compilation on MPS
        r0 = sub.iloc[0]
        scorer.score(r0["question"], r0["choices"], int(r0["gold"]))

        idle = {} if args.no_energy else energy.idle_baseline(args.idle_secs, tag)
        scorer.reset_peak()
        rows = []
        if args.no_energy:
            from contextlib import nullcontext
            ctx = nullcontext({"data": {}})
        else:
            ctx = energy.tracked(f"score_{tag}")
        with ctx as run:
            for k, (_, r) in enumerate(sub.iterrows()):
                s = scorer.score(r["question"], r["choices"], int(r["gold"]))
                rows.append(dict(id=r["id"], dataset=ds, subject=r["subject"], fold=r.get("fold"),
                                 model=role, chat_template=int(ct), n_options=int(r["n_options"]),
                                 gold=int(r["gold"]), pred=s["pred"], pred_letter=s["pred_letter"],
                                 correct=s["correct"], graded=s["graded"], p_gold=s["p_gold"],
                                 logprobs=json.dumps(s["logprobs"]), prompt_tokens=s["prompt_tokens"],
                                 seconds=s["seconds"], peak_mem_gb=s["peak_mem_gb"],
                                 energy_kwh_attr=np.nan, run_id=None, repeat=args.repeat))
                if (k + 1) % 100 == 0:
                    acc = np.mean([x["correct"] for x in rows])
                    print(f"  {ds}: {k + 1}/{len(sub)}  acc {acc:.3f}  "
                          f"{np.mean([x['seconds'] for x in rows]):.2f} s/q", flush=True)
        df = pd.DataFrame(rows)
        if not args.no_energy:
            summ = energy.summarize_run(run, role=role, model_id=m["id"], dataset=ds,
                                        repeat=args.repeat, n=len(df), idle=idle,
                                        extra=dict(chat_template=int(ct), device=device,
                                                   accuracy=float(df["correct"].mean()),
                                                   mean_s_per_q=float(df["seconds"].mean()),
                                                   median_s_per_q=float(df["seconds"].median()),
                                                   peak_mem_gb=float(df["peak_mem_gb"].max()),
                                                   limit=args.limit))
            energy.write_summary(summ, tag)
            df["energy_kwh_attr"] = energy.attribute(summ["total_kwh"], df["seconds"])
            df["run_id"] = summ["run_id"]
            print(f"  {ds}: n={len(df)} acc={df['correct'].mean():.3f}  "
                  f"{summ['total_kwh'] * 1000:.3f} Wh  {summ['gco2e']:.2f} gCO2e  "
                  f"mean {summ['mean_w'] or 0:.1f} W  idle {idle.get('idle_w') or 0:.1f} W  "
                  f"powermetrics={summ['powermetrics']}")
        else:
            print(f"  {ds}: n={len(df)} acc={df['correct'].mean():.3f}  (energy not tracked)")
        chance = (1.0 / df["n_options"]).mean()
        if df["correct"].mean() < chance + 0.05:
            print(f"  WARNING: {role} is at chance on {ds} ({df['correct'].mean():.3f} vs {chance:.3f}); "
                  f"its confidence there is noise.")
        new_rows.append(df)
        if args.no_energy is False and args.idle_secs > 0:
            time.sleep(5)

    scorer.unload()
    new = pd.concat(new_rows, ignore_index=True)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[~old["dataset"].isin(new["dataset"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(new)} rows)   total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
