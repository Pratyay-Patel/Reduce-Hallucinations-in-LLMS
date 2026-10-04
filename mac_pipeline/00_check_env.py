"""
mac_pipeline/00_check_env.py
----------------------------
Run this first, once, on the Mac.

  1. torch backend (MPS), fp16, CodeCarbon version, password-less `sudo powermetrics`,
     HF access to all four model repos.
  2. A 30 s tracked load: asserts CPU and GPU energy are non-zero (i.e. really measured).
  3. For each model: score the first 200 MMLU questions raw and with the chat template,
     compare to the published MMLU number, keep the better format in config_overrides.json.

  python mac_pipeline/00_check_env.py                 # everything
  python mac_pipeline/00_check_env.py --models base   # one model
  python mac_pipeline/00_check_env.py --skip-models   # environment + energy only
"""

import argparse
import os
import platform
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402


def env_checks():
    import torch
    import codecarbon
    import transformers
    print(f"python {platform.python_version()}  {platform.system()} {platform.machine()}")
    print(f"torch {torch.__version__}  transformers {transformers.__version__}  "
          f"codecarbon {codecarbon.__version__}")
    dev = config.pick_device()
    print(f"device: {dev}")
    ok = True
    if platform.system() == "Darwin" and dev != "mps":
        print("  FAIL: MPS backend not available -- install a torch build with MPS support")
        ok = False
    x = torch.randn(64, 64, device=dev, dtype=torch.float16)
    assert torch.isfinite(x @ x).all(), "fp16 matmul failed"
    print("  fp16 matmul: ok")

    if platform.system() == "Darwin":
        r = subprocess.run(["sudo", "-n", "powermetrics", "--samplers", "cpu_power", "-n", "1", "-i", "100"],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=15)
        if r.returncode == 0:
            print("  sudo -n powermetrics: ok")
        else:
            ok = False
            print("  FAIL: powermetrics needs a password. Run `sudo visudo` and add:\n"
                  f"        {os.environ.get('USER', '<user>')} ALL = (root) NOPASSWD: /usr/bin/powermetrics")

    from huggingface_hub import model_info
    for role, m in config.MODELS.items():
        try:
            model_info(m["id"])
            print(f"  HF access {role:<9} {m['id']}: ok")
        except Exception as e:
            ok = False
            print(f"  FAIL: cannot access {m['id']} ({type(e).__name__}); accept the licence on HF / login")
    return ok


def energy_check():
    from common import energy
    r = energy.check_measurement(30)
    print(f"tracked 30 s load on {r['device']}:")
    if r.get("error"):
        print(f"  FAIL: {r['error']}")
        return False
    hw = r.get("cc_hardware", {})
    print(f"  powermetrics SoC: {(r['soc_kwh'] or 0) * 1000:.3f} Wh, mean {r['soc_mean_w'] or 0:.1f} W, "
          f"{r['soc_samples']} samples in 30 s   <- energy of record")
    print(f"  CodeCarbon:       {(r['cc_total_kwh'] or 0) * 1000:.3f} Wh  cpu {r['cc_cpu_w'] or 0:.1f} W  "
          f"gpu {r['cc_gpu_w'] or 0:.1f} W   carbon factor {r['carbon_factor'] or 0:.0f} gCO2e/kWh")
    print(f"  CodeCarbon trackers: {', '.join(hw.get('trackers', [])) or 'unknown'}")
    if hw.get("cpu_is_tdp_model"):
        print("  note: CodeCarbon's CPU figure is a TDP model, not a reading. It is kept for comparison "
              "only; the SoC figure above is what the pipeline records.")
    if platform.system() == "Darwin":
        ok = bool(r["ok_soc"]) and bool(r["powermetrics"])
        expected = 30 // config.MEASURE_POWER_SECS
        if r["soc_samples"] < expected // 2:
            print(f"  FAIL: only {r['soc_samples']} powermetrics samples in 30 s (expected ~{expected}).")
            ok = False
        if (r["soc_mean_w"] or 0) <= 0:
            print("  FAIL: powermetrics reported zero SoC power.")
            ok = False
        if not ok:
            print("  STOP: the pipeline would fall back to CodeCarbon's TDP model. Every tracked step\n"
                  "        refuses to run in that state (config.REQUIRE_POWERMETRICS). Fix sudoers:\n"
                  f"        {os.environ.get('USER', '<user>')} ALL = (root) NOPASSWD: /usr/bin/powermetrics")
    else:
        ok = r["ok_cpu"]
        print("  (not a Mac: powermetrics path not applicable; CPU energy is a TDP estimate)")
    return ok


def model_checks(roles):
    import pandas as pd
    from common.data import load_questions, subset
    from common.scorer import OptionScorer
    dev = config.pick_device()
    q = load_questions()
    # train rows only: the prompt-format choice must not look at test questions
    sub = subset(q[(q["dataset"] == "mmlu") & (q["fold"] == "train")], config.SANITY_N)
    print(f"\nsanity subset: {len(sub)} MMLU questions spread across the 20 subjects")
    print(f"{'role':<10}{'format':<10}{'acc':>7}{'published':>11}{'s/q':>7}{'peak GB':>9}")
    for role in roles:
        m = config.MODELS[role]
        res = {}
        for ct in (False, True):
            try:
                sc = OptionScorer(m["id"], dev, chat_template=ct)
            except Exception as e:
                print(f"{role:<10}{'chat' if ct else 'raw':<10} LOAD FAILED: {type(e).__name__}: {e}")
                continue
            rows = [sc.score(r["question"], r["choices"], int(r["gold"])) for _, r in sub.iterrows()]
            df = pd.DataFrame(rows)
            res[ct] = df["correct"].mean()
            print(f"{role:<10}{'chat' if ct else 'raw':<10}{res[ct]:>7.3f}{m['published_mmlu'] / 100:>11.3f}"
                  f"{df['seconds'].mean():>7.2f}{df['peak_mem_gb'].max():>9.2f}")
            sc.unload()
            time.sleep(3)
        if res:
            best = max(res, key=res.get)
            config.save_override(role, best)
            gap = m["published_mmlu"] / 100 - res[best]
            flag = "   <-- >15 pts under published: check prompt format before scoring 2,000" if gap > 0.15 else ""
            print(f"{'':<10}-> chat_template={best}  (gap to published {gap:+.3f}){flag}")
    print(f"\nchoices written to {config.OVERRIDES_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=config.ALL_ROLES)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-energy", action="store_true")
    args = ap.parse_args()

    ok = env_checks()
    if not args.skip_energy:
        ok = energy_check() and ok
    if not ok:
        print("\nEnvironment problems above must be fixed before scoring. Stopping.")
        sys.exit(1)
    if not args.skip_models:
        if not config.QUESTIONS_CSV.exists():
            print("\nquestions.csv missing -- run 01_build_questions.py first")
            sys.exit(1)
        model_checks(args.models)


if __name__ == "__main__":
    main()
