#!/usr/bin/env python3
"""Warm the Mac with a sustained MPS workload until thermal readings stabilize.

Run before the scoring batch when you want to compare runs at a similarly warmed
thermal state. Requires password-less sudo access to powermetrics, as the pipeline
energy sampler does.
"""

import argparse
import csv
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

PRESSURE_ORDER = {"nominal": 0, "moderate": 1, "fair": 1,
                  "heavy": 2, "serious": 3, "critical": 4}


def sample_powermetrics(seconds: int) -> dict:
    cmd = ["sudo", "-n", "powermetrics", "--samplers", "thermal,cpu_power,gpu_power",
           "-i", "1000", "-n", str(seconds)]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 15,
                       stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or "powermetrics failed")
    out = p.stdout
    pressures = re.findall(r"pressure level:\s*(\w+)", out, re.I)
    def nums(pattern):
        return [float(x) for x in re.findall(pattern, out, re.I)]
    cpu = nums(r"CPU Power:\s*([\d.]+)\s*mW")
    gpu = nums(r"GPU Power:\s*([\d.]+)\s*mW")
    combined = nums(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
    ghz = [x / 1000 for x in nums(r"frequency:\s*([\d.]+)\s*MHz")]
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    # Thermal pressure is categorical. Keep the most severe state observed in this window.
    pressure = max(pressures, key=lambda x: PRESSURE_ORDER.get(x.lower(), -1)) if pressures else None
    return {"pressure": pressure, "cpu_w": mean(cpu), "gpu_w": mean(gpu),
            "soc_w": mean(combined), "cpu_ghz": mean(ghz), "samples": len(combined)}


def workload(stop: threading.Event, size: int, device: str):
    import torch
    torch.set_grad_enabled(False)
    x = torch.randn((size, size), device=device, dtype=torch.float16)
    y = torch.randn((size, size), device=device, dtype=torch.float16)
    while not stop.is_set():
        z = x @ y
        # Force completion so queued MPS work cannot accumulate and distort timing.
        torch.mps.synchronize() if device == "mps" else None
        del z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-seconds", type=int, default=60,
                    help="powermetrics sampling window (default: 60)")
    ap.add_argument("--stable-windows", type=int, default=3,
                    help="consecutive stable windows required (default: 3)")
    ap.add_argument("--max-minutes", type=float, default=45,
                    help="maximum warm-up duration (default: 45; then exits without claiming stability)")
    ap.add_argument("--tolerance", type=float, default=0.08,
                    help="max relative change in SoC power and CPU clock between windows (default: 0.08)")
    ap.add_argument("--target-pressure", choices=("nominal", "moderate", "heavy", "critical"),
                    default=None, help="require this thermal pressure or higher before declaring stable")
    ap.add_argument("--matrix-size", type=int, default=3072,
                    help="MPS matrix dimension for sustained load (default: 3072)")
    ap.add_argument("--output", type=Path, default=None,
                    help="CSV output path (default: mac_pipeline/energy/thermal_warmup_<time>.csv)")
    ap.add_argument("--then", nargs=argparse.REMAINDER, metavar="COMMAND",
                    help="stop the warm-up and immediately run this command after stability is reached")
    args = ap.parse_args()

    if platform.system() != "Darwin":
        ap.error("thermal_stabilize.py is for macOS")
    if args.window_seconds < 10 or args.stable_windows < 2 or args.max_minutes <= 0:
        ap.error("use window-seconds >= 10, stable-windows >= 2, and max-minutes > 0")

    try:
        import torch
    except ImportError as e:
        ap.error(f"PyTorch is required: {e}")
    if not torch.backends.mps.is_available():
        ap.error("PyTorch MPS is unavailable; activate the pipeline environment")
    try:
        first = sample_powermetrics(1)
    except Exception as e:
        ap.error(f"powermetrics unavailable ({e}); check password-less sudo setup")
    if first["samples"] == 0:
        ap.error("powermetrics returned no Combined Power samples")

    if args.output is None:
        root = Path(__file__).resolve().parent
        outdir = root / "energy"
        outdir.mkdir(parents=True, exist_ok=True)
        args.output = outdir / f"thermal_warmup_{datetime.now():%Y%m%d_%H%M%S}.csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()
    worker = threading.Thread(target=workload, args=(stop, args.matrix_size, "mps"), daemon=True)
    worker.start()
    fields = ["window", "timestamp", "pressure", "cpu_w", "gpu_w", "soc_w", "cpu_ghz", "samples"]
    history = []
    start = time.monotonic()
    stable_count = 0
    print(f"Sustained MPS load started; sampling {args.window_seconds}s windows.")
    print("Stability means consecutive windows have the same thermal pressure and SoC power / CPU clock within tolerance.")
    try:
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            window = 0
            while time.monotonic() - start < args.max_minutes * 60:
                row = sample_powermetrics(args.window_seconds)
                window += 1
                row = {"window": window, "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"), **row}
                writer.writerow(row)
                f.flush()
                history.append(row)
                prev = history[-2] if len(history) >= 2 else None
                target_met = (args.target_pressure is None or
                              (row["pressure"] is not None and
                               PRESSURE_ORDER.get(row["pressure"].lower(), -1) >=
                               PRESSURE_ORDER[args.target_pressure]))
                same = bool(prev and row["pressure"] is not None and
                            row["pressure"] == prev["pressure"] and
                            target_met and
                            row["samples"] >= args.window_seconds // 2 and
                            prev["samples"] >= args.window_seconds // 2 and
                            all(row[k] is not None and prev[k] is not None and
                                abs(row[k] - prev[k]) / max(abs(prev[k]), 0.1) <= args.tolerance
                                for k in ("soc_w", "cpu_ghz")))
                stable_count = stable_count + 1 if same else 0
                print(f"window {window}: pressure={row['pressure'] or 'unknown'}  "
                      f"SoC={row['soc_w'] or 0:.1f} W  CPU={row['cpu_ghz'] or 0:.2f} GHz  "
                      f"stable={stable_count}/{args.stable_windows}", flush=True)
                if stable_count >= args.stable_windows:
                    print(f"Stable state reached. Log: {args.output}")
                    if args.then:
                        command = args.then
                        if command and command[0] == "--":
                            command = command[1:]
                        if not command:
                            ap.error("--then needs a command")
                        print(f"Stopping warm-up and starting: {' '.join(command)}", flush=True)
                        stop.set()
                        worker.join(timeout=10)
                        return subprocess.run(command, check=False).returncode
                    print("Warm-up workload will now stop. Run the benchmark promptly; its own sustained load will determine its thermal state.")
                    return 0
            print(f"Maximum warm-up time reached without meeting stability criteria. Log: {args.output}")
            return 2
    except KeyboardInterrupt:
        print("Interrupted; stopping workload.")
        return 130
    finally:
        stop.set()
        worker.join(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
