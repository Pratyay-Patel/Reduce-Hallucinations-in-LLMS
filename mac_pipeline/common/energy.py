"""
mac_pipeline/common/energy.py
-----------------------------
CodeCarbon wrapper for the Mac. One tracked run per (model, dataset); per-question energy is
attributed afterwards by share of wall time. Also: idle baseline, thermal/clock readout via
powermetrics, and the JSON summary schema every run writes.

On Apple Silicon CodeCarbon reads CPU+GPU power from `sudo powermetrics`. If sudo is not
password-less it silently falls back to a constant TDP guess -- `check_measurement()` catches
that by asserting the energy fields are non-zero and the powermetrics path is active.
"""

import json
import platform
import re
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

import config

IS_MAC = platform.system() == "Darwin"


def _tracker(project_name: str):
    from codecarbon import OfflineEmissionsTracker
    kw = dict(project_name=project_name,
              country_iso_code=config.COUNTRY,
              measure_power_secs=config.MEASURE_POWER_SECS,
              tracking_mode="machine",
              output_dir=str(config.ENERGY_DIR),
              output_file="emissions.csv",
              save_to_file=True,
              log_level="error",
              allow_multiple_runs=True)
    if config.CARBON_INTENSITY_G_PER_KWH:
        kw["force_carbon_intensity_g_co2e_kwh"] = float(config.CARBON_INTENSITY_G_PER_KWH)
    return OfflineEmissionsTracker(**kw)


def powermetrics_available() -> bool:
    if not IS_MAC:
        return False
    try:
        from codecarbon.core.powermetrics import is_powermetrics_available
        return bool(is_powermetrics_available())
    except Exception:
        return False


def thermal_info() -> dict:
    """Thermal pressure and mean CPU clock from one powermetrics sample (Mac only)."""
    if not IS_MAC:
        return {"thermal_pressure": None, "mean_cpu_ghz": None}
    try:
        out = subprocess.run(["sudo", "-n", "powermetrics", "--samplers", "thermal,cpu_power",
                              "-n", "1", "-i", "500"], capture_output=True, text=True,
                             timeout=10, stdin=subprocess.DEVNULL).stdout
        tp = re.search(r"pressure level:\s*(\w+)", out)
        freqs = [float(x) for x in re.findall(r"frequency:\s*([\d.]+)\s*MHz", out)]
        return {"thermal_pressure": tp.group(1) if tp else None,
                "mean_cpu_ghz": round(float(np.mean(freqs)) / 1000, 3) if freqs else None}
    except Exception:
        return {"thermal_pressure": None, "mean_cpu_ghz": None}


def _data_to_dict(d) -> dict:
    keys = ["run_id", "duration", "emissions", "cpu_power", "gpu_power", "ram_power",
            "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed", "country_iso_code",
            "country_name", "codecarbon_version", "cpu_model", "gpu_model", "tracking_mode"]
    return {k: getattr(d, k, None) for k in keys}


@contextmanager
def tracked(project_name: str):
    """Usage:  with tracked("score_base_mmlu") as run: ...; run["data"] holds the summary."""
    t = _tracker(project_name)
    run = {"project": project_name, "t0": time.perf_counter()}
    t.start()
    try:
        yield run
    finally:
        t.stop()
        run["wall_s"] = time.perf_counter() - run["t0"]
        run["data"] = _data_to_dict(t.final_emissions_data)


def idle_baseline(seconds: int, tag: str) -> dict:
    """Track `seconds` of nothing. Returns mean idle watts and writes energy/idle_<tag>.json."""
    if seconds <= 0:
        return {"idle_w": None, "idle_kwh_per_s": None}
    with tracked(f"idle_{tag}") as run:
        time.sleep(seconds)
    d = run["data"]
    kwh = d["energy_consumed"] or 0.0
    out = {"tag": tag, "seconds": seconds, "kwh": kwh,
           "idle_w": kwh * 3.6e6 / max(d["duration"] or seconds, 1e-9),
           "idle_kwh_per_s": kwh / max(d["duration"] or seconds, 1e-9),
           "cpu_kwh": d["cpu_energy"], "gpu_kwh": d["gpu_energy"], "ram_kwh": d["ram_energy"],
           "timestamp": datetime.now(timezone.utc).isoformat()}
    (config.ENERGY_DIR / f"idle_{tag}.json").write_text(json.dumps(out, indent=2))
    return out


def attribute(total_kwh: float, seconds) -> np.ndarray:
    """Split a run's energy over its questions by share of wall time."""
    s = np.asarray(seconds, dtype=float)
    if s.sum() <= 0:
        return np.full(len(s), total_kwh / max(len(s), 1))
    return total_kwh * s / s.sum()


def summarize_run(run: dict, *, role: str, model_id: str, dataset: str, repeat: int, n: int,
                  idle: dict, extra: dict | None = None) -> dict:
    d = run["data"]
    total = d["energy_consumed"] or 0.0
    dur = d["duration"] or run["wall_s"]
    idle_kwh = (idle.get("idle_kwh_per_s") or 0.0) * dur
    factor = (d["emissions"] * 1000 / total) if (total and d["emissions"] is not None) else None
    out = dict(role=role, model_id=model_id, dataset=dataset, repeat=repeat, run_id=d["run_id"],
               n=n, duration_s=dur, cpu_kwh=d["cpu_energy"], gpu_kwh=d["gpu_energy"],
               ram_kwh=d["ram_energy"], total_kwh=total, gco2e=(d["emissions"] or 0.0) * 1000,
               carbon_intensity_g_per_kwh=factor, country=d["country_iso_code"],
               mean_w=total * 3.6e6 / dur if dur else None, idle_w=idle.get("idle_w"),
               idle_kwh_over_run=idle_kwh, net_kwh=total - idle_kwh,
               cpu_w=d["cpu_power"], gpu_w=d["gpu_power"], ram_w=d["ram_power"],
               powermetrics=powermetrics_available(), codecarbon_version=d["codecarbon_version"],
               cpu_model=d["cpu_model"], gpu_model=d["gpu_model"],
               timestamp=datetime.now(timezone.utc).isoformat())
    out.update(thermal_info())
    if extra:
        out.update(extra)
    return out


def write_summary(summary: dict, name: str):
    p = config.ENERGY_DIR / f"{name}.json"
    p.write_text(json.dumps(summary, indent=2))
    return p


def load_summary(name: str) -> dict | None:
    p = config.ENERGY_DIR / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def check_measurement(seconds: int = 30) -> dict:
    """Run the tracker on a small CPU+GPU load and report whether it actually measured."""
    import torch
    dev = config.pick_device()
    with tracked("check_measurement") as run:
        x = torch.randn(1024, 1024, device=dev)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            x = (x @ x).clamp(-1, 1)
        if dev == "mps":
            torch.mps.synchronize()
    d = run["data"]
    ok_cpu = (d["cpu_energy"] or 0) > 0
    ok_gpu = (d["gpu_energy"] or 0) > 0
    return dict(device=dev, powermetrics=powermetrics_available(), cpu_kwh=d["cpu_energy"],
                gpu_kwh=d["gpu_energy"], ram_kwh=d["ram_energy"], total_kwh=d["energy_consumed"],
                cpu_w=d["cpu_power"], gpu_w=d["gpu_power"], ok_cpu=ok_cpu, ok_gpu=ok_gpu,
                codecarbon_version=d["codecarbon_version"])
