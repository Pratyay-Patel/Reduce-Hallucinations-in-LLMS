"""
mac_pipeline/common/energy.py
-----------------------------
Energy measurement for the Mac. One tracked run per (model, dataset); per-question energy is
attributed afterwards by share of wall time.

Two sources are recorded for every run:

  * `soc_kwh`   -- the SoC's own power from `sudo powermetrics` (CPU + GPU + ANE, "Combined
                   Power"), sampled every second and integrated here. This is the energy of
                   record on the Mac.
  * `total_kwh` -- CodeCarbon's figure. On Apple Silicon (CodeCarbon 3.3.1) its CPU term is a
                   generic 85 W TDP load model (floor 8.5 W), only its GPU term comes from
                   powermetrics, and its emissions figure omits the GPU energy. It is kept for
                   the carbon factor and for comparison, not as the measurement.

Emissions are computed here as `record_kwh x carbon factor`, where the factor is the one
CodeCarbon applied for the configured country (recovered from its own emissions / energy).
`record_kwh(summary)` returns soc_kwh when present, else total_kwh, so every downstream
script uses one number.
"""

import json
import platform
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

import config

IS_MAC = platform.system() == "Darwin"


# ----------------------------------------------------------------------------- CodeCarbon
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


def codecarbon_hardware(tracker) -> dict:
    """Which hardware trackers CodeCarbon instantiated, and whether its CPU figure is a model.
    `CPU(mode=cpu_load|constant|windows_emi, tdp=85, generic=True)` means an 85 W-TDP estimate;
    `AppleSiliconChip(CPU)` would mean powermetrics."""
    items, cpu_is_model = [], False
    for h in getattr(tracker, "_hardware", []):
        name = type(h).__name__
        mode = getattr(h, "_mode", None)
        tdp = getattr(h, "_tdp", None)
        generic = getattr(h, "_is_generic_tdp", None)
        part = getattr(h, "chip_part", None)
        desc = name + (f"({part})" if part else "") + \
            (f"[mode={mode}, tdp={tdp} W, generic_tdp={generic}]" if mode else "")
        items.append(desc)
        if name == "CPU" and mode in ("cpu_load", "constant", "windows_emi", "intel_power_gadget"):
            cpu_is_model = mode in ("cpu_load", "constant")
    return {"trackers": items, "cpu_is_tdp_model": cpu_is_model}


def _data_to_dict(d) -> dict:
    keys = ["run_id", "duration", "emissions", "cpu_power", "gpu_power", "ram_power",
            "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed", "country_iso_code",
            "country_name", "codecarbon_version", "cpu_model", "gpu_model", "tracking_mode"]
    return {k: getattr(d, k, None) for k in keys}


# ----------------------------------------------------------------------------- powermetrics
class PowermetricsSampler:
    """Streams `sudo powermetrics --samplers cpu_power,gpu_power -i <ms>` and integrates the
    reported CPU, GPU, ANE and Combined power over each sample's elapsed time. Mac only;
    elsewhere start() is a no-op and stop() returns Nones."""

    _ELAPSED = re.compile(r"\((\d+(?:\.\d+)?)ms elapsed\)")
    _FIELDS = {"cpu": re.compile(r"^CPU Power:\s*(\d+)\s*mW"),
               "gpu": re.compile(r"^GPU Power:\s*(\d+)\s*mW"),
               "ane": re.compile(r"^ANE Power:\s*(\d+)\s*mW"),
               "combined": re.compile(r"^Combined Power \(CPU \+ GPU \+ ANE\):\s*(\d+)\s*mW")}

    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.active = False
        self.mws = {k: 0.0 for k in self._FIELDS}      # sum of mW * ms
        self.samples = 0
        self.ms_total = 0.0
        self._cur = {}
        self._elapsed = None
        self._p = None
        self._t = None

    def start(self):
        if not IS_MAC or not powermetrics_available():
            return
        try:
            self._p = subprocess.Popen(
                ["sudo", "-n", "powermetrics", "--samplers", "cpu_power,gpu_power",
                 "-i", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                text=True, bufsize=1)
        except Exception:
            self._p = None
            return
        self.active = True
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        for line in self._p.stdout:
            m = self._ELAPSED.search(line)
            if m:
                self._elapsed = float(m.group(1))
                self._cur = {}
                continue
            for k, rx in self._FIELDS.items():
                mm = rx.match(line.strip())
                if mm:
                    self._cur[k] = float(mm.group(1))
                    if k == "combined" and self._elapsed:
                        for kk, v in self._cur.items():
                            self.mws[kk] += v * self._elapsed
                        self.ms_total += self._elapsed
                        self.samples += 1
                    break

    def stop(self) -> dict:
        if not self.active:
            return {"soc_kwh": None, "soc_cpu_kwh": None, "soc_gpu_kwh": None, "soc_ane_kwh": None,
                    "soc_mean_w": None, "soc_samples": 0}
        try:
            self._p.terminate()
            try:
                self._p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "-n", "kill", "-9", str(self._p.pid)], stdin=subprocess.DEVNULL)
        except Exception:
            pass
        if self._t:
            self._t.join(timeout=5)
        kwh = {k: v / 3.6e12 for k, v in self.mws.items()}        # mW*ms -> J is 1e-6; J -> kWh is /3.6e6
        secs = self.ms_total / 1000.0
        return {"soc_kwh": kwh["combined"], "soc_cpu_kwh": kwh["cpu"], "soc_gpu_kwh": kwh["gpu"],
                "soc_ane_kwh": kwh["ane"], "soc_mean_w": (kwh["combined"] * 3.6e6 / secs) if secs else None,
                "soc_samples": self.samples}


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


# ----------------------------------------------------------------------------- tracked runs
@contextmanager
def tracked(project_name: str):
    """Usage:  with tracked("score_base_mmlu") as run: ...
    run["data"] = CodeCarbon summary, run["soc"] = powermetrics integration."""
    t = _tracker(project_name)
    pm = PowermetricsSampler(config.MEASURE_POWER_SECS * 1000)
    run = {"project": project_name, "t0": time.perf_counter()}
    t.start()
    pm.start()
    strict = IS_MAC and getattr(config, "REQUIRE_POWERMETRICS", True)
    if strict and not pm.active:
        t.stop()
        raise RuntimeError("powermetrics is not delivering power samples (sudo -n powermetrics failed). "
                           "Refusing to run: CodeCarbon's fallback is an 85 W-TDP model, not a measurement. "
                           "Fix sudoers (see PLAN.md §2) and rerun.")
    try:
        yield run
    finally:
        run["soc"] = pm.stop()
        t.stop()
        run["wall_s"] = time.perf_counter() - run["t0"]
        run["data"] = _data_to_dict(t.final_emissions_data)
        run["cc_hardware"] = codecarbon_hardware(t)
        if strict and not (run["soc"].get("soc_samples") or 0):
            raise RuntimeError(f"powermetrics produced no samples during '{project_name}' "
                               f"({run['wall_s']:.0f}s). The run's energy would be CodeCarbon's TDP model. Aborting.")
        try:
            # the gCO2e/kWh CodeCarbon applied for config.COUNTRY, read directly (after stop(),
            # when its Emissions object exists) so it does not depend on which energy terms
            # CodeCarbon happens to include in its own emissions figure
            from codecarbon.core.units import Energy
            run["factor"] = t._emissions.get_private_infra_emissions(Energy.from_energy(kWh=1.0), t._geo) * 1000
        except Exception:
            run["factor"] = None


def idle_baseline(seconds: int, tag: str) -> dict:
    """Track `seconds` of nothing. Returns idle watts (SoC and CodeCarbon) and writes
    energy/idle_<tag>.json."""
    if seconds <= 0:
        return {"idle_w": None, "idle_kwh_per_s": None}
    with tracked(f"idle_{tag}") as run:
        time.sleep(seconds)
    d, soc = run["data"], run["soc"]
    dur = max(d["duration"] or seconds, 1e-9)
    cc_kwh = d["energy_consumed"] or 0.0
    rec = soc["soc_kwh"] if soc["soc_kwh"] is not None else cc_kwh
    out = {"tag": tag, "seconds": seconds,
           "kwh": rec, "idle_w": rec * 3.6e6 / dur, "idle_kwh_per_s": rec / dur,
           "codecarbon_kwh": cc_kwh, "codecarbon_idle_w": cc_kwh * 3.6e6 / dur,
           "cpu_kwh": d["cpu_energy"], "gpu_kwh": d["gpu_energy"], "ram_kwh": d["ram_energy"],
           **soc, "timestamp": datetime.now(timezone.utc).isoformat()}
    (config.ENERGY_DIR / f"idle_{tag}.json").write_text(json.dumps(out, indent=2))
    return out


def attribute(total_kwh: float, seconds) -> np.ndarray:
    """Split a run's energy over its questions by share of wall time."""
    s = np.asarray(seconds, dtype=float)
    if s.sum() <= 0:
        return np.full(len(s), total_kwh / max(len(s), 1))
    return total_kwh * s / s.sum()


# ----------------------------------------------------------------------------- summaries
def codecarbon_factor(d: dict) -> float | None:
    """The gCO2e/kWh CodeCarbon actually applied: its emissions over the energy it counted.
    On Apple Silicon it counts CPU + RAM only (GPU is dropped from emissions), so divide by
    that, not by energy_consumed."""
    em = d.get("emissions") if "emissions" in d else None
    if em is None:
        return None
    base = (d.get("cpu_energy") or 0.0) + (d.get("ram_energy") or 0.0)
    if base <= 0:
        base = d.get("energy_consumed") or 0.0
    return (em * 1000.0 / base) if base > 0 else None


def record_kwh(summary: dict) -> float:
    """Energy of record for a run summary: powermetrics SoC if measured, else CodeCarbon."""
    v = summary.get("soc_kwh")
    return float(v) if v is not None else float(summary.get("total_kwh") or 0.0)


def factor_of(summary: dict) -> float | None:
    """Carbon factor for a summary, old or new format."""
    if summary.get("carbon_factor_g_per_kwh"):
        return summary["carbon_factor_g_per_kwh"]
    # old summaries: gco2e was CodeCarbon's raw figure over CPU+RAM energy
    base = (summary.get("cpu_kwh") or 0.0) + (summary.get("ram_kwh") or 0.0)
    g = summary.get("gco2e")
    return (g / base) if (g is not None and base > 0) else None


def emissions_g(summary: dict) -> float | None:
    """gCO2e for a summary = energy of record x carbon factor (recomputed for old summaries)."""
    f = factor_of(summary)
    return (record_kwh(summary) * f) if f else None


def summarize_run(run: dict, *, role: str, model_id: str, dataset: str, repeat: int, n: int,
                  idle: dict, extra: dict | None = None) -> dict:
    d, soc = run["data"], run.get("soc", {})
    cc_total = d["energy_consumed"] or 0.0
    dur = d["duration"] or run["wall_s"]
    factor = run.get("factor") or codecarbon_factor(d)
    rec = soc.get("soc_kwh") if soc.get("soc_kwh") is not None else cc_total
    idle_kwh = (idle.get("idle_kwh_per_s") or 0.0) * dur
    out = dict(role=role, model_id=model_id, dataset=dataset, repeat=repeat, run_id=d["run_id"],
               n=n, duration_s=dur,
               # energy of record (powermetrics SoC) and its parts
               soc_kwh=soc.get("soc_kwh"), soc_cpu_kwh=soc.get("soc_cpu_kwh"),
               soc_gpu_kwh=soc.get("soc_gpu_kwh"), soc_ane_kwh=soc.get("soc_ane_kwh"),
               soc_mean_w=soc.get("soc_mean_w"), soc_samples=soc.get("soc_samples"),
               # CodeCarbon's figures, kept for comparison
               cpu_kwh=d["cpu_energy"], gpu_kwh=d["gpu_energy"], ram_kwh=d["ram_energy"],
               total_kwh=cc_total, codecarbon_gco2e=(d["emissions"] or 0.0) * 1000,
               cpu_w=d["cpu_power"], gpu_w=d["gpu_power"], ram_w=d["ram_power"],
               # emissions of record
               carbon_factor_g_per_kwh=factor, gco2e=(rec * factor) if factor else None,
               country=d["country_iso_code"],
               mean_w=rec * 3.6e6 / dur if dur else None,
               idle_w=idle.get("idle_w"), idle_kwh_over_run=idle_kwh, net_kwh=rec - idle_kwh,
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
    """Run the tracker on a small GPU load and report what was actually measured."""
    import torch
    dev = config.pick_device()
    run = {}
    try:
        with tracked("check_measurement") as run:
            x = torch.randn(1024, 1024, device=dev)
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < seconds:
                x = (x @ x).clamp(-1, 1)
            if dev == "mps":
                torch.mps.synchronize()
    except RuntimeError as e:
        # strict mode refused: report it instead of crashing the check script
        return dict(device=dev, powermetrics=powermetrics_available(), error=str(e), ok_soc=False,
                    ok_cpu=False, ok_gpu=False, soc_kwh=None, soc_mean_w=None, soc_samples=0,
                    soc_cpu_kwh=None, soc_gpu_kwh=None, cc_cpu_kwh=None, cc_gpu_kwh=None, cc_ram_kwh=None,
                    cc_total_kwh=None, cc_cpu_w=None, cc_gpu_w=None, carbon_factor=None,
                    cc_hardware=run.get("cc_hardware", {}), codecarbon_version=None)
    d, soc = run["data"], run["soc"]
    return dict(device=dev, powermetrics=powermetrics_available(), error=None,
                cc_hardware=run.get("cc_hardware", {}),
                soc_kwh=soc["soc_kwh"], soc_mean_w=soc["soc_mean_w"], soc_samples=soc["soc_samples"],
                soc_cpu_kwh=soc["soc_cpu_kwh"], soc_gpu_kwh=soc["soc_gpu_kwh"],
                cc_cpu_kwh=d["cpu_energy"], cc_gpu_kwh=d["gpu_energy"], cc_ram_kwh=d["ram_energy"],
                cc_total_kwh=d["energy_consumed"], cc_cpu_w=d["cpu_power"], cc_gpu_w=d["gpu_power"],
                carbon_factor=run.get("factor") or codecarbon_factor(d),
                ok_soc=(soc["soc_kwh"] or 0) > 0 and (soc["soc_samples"] or 0) >= seconds // 2,
                ok_cpu=(d["cpu_energy"] or 0) > 0, ok_gpu=(d["gpu_energy"] or 0) > 0,
                codecarbon_version=d["codecarbon_version"])
