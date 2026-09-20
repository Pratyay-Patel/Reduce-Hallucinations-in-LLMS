import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import math
import subprocess
import threading
import torch
import pandas as pd
import numpy as np
import time
import csv
import re
import sys
import gc
import signal
from datetime import datetime
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from codecarbon import EmissionsTracker
from llmlingua import PromptCompressor
from dotenv import load_dotenv
import requests
import json

load_dotenv()
torch.manual_seed(42)

# ── resolve project root (one level above scenarios_evaluation/) ─────────────
_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)

# nvidia_classifier lives in "Nvidia prompt class/" — add to sys.path
_nemo_dir = os.path.join(_project_root, "Nvidia prompt class")
if _nemo_dir not in sys.path:
    sys.path.insert(0, _nemo_dir)

_classification_dir = os.path.join(_project_root, "Classification_model")
if _classification_dir not in sys.path:
    sys.path.insert(0, _classification_dir)

try:
    import nvidia_classifier  # type: ignore
    print("done — loaded nvidia_classifier")
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {_nemo_dir}: {e}")
    nvidia_classifier = None

try:
    import predict_nemo  # type: ignore
    print("done — loaded predict_nemo")
except ImportError as e:
    print(f"Warning: Could not import predict_nemo from {_classification_dir}: {e}")
    predict_nemo = None


# ---------------------------------------------------------------------------
# Carbon / energy tracking helper
# ---------------------------------------------------------------------------

def _empty_carbon_stats():
    """Zero-filled dict used when tracking is disabled or fails entirely."""
    return {
        "emissions_kg_co2":   0.0,
        "energy_consumed_kwh": 0.0,
        "gen_duration_s":     0.0,
        "cpu_power_w":        0.0,
        "gpu_power_w":        0.0,
        "ram_power_w":        0.0,
        "cpu_energy_kwh":     0.0,
        "gpu_energy_kwh":     0.0,
        "ram_energy_kwh":     0.0,
        "tracking_method":    "none",
    }


def _die_carbon(msg):
    print(f"[Carbon] ERROR: {msg}")
    sys.exit(1)


def _require_finite(value, name, allow_zero=True):
    """Reject missing/NaN measurements. Never substitute a fallback number."""
    if value is None:
        _die_carbon(f"{name} is missing. CodeCarbon is not tracking.")
    try:
        x = float(value)
    except (TypeError, ValueError):
        _die_carbon(f"{name} is not numeric ({value!r}). CodeCarbon is not tracking.")
    if math.isnan(x) or math.isinf(x):
        _die_carbon(f"{name} is {x}. CodeCarbon is not tracking.")
    if not allow_zero and x <= 0:
        _die_carbon(f"{name} is {x}. CodeCarbon did not record a real measurement.")
    return round(x, 10)


def _kwh_attr(obj):
    if obj is None:
        return None
    return getattr(obj, "kWh", obj)


def _field(ed, tracker, ed_key, tracker_attr=None):
    if ed is not None:
        v = getattr(ed, ed_key, None)
        if v is not None:
            return v
    if tracker_attr:
        return _kwh_attr(getattr(tracker, tracker_attr, None))
    return None


def _extract_tracker_stats(tracker, emissions_kg=None):
    """Pull CO2 / energy / power fields. Abort if any required value is invalid."""
    ed = getattr(tracker, "final_emissions_data", None)
    if ed is None:
        _die_carbon("CodeCarbon did not produce final_emissions_data.")

    stats = _empty_carbon_stats()
    stats["tracking_method"] = "codecarbon_batch"
    stats["cpu_energy_kwh"] = _require_finite(
        _field(ed, tracker, "cpu_energy", "_total_cpu_energy"), "cpu_energy", allow_zero=False
    )
    stats["gpu_energy_kwh"] = _require_finite(
        _field(ed, tracker, "gpu_energy", "_total_gpu_energy"), "gpu_energy",
        allow_zero=True,
    )
    stats["ram_energy_kwh"] = _require_finite(
        _field(ed, tracker, "ram_energy", "_total_ram_energy"), "ram_energy", allow_zero=False
    )
    stats["energy_consumed_kwh"] = _require_finite(
        _field(ed, tracker, "energy_consumed", "_total_energy"),
        "energy_consumed",
        allow_zero=False,
    )
    stats["cpu_power_w"] = _require_finite(_field(ed, tracker, "cpu_power"), "cpu_power", allow_zero=False)
    stats["gpu_power_w"] = _require_finite(
        _field(ed, tracker, "gpu_power"), "gpu_power", allow_zero=True
    )
    stats["ram_power_w"] = _require_finite(_field(ed, tracker, "ram_power"), "ram_power", allow_zero=False)
    co2 = emissions_kg if emissions_kg is not None else getattr(ed, "emissions", None)
    stats["emissions_kg_co2"] = _require_finite(co2, "emissions_kg_co2", allow_zero=False)
    return stats


# macOS sudo timestamps expire after ~5 minutes. CodeCarbon's background
# `sudo powermetrics` does not reliably refresh that ticket, so GPU energy
# becomes NaN on runs longer than the timeout.
_sudo_keepalive_stop = None
_sudo_keepalive_thread = None
_sudo_keepalive_failed = False
_SUDO_REFRESH_SECS = 4 * 60  # under macOS ~5 min sudo timeout; not per prompt


def _sudo_refresh():
    """Extend the sudo timestamp without prompting. Returns True on success."""
    try:
        return subprocess.run(["sudo", "-n", "-v"], capture_output=True).returncode == 0
    except FileNotFoundError:
        return False


def _start_sudo_keepalive():
    """Refresh sudo every 4 minutes so PowerMetrics does not die mid-run."""
    global _sudo_keepalive_stop, _sudo_keepalive_thread, _sudo_keepalive_failed
    if sys.platform != "darwin":
        return
    _sudo_keepalive_failed = False
    _sudo_keepalive_stop = threading.Event()

    def _loop():
        global _sudo_keepalive_failed
        while not _sudo_keepalive_stop.wait(_SUDO_REFRESH_SECS):
            if _sudo_refresh():
                print("[Carbon] sudo timestamp refreshed (every 4 min).")
            else:
                _sudo_keepalive_failed = True
                print(
                    "[Carbon] ERROR: sudo timestamp expired (macOS default ~5 min). "
                    "PowerMetrics will return empty GPU/CPU samples."
                )
                return

    _sudo_keepalive_thread = threading.Thread(
        target=_loop, daemon=True, name="sudo-keepalive"
    )
    _sudo_keepalive_thread.start()
    print("[Carbon] sudo keepalive started (refresh every 4 min).")


def _stop_sudo_keepalive():
    global _sudo_keepalive_stop, _sudo_keepalive_thread
    if _sudo_keepalive_stop is not None:
        _sudo_keepalive_stop.set()
    _sudo_keepalive_thread = None
    _sudo_keepalive_stop = None


_POWERMETRICS_PATCHED = False


def _finite_or_none(value):
    v = _kwh_attr(value)
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _patch_apple_powermetrics():
    """
    CodeCarbon uses np.mean([]) when a PowerMetrics sample has no GPU Power
    line. That NaN is added into the running GPU/total energy and cannot be
    recovered. Parse floats ourselves and never return NaN.
    """
    global _POWERMETRICS_PATCHED
    if _POWERMETRICS_PATCHED or sys.platform != "darwin":
        return
    from codecarbon.core.powermetrics import ApplePowermetrics

    def _log_values(self):
        cmd = [
            "sudo", "-n",
            "powermetrics",
            "-n", str(self._n_points),
            "--samplers", "cpu_power",
            "-i", str(self._interval),
            "-o", self._log_file_path,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"powermetrics failed (exit {rc})")

    def _mean_watts(logfile, label):
        vals = [float(x) / 1000.0 for x in re.findall(rf"{label}: ([\d.]+) mW", logfile)]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def get_details(self, delay=None):
        last_err = None
        for _ in range(3):
            try:
                self._log_values()
                with open(self._log_file_path) as f:
                    logfile = f.read()
            except Exception as e:
                last_err = e
                time.sleep(0.15)
                continue
            cpu = _mean_watts(logfile, "CPU Power")
            gpu = _mean_watts(logfile, "GPU Power")
            if cpu is None:
                last_err = RuntimeError("powermetrics log had no CPU Power lines")
                time.sleep(0.15)
                continue
            if gpu is None:
                gpu = 0.0
            interval_s = float(self._interval) / 1000.0
            n_cpu = max(len(re.findall(r"CPU Power: ([\d.]+) mW", logfile)), 1)
            n_gpu = max(len(re.findall(r"GPU Power: ([\d.]+) mW", logfile)), 1)
            return {
                "CPU Power": cpu,
                "GPU Power": gpu,
                "CPU Energy Delta": interval_s * cpu * n_cpu,
                "GPU Energy Delta": interval_s * gpu * n_gpu,
            }
        raise RuntimeError(f"powermetrics parse failed: {last_err}")

    ApplePowermetrics._log_values = _log_values
    ApplePowermetrics.get_details = get_details
    _POWERMETRICS_PATCHED = True
    print("[Carbon] Patched PowerMetrics parser so empty GPU samples are 0 W, not NaN.")


def _guard_tracker_against_nan(tracker):
    """If one background sample is still NaN, keep prior GPU/total energy."""
    from codecarbon.core.units import Energy

    orig = tracker._do_measurements

    def wrapped():
        gpu_before = _finite_or_none(getattr(tracker, "_total_gpu_energy", None))
        energy_before = _finite_or_none(getattr(tracker, "_total_energy", None))
        gpu_sum_before = getattr(tracker, "_gpu_power_sum", 0.0)
        orig()
        gpu_after = _finite_or_none(getattr(tracker, "_total_gpu_energy", None))
        if gpu_after is None:
            tracker._total_gpu_energy = Energy.from_energy(kWh=gpu_before or 0.0)
            tracker._gpu_power_sum = gpu_sum_before
            gpu_after = gpu_before or 0.0
        cpu = _finite_or_none(getattr(tracker, "_total_cpu_energy", None)) or 0.0
        ram = _finite_or_none(getattr(tracker, "_total_ram_energy", None)) or 0.0
        total = _finite_or_none(getattr(tracker, "_total_energy", None))
        if total is None:
            tracker._total_energy = Energy.from_energy(
                kWh=cpu + gpu_after + ram if energy_before is None else max(energy_before, cpu + gpu_after + ram)
            )

    tracker._do_measurements = wrapped


def _enable_macos_powermetrics():
    """CodeCarbon never prompts for sudo; without it it silently uses TDP. Refuse that."""
    if sys.platform != "darwin":
        return
    print("[Carbon] Apple PowerMetrics requires sudo. CodeCarbon will not prompt on its own.")
    if not sys.stdin.isatty():
        _die_carbon(
            "No interactive terminal for sudo. Run this script in a terminal and enter "
            "your password so PowerMetrics can measure CPU/GPU. Refusing TDP fallback."
        )
    try:
        rc = subprocess.run(["sudo", "-v"]).returncode
    except FileNotFoundError:
        _die_carbon("sudo not found. Cannot enable PowerMetrics.")
    if rc != 0:
        _die_carbon("sudo was not granted. Refusing to run with CodeCarbon TDP fallback.")
    from codecarbon.core import powermetrics
    if not powermetrics.is_powermetrics_available():
        _die_carbon(
            "PowerMetrics still unavailable after sudo. "
            "CodeCarbon would fall back to TDP estimates. Stopping."
        )
    print("[Carbon] PowerMetrics sudo ok.")
    _start_sudo_keepalive()


def _assert_real_hardware_tracking(tracker):
    """Abort if CodeCarbon selected TDP / CPU-load estimates instead of meters."""
    from codecarbon.external.hardware import AppleSiliconChip, CPU, MODE_CPU_LOAD

    hardware = list(getattr(tracker, "_hardware", None) or [])
    summary = []
    for h in hardware:
        mode = getattr(h, "_mode", None)
        part = getattr(h, "chip_part", None)
        extra = mode or part or ""
        summary.append(f"{type(h).__name__}:{extra}" if extra else type(h).__name__)
    print(f"[Carbon] Hardware backends: {summary}")

    fallback_cpus = [
        h for h in hardware
        if isinstance(h, CPU) and getattr(h, "_mode", None) in {MODE_CPU_LOAD, "constant"}
    ]
    if fallback_cpus:
        modes = [h._mode for h in fallback_cpus]
        _die_carbon(
            f"CodeCarbon CPU backend is a fallback ({modes}), not RAPL/PowerMetrics. Stopping."
        )

    if sys.platform == "darwin":
        parts = {h.chip_part for h in hardware if isinstance(h, AppleSiliconChip)}
        if "CPU" not in parts or "GPU" not in parts:
            _die_carbon(
                f"Need Apple PowerMetrics CPU+GPU chips, got {parts or 'none'}. Stopping."
            )


def _assert_first_samples_finite(tracker):
    """Catch NaN GPU/CPU energy before the experiment loop starts."""
    try:
        tracker._measure_power_and_energy()
    except Exception as e:
        _die_carbon(f"CodeCarbon first measurement failed ({e}). Stopping.")
    for name, attr in (
        ("cpu_energy", "_total_cpu_energy"),
        ("gpu_energy", "_total_gpu_energy"),
        ("energy_consumed", "_total_energy"),
    ):
        _require_finite(_kwh_attr(getattr(tracker, attr, None)), f"first sample {name}")


def start_batch_tracker():
    """Start a single CodeCarbon tracker. Exit if real metering is not active."""
    _enable_macos_powermetrics()
    _patch_apple_powermetrics()
    try:
        tracker = EmissionsTracker(
            project_name="ecoprompt_batch",
            measure_power_secs=1,
            save_to_file=False,
            log_level="warning",
            allow_multiple_runs=True,
        )
    except Exception as e:
        _die_carbon(f"Tracker init failed ({e}).")
    _assert_real_hardware_tracking(tracker)
    try:
        tracker.start()
    except Exception as e:
        _die_carbon(f"Tracker start failed ({e}).")
    _guard_tracker_against_nan(tracker)
    time.sleep(1.2)
    _assert_first_samples_finite(tracker)
    print("[Carbon] Batch tracker started with real PowerMetrics/RAPL metering.")
    return tracker


def stop_batch_tracker(tracker):
    """Stop the run-level tracker. Exit if CO2/energy were not actually measured."""
    _stop_sudo_keepalive()
    if tracker is None:
        _die_carbon("CodeCarbon tracker was never started.")
    if _sudo_keepalive_failed:
        _die_carbon(
            "sudo expired before CodeCarbon stopped, so GPU/CPU energy became NaN. "
            "Re-run in this terminal; sudo is refreshed every 4 min."
        )
    try:
        emissions_kg = tracker.stop()
    except Exception as e:
        _die_carbon(f"Tracker stop failed ({e}).")
    stats = _extract_tracker_stats(tracker, emissions_kg)
    print(
        f"[Carbon] Batch tracker stopped | "
        f"CO2={stats['emissions_kg_co2']:.6f} kg | "
        f"Energy={stats['energy_consumed_kwh']:.6f} kWh"
    )
    return stats


def track_generation(generate_fn, no_tracking=False):
    """
    Time a single generation call. CodeCarbon is owned by the batch tracker
    (start/stop once per run), not per prompt.

    Returns
    -------
    output : str
    stats  : dict  — gen_duration_s filled now; CO2/energy filled after batch stop
    """
    t0 = time.time()
    try:
        output = generate_fn()
    except Exception as e:
        print(f"[Carbon] Generation error: {e}")
        output = f"Error: {e}"
    stats = _empty_carbon_stats()
    stats["gen_duration_s"] = round(time.time() - t0, 4)
    stats["tracking_method"] = "disabled" if no_tracking else "codecarbon_batch"
    return output, stats


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
# Both Llama 3.2 1B-Instruct and Phi-3 Mini support the standard
# system/user chat format via apply_chat_template, so we produce a single
# (user_content, system_content) pair for every task — no model-specific
# markup needed here at all.
#
# Returns: (user_content, system_content, reference)
# ModelManager.generate() feeds these into apply_chat_template.
# ---------------------------------------------------------------------------

def build_prompt(item, ds_name, subset):
    """
    Returns (user_content: str, system_content: str, reference).
    Plain text only — no [INST]/<<SYS>> markup.
    apply_chat_template in ModelManager handles model-specific wrapping.
    """

    # ── MNLI ──────────────────────────────────────────────────────────────
    if ds_name == "glue" and subset == "mnli":
        ref = item["label"]
        system = (
            "You are a Natural Language Inference classifier. "
            "Your reply must be exactly one word: entailment, contradiction, or neutral. "
            "No explanation, no punctuation, no extra text."
        )
        user = (
            f"Premise: {item['premise']}\n"
            f"Hypothesis: {item['hypothesis']}\n"
            "Label:"
        )
        return user, system, ref

    # ── SST-2 ─────────────────────────────────────────────────────────────
    elif ds_name == "glue" and subset == "sst2":
        ref = item["label"]
        system = (
            "You are a sentiment classifier. "
            "Reply with exactly one word: positive or negative. Nothing else."
        )
        user = (
            f"Sentence: {item['sentence']}\n"
            "Sentiment:"
        )
        return user, system, ref

    # ── SQuAD v2 ──────────────────────────────────────────────────────────
    elif ds_name == "squad_v2":
        ref = item["answers"]
        system = (
            "You are a reading-comprehension assistant. "
            "Answer using only words found in the context. "
            'If the answer is not in the context, reply with exactly: unanswerable\n'
            "Give only the answer — no explanation, no full sentence."
        )
        user = (
            f"Context: {item['context']}\n"
            f"Question: {item['question']}\n"
            "Answer:"
        )
        return user, system, ref

    # ── CNN / DailyMail ───────────────────────────────────────────────────
    elif ds_name == "cnn_dailymail":
        ref     = item["highlights"]
        article = item["article"][:2000]
        system  = (
            "You are a news summariser. "
            "Write a concise 2-3 sentence summary covering the key facts. "
            "No bullet points."
        )
        user = (
            f"Article:\n{article}\n\n"
            "Summary:"
        )
        return user, system, ref

    # ── GSM8K ─────────────────────────────────────────────────────────────
    elif ds_name == "gsm8k":
        ref    = item["answer"]
        system = (
            "You are a math solver. Think step by step, then write your final "
            "numeric answer on the last line in this exact format: #### <number>"
        )
        user = (
            f"Question: {item['question']}\n"
            "Solution:"
        )
        return user, system, ref

    # ── BoolQ ─────────────────────────────────────────────────────────────
    elif ds_name == "boolq":
        ref    = item["answer"]
        system = (
            "You are a Boolean question answering assistant. "
            "Read the passage and answer the question using exactly one word: yes or no. "
            "Do not explain your answer."
        )
        user = (
            f"Passage: {item['passage']}\n\n"
            f"Question: {item['question']}\n\n"
            "Answer:"
        )
        return user, system, ref

    # ── AI2 ARC ───────────────────────────────────────────────────────────
    elif ds_name == "allenai/ai2_arc" and subset in ("ARC-Easy", "ARC-Challenge"):
        ref = item["answerKey"]
        system = (
            "You are a science question answering assistant. "
            "Respond with only the option identifier exactly as it appears in the choices. "
            "The identifier may be A, B, C, D or 1, 2, 3, 4. "
            "Do not explain your answer."
        )
        labels = item["choices"]["label"]
        texts  = item["choices"]["text"]
        choice_lines = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
        user = (
            f"Question: {item['question']}\n\n"
            f"Choices:\n{choice_lines}\n\n"
            "Answer:"
        )
        return user, system, ref

    # ── PIQA ──────────────────────────────────────────────────────────────
    elif ds_name == "piqa":
        ref    = item["label"]
        system = (
            "You are a physical commonsense reasoning assistant. "
            "Choose the solution that better achieves the goal. "
            "Reply with only the solution number, 1 or 2. "
            "Do not explain your answer."
        )
        user = (
            f"Goal: {item['goal']}\n\n"
            f"Solution 1: {item['sol1']}\n\n"
            f"Solution 2: {item['sol2']}\n\n"
            "Answer:"
        )
        return user, system, ref

    # ── AG News ───────────────────────────────────────────────────────────
    elif ds_name == "ag_news":
        ref    = item["label"]
        system = (
            "You are a news topic classifier. "
            "Your reply must be exactly one of: World, Sports, Business, Sci/Tech. "
            "No explanation, no punctuation, no extra text."
        )
        user = (
            f"Article: {item['text']}\n\n"
            "Category:"
        )
        return user, system, ref

    # ── fallback ──────────────────────────────────────────────────────────
    return str(item), "", ""


# ---------------------------------------------------------------------------
# ModelManager
# Tier 1 = Llama 3.2 1B Instruct  (small / fast / low-energy)
# Tier 3 = Phi-3 Mini 4k Instruct  (larger / higher accuracy)
#
# Both models use apply_chat_template — no model-specific prompt markup.
# ---------------------------------------------------------------------------
class ModelManager:

    def __init__(self):
        self.models     = {"tier1": None, "tier3": None, "nemo": None}
        self.tokenizers = {"tier1": None, "tier3": None, "nemo": None}

    def get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_tier1(self):
        if self.models["tier1"] is not None:
            return self.models["tier1"], self.tokenizers["tier1"]
        print("Loading Tier 1 (Llama 3.2 1B Instruct)...")
        model_id = "meta-llama/Llama-3.2-1B-Instruct"
        # Llama 3.2 is a gated model — HF_TOKEN must be set in .env
        return self._load_generic_model("tier1", model_id, use_auth=True)

    def load_tier3(self):
        if self.models["tier3"] is not None:
            return self.models["tier3"], self.tokenizers["tier3"]
        print("Loading Tier 3 (Phi-3 Mini 4k Instruct)...")
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        return self._load_generic_model("tier3", model_id, use_auth=False)

    def _load_generic_model(self, tier_key, model_id, use_auth=False):
        token = os.getenv("HF_TOKEN") if use_auth else None
        if use_auth and not token:
            print(
                "Warning: HF_TOKEN not found. "
                "Llama 3.2 is a gated model and requires a Hugging Face token."
            )
        try:
            print(f"Checking for local weights for {model_id}...")
            device = self.get_device()

            if device == "mps":
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, token=token, local_files_only=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, token=token,
                        local_files_only=True, low_cpu_mem_usage=True,
                    )
                    model = model.to(device)
                    print(f"Loaded {tier_key} from local cache to MPS.")
                except OSError:
                    print(f"Local weights not found for {tier_key}, downloading...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, token=token,
                        low_cpu_mem_usage=True,
                    )
                    model = model.to(device)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id, token=token, local_files_only=True
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map="cuda",
                        token=token
                    )
                    print(f"Loaded {tier_key} from local cache.")
                except OSError:
                    print(f"Local weights not found for {tier_key}, downloading...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map="auto",
                        token=token
                    )

            # Llama tokenizers sometimes have no pad token — default to eos
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Disable KV-cache for both models (consistent with original Phi-3 setting)
            model.generation_config.use_cache = False

            self.models[tier_key]     = model
            self.tokenizers[tier_key] = tokenizer
            return model, tokenizer

        except Exception as e:
            print(f"Error loading {tier_key}: {e}. Terminating script.")
            sys.exit(1)

    def get_nemo_model(self):
        if nvidia_classifier is None:
            return None, None
        if self.models["nemo"] is None:
            print("Loading NeMo Curator model...")
            try:
                self.models["nemo"], self.tokenizers["nemo"] = nvidia_classifier.load_model()
            except Exception as e:
                print(f"Error loading NeMo model: {e}. Terminating script.")
                sys.exit(1)
        return self.models["nemo"], self.tokenizers["nemo"]

    def unload_model(self, tier):
        if self.models.get(tier) is not None:
            print(f"Unloading {tier}...")
            del self.models[tier]
            del self.tokenizers[tier]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
            self.models[tier]     = None
            self.tokenizers[tier] = None

    def generate(self, tier, user_content, system_content=""):
        """
        Unified generation for both Llama 3.2 1B and Phi-3 Mini.
        Both models support apply_chat_template with system + user roles.

        Parameters
        ----------
        tier           : "tier1" (Llama 3.2 1B) | "tier3" (Phi-3 Mini)
        user_content   : plain-text user turn
        system_content : plain-text system instruction (may be empty)
        """
        print(f"[DEBUG] Generating ({tier})... User prompt len: {len(user_content)}")

        methods = {"tier1": self.load_tier1, "tier3": self.load_tier3}
        if tier not in methods:
            return "Error: Invalid logic tier"

        model_weights = methods[tier]()
        if not model_weights or model_weights[0] is None:
            return "Error: Model loading failed"

        model, tokenizer = model_weights

        # Build message list — include system turn only when non-empty
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": user_content})

        try:
            # ── decode=False path: get token ids ──────────────────────────────
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            # ── TEMPLATE VERIFICATION LOG ─────────────────────────────────────
            # Decode the same messages to a string (tokenize=False) so we can
            # print the exact template markup that was applied, then count tokens.
            templated_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            n_input_tokens = inputs["input_ids"].shape[-1]
            model_label = "Llama 3.2 1B" if tier == "tier1" else "Phi-3 Mini"
            print(
                f"\n[TEMPLATE] ── {model_label} ({tier}) ──────────────────────────\n"
                f"  Templated string (repr):\n"
                f"  {repr(templated_str)}\n"
                f"  Total input tokens : {n_input_tokens}\n"
                f"  User content chars : {len(user_content)}\n"
                f"────────────────────────────────────────────────────────────────\n"
            )
            # ─────────────────────────────────────────────────────────────────

        except Exception as e:
            # Graceful fallback: concatenate and plain-tokenise
            print(f"[WARN] apply_chat_template failed ({e}), falling back to plain tokenisation.")
            fallback = (
                f"{system_content}\n\n{user_content}" if system_content else user_content
            )
            inputs = tokenizer(fallback, return_tensors="pt").to(model.device)
            print(f"[TEMPLATE] Fallback plain tokenisation — input tokens: {inputs['input_ids'].shape[-1]}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,       # greedy decoding — deterministic outputs
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return generated_text.strip()


# ---------------------------------------------------------------------------
# IntelligenceEngine
# ---------------------------------------------------------------------------
class IntelligenceEngine:

    def __init__(self, model_manager):
        self.mm = model_manager
        self.compressor = None

    def get_compressor(self):
        if self.compressor is None:
            print("Initializing LLM Lingua-2...")
            try:
                self.compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                    device_map="cpu",
                )
            except Exception as e:
                print(f"Error loading compressor (LLM Lingua): {e}. Terminating script.")
                sys.exit(1)
        return self.compressor

    def classify_complexity(self, prompt):
        if predict_nemo is None:
             print("[WARN] predict_nemo not available. Default to Label0.")
             return "Label0", {}
        try:
            if not hasattr(self, 'nemo_extractor'):
                self.nemo_extractor = predict_nemo.NemoFeatureExtractor()
            
            features = self.nemo_extractor.get_features(prompt)

            if features:
                nemo_raw_json = features["nemo_raw_json"]
                num_tokens = features["num_tokens"]

                # Now use predict_label
                scaler_path = os.path.join(_classification_dir, 'advanced_scaler.pkl')
                model_path = os.path.join(_classification_dir, 'best_advanced_model.pkl')

                label_int = predict_nemo.predict_label(
                    nemo_raw_output_json=nemo_raw_json, 
                    compressed_prompt_len=num_tokens,
                    scaler_path=scaler_path,
                    model_path=model_path
                )
                
                category = f"Label{label_int}"
                print(f"[DEBUG] predict_nemo Model Output: {label_int} -> {category}")
                return category, nemo_raw_json
            else:
                return "Label0", {}
        except Exception as e:
            print(f"Classifier error: {e}")
            return "Label0", {}

    def route_prompt(self, category):
        """Label1 -> tier1 (Llama 3.1), Label0 -> tier3 (Phi-3 Mini)"""
        return "tier1" if category == "Label1" else "tier3"

    def compress_prompt(self, prompt):
        compressor = self.get_compressor()
        if not compressor:
            return {"text": prompt, "rate": 1.0, "init": 0, "final": 0}
        try:
            token_count = len(compressor.tokenizer.encode(prompt))
            slope = 9.5 / 8000
            ratio = 2.5 + (token_count - 2000) * slope
            if ratio < 1.0:
                ratio = 1.0
            if 1.0 < ratio < 2.0:
                ratio = 2.0
            if ratio == 1.0:
                return {"text": prompt, "rate": 1.0, "init": len(prompt.split()), "final": len(prompt.split())}
            rate = 1 / ratio
            res = compressor.compress_prompt(prompt, rate=rate, force_tokens=['\n', '?'])
            compressed_text = res['compressed_prompt'] if isinstance(res, dict) else res
            return {"text": compressed_text, "rate": rate, "init": len(prompt.split()), "final": len(compressed_text.split())}
        except Exception as e:
            print(f"Compression failed: {e}")
            return {"text": prompt, "rate": 1.0, "init": 0, "final": 0}


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------
class DatasetLoader:

    def __init__(self, data_root):
        self.data_root = data_root

    def get_local_path(self, ds_name, subset, split):
        paths = {
            ("glue",         "mnli"):  (os.path.join(self.data_root, "data", "NLI_MNLI"),     "validation_matched"),
            ("glue",         "sst2"):  (os.path.join(self.data_root, "data", "SST-2"),         "validation"),
            ("squad_v2",     None):    (os.path.join(self.data_root, "data", "SQuAD_v2"),      "validation"),
            ("cnn_dailymail","3.0.0"): (os.path.join(self.data_root, "data", "CNN_DailyMail"), "test"),
            ("gsm8k",        "main"):  (os.path.join(self.data_root, "data", "GSM8K"),         "train"),
            ("boolq",        None):    (os.path.join(self.data_root, "data", "BoolQ"),         "validation"),
            ("allenai/ai2_arc", "ARC-Easy"):      (os.path.join(self.data_root, "data", "ARC_Easy"),      "validation"),
            ("allenai/ai2_arc", "ARC-Challenge"): (os.path.join(self.data_root, "data", "ARC_Challenge"), "validation"),
            ("piqa",         None):    (os.path.join(self.data_root, "data", "PIQA"),          "validation"),
            ("ag_news",      None):    (os.path.join(self.data_root, "data", "AG_News"),       "test"),
        }
        return paths.get((ds_name, subset))

    def load(self, ds_name, subset, split, samples, start_index=0):
        path_info = self.get_local_path(ds_name, subset, split)
        if not path_info:
            print(f"Unknown local path for {ds_name}/{subset}")
            return []
        dataset_dir, split_name = path_info
        try:
            ds_dict = load_from_disk(dataset_dir)
            if split_name not in ds_dict:
                print(f"Split '{split_name}' not found. Available: {list(ds_dict.keys())}")
                return []
            ds = ds_dict[split_name]
            total = len(ds)
            start = min(start_index, total)
            end = min(total, start + samples)
            ds = ds.select(range(start, end))
            print(f"[INFO] {split_name}: {total} total, using {end - start} samples from index {start}")
            return ds
        except Exception as e:
            print(f"Dataset load error ({dataset_dir}, split={split_name}): {e}")
            return []


# ---------------------------------------------------------------------------
# Evaluator — robust extraction handles chatty model outputs
# ---------------------------------------------------------------------------
class Evaluator:

    @staticmethod
    def evaluate(output, reference, ds_name, subset):
        if ds_name == "glue":
            if subset == "mnli":
                return Evaluator.mnli(output, reference), "accuracy"
            if subset == "sst2":
                return Evaluator.sst2(output, reference), "accuracy"
        elif ds_name == "squad_v2":
            return Evaluator.squad(output, reference), "EM"
        elif ds_name == "cnn_dailymail":
            return Evaluator.rouge(output, reference), "ROUGE-L"
        elif ds_name == "gsm8k":
            return Evaluator.gsm8k(output, reference), "EM"
        elif ds_name == "boolq":
            return Evaluator.boolq(output, reference), "accuracy"
        elif ds_name == "allenai/ai2_arc":
            return Evaluator.arc(output, reference), "accuracy"
        elif ds_name == "piqa":
            return Evaluator.piqa(output, reference), "accuracy"
        elif ds_name == "ag_news":
            return Evaluator.ag_news(output, reference), "accuracy"
        return 0.0, "unknown"

    @staticmethod
    def mnli(pred, label):
        """Find the first valid label word anywhere in the output."""
        pred = pred.lower()
        map_ = {0: "entailment", 1: "neutral", 2: "contradiction"}
        lbl  = map_.get(label, "") if isinstance(label, int) else str(label).lower()

        positions = {
            "entailment":    pred.find("entailment"),
            "neutral":       pred.find("neutral"),
            "contradiction": pred.find("contradiction"),
        }
        found = {k: v for k, v in positions.items() if v != -1}
        if not found:
            return 0
        p = min(found, key=found.get)
        return 1 if p == lbl else 0

    @staticmethod
    def sst2(pred, label):
        """First mention of positive/negative wins."""
        pred = pred.lower()
        map_ = {0: "negative", 1: "positive"}
        lbl  = map_.get(label, "") if isinstance(label, int) else str(label).lower()

        pos = pred.find("positive")
        neg = pred.find("negative")
        if pos == -1 and neg == -1:
            return 0
        if pos == -1:
            p = "negative"
        elif neg == -1:
            p = "positive"
        else:
            p = "positive" if pos < neg else "negative"
        return 1 if p == lbl else 0

    @staticmethod
    def squad(pred, answers):
        """Strip preamble, take first line/sentence, exact-match."""
        candidates = answers.get('text', [])
        if not candidates:
            return 1 if "unanswerable" in pred.lower() else 0

        def normalize(s):
            import string
            s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            return ' '.join(s.split())

        clean = re.sub(r'^(answer\s*[:\-]\s*)', '', pred.strip(), flags=re.IGNORECASE)
        clean = re.split(r'[\n\.]', clean)[0].strip()
        pred_norm = normalize(clean)
        return max([1 if normalize(a) == pred_norm else 0 for a in candidates])

    @staticmethod
    def rouge(pred, ref):
        if not pred:
            return 0.0
        try:
            scorer = evaluate.load("rouge")
            res = scorer.compute(predictions=[pred], references=[ref])
            return res.get('rougeL', 0.0)
        except:
            return 0.0

    @staticmethod
    def gsm8k(pred, ref_str):
        """Accept #### format or fall back to last number in output."""
        gold_str = ref_str.split("####")[-1].strip()
        try:
            gold = float(gold_str.replace(",", ""))
        except ValueError:
            return 0

        if "####" in pred:
            after = pred.split("####")[-1].strip()
            nums  = re.findall(r'-?[\d,]+\.?\d*', after)
        else:
            nums = re.findall(r'-?[\d,]+\.?\d*', pred)

        if not nums:
            return 0
        try:
            pred_val = float(nums[-1].replace(",", ""))
            return 1 if abs(pred_val - gold) < 1e-5 else 0
        except ValueError:
            return 0

    @staticmethod
    def boolq(pred, label):
        """First whole-word mention of yes/no wins (word-bounded so 'not'/'know' don't count as 'no')."""
        pred = pred.lower()
        if isinstance(label, str):
            lbl = "yes" if label.strip().lower() in ("yes", "true", "1") else "no"
        else:
            lbl = "yes" if label else "no"

        yes_m = re.search(r"\byes\b", pred)
        no_m  = re.search(r"\bno\b", pred)
        if not yes_m and not no_m:
            return 0
        if not yes_m:
            p = "no"
        elif not no_m:
            p = "yes"
        else:
            p = "yes" if yes_m.start() < no_m.start() else "no"
        return 1 if p == lbl else 0

    @staticmethod
    def arc(pred, answer_key):
        """Exact multiple-choice accuracy over A-D / 1-4 option identifiers."""
        gold = str(answer_key).strip().upper()
        pred = str(pred).upper()
        match = re.search(r"\b([A-D]|[1-4])\b", pred)
        if not match:
            return 0
        return 1 if match.group(1) == gold else 0

    @staticmethod
    def piqa(pred, label):
        """Exact two-way accuracy over solution identifiers 1 / 2."""
        gold = str(label + 1) if isinstance(label, int) else str(label).strip()
        match = re.search(r"\b([12])\b", str(pred))
        if not match:
            return 0
        return 1 if match.group(1) == gold else 0

    @staticmethod
    def ag_news(pred, label):
        """Find the first valid category word anywhere in the output."""
        pred = pred.lower()
        map_ = {0: "world", 1: "sports", 2: "business", 3: "sci/tech"}
        lbl  = map_.get(label, "") if isinstance(label, int) else str(label).lower()

        # 'tech' occurs inside other words, so these are regexes not str.find
        patterns = {
            "world":    r"\bworld\b",
            "sports":   r"\bsports?\b",
            "business": r"\bbusiness\b",
            "sci/tech": r"\b(?:sci/tech|sci-tech|tech\b|technolog\w*|scien\w*|computer\w*)",
        }
        positions = {}
        for name, pat in patterns.items():
            m = re.search(pat, pred)
            if m:
                positions[name] = m.start()
        if not positions:
            return 0
        p = min(positions, key=positions.get)
        return 1 if p == lbl else 0


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------
class ExperimentRunner:

    def __init__(self, args, data_root):
        self.args = args
        self.mm   = ModelManager()
        self.ie   = IntelligenceEngine(self.mm)
        self.results         = []
        self.csv_initialized = False
        self.interrupted     = False
        self.batch_tracker   = None
        self.batch_totals    = _empty_carbon_stats()
        self._carbon_stopped = False

        # S1  Upper Bound      – always tier3 (Phi-3 Mini),       no compression
        # S2  Lower Bound      – always tier1 (Llama 3.2 1B),     no compression
        # S3  Compression      – always tier3 (Phi-3 Mini),       with compression
        # S4  Routing Only     – Label0->tier1, Label1->tier3,    no compression
        # S5  EcoPrompt        – Label0->tier1, Label1->tier3,    with compression
        self.scenarios = [
            {"id": "S1", "name": "Upper Bound (Phi-3 Mini)",          "routing": False, "compression": False, "fixed": "tier3"},
            {"id": "S2", "name": "Lower Bound (Llama 3.2 1B)",        "routing": False, "compression": False, "fixed": "tier1"},
            {"id": "S3", "name": "Compression (Phi-3 Mini + Comp)",   "routing": False, "compression": True,  "fixed": "tier3"},
            {"id": "S4", "name": "Routing Only",                       "routing": True,  "compression": False, "fixed": None},
            {"id": "S5", "name": "EcoPrompt (Routing + Comp)",         "routing": True,  "compression": True,  "fixed": None},
        ]

        self.data_root = data_root
        signal.signal(signal.SIGINT,  self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n\n⚠️  Process interrupted! Saving results before exit...")
        self.interrupted = True
        self._finalize_batch_carbon()
        self._save_results()
        print("✅ Results saved. Exiting gracefully.")
        sys.exit(0)

    def _preload_models(self, active_scenarios):
        """
        Load (and warm up) models before CodeCarbon starts so the first
        real prompts are not charged for download/weight-load time.
        """
        print("[INFO] Preloading models before carbon tracking...")
        # classify_complexity always runs, even for non-routing scenarios
        self.mm.get_nemo_model()

        # Match real SST-2 / ARC chat length. A 2-token "ok" warmup does not
        # compile MPS/CUDA kernels for the ~70–90 token templates used later,
        # so the first real prompts still pay first-shape compile time.
        warmup_system = (
            "You are a sentiment classifier. "
            "Reply with exactly one word: positive or negative. Nothing else."
        )
        warmup_user = (
            "Sentence: a thoughtful character study whose performances and "
            "pacing hold together even when the plot turns familiar and "
            "the ending feels a little too neat for its own good\n"
            "Sentiment:"
        )
        warmup_rounds = 3

        need_t1 = any(sc.get("routing") or sc.get("fixed") == "tier1" for sc in active_scenarios)
        need_t3 = any(sc.get("routing") or sc.get("fixed") == "tier3" for sc in active_scenarios)
        if need_t1:
            self.mm.load_tier1()
            print(f"[INFO] Warmup generate (tier1) x{warmup_rounds}...")
            for _ in range(warmup_rounds):
                self.mm.generate("tier1", warmup_user, warmup_system)
        if need_t3:
            self.mm.load_tier3()
            print(f"[INFO] Warmup generate (tier3) x{warmup_rounds}...")
            for _ in range(warmup_rounds):
                self.mm.generate("tier3", warmup_user, warmup_system)
        if any(sc.get("compression") for sc in active_scenarios):
            self.ie.get_compressor()
        print("[INFO] Models ready. Starting carbon tracker.")

    def _start_batch_tracker(self):
        if getattr(self.args, "no_tracking", False):
            print("[Carbon] Tracking disabled (--no_tracking).")
            return
        self.batch_tracker = start_batch_tracker()

    def _finalize_batch_carbon(self):
        """Stop the run-level tracker once and allocate totals across prompts."""
        if self._carbon_stopped:
            return
        self._carbon_stopped = True
        if getattr(self.args, "no_tracking", False):
            return
        if self.batch_tracker is None:
            print("[Carbon] Tracker was not started; not writing carbon columns.")
            return
        self.batch_totals = stop_batch_tracker(self.batch_tracker)
        self.batch_tracker = None
        self._allocate_batch_carbon()
        self._rewrite_carbon_columns()

    def _allocate_batch_carbon(self):
        """Split batch CO2/energy across rows in proportion to generation duration."""
        if not self.results:
            return
        totals = self.batch_totals
        total_dur = sum(float(r.get("gen_duration_s") or 0.0) for r in self.results)
        energy_keys = (
            "emissions_kg_co2",
            "energy_consumed_kwh",
            "cpu_energy_kwh",
            "gpu_energy_kwh",
            "ram_energy_kwh",
        )
        power_keys = ("cpu_power_w", "gpu_power_w", "ram_power_w")
        method = totals.get("tracking_method") or "codecarbon_batch"
        for row in self.results:
            dur = float(row.get("gen_duration_s") or 0.0)
            frac = (dur / total_dur) if total_dur > 0 else 0.0
            for k in energy_keys:
                row[k] = round(_require_finite(totals.get(k), k) * frac, 10)
            for k in power_keys:
                row[k] = _require_finite(totals.get(k), k, allow_zero=(k == "gpu_power_w"))
            row["tracking_method"] = method

    def _rewrite_carbon_columns(self):
        """Patch carbon fields on this run's rows in the output CSV (resume-safe)."""
        path = getattr(self.args, "output_csv", None)
        if not path or not os.path.exists(path) or not self.results:
            return
        carbon_cols = [
            "emissions_kg_co2", "energy_consumed_kwh", "gen_duration_s",
            "cpu_power_w", "gpu_power_w", "ram_power_w",
            "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
            "tracking_method",
        ]
        key = ["scenario_id", "dataset", "sample_index", "timestamp"]
        try:
            df = pd.read_csv(path)
            updates = pd.DataFrame(self.results)
            if df.empty or updates.empty:
                return
            df["_k"] = df[key].astype(str).agg("|".join, axis=1)
            updates["_k"] = updates[key].astype(str).agg("|".join, axis=1)
            upd_map = updates.set_index("_k")[carbon_cols]
            mask = df["_k"].isin(upd_map.index)
            for col in carbon_cols:
                df.loc[mask, col] = df.loc[mask, "_k"].map(upd_map[col])
            df.drop(columns=["_k"]).to_csv(path, index=False)
            print(f"[Carbon] Wrote allocated batch metrics into {path}")
        except Exception as e:
            print(f"[Carbon] Could not patch CSV carbon columns ({e}).")

    def run(self):
        all_dsets = [
            ("glue",         "mnli",  "validation_matched"),
            ("glue",         "sst2",  "validation"),
            ("squad_v2",     None,    "validation"),
            ("cnn_dailymail","3.0.0", "test"),
            ("gsm8k",        "main",  "train"),
            ("boolq",        None,    "validation"),
            ("allenai/ai2_arc", "ARC-Easy",      "validation"),
            ("allenai/ai2_arc", "ARC-Challenge", "validation"),
            ("piqa",         None,    "validation"),
            ("ag_news",      None,    "test"),
        ]

        target_dsets = []
        if "all" in self.args.datasets:
            target_dsets = all_dsets
        else:
            for d in self.args.datasets:
                for cand in all_dsets:
                    name_full = f"{cand[0]}/{cand[1]}" if cand[1] else cand[0]
                    if d == cand[0] or d == name_full:
                        target_dsets.append(cand)

        active_scenarios = self.scenarios
        if "all" not in self.args.scenarios:
            active_scenarios = [s for s in self.scenarios if s["id"] in self.args.scenarios]

        loader = DatasetLoader(self.data_root)
        self._preload_models(active_scenarios)
        self._start_batch_tracker()
        try:
            for ds_name, subset, split in target_dsets:
                if self.interrupted:
                    break

                print(f"\ndataset: {ds_name} ({subset or ''})")

                start_idx = getattr(self.args, 'start_index', 0)
                data = loader.load(ds_name, subset, split, self.args.samples, start_idx)
                if not data:
                    continue

                for i, item in enumerate(tqdm(data), start=start_idx):
                    if self.interrupted:
                        break

                    # Classify using only the user content (no system prompt noise)
                    plain_user, _, _ = build_prompt(item, ds_name, subset)
                    category, nemo_result = self.ie.classify_complexity(plain_user)

                    for sc in active_scenarios:
                        if self.interrupted:
                            break
                        self._run_scenario(sc, i, item, ds_name, subset, category, nemo_result)
        finally:
            self._finalize_batch_carbon()
            self._save_results()

    def _run_scenario(self, sc, idx, item, ds_name, subset, category, nemo_result):
        # 1. Determine tier
        tier = self.ie.route_prompt(category) if sc["routing"] else sc["fixed"]
        model_display = f"{tier} ({category})" if sc["routing"] else tier

        # 2. Build prompt (same for both models)
        user_content, system_content, ref = build_prompt(item, ds_name, subset)
        display_name = f"{ds_name}/{subset}" if subset else ds_name

        # 3. Optional compression (S3, S5)
        # Compress the combined text; feed result as the user turn, drop system
        # to avoid duplication of an already-compressed system instruction.
        final_user   = user_content
        final_system = system_content
        c_stats = {
            "rate":  1.0,
            "init":  len(user_content.split()),
            "final": len(user_content.split()),
        }
        if sc["compression"]:
            combined = f"{system_content}\n\n{user_content}" if system_content else user_content
            res = self.ie.compress_prompt(combined)
            final_user   = res["text"]
            final_system = ""          # system already baked into compressed text
            c_stats = {"rate": res["rate"], "init": res["init"], "final": res["final"]}

        # 4. Generate + track carbon/energy
        output, carbon = track_generation(
            lambda: self.mm.generate(tier, final_user, final_system),
            no_tracking=getattr(self.args, "no_tracking", False),
        )
        if output.startswith("Error"):
            print(f"Gen Error: {output}")
        else:
            print("ek prompt hogaya")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(
            f"[Carbon] {carbon['tracking_method']} | "
            f"Duration={carbon['gen_duration_s']:.2f}s "
            f"(batch CO2/energy assigned at end of run)"
        )

        # 5. Score
        score  = 0.0
        stype  = "acc"
        if not output.startswith("Error"):
            score, stype = Evaluator.evaluate(output, ref, ds_name, subset)

        # 6. Record
        nemo_str = json.dumps(nemo_result) if nemo_result else "{}"
        full_prompt_logged = (
            f"[SYSTEM]: {final_system}\n\n[USER]: {final_user}"
            if final_system else f"[USER]: {final_user}"
        )
        row = {
            "scenario_id":           sc["id"],
            "scenario_name":         sc["name"],
            "dataset":               display_name,
            "sample_index":          idx,
            "prompt_complexity":     category,
            "nemo_complexity_score": nemo_result.get("prompt_complexity_score", [""])[0] if nemo_result else "",
            "nemo_raw_output":       nemo_str,
            "model_used":            model_display,
            "original_prompt_len":   c_stats["init"],
            "compressed_prompt_len": c_stats["final"],
            "compression_rate":      c_stats["rate"],
            "accuracy_score":        score,
            "score_type":            stype,
            # ── CodeCarbon metrics ──────────────────────────────────────
            "emissions_kg_co2":      carbon["emissions_kg_co2"],
            "energy_consumed_kwh":   carbon["energy_consumed_kwh"],
            "gen_duration_s":        carbon["gen_duration_s"],
            "cpu_power_w":           carbon["cpu_power_w"],
            "gpu_power_w":           carbon["gpu_power_w"],
            "ram_power_w":           carbon["ram_power_w"],
            "cpu_energy_kwh":        carbon["cpu_energy_kwh"],
            "gpu_energy_kwh":        carbon["gpu_energy_kwh"],
            "ram_energy_kwh":        carbon["ram_energy_kwh"],
            "tracking_method":       carbon["tracking_method"],
            # ───────────────────────────────────────────────────────────
            "full_prompt":           full_prompt_logged,
            "full_output":           output,
            "output_excerpt":        output[:100].replace("\n", " "),
            "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.results.append(row)
        self._append_to_csv(row)

    # Single source of truth for all CSV columns
    _CSV_FIELDS = [
        "scenario_id", "scenario_name", "dataset", "sample_index",
        "prompt_complexity", "nemo_complexity_score", "nemo_raw_output",
        "model_used", "original_prompt_len", "compressed_prompt_len",
        "compression_rate", "accuracy_score", "score_type",
        # CodeCarbon columns
        "emissions_kg_co2", "energy_consumed_kwh", "gen_duration_s",
        "cpu_power_w", "gpu_power_w", "ram_power_w",
        "cpu_energy_kwh", "gpu_energy_kwh", "ram_energy_kwh",
        "tracking_method",
        # text columns last (wide)
        "full_prompt", "full_output", "output_excerpt", "timestamp",
    ]

    def _initialize_csv(self):
        if not self.csv_initialized:
            file_exists = os.path.exists(self.args.output_csv)
            file_empty  = True
            if file_exists:
                with open(self.args.output_csv, 'r') as f:
                    file_empty = len(f.read().strip()) == 0
            if not file_exists or file_empty:
                with open(self.args.output_csv, 'w', newline='') as f:
                    csv.DictWriter(f, fieldnames=self._CSV_FIELDS).writeheader()
                print(f"Initialized CSV file: {self.args.output_csv}")
            self.csv_initialized = True

    def _append_to_csv(self, row):
        self._initialize_csv()
        try:
            with open(self.args.output_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDS)
                writer.writerow(row)
                f.flush()
        except Exception as e:
            print(f"Error appending to CSV: {e}")

    def _save_results(self):
        if not self.results:
            print("No results to save.")
            return
        df = pd.DataFrame(self.results)
        print(f"\nMain results already saved to {self.args.output_csv}")
        if not getattr(self.args, "no_tracking", False):
            t = self.batch_totals
            print(
                f"[Carbon] Batch totals | CO2={t.get('emissions_kg_co2', 0):.6f} kg | "
                f"Energy={t.get('energy_consumed_kwh', 0):.6f} kWh | "
                f"method={t.get('tracking_method')}"
            )

        summary = df.groupby(["scenario_id", "scenario_name", "dataset"])["accuracy_score"].mean().reset_index()
        summary_path = self.args.output_csv.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

        try:
            pivot = df.pivot_table(
                index=['dataset', 'sample_index'],
                columns='scenario_id',
                values=['accuracy_score', 'model_used'],
                aggfunc='first',
            )
            pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
            pivot.reset_index(inplace=True)
            pivot_path = self.args.output_csv.replace(".csv", "_per_prompt.csv")
            pivot.to_csv(pivot_path, index=False)
            print(f"Saved pivot comparison to {pivot_path}")
        except Exception as e:
            print(f"Pivot error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples",     type=int, default=5)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--output_csv",  type=str, default="ecoprompt_results_final.csv")
    parser.add_argument("--no_tracking", action="store_true")
    parser.add_argument("--datasets",    nargs="+", default=["all"])
    parser.add_argument("--scenarios",   nargs="+", default=["all"])
    parser.add_argument("--data_root",   type=str, default=None)
    args = parser.parse_args()

    if args.data_root:
        project_root = args.data_root
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        if os.path.exists(os.path.join(parent_dir, "data")):
            project_root = parent_dir
        elif os.path.exists(os.path.join(script_dir, "data")):
            project_root = script_dir
        elif os.path.exists(os.path.join(os.path.expanduser("~"), "data")):
            project_root = os.path.expanduser("~")
        else:
            project_root = parent_dir

    print(f"[INFO] Using data root: {project_root}")
    print(f"[INFO] Looking for data in: {os.path.join(project_root, 'data')}")

    runner = ExperimentRunner(args, data_root=project_root)
    runner.run()


if __name__ == "__main__":
    main()