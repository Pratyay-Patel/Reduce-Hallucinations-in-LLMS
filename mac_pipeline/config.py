"""
mac_pipeline/config.py
----------------------
Every fixed choice of the pipeline in one place: models, datasets, split, energy settings,
paths. Scripts import from here and never hard-code any of it.

Per-model chat-template choices are written by 00_check_env.py to config_overrides.json and
merged in at import time, so config.py itself never needs editing after a check run.
"""

import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# MACPIPE_DATA lets a smoke test write somewhere else (scratch dir) without touching the real outputs
DATA_ROOT = Path(os.environ.get("MACPIPE_DATA", ROOT))
QUESTIONS_DIR = DATA_ROOT / "questions"
SCORES_DIR = DATA_ROOT / "scores"
ENERGY_DIR = DATA_ROOT / "energy"
ROUTER_DIR = DATA_ROOT / "router"
RESULTS_DIR = DATA_ROOT / "results"
for _d in (QUESTIONS_DIR, SCORES_DIR, ENERGY_DIR, ROUTER_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

QUESTIONS_CSV = QUESTIONS_DIR / "questions.csv"
SPLIT_CSV = QUESTIONS_DIR / "split.csv"
OVERRIDES_JSON = ROOT / "config_overrides.json"

# On the Windows dev box the default HF cache points at a drive that does not exist.
# On the Mac the default (~/.cache/huggingface) is fine, so only override on Windows.
if platform.system() == "Windows":
    _home = os.environ.get("HF_HOME")
    if not _home or not Path(_home).exists():
        os.environ["HF_HOME"] = "C:/tmp/hf"
        os.environ["HF_HUB_CACHE"] = "C:/tmp/hf"
        os.environ["HF_DATASETS_CACHE"] = "C:/tmp/hf/datasets"
        os.environ.pop("TRANSFORMERS_CACHE", None)

# ----------------------------------------------------------------------------- models
BASE = "base"
POOL = ["qwen3b", "phi4mini", "llama3b"]
MODELS = {
    "base":     dict(id="Qwen/Qwen2.5-1.5B-Instruct",      chat_template=False, published_mmlu=60.0),
    "qwen3b":   dict(id="Qwen/Qwen2.5-3B-Instruct",        chat_template=False, published_mmlu=65.6),
    "phi4mini": dict(id="microsoft/Phi-4-mini-instruct",   chat_template=True,  published_mmlu=67.3),
    "llama3b":  dict(id="meta-llama/Llama-3.2-3B-Instruct", chat_template=True,  published_mmlu=63.4),
}
ALL_ROLES = [BASE] + POOL

# ----------------------------------------------------------------------------- datasets
MMLU_SUBJECTS = {
    "stem":       ["abstract_algebra", "college_mathematics", "high_school_mathematics",
                   "college_physics", "high_school_chemistry"],
    "humanities": ["philosophy", "world_religions", "moral_scenarios", "prehistory",
                   "high_school_world_history"],
    "social":     ["high_school_psychology", "sociology", "econometrics",
                   "high_school_macroeconomics", "us_foreign_policy"],
    "other":      ["clinical_knowledge", "nutrition", "management", "marketing",
                   "computer_security"],
}
MMLU_PER_SUBJECT = 50
AQUA_SPLITS = ("test", "validation")     # 254 + 254
ARC_N = 500
DATASETS = ["mmlu", "aqua", "arc"]

SEED = 0
TEST_FRAC = 0.30
SANITY_N = 200                            # first-N MMLU subset used by 00_check_env and the repeats

# ----------------------------------------------------------------------------- energy
COUNTRY = "IND"                           # ISO-3; change to where the Mac actually runs
CARBON_INTENSITY_G_PER_KWH = None         # set a number to force the factor the paper states
MEASURE_POWER_SECS = 1
IDLE_SECS = 60
ROUTER_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------- routing
TAU_GRID = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)  # 1.01 = escalate all
PCA_DIMS = 32
ROUTER_SEEDS = (0, 1, 42, 7)

# ----------------------------------------------------------------------------- device
def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


# ----------------------------------------------------------------------------- overrides
def _apply_overrides():
    # dev only: MACPIPE_MODEL_OVERRIDE='{"qwen3b": "Qwen/Qwen2.5-0.5B-Instruct"}' swaps model ids
    # so the pipeline can be smoke-tested on a machine that cannot hold the real pool.
    if os.environ.get("MACPIPE_MODEL_OVERRIDE"):
        for role, mid in json.loads(os.environ["MACPIPE_MODEL_OVERRIDE"]).items():
            MODELS[role]["id"] = mid
    if OVERRIDES_JSON.exists():
        ov = json.loads(OVERRIDES_JSON.read_text())
        for role, ct in ov.get("chat_template", {}).items():
            if role in MODELS:
                MODELS[role]["chat_template"] = bool(ct)


_apply_overrides()


def save_override(role: str, chat_template: bool):
    ov = json.loads(OVERRIDES_JSON.read_text()) if OVERRIDES_JSON.exists() else {}
    ov.setdefault("chat_template", {})[role] = bool(chat_template)
    OVERRIDES_JSON.write_text(json.dumps(ov, indent=2))
    MODELS[role]["chat_template"] = bool(chat_template)
