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
# Decision rule (Oct 24): route-then-cascade.
#   router (question text only) -> one of the ROUTED 1.5B models answers
#   its confidence < tau[route]  -> the ESCALATION 3B model answers instead
# Results_Run2 used base=Qwen2.5-1.5B with a pool of three 3B models (qwen3b, phi4mini, llama3b)
# chosen by the router after the base had answered; see RESULTS_RUN2.md.
BASE = "generalist"                       # the plain cascade's first model; also "base" anchors
ROUTED = ["generalist", "math"]           # the 1.5B tier the router chooses among
ESCALATION = "qwen3b"
POOL = [ESCALATION]                       # models a question can be escalated to
MODELS = {
    "generalist": dict(id="Qwen/Qwen2.5-1.5B-Instruct",      chat_template=False, published_mmlu=60.0),
    # Qwen2.5-Math-1.5B-Instruct: no published MMLU; raw letter format like the other Qwen2.5
    # models (00_check_env with CHECK_MODELS=1 re-checks raw vs chat on 200 MMLU questions)
    "math":       dict(id="Qwen/Qwen2.5-Math-1.5B-Instruct", chat_template=False, published_mmlu=None),
    "qwen3b":     dict(id="Qwen/Qwen2.5-3B-Instruct",        chat_template=False, published_mmlu=65.6),
    # Results_Run2 pool members, not in the current workflow:
    # "phi4mini": dict(id="microsoft/Phi-4-mini-instruct",   chat_template=True,  published_mmlu=67.3),
    # "llama3b":  dict(id="meta-llama/Llama-3.2-3B-Instruct", chat_template=True,  published_mmlu=63.4),
    # LoRA roles: a base id plus an adapter (HF repo id from lora/colab_train_lora.ipynb, or a
    # local dir). The adapter must have been trained on exactly that base. Uncomment and edit
    # once the adapters exist, then add the roles to POOL.
    # "qwen3b_math": dict(id="Qwen/Qwen2.5-3B-Instruct", adapter="<hf-user>/qwen2.5-3b-instruct-lora-math",
    #                     chat_template=False, published_mmlu=65.6),
    # "qwen3b_code": dict(id="Qwen/Qwen2.5-3B-Instruct", adapter="<hf-user>/qwen2.5-3b-instruct-lora-code",
    #                     chat_template=False, published_mmlu=65.6),
    # "qwen3b_knowledge": dict(id="Qwen/Qwen2.5-3B-Instruct", adapter="<hf-user>/qwen2.5-3b-instruct-lora-knowledge",
    #                          chat_template=False, published_mmlu=65.6),
}
ALL_ROLES = ROUTED + [ESCALATION]

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
CSQA_N = 500                              # CommonsenseQA validation (1,221 labelled), seeded sample
ARC_N = 500
# AQuA-RAT: multi-step arithmetic word problems, 5 options. Letter scoring alone is at chance
# on it (Results_Run1), so it is scored with the reasoning protocol: the model reasons first
# (greedy, bounded), then the option letters are scored after a fixed answer stem. The
# test split (254) is the whole set; ~25 s per question for a 1.5B model on the Air.
AQUA_SPLITS = ("test",)                   # 254 questions; add "validation" for 254 more
COT_DATASETS = ["aqua"]                   # scored by reason-then-letter instead of letter only
COT_MAX_NEW_TOKENS = 512                  # 256 truncated 90% of the math model's reasoning
# Code datasets, not in the workflow (Oct 24 decision): the 1.5B coder specialist scored
# below the 1.5B generalist on CodeMMLU's knowledge subsets, and code generation costs ~30x a
# letter score. The builders and the generate-and-test scorer are kept; add "codemmlu" and/or
# "codegen" back to DATASETS and rerun 01 to use them.
CODEMMLU_SUBSETS = ["api_frameworks", "dbms_sql", "programming_syntax", "software_principles", "others"]
CODEMMLU_PER_SUBSET = 100
HUMANEVAL_N = 164                         # 164 = all
MBPP_N = 257                              # 257 = all of the sanitized test split
CODEGEN_MAX_NEW_TOKENS = 512
CODEGEN_EXEC_TIMEOUT = 10                 # seconds per test program
DATASETS = ["mmlu", "csqa", "arc", "aqua"]
# AQuA-RAT was in the first Mac run under plain letter scoring and sat at chance (60-75%
# position habit); it was replaced by CommonsenseQA for Run2 and is back now with reasoning.

SEED = 0
TEST_FRAC = 0.30
SANITY_N = 200                            # first-N MMLU subset used by 00_check_env and the repeats

# ----------------------------------------------------------------------------- energy
COUNTRY = "IND"                           # ISO-3; change to where the Mac actually runs
CARBON_INTENSITY_G_PER_KWH = None         # set a number to force the factor the paper states
MEASURE_POWER_SECS = 1
IDLE_SECS = 60
# On the Mac, refuse to run any tracked step unless powermetrics is delivering SoC samples.
# CodeCarbon's fallback is an 85 W-TDP model, which is not a measurement.
REQUIRE_POWERMETRICS = True
ROUTER_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

# ----------------------------------------------------------------------------- routing
TAU_GRID = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)  # 1.01 = escalate all
PCA_DIMS = 32
ROUTER_SEEDS = (0, 1, 42, 7)
# router heads compared out-of-fold in 04; "auto" picks the best OOF routed accuracy (ties by AUC).
# lr is the literature default for dense embeddings; rf/xgb/svc/mlp/voting are the classifier
# families kept for the out-of-fold router-head ablation.
ROUTER_HEADS = ("lr", "rf", "xgb", "svc", "mlp", "voting")
ROUTER_HEAD = "auto"

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
    # dev only: MACPIPE_ADAPTER_OVERRIDE='{"base": "/path/or/repo"}' attaches an adapter to a role
    if os.environ.get("MACPIPE_ADAPTER_OVERRIDE"):
        for role, ad in json.loads(os.environ["MACPIPE_ADAPTER_OVERRIDE"]).items():
            MODELS[role]["adapter"] = ad
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
