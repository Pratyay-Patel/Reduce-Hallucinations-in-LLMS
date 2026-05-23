import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
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

_classification_dir = os.path.join(_project_root, "classification_model")
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


def track_generation(generate_fn, no_tracking=False):
    """
    Wraps a zero-argument callable `generate_fn` with CodeCarbon tracking.

    Usage:
        output, carbon = track_generation(lambda: model_manager.generate(tier, user, sys))

    Returns
    -------
    output : str   — whatever generate_fn() returns
    stats  : dict  — carbon / energy metrics (safe to write to CSV)
    """
    if no_tracking:
        t0 = time.time()
        try:
            output = generate_fn()
        except Exception as e:
            output = f"Error: {e}"
        stats = _empty_carbon_stats()
        stats["gen_duration_s"]  = round(time.time() - t0, 4)
        stats["tracking_method"] = "disabled"
        return output, stats

    tracker = None
    output  = "Error"
    t0 = time.time()

    try:
        tracker = EmissionsTracker(
            project_name="ecoprompt_per_prompt",
            measure_power_secs=1,
            save_to_file=False,
            log_level="error",
            allow_multiple_runs=True,
        )
        tracker.start()
    except Exception as e:
        print(f"[Carbon] Tracker init failed ({e}). Terminating script.")
        sys.exit(1)

    try:
        output = generate_fn()
    except Exception as e:
        print(f"[Carbon] Generation error: {e}")
        output = f"Error: {e}"

    t1 = time.time()
    gen_duration = round(t1 - t0, 4)

    emissions_kg = 0.0
    stats = _empty_carbon_stats()
    stats["gen_duration_s"] = gen_duration

    if tracker is not None:
        try:
            emissions_kg = tracker.stop() or 0.0
            stats["emissions_kg_co2"] = round(float(emissions_kg), 10)
            stats["tracking_method"]  = "codecarbon"

            ed = getattr(tracker, "final_emissions_data", None) \
              or getattr(tracker, "_emissions",            None) \
              or getattr(tracker, "final_emissions",       None)

            if ed is not None:
                def _g(obj, *keys):
                    for k in keys:
                        v = getattr(obj, k, None)
                        if v is not None:
                            try:
                                return round(float(v), 10)
                            except (TypeError, ValueError):
                                pass
                    return 0.0

                stats["energy_consumed_kwh"] = _g(ed, "energy_consumed")
                stats["cpu_power_w"]         = _g(ed, "cpu_power")
                stats["gpu_power_w"]         = _g(ed, "gpu_power")
                stats["ram_power_w"]         = _g(ed, "ram_power")
                stats["cpu_energy_kwh"]      = _g(ed, "cpu_energy")
                stats["gpu_energy_kwh"]      = _g(ed, "gpu_energy")
                stats["ram_energy_kwh"]      = _g(ed, "ram_energy")
            else:
                stats["tracking_method"] = "codecarbon_co2_only"

        except Exception as e:
            print(f"[Carbon] Tracker stop/extract failed ({e}), using fallback.")
            stats["tracking_method"] = "wallclock_fallback"
            try:
                tracker.stop()
            except Exception:
                pass

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
                scaler_path = os.path.join(_project_root, "classification_model", 'advanced_scaler.pkl')
                model_path = os.path.join(_project_root, "classification_model", 'best_advanced_model.pkl')

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
        self._save_results()
        print("✅ Results saved. Exiting gracefully.")
        sys.exit(0)

    def run(self):
        all_dsets = [
            ("glue",         "mnli",  "validation_matched"),
            ("glue",         "sst2",  "validation"),
            ("squad_v2",     None,    "validation"),
            ("cnn_dailymail","3.0.0", "test"),
            ("gsm8k",        "main",  "train"),
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
            f"CO2={carbon['emissions_kg_co2']:.6f} kg | "
            f"Energy={carbon['energy_consumed_kwh']:.6f} kWh | "
            f"Duration={carbon['gen_duration_s']:.2f}s"
        )

        # 5. Score
        score  = 0.0
        stype  = "acc"
        ds_raw = display_name.split("/")[0]
        sub    = display_name.split("/")[1] if "/" in display_name else None
        if not output.startswith("Error"):
            score, stype = Evaluator.evaluate(output, ref, ds_raw, sub)

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