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

current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

prompt_class_dir = os.path.join(parent_dir, "Nvidia prompt class")
if os.path.exists(prompt_class_dir) and prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import nvidia_classifier
    import classify_prompt
    print("done")
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {prompt_class_dir}: {e}")
    nvidia_classifier = None
    classify_prompt = None


# ---------------------------------------------------------------------------
# Carbon / energy tracking helper
# ---------------------------------------------------------------------------
# Returns a flat dict of metrics that can be merged directly into a CSV row.
# All failures are caught internally — the rest of the pipeline is never
# affected.  If CodeCarbon has no RAPL access (common on Apple Silicon and
# cloud VMs) we fall back to wall-clock timing only.
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
        output, carbon = track_generation(lambda: model_manager.generate(tier, prompt))

    Returns
    -------
    output : str          — whatever generate_fn() returns
    stats  : dict         — carbon / energy metrics (safe to write to CSV)
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

    # ── Attempt CodeCarbon tracking ──────────────────────────────────────
    tracker = None
    output  = "Error"
    t0 = time.time()

    try:
        tracker = EmissionsTracker(
            project_name="ecoprompt_per_prompt",
            measure_power_secs=1,
            save_to_file=False,          # we extract values ourselves
            log_level="error",           # suppress noisy INFO lines
            allow_multiple_runs=True,
        )
        tracker.start()
    except Exception as e:
        print(f"[Carbon] Tracker init failed ({e}), running without tracking.")
        tracker = None

    try:
        output = generate_fn()
    except Exception as e:
        print(f"[Carbon] Generation error: {e}")
        output = f"Error: {e}"

    t1 = time.time()
    gen_duration = round(t1 - t0, 4)

    # ── Stop tracker and extract metrics ────────────────────────────────
    emissions_kg = 0.0
    stats = _empty_carbon_stats()
    stats["gen_duration_s"] = gen_duration

    if tracker is not None:
        try:
            emissions_kg = tracker.stop() or 0.0
            stats["emissions_kg_co2"] = round(float(emissions_kg), 10)
            stats["tracking_method"]  = "codecarbon"

            # Pull per-component values from the internal EmissionsData object.
            # Attribute names differ slightly across CodeCarbon versions so we
            # try several spellings gracefully.
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
                # EmissionsData not available — energy unknown but CO2 was captured
                stats["tracking_method"] = "codecarbon_co2_only"

        except Exception as e:
            print(f"[Carbon] Tracker stop/extract failed ({e}), using fallback.")
            # ── Fallback: wall-clock only, no power data ─────────────────
            stats["tracking_method"] = "wallclock_fallback"
            # Try to stop silently so the tracker doesn't leak
            try:
                tracker.stop()
            except Exception:
                pass

    return output, stats


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
# Mistral 7B Instruct needs:
#   1. Hand-crafted [INST]/<<SYS>> markup (NOT re-applied by the tokenizer)
#   2. 2-shot examples so it learns "output only the label" literally
#   3. A closing [/INST] trigger so the very next token is the answer
#
# Phi-3 Mini follows plain terse instructions fine; the tokenizer's
# apply_chat_template is used for it as before.
# ---------------------------------------------------------------------------

def build_prompt(item, ds_name, subset, tier):
    """
    Returns (prompt_str, reference).
    tier: "tier1" (Mistral 7B) | "tier3" (Phi-3 Mini)
    """

    # ── MNLI ──────────────────────────────────────────────────────────────
    if ds_name == "glue" and subset == "mnli":
        ref = item["label"]
        if tier == "tier1":
            prompt = (
                "<s>[INST] <<SYS>>\n"
                "You are a precise NLI classifier. "
                "Your reply must be exactly one word — entailment, contradiction, or neutral. "
                "No explanation, no punctuation, no extra text.\n"
                "<</SYS>>\n\n"
                "Premise: Two people are outside.\n"
                "Hypothesis: People are indoors.\n"
                "Label: [/INST] contradiction </s>"
                "<s>[INST] "
                "Premise: A man is playing guitar.\n"
                "Hypothesis: Someone is making music.\n"
                "Label: [/INST] entailment </s>"
                "<s>[INST] "
                f"Premise: {item['premise']}\n"
                f"Hypothesis: {item['hypothesis']}\n"
                "Label: [/INST]"
            )
        else:
            prompt = (
                'Determine if the premise entails, contradicts, or is neutral to the hypothesis.\n'
                'Output only one word: "entailment", "contradiction", or "neutral". No explanation.\n\n'
                f'Premise: {item["premise"]}\n'
                f'Hypothesis: {item["hypothesis"]}\n'
                'Label:'
            )
        return prompt, ref

    # ── SST-2 ─────────────────────────────────────────────────────────────
    elif ds_name == "glue" and subset == "sst2":
        ref = item["label"]
        if tier == "tier1":
            prompt = (
                "<s>[INST] <<SYS>>\n"
                "You are a sentiment classifier. "
                "Reply with exactly one word: positive or negative. Nothing else.\n"
                "<</SYS>>\n\n"
                "Sentence: The film was a total bore.\n"
                "Sentiment: [/INST] negative </s>"
                "<s>[INST] "
                "Sentence: I absolutely loved every moment.\n"
                "Sentiment: [/INST] positive </s>"
                "<s>[INST] "
                f'Sentence: {item["sentence"]}\n'
                "Sentiment: [/INST]"
            )
        else:
            prompt = (
                'Classify the sentiment of the following sentence.\n'
                'Output only one word: "positive" or "negative". No explanation.\n\n'
                f'Sentence: {item["sentence"]}\n'
                'Sentiment:'
            )
        return prompt, ref

    # ── SQuAD v2 ──────────────────────────────────────────────────────────
    elif ds_name == "squad_v2":
        ref = item["answers"]
        ctx = item["context"]
        q   = item["question"]
        if tier == "tier1":
            prompt = (
                "<s>[INST] <<SYS>>\n"
                "You are a reading-comprehension assistant. "
                "Answer using only words found in the context. "
                "If the answer is not in the context, reply with exactly: unanswerable\n"
                "Give only the answer — no explanation, no full sentence.\n"
                "<</SYS>>\n\n"
                f"Context: {ctx}\n"
                f"Question: {q}\n"
                "Answer: [/INST]"
            )
        else:
            prompt = (
                'Answer the question from the context. '
                'If unanswerable, output "unanswerable". No explanation.\n\n'
                f'Context: {ctx}\n'
                f'Question: {q}\n'
                'Answer:'
            )
        return prompt, ref

    # ── CNN / DailyMail ───────────────────────────────────────────────────
    elif ds_name == "cnn_dailymail":
        ref     = item["highlights"]
        article = item["article"][:2000]
        if tier == "tier1":
            prompt = (
                "<s>[INST] <<SYS>>\n"
                "You are a news summariser. "
                "Write a concise 2-3 sentence summary covering the key facts. "
                "No bullet points.\n"
                "<</SYS>>\n\n"
                f"Article:\n{article}\n\n"
                "Summary: [/INST]"
            )
        else:
            prompt = (
                "Summarize the following article in 2-3 sentences.\n\n"
                f"Article:\n{article}\n\n"
                "Summary:"
            )
        return prompt, ref

    # ── GSM8K ─────────────────────────────────────────────────────────────
    elif ds_name == "gsm8k":
        ref = item["answer"]
        q   = item["question"]
        if tier == "tier1":
            prompt = (
                "<s>[INST] <<SYS>>\n"
                "You are a math solver. Think step by step, then write your final "
                "numeric answer on the last line in this exact format:  #### <number>\n"
                "<</SYS>>\n\n"
                f"Question: {q}\n"
                "Solution: [/INST]"
            )
        else:
            prompt = (
                f"Question: {q}\n"
                "Let's think step by step. "
                "Put your final numeric answer after ####.\n####"
            )
        return prompt, ref

    # ── fallback ──────────────────────────────────────────────────────────
    return str(item), ""


# ---------------------------------------------------------------------------
# ModelManager
# Tier 1 = Mistral 7B   (small / fast)
# Tier 3 = Phi-3 Mini   (large / local)
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
        print("Loading Tier 1 (Mistral 7B)...")
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        return self._load_generic_model("tier1", model_id, use_auth=False, use_cache_config=True)

    def load_tier3(self):
        if self.models["tier3"] is not None:
            return self.models["tier3"], self.tokenizers["tier3"]
        print("Loading Tier 3 (Phi-3 Mini)...")
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        return self._load_generic_model("tier3", model_id, use_cache_config=False)

    def _load_generic_model(self, tier_key, model_id, use_auth=False, use_cache_config=True):
        token = os.getenv("HF_TOKEN") if use_auth else None
        if use_auth and not token:
            print("Warning: HF_TOKEN not found for authenticated model.")
        try:
            print(f"Checking for local weights for {model_id}...")
            device = self.get_device()

            if device == "mps":
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, token=token,
                        trust_remote_code=True, local_files_only=True, low_cpu_mem_usage=True,
                    )
                    model = model.to(device)
                    print(f"Loaded {tier_key} from local cache to MPS.")
                except OSError:
                    print(f"Local weights not found for {tier_key}, downloading...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, token=token,
                        trust_remote_code=True, low_cpu_mem_usage=True,
                    )
                    model = model.to(device)
            else:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map="cuda", token=token,
                    )
                    print(f"Loaded {tier_key} from local cache.")
                except OSError:
                    print(f"Local weights not found for {tier_key}, downloading...")
                    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map="auto",
                        token=token, trust_remote_code=True,
                    )

            if not use_cache_config:
                model.generation_config.use_cache = False

            self.models[tier_key]     = model
            self.tokenizers[tier_key] = tokenizer
            return model, tokenizer
        except Exception as e:
            print(f"Error loading {tier_key}: {e}")
            return None, None

    def get_nemo_model(self):
        if nvidia_classifier is None:
            return None, None
        if self.models["nemo"] is None:
            print("Loading NeMo Curator model...")
            try:
                self.models["nemo"], self.tokenizers["nemo"] = nvidia_classifier.load_model()
            except Exception as e:
                print(f"Error loading NeMo model: {e}")
                return None, None
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

    def generate(self, tier, prompt):
        """
        tier1 (Mistral): prompt already contains [INST]/<<SYS>> markup —
          tokenize directly, do NOT apply chat template again.
        tier3 (Phi-3):   prompt is plain text — let apply_chat_template wrap it.
        """
        print(f"[DEBUG] Generating ({tier})... Prompt len: {len(prompt)}")

        methods = {"tier1": self.load_tier1, "tier3": self.load_tier3}
        if tier not in methods:
            return "Error: Invalid logic tier"

        model_weights = methods[tier]()
        if not model_weights or model_weights[0] is None:
            return "Error: Model loading failed"

        model, tokenizer = model_weights

        if tier == "tier1":
            # Raw tokenise — markup already in the string
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        else:
            # Phi-3 chat template
            messages = [{"role": "user", "content": prompt}]
            try:
                inputs = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)
            except Exception:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,       # greedy — more deterministic label output
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
                print(f"Error loading compressor: {e}")
                return None
        return self.compressor

    def classify_complexity(self, prompt):
        if classify_prompt is None:
            return "Label0", {}
        try:
            # get_prompt_class_full returns a rich dict including
            # "prompt_complexity_score" so the CSV column gets populated.
            if hasattr(classify_prompt, "get_prompt_class_full"):
                result          = classify_prompt.get_prompt_class_full(prompt)
                predicted_class = result["predicted_class"]
                nemo_result     = result
            else:
                predicted_class = classify_prompt.get_prompt_class(prompt)
                nemo_result     = {"predicted_class": predicted_class}

            print(f"[DEBUG] Advanced Classifier Predicted Class: {predicted_class}")
            category = "Label1" if predicted_class == 1 else "Label0"
            return category, nemo_result
        except Exception as e:
            print(f"Classifier error: {e}")
            return "Label0", {}

    def route_prompt(self, category):
        """Label0 -> tier1 (Mistral), Label1 -> tier3 (Phi-3 Mini)"""
        return "tier3" if category == "Label1" else "tier1"

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

    def load(self, ds_name, subset, split, samples):
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
            ds = ds.select(range(0, min(total, samples)))
            print(f"[INFO] {split_name}: {total} total, using first {min(total, samples)} samples")
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
        p = min(found, key=found.get)   # earliest mention wins
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

        # Strip "Answer:" preamble, take first line
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

        # S1  Upper Bound   – always tier3 (Phi-3),   no compression
        # S2  Lower Bound   – always tier1 (Mistral), no compression
        # S3  Compression   – always tier3 (Phi-3),   with compression
        # S4  Routing Only  – Label0->tier1, Label1->tier3, no compression
        # S5  EcoPrompt     – Label0->tier1, Label1->tier3, with compression
        self.scenarios = [
            {"id": "S1", "name": "Upper Bound (Phi-3 Mini)",          "routing": False, "compression": False, "fixed": "tier3"},
            {"id": "S2", "name": "Lower Bound (Mistral 7B)",           "routing": False, "compression": False, "fixed": "tier1"},
            {"id": "S3", "name": "Compression (Phi-3 Mini + Comp)",    "routing": False, "compression": True,  "fixed": "tier3"},
            {"id": "S4", "name": "Routing Only",                        "routing": True,  "compression": False, "fixed": None},
            {"id": "S5", "name": "EcoPrompt (Routing + Comp)",          "routing": True,  "compression": True,  "fixed": None},
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
            data = loader.load(ds_name, subset, split, self.args.samples)
            if not data:
                continue

            for i, item in enumerate(tqdm(data)):
                if self.interrupted:
                    break

                # Use plain Phi-3 prompt for the classifier (no [INST] noise)
                plain_prompt, _ = build_prompt(item, ds_name, subset, "tier3")
                category, nemo_result = self.ie.classify_complexity(plain_prompt)

                for sc in active_scenarios:
                    if self.interrupted:
                        break
                    self._run_scenario(sc, i, item, ds_name, subset, category, nemo_result)

        self._save_results()

    def _run_scenario(self, sc, idx, item, ds_name, subset, category, nemo_result):
        # 1. Determine tier
        tier = self.ie.route_prompt(category) if sc["routing"] else sc["fixed"]
        model_display = f"{tier} ({category})" if sc["routing"] else tier

        # 2. Build tier-appropriate prompt
        prompt, ref = build_prompt(item, ds_name, subset, tier)
        display_name = f"{ds_name}/{subset}" if subset else ds_name

        # 3. Compression (S3, S5 only)
        final_prompt = prompt
        c_stats = {"rate": 1.0, "init": len(prompt.split()), "final": len(prompt.split())}
        if sc["compression"]:
            res = self.ie.compress_prompt(prompt)
            final_prompt = res["text"]
            c_stats = {"rate": res["rate"], "init": res["init"], "final": res["final"]}

        # 4. Generate + track carbon/energy
        output, carbon = track_generation(
            lambda: self.mm.generate(tier, final_prompt),
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
        if output != "Error":
            score, stype = Evaluator.evaluate(output, ref, ds_raw, sub)

        # 6. Record
        nemo_str = json.dumps(nemo_result) if nemo_result else "{}"
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
            "full_prompt":           prompt,
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
    parser.add_argument("--output_csv",  type=str, default="ecoprompt_results.csv")
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