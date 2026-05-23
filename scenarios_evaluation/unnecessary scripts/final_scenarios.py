import argparse
import os
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

# Load environment variables
load_dotenv()

# --- Configuration & Global Setup ---
torch.manual_seed(42)

# Ensure Nvidia Classifier is in path
current_dir = os.getcwd()
prompt_class_dir = os.path.join(current_dir, "Nvidia prompt class")
if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import nvidia_classifier
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {prompt_class_dir}: {e}")
    nvidia_classifier = None


class ModelManager:
    """Manages loading, caching, and unloading of LLM models and Tokenizers."""
    
    def __init__(self):
        self.models = {
            "tier1": None,  # Phi-3 Mini
            "tier2": None,  # Mistral 7B
            "tier3": None,  # Llama 2 13B
            "nemo": None
        }
        self.tokenizers = {
            "tier1": None,
            "tier2": None,
            "tier3": None,
            "nemo": None
        }

    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load_tier1(self):
        """Loads Tier 1: Phi-3 Mini"""
        if self.models["tier1"] is not None:
            return self.models["tier1"], self.tokenizers["tier1"]

        print("Loading Tier 1 (Phi-3 Mini)...")
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        return self._load_generic_model("tier1", model_id, use_cache_config=False)

    def load_tier2(self):
        """Loads Tier 2: Mistral 7B"""
        if self.models["tier2"] is not None:
            return self.models["tier2"], self.tokenizers["tier2"]

        print("Loading Tier 2 (Mistral 7B)...")
        model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        return self._load_generic_model("tier2", model_id)

    def load_tier3(self):
        """Loads Tier 3: Llama 2 13B (using Mistral path as per original code??)"""
       
        if self.models["tier3"] is not None:
            return self.models["tier3"], self.tokenizers["tier3"]

        print("Loading Tier 3 (Llama 2 13B)...") 
        model_id = "mistralai/Mistral-7B-Instruct-v0.3" # Keeping as per original file
        return self._load_generic_model("tier3", model_id, use_auth=True)

    def _load_generic_model(self, tier_key, model_id, use_auth=False, use_cache_config=True):
        model_id = "microsoft/Phi-3-mini-4k-instruct"
        token = os.getenv("HF_TOKEN") if use_auth else None
        if use_auth and not token:
             print("Warning: HF_TOKEN not found for authenticated model.")

        try:
            print(f"Checking for local weights for {model_id}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    device_map="auto",
                    token=token,
                    trust_remote_code=True,
                    local_files_only=True
                )
                print(f"Loaded {tier_key} from local cache.")
            except OSError:
                print(f"Local weights not found for {tier_key}, downloading from Hub...")
                tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, 
                    torch_dtype=torch.float16, 
                    device_map="auto",
                    token=token,
                    trust_remote_code=True
                )
            
            # Phi-3 specific fix from original code
            if not use_cache_config:
                 model.generation_config.use_cache = False

            self.models[tier_key] = model
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
            torch.cuda.empty_cache()
            gc.collect()
            self.models[tier] = None
            self.tokenizers[tier] = None
    
    def generate(self, tier, prompt):
        """Generates text using the loaded model for the given tier."""
        print(f"[DEBUG] Generating ({tier})... Prompt len: {len(prompt)}")
        
        methods = {"tier1": self.load_tier1, "tier2": self.load_tier2, "tier3": self.load_tier3}
        if tier not in methods:
            return "Error: Invalid logic tier"

        model_weights = methods[tier]()
        
        if not model_weights or model_weights[0] is None:
            return "Error: Model loading failed"
            
        model, tokenizer = model_weights
        
        # Formatting
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        return generated_text.strip()


class IntelligenceEngine:
    """Handles complexity analysis (Classifier) and Prompt Compression."""

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
                    device_map="cpu"
                )
            except Exception as e:
                print(f"Error loading compressor: {e}")
                return None
        return self.compressor

    def classify_complexity(self, prompt):
        if nvidia_classifier is None:
            return "Medium"
            
        try:
            model, tokenizer = self.mm.get_nemo_model()
            if not model: return "Medium"
            
            result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
            score = 0.0
            if "prompt_complexity_score" in result:
                try: score = float(result["prompt_complexity_score"][0])
                except: score = 0.0
            
            if score < 0.35: category = "Easy"
            elif score < 0.65: category = "Medium"
            else: category = "Hard"
            
            print(f"[DEBUG] Classifier Score: {score:.4f} -> {category}")
            return category
        except Exception as e:
            print(f"Classifier error: {e}")
            return "Medium"

    def route_prompt(self, category):
        if category == "Easy": return "tier1"
        elif category == "Medium": return "tier2"
        elif category == "Hard": return "tier3"
        return "tier2"

    def compress_prompt(self, prompt):
        compressor = self.get_compressor()
        if not compressor:
            return {"text": prompt, "rate": 1.0, "init": 0, "final": 0}
            
        try:
            token_count = len(compressor.tokenizer.encode(prompt))
            # Heuristic
            slope = 9.5 / 8000
            ratio = 2.5 + (token_count - 2000) * slope
            if ratio < 1.0: ratio = 1.0
            
            # Min compression ratio constraint
            if ratio < 2.0 and ratio > 1.0: ratio = 2.0
            
            if ratio == 1.0:
                 return {"text": prompt, "rate": 1.0, "init": token_count, "final": token_count}

            rate = 1 / ratio
            res = compressor.compress_prompt(prompt, rate=rate, force_tokens=['\n', '?'])
            
            compressed_text = res['compressed_prompt'] if isinstance(res, dict) else res
            final_tokens = len(compressor.tokenizer.encode(compressed_text))
            
            return {
                "text": compressed_text,
                "rate": rate,
                "init": token_count,
                "final": final_tokens
            }
        except Exception as e:
            print(f"Compression failed: {e}")
            return {"text": prompt, "rate": 1.0, "init": 0, "final": 0}


class DatasetLoader:
    """Handles loading of datasets from local files and formatting prompts."""
    
    def __init__(self, data_root):
        self.data_root = data_root

    def get_local_path(self, ds_name, subset, split):
        # Map dataset args to local dataset directories and their splits
        # Returns tuple of (dataset_dir, split_name)
        paths = {
            ("glue", "mnli"): (os.path.join(self.data_root, "data", "NLI_MNLI"), "validation_matched"),
            ("glue", "sst2"): (os.path.join(self.data_root, "data", "SST-2"), "validation"),
            ("squad_v2", None): (os.path.join(self.data_root, "data", "SQuAD_v2"), "validation"),
            ("cnn_dailymail", "3.0.0"): (os.path.join(self.data_root, "data", "CNN_DailyMail"), "test"),
            ("gsm8k", "main"): (os.path.join(self.data_root, "data", "GSM8K"), "test")
        }
        return paths.get((ds_name, subset))

    def load(self, ds_name, subset, split, samples):
        path_info = self.get_local_path(ds_name, subset, split)
        if not path_info:
            print(f"Unknown local path for {ds_name}/{subset}")
            return []

        dataset_dir, split_name = path_info
        
        try:
            # Load the dataset dictionary from the local directory using load_from_disk
            # HuggingFace datasets cache with dataset_dict.json at root
            ds_dict = load_from_disk(dataset_dir)
            
            # Access the specific split
            if split_name not in ds_dict:
                print(f"Split '{split_name}' not found. Available: {list(ds_dict.keys())}")
                return []
            
            ds = ds_dict[split_name]
            ds = ds.select(range(min(len(ds), samples)))
            return ds
        except Exception as e:
            print(f"Dataset load error ({dataset_dir}, split={split_name}): {e}")
            return []

    def format_sample(self, item, ds_name, subset):
        """Returns (prompt, reference_answer, dataset_display_name)"""
        prompt = ""
        ref = ""
        
        # Unique display name to avoid collision
        display_name = f"{ds_name}/{subset}" if subset else ds_name

        if ds_name == "glue" and subset == "mnli":
            prompt = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nlabel (entailment, neutral, contradiction):"
            ref = item['label']
        elif ds_name == "glue" and subset == "sst2":
            prompt = f"Sentence: {item['sentence']}\nSentiment (positive, negative):"
            ref = item['label']
        elif ds_name == "squad_v2":
            prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
            ref = item['answers']
        elif ds_name == "cnn_dailymail":
            prompt = f"Summarize:\n{item['article'][:2000]}"
            ref = item['highlights']
        elif ds_name == "gsm8k":
            prompt = f"Question: {item['question']}\nThink step by step:"
            ref = item['answer']
            
        return prompt, ref, display_name


class Evaluator:
    """Static evaluation metrics."""
    
    @staticmethod
    def evaluate(output, reference, ds_name, subset):
        if ds_name == "glue":
            if subset == "mnli": return Evaluator.mnli(output, reference), "accuracy"
            if subset == "sst2": return Evaluator.sst2(output, reference), "accuracy"
        elif ds_name == "squad_v2":
            return Evaluator.squad(output, reference), "EM"
        elif ds_name == "cnn_dailymail":
            return Evaluator.rouge(output, reference), "ROUGE-L"
        elif ds_name == "gsm8k":
            return Evaluator.gsm8k(output, reference), "EM"
        return 0.0, "unknown"

    @staticmethod
    def mnli(pred, label):
        pred = pred.lower()
        map_ = {0: "entailment", 1: "neutral", 2: "contradiction"}
        lbl = map_.get(label, "") if isinstance(label, int) else str(label).lower()
        
        if "entailment" in pred: p = "entailment"
        elif "neutral" in pred: p = "neutral"
        elif "contradiction" in pred: p = "contradiction"
        else: p = "unknown"
        return 1 if p == lbl else 0

    @staticmethod
    def sst2(pred, label):
        pred = pred.lower()
        map_ = {0: "negative", 1: "positive"}
        lbl = map_.get(label, "") if isinstance(label, int) else str(label).lower()
        
        if "positive" in pred: p = "positive"
        elif "negative" in pred: p = "negative"
        else: p = "unknown"
        return 1 if p == lbl else 0

    @staticmethod
    def squad(pred, answers):
        candidates = answers.get('text', [])
        if not candidates: return 0
        
        def normalize(s):
            import string
            s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            return ' '.join(s.split())

        pred_norm = normalize(pred)
        return max([1 if normalize(a) == pred_norm else 0 for a in candidates])

    @staticmethod
    def rouge(pred, ref):
        if not pred: return 0.0
        try:
            scorer = evaluate.load("rouge")
            res = scorer.compute(predictions=[pred], references=[ref])
            return res.get('rougeL', 0.0)
        except: return 0.0

    @staticmethod
    def gsm8k(pred, ref_str):
        gold = ref_str.split("####")[-1].strip()
        nums = re.findall(r'-?\d+\.?\d*', pred)
        pred_num = nums[-1] if nums else ""
        try:
            return 1 if abs(float(pred_num) - float(gold)) < 1e-5 else 0
        except: return 0


class ExperimentRunner:
    """Main Orchestrator."""
    
    def __init__(self, args, data_root):
        self.args = args
        self.mm = ModelManager()
        self.ie = IntelligenceEngine(self.mm)
        self.loader = DatasetLoader(data_root)
        self.results = []
        self.csv_initialized = False
        self.interrupted = False
        
        self.scenarios = [
            {"id": "S1", "name": "Upper Bound (Tier 3)", "routing": False, "compression": False, "fixed": "tier3"},
            {"id": "S2", "name": "Lower Bound (Tier 1)", "routing": False, "compression": False, "fixed": "tier1"},
            {"id": "S3", "name": "Compression (Tier 3 + Comp)", "routing": False, "compression": True, "fixed": "tier3"},
            {"id": "S4", "name": "Routing Only", "routing": True, "compression": False, "fixed": None},
            {"id": "S5", "name": "EcoPrompt (Routing + Comp)", "routing": True, "compression": True, "fixed": None},
        ]
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        print("\n\n⚠️  Process interrupted! Saving results before exit...")
        self.interrupted = True
        self._save_results()
        print("✅ Results saved. Exiting gracefully.")
        sys.exit(0)

    def run(self):
        # 1. Filter Datasets
        all_dsets = [
            ("glue", "mnli", "validation_matched"), 
            ("glue", "sst2", "validation"),
            ("squad_v2", None, "validation"),
            ("cnn_dailymail", "3.0.0", "test"),#why not
            ("gsm8k", "main", "test")
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

        # 2. Filter Scenarios
        active_scenarios = self.scenarios
        if "all" not in self.args.scenarios:
             active_scenarios = [s for s in self.scenarios if s["id"] in self.args.scenarios]

        # 3. CodeCarbon
        tracker = None
        if not self.args.no_tracking:
            tracker = EmissionsTracker(project_name="ecoprompt", measure_power_secs=1, save_to_file=False, log_level="error")

        print(f"Starting Experiment: {self.args.samples} samples...")

        # 4. Main Loop
        for ds_name, subset, split in target_dsets:
            if self.interrupted:
                break
                
            print(f"\ndataset: {ds_name} ({subset or ''})")
            data = self.loader.load(ds_name, subset, split, self.args.samples)
            if not data: continue
            
            for i, item in enumerate(tqdm(data)):
                if self.interrupted:
                    break
                    
                prompt, ref, unique_ds_name = self.loader.format_sample(item, ds_name, subset)
                
                # Analyze once per sample
                category = self.ie.classify_complexity(prompt)
                
                for sc in active_scenarios:
                    if self.interrupted:
                        break
                    self._run_scenario(sc, i, prompt, ref, category, unique_ds_name, tracker)

        self._save_results()

    def _run_scenario(self, sc, idx, prompt, ref, category, ds_name, tracker):
        # 1. Determine Model
        tier = ""
        if sc["routing"]:
            tier = self.ie.route_prompt(category)
        else:
            tier = sc["fixed"]
            
        model_display = f"{tier} ({category})" if sc["routing"] else tier

        # 2. Compression
        final_prompt = prompt
        c_stats = {"rate": 1.0, "init": len(prompt.split()), "final": len(prompt.split())}
        
        if sc["compression"]:
            res = self.ie.compress_prompt(prompt)
            final_prompt = res["text"]
            c_stats = {"rate": res["rate"], "init": res["init"], "final": res["final"]}

        # 3. Generation & Emissions
        emissions = 0.0
        output = ""
        try:
            if tracker: tracker.start()
            output = self.mm.generate(tier, final_prompt)
            if tracker: emissions = tracker.stop()
            print("ek prompt hogaya")
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:
            print(f"Gen Error: {e}")
            output = "Error"
            if tracker and tracker._start_time: tracker.stop()

        # 4. Score
        score = 0.0
        stype = "acc"
        ds_raw = ds_name.split("/")[0] # 'glue/mnli' -> 'glue' for evaluator dispatch
        subset = ds_name.split("/")[1] if "/" in ds_name else None
        
        if output != "Error":
             score, stype = Evaluator.evaluate(output, ref, ds_raw, subset)

        # 5. Record
        row = {
            "scenario_id": sc["id"],
            "scenario_name": sc["name"],
            "dataset": ds_name, # Now contains unique name (e.g. glue/mnli)
            "sample_index": idx,
            "prompt_complexity": category,
            "model_used": model_display,
            "original_prompt_len": c_stats["init"],
            "compressed_prompt_len": c_stats["final"],
            "compression_rate": c_stats["rate"],
            "carbon_kg": emissions,
            "accuracy_score": score,
            "score_type": stype,
            "output_excerpt": output[:100].replace("\n", " "),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.results.append(row)
        
        # Immediately save to CSV to prevent data loss
        self._append_to_csv(row)

    def _initialize_csv(self):
        """Initialize CSV file with headers if it doesn't exist or is empty."""
        if not self.csv_initialized:
            fieldnames = [
                "scenario_id", "scenario_name", "dataset", "sample_index",
                "prompt_complexity", "model_used", "original_prompt_len",
                "compressed_prompt_len", "compression_rate", "carbon_kg",
                "accuracy_score", "score_type", "output_excerpt", "timestamp"
            ]
            
            # Check if file exists and has content
            file_exists = os.path.exists(self.args.output_csv)
            if file_exists:
                with open(self.args.output_csv, 'r') as f:
                    file_empty = len(f.read().strip()) == 0
            else:
                file_empty = True
            
            # Write headers if file is new or empty
            if not file_exists or file_empty:
                with open(self.args.output_csv, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                print(f"Initialized CSV file: {self.args.output_csv}")
            
            self.csv_initialized = True
    
    def _append_to_csv(self, row):
        """Append a single row to the CSV file immediately."""
        self._initialize_csv()
        
        try:
            with open(self.args.output_csv, 'a', newline='') as f:
                fieldnames = [
                    "scenario_id", "scenario_name", "dataset", "sample_index",
                    "prompt_complexity", "model_used", "original_prompt_len",
                    "compressed_prompt_len", "compression_rate", "carbon_kg",
                    "accuracy_score", "score_type", "output_excerpt", "timestamp"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
                f.flush()  # Force write to disk immediately
                os.fsync(f.fileno())  # Ensure OS buffers are written to disk
        except Exception as e:
            print(f"Error appending to CSV: {e}")
    
    def _save_results(self):
        """Generate summary and pivot tables from collected results."""
        if not self.results:
            print("No results to save.")
            return

        df = pd.DataFrame(self.results)
        print(f"\nMain results already saved to {self.args.output_csv}")

        # Summary
        summary = df.groupby(["scenario_id", "scenario_name", "dataset"])["accuracy_score"].mean().reset_index()
        summary_path = self.args.output_csv.replace(".csv", "_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary to {summary_path}")

        # Pivot (Wide Format)
        try:
            pivot = df.pivot_table(
                index=['dataset', 'sample_index'],
                columns='scenario_id',
                values=['accuracy_score', 'carbon_kg', 'model_used'],
                aggfunc='first'
            )
            # Flatten columns
            pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
            pivot.reset_index(inplace=True)
            
            pivot_path = self.args.output_csv.replace(".csv", "_per_prompt.csv")
            pivot.to_csv(pivot_path, index=False)
            print(f"Saved pivot comparison to {pivot_path}")
        except Exception as e:
            print(f"Pivot error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output_csv", type=str, default="evaluation_scenarios_results.csv")
    parser.add_argument("--no_tracking", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=["all"])
    parser.add_argument("--scenarios", nargs="+", default=["all"])
    parser.add_argument("--data_root", type=str, default=None, help="Root directory containing data folder")
    args = parser.parse_args()

    # Determine project root - auto-detect data directory location
    if args.data_root:
        project_root = args.data_root
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try multiple locations for data directory
        # 1. Parent directory (for scenarios_evaluation subfolder structure)
        parent_dir = os.path.dirname(script_dir)
        if os.path.exists(os.path.join(parent_dir, "data")):
            project_root = parent_dir
        # 2. Current script directory (for co-located scripts)
        elif os.path.exists(os.path.join(script_dir, "data")):
            project_root = script_dir
        # 3. Home directory (for GPU server setup)
        elif os.path.exists(os.path.join(os.path.expanduser("~"), "data")):
            project_root = os.path.expanduser("~")
        else:
            # Fallback to parent directory
            project_root = parent_dir
    
    print(f"[INFO] Using data root: {project_root}")
    print(f"[INFO] Looking for data in: {os.path.join(project_root, 'data')}")
    
    runner = ExperimentRunner(args, data_root=project_root)
    runner.run()


if __name__ == "__main__":
    main()
