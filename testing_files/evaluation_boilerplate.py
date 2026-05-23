import argparse
import os
import torch
import pandas as pd
import numpy as np
import time
import csv
import re
import sys
from datetime import datetime
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from codecarbon import EmissionsTracker
from llmlingua import PromptCompressor
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (for Gemini API key)
load_dotenv()

# --- Nvidia Classifier Setup ---
# Add the directory containing the Nvidia prompt classifier code to sys.path
# We assume the script is run from project root, so 'Nvidia prompt class' is in CWD or nearby
# Based on file structure: project/Nvidia prompt class
current_dir = os.getcwd()
prompt_class_dir = os.path.join(current_dir, "Nvidia prompt class")

if prompt_class_dir not in sys.path:
    sys.path.append(prompt_class_dir)

try:
    import temp as nvidia_classifier
except ImportError as e:
    print(f"Warning: Could not import nvidia_classifier from {prompt_class_dir}: {e}")
    nvidia_classifier = None

# Set seed for reproducibility
torch.manual_seed(42)

# Global variables for models
MODEL_1 = None # Phi-3
TOKENIZER_1 = None
MODEL_2 = None # TinyLlama
TOKENIZER_2 = None
COMPRESSOR = None
NEMO_MODEL = None
NEMO_TOKENIZER = None

# --- 1. Model Loading Functions ---

def load_model_1():
    """Loads Phi-3 Mini (Medium Complexity)"""
    global MODEL_1, TOKENIZER_1
    if MODEL_1 is not None:
        return MODEL_1, TOKENIZER_1
        
    print("Loading Model 1 (Phi-3 Mini)...")
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device,
            attn_implementation="eager"
        )
        MODEL_1 = model
        TOKENIZER_1 = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading Phi-3: {e}")
        return None, None

def load_model_2():
    """Loads TinyLlama (Easy Complexity)"""
    global MODEL_2, TOKENIZER_2
    if MODEL_2 is not None:
        return MODEL_2, TOKENIZER_2

    print("Loading Model 2 (TinyLlama)...")
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map=device
        )
        MODEL_2 = model
        TOKENIZER_2 = tokenizer
        return model, tokenizer
    except Exception as e:
        print(f"Error loading TinyLlama: {e}")
        return None, None

def load_model_3():
    """Loads Gemini (API Client) (Hard Complexity)"""
    # Gemini is API-based, so we just check for the key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found.")
        return None
    
    genai.configure(api_key=api_key)
    return "gemini_model"

# --- 2. Nvidia Classifier Logic ---

def get_nemo_model():
    global NEMO_MODEL, NEMO_TOKENIZER
    if nvidia_classifier is None:
        return None, None
        
    if NEMO_MODEL is None or NEMO_TOKENIZER is None:
        print("Loading NeMo Curator model for classification...")
        try:
            NEMO_MODEL, NEMO_TOKENIZER = nvidia_classifier.load_model()
        except Exception as e:
            print(f"Error loading NeMo model: {e}")
            return None, None
    return NEMO_MODEL, NEMO_TOKENIZER

def classify_complexity(score):
    if score < 0.35:
        return "Easy"
    elif score < 0.65:
        return "Medium"
    else:
        return "Hard"

def get_prompt_category(prompt):
    """
    Analyzes prompt complexity using Nvidia classifier.
    Returns: 'Easy', 'Medium', or 'Hard'
    """
    if nvidia_classifier is None:
        return "Medium" # Fallback
        
    try:
        model, tokenizer = get_nemo_model()
        if model is None:
            return "Medium"
            
        result = nvidia_classifier.analyze_prompt(model, tokenizer, prompt)
        
        complexity_score = 0.0
        if "prompt_complexity_score" in result:
             try:
                complexity_score = float(result["prompt_complexity_score"][0])
             except:
                complexity_score = 0.0
        
        return classify_complexity(complexity_score)
        
    except Exception as e:
        print(f"Error classifying prompt: {e}")
        return "Medium"

def select_model_for_category(category):
    """
    Selects model based on category.
    Easy -> TinyLlama (Model 2)
    Medium -> Phi-3 (Model 1)
    Hard -> Gemini (Model 3)
    """
    if category == "Easy":
        return "tinyllama", load_model_2()
    elif category == "Hard":
        return "gemini", load_model_3()
    else: # Medium or fallback
        return "phi3mini", load_model_1()

# --- 3. Compression Function ---

def get_compressor():
    global COMPRESSOR
    if COMPRESSOR is None:
        print("Initializing LLM Lingua-2 Compressor...")
        try:
            COMPRESSOR = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu" 
            )
        except Exception as e:
            print(f"Error loading compressor: {e}")
            return None
    return COMPRESSOR

def compress_prompt(prompt):
    """
    Compresses the prompt using LLM Lingua.
    Returns dictionary with compressed prompt and metadata.
    """
    compressor = get_compressor()
    if not compressor:
        return {
            "compressed_prompt": prompt,
            "initial_tokens": len(prompt.split()), 
            "final_tokens": len(prompt.split())
        }
        
    try:
        token_count = len(compressor.tokenizer.encode(prompt))
        
        # Simple dynamic rate logic
        rate = 0.5 
        if token_count < 50:
             rate = 0.9 # Little compression for short prompts
        elif token_count > 1000:
             rate = 0.3
             
        compressed_result = compressor.compress_prompt(
            prompt, 
            rate=rate, 
            force_tokens=['\n', '?']
        )
        
        if isinstance(compressed_result, dict):
            compressed_text = compressed_result['compressed_prompt']
        else:
            compressed_text = compressed_result
            
        final_tokens = len(compressor.tokenizer.encode(compressed_text))
        
        return {
            "compressed_prompt": compressed_text,
            "initial_tokens": token_count,
            "final_tokens": final_tokens
        }

    except Exception as e:
        print(f"Compression error: {e}")
        return {
            "compressed_prompt": prompt, 
            "initial_tokens": 0, 
            "final_tokens": 0
        }

# --- 4. Generate Function ---

def generate_prompt(model_weights, prompt, model_name):
    """
    Generates output using the selected model.
    model_weights: tuple (model, tokenizer) or string for Gemini
    """
    if model_name == "gemini":
        try:
            # Re-init if needed or use global config from load_model_3
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Gemini Error: {e}"
            
    else:
        # Local HF models
        if not model_weights or model_weights[0] is None:
            return "Error: Model not loaded"
            
        model, tokenizer = model_weights
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return generated_text.strip()

# --- 5. Evaluation Helper Functions ---

def evaluate_mnli(pred, label):
    pred = pred.lower()
    if "entailment" in pred: p = 0
    elif "neutral" in pred: p = 1
    elif "contradiction" in pred: p = 2
    else: p = -1
    return 1 if p == label else 0

def evaluate_sst2(pred, label):
    pred = pred.lower()
    if "positive" in pred: p = 1
    elif "negative" in pred: p = 0
    else: p = -1
    return 1 if p == label else 0

def evaluate_squad(pred, answers):
    possible_answers = answers.get('text', [])
    if not possible_answers:
        return 1 if not pred else 0 
    
    def normalize(s):
        import string
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    pred_norm = normalize(pred)
    return max([1 if normalize(a) == pred_norm else 0 for a in possible_answers])

def evaluate_rouge(pred, ref):
    scorer = evaluate.load("rouge")
    results = scorer.compute(predictions=[pred], references=[ref])
    return results

def evaluate_gsm8k(pred, answer_str):
    gold = answer_str.split("####")[-1].strip()
    pred_nums = re.findall(r'-?\d+\.?\d*', pred)
    pred_num = pred_nums[-1] if pred_nums else ""
    try:
        return 1 if float(pred_num) == float(gold) else 0
    except:
        return 0

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5, help="Samples per dataset")
    parser.add_argument("--output_csv", type=str, default="evaluation_report_full.csv")
    args = parser.parse_args()

    # Pre-load all models to avoid reloading penalty
    # (Though we check existence in load functions, good to init explicitly)
    print("Pre-loading models...")
    load_model_1()
    load_model_2()
    load_model_3()
    # Check if classifier model needs loading
    if nvidia_classifier:
        get_nemo_model()
    
    # Dataset Config - User manually commented out NarrativeQA
    datasets_to_eval = [
        ("glue", "mnli"),
        ("glue", "sst2"),
        ("squad_v2", None),
        ("cnn_dailymail", "3.0.0"),
        # ("narrativeqa", None), 
        ("gsm8k", "main")
    ]
    
    results = []
    
    tracker = EmissionsTracker(
        project_name="ecoprompt_eval",
        measure_power_secs=1,
        save_to_file=False,
        log_level="error"
    )
    
    print("Starting Evaluation Loop...")
    
    for ds_name, subset in datasets_to_eval:
        print(f"\nProcessing {ds_name}...")
        try:
            if subset:
                data = load_dataset(ds_name, subset, split=f"validation[:{args.samples}]" if ds_name != "gsm8k" else f"test[:{args.samples}]")
            else:
                data = load_dataset(ds_name, split=f"test[:{args.samples}]" if ds_name != "squad_v2" else f"validation[:{args.samples}]")
        except Exception as e:
            print(f"Failed to load {ds_name}: {e}")
            continue
            
        for i, item in enumerate(tqdm(data)):
            prompt = ""
            reference = ""
            
            # Construct Prompts
            if ds_name == "glue" and subset == "mnli":
                prompt = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}\nlabel (entailment, neutral, contradiction):"
                reference = item['label']
            elif ds_name == "glue" and subset == "sst2":
                prompt = f"Sentence: {item['sentence']}\nSentiment (positive, negative):"
                reference = item['label']
            elif ds_name == "squad_v2":
                prompt = f"Context: {item['context']}\nQuestion: {item['question']}\nAnswer:"
                reference = item['answers']
            elif ds_name == "cnn_dailymail":
                prompt = f"Summarize:\n{item['article'][:2000]}"
                reference = item['highlights']
            elif ds_name == "narrativeqa":
                prompt = f"Context: {item['document']['summary']['text']}\nQuestion: {item['question']['text']}\nAnswer:"
                reference = str(item['answers'])
            elif ds_name == "gsm8k":
                prompt = f"Question: {item['question']}\nThink step by step:"
                reference = item['answer']
            
            # 1. Determine Category (Dynamic Logic)
            category = get_prompt_category(prompt)
            
            # 2. Select Model based on Category
            model_choice_name, model_weights = select_model_for_category(category)
            model_choice_name="gemini"
            # 3. Compress
            comp_res = compress_prompt(prompt)
            
            #compressed_prompt_text = comp_res["compressed_prompt"]
            compressed_prompt_text = prompt
            # 4. Generate & Track
            emissions = 0.0
            output = ""
            try:
                tracker.start()
                output = generate_prompt(model_weights, compressed_prompt_text, 'gemini')
                emissions = tracker.stop()
            except Exception as e:
                print(f"Tracking/Gen Error: {e}")
                output = "Error"
            
            # 5. Evaluate
            score = 0.0
            score_type = "accuracy"
            
            try:
                if output != "Error":
                    if ds_name == "glue":
                        if subset == "mnli":
                            score = evaluate_mnli(output, reference)
                        elif subset == "sst2":
                            score = evaluate_sst2(output, reference)
                    elif ds_name == "squad_v2":
                        score = evaluate_squad(output, reference)
                        score_type = "EM"
                    elif ds_name == "cnn_dailymail":
                        rouge_scores = evaluate_rouge(output, reference)
                        score = rouge_scores['rougeL'] 
                        score_type = "ROUGE-L"
                    elif ds_name == "gsm8k":
                        score = evaluate_gsm8k(output, reference)
                        score_type = "EM"
            except Exception as e:
                print(f"Eval Error: {e}")
            
            # 6. Log
            results.append({
                "timestamp": datetime.now().isoformat(),
                "dataset": ds_name,
                "prompt": prompt[:500],
                "reference": str(reference)[:500],
                "complexity": category,
                "compressed_prompt": compressed_prompt_text[:500],
                "initial_tokens": comp_res["initial_tokens"],
                "final_tokens": comp_res["final_tokens"],
                "model_chosen": model_choice_name,
                "output": output[:500],
                "carbon_footprint_kg": emissions,
                "evaluation_score": score,
                "score_type": score_type
            })

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")
    else:
        print("No results.")

if __name__ == "__main__":
    main()
