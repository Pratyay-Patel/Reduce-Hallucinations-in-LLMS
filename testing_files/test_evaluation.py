import pandas as pd
import evaluate
import json
import ast
import re

# Load metrics
try:
    accuracy_metric = evaluate.load("accuracy")
    rouge_metric = evaluate.load("rouge")
    squad_metric = evaluate.load("squad_v2")
except Exception as e:
    print(f"Warning: Could not load some metrics. Ensure 'evaluate' and 'bert_score' etc are installed. {e}")

def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def calculate_exact_match(predictions, references):
    """
    Exact Match for QA. 
    References can be a list of valid answers.
    Returns average EM score.
    """
    total = 0
    correct = 0
    for pred, refs in zip(predictions, references):
        # Handle if refs is string representation of list
        if isinstance(refs, str):
            try:
                refs = ast.literal_eval(refs)
            except:
                refs = [refs]
        
        # Normalize
        norm_pred = normalize_text(str(pred))
        norm_refs = [normalize_text(str(r)) for r in refs]
        
        if norm_pred in norm_refs:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def calculate_accuracy(predictions, references):
    """
    Accuracy for classification (SST-2, MNLI).
    """
    # Simple direct comparison after normalization
    clean_preds = [normalize_text(str(p)) for p in predictions]
    clean_refs = [normalize_text(str(r)) for r in references]
    
    # Manual calculation to avoid ValueError with evaluate library if inputs are not ints
    correct = sum(1 for p, r in zip(clean_preds, clean_refs) if p == r)
    total = len(clean_preds)
    return {"accuracy": correct / total if total > 0 else 0.0}

def calculate_rouge(predictions, references):
    """
    ROUGE for Summarization (CNN/DailyMail).
    """
    # Ensure references are strings (sometimes they are lists of strings)
    clean_preds = [str(p) for p in predictions]
    clean_refs = []
    for r in references:
        if isinstance(r, list):
            clean_refs.append(r[0]) # ROUGE usually takes one ref or list
        else:
            clean_refs.append(str(r))
            
    return rouge_metric.compute(predictions=clean_preds, references=clean_refs)

def extract_number(text):
    """Helper for GSM8K to extract the last number"""
    # Simple heuristic: find the last number in the text
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if numbers:
        return numbers[-1]
    return ""

def calculate_gsm8k_em(predictions, references):
    """
    GSM8K Exact Match.
    Compares the extracted numerical answer.
    """
    total = 0
    correct = 0
    for pred, ref in zip(predictions, references):
        pred_num = extract_number(str(pred))
        ref_num = extract_number(str(ref))
        
        if pred_num and ref_num and float(pred_num) == float(ref_num):
            correct += 1
        total += 1
    return correct / total if total > 0 else 0

def evaluate_experiments(csv_file="all_experiments_results.csv"):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File {csv_file} not found. Run experiments first.")
        return

    results_summary = []

    # Group by Dataset and Experiment
    grouped = df.groupby(["Dataset", "Experiment"])

    for (dataset, unique_experiment), group in grouped:
        print(f"Evaluating {dataset} - {unique_experiment}...")
        
        preds = group["Generated_Output"].tolist()
        refs = group["Reference_Output"].tolist()
        
        metrics = {}
        
        if dataset in ["SST-2", "NLI_MNLI"]:
            acc = calculate_accuracy(preds, refs)
            metrics["Accuracy"] = acc["accuracy"]
            
        elif dataset in ["SQuAD_v2"]:
            em = calculate_exact_match(preds, refs)
            metrics["Exact_Match"] = em
            # SQuAD F1 could be added here similar to official script
            
        elif dataset in ["CNN_DailyMail"]:
            rouge = calculate_rouge(preds, refs)
            metrics.update(rouge)
            
        elif dataset in ["GSM8K"]:
            em = calculate_gsm8k_em(preds, refs)
            metrics["Exact_Match"] = em
            
        elif "LongBench" in dataset:
            # Simplified LongBench logic: compute both QA and Summ metrics
            em = calculate_exact_match(preds, refs)
            metrics["Exact_Match"] = em
            try:
                rouge = calculate_rouge(preds, refs)
                metrics.update(rouge)
            except:
                pass
                
        elif dataset in ["MT_Bench", "MT-Bench"]:
            print(f"MT-Bench requires LLM-as-a-Judge. Please run the judge script separately using the outputs in 'Generated_Output'.")
            metrics["Judge_Score"] = "N/A (Run Judge)"
            
        else:
            print(f"No specific metric defined for {dataset}. validation manually.")
            
        # Add to summary
        for m_name, m_val in metrics.items():
            results_summary.append({
                "Dataset": dataset,
                "Experiment": unique_experiment,
                "Metric": m_name,
                "Value": m_val
            })

    # Display results
    print("\n--- Evaluation Results ---")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df)
    
    # Save results
    summary_df.to_csv("evaluation_report.csv", index=False)
    print("Saved evaluation report to evaluation_report.csv")

if __name__ == "__main__":
    evaluate_experiments()
