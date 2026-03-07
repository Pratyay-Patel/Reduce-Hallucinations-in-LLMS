"""
Runner module for LLM Hallucination + Prompt Compression Pipeline.

This module orchestrates the complete experiment pipeline:
- Loads datasets
- Loads models
- Builds prompts
- Generates answers
- Computes metrics
- Logs results to CSV
"""

import csv
import random
import numpy as np
import torch
import os

from src.config import (
    LLM_MODELS,
    DATA_PATHS,
    RESULTS_PATH,
    SELF_CONSISTENCY_SAMPLES,
    SEED,
    COMPRESSION_ENABLED,
    NLI_HALLUCINATION_THRESHOLD
)

from src.dataset import load_all_datasets
from src.models import (
    load_llm,
    generate_answers,
    load_nli_model,
    load_embedding_model
)

from src.metrics import (
    exact_match,
    keyword_match_score,
    self_consistency_score,
    nli_support_score
)

# Compression safely ignored for now
if COMPRESSION_ENABLED:
    from src.compression import init_compression, maybe_compress_prompt


def set_random_seeds(seed: int):
    """
    Set seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_prompt(context: str, question: str) -> str:
    if context and context.strip():
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Respond with ONLY the final answer.\n"
            f"If the answer is numeric, output only the number.\n"
            f"Do not explain.\n"
            f"Answer:"
        )

    return (
        f"Question: {question}\n"
        f"Respond with ONLY the final answer.\n"
        f"If the answer is numeric, output only the number.\n"
        f"Do not explain.\n"
        f"Answer:"
    )


def main():
    print("Setting random seeds...")
    set_random_seeds(SEED)

    print("Loading NLI model...")
    nli_model, nli_tokenizer = load_nli_model()

    print("Loading embedding model...")
    emb_model = load_embedding_model()

    print("Loading datasets...")
    samples = list(load_all_datasets(DATA_PATHS))

    # # Add long-context dataset (HotpotQA)
    # samples.extend(load_hotpotqa_samples(limit=30))

    print(f"Loaded {len(samples)} samples.")

    results = []

    with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "dataset",
                "model_name",
                "compressed",
                "orig_tokens",
                "compressed_tokens",
                "prediction",
                "exact_match",
                "keyword_match",
                "self_consistency",
                "nli_support",
                "hallucination"
            ]
        )
        writer.writeheader()

        for model_name in LLM_MODELS:
            print(f"\nLoading LLM: {model_name}")
            llm, tokenizer = load_llm(model_name)

            # Compression init only if enabled
            if COMPRESSION_ENABLED:
                init_compression(tokenizer)

            #compression_modes = [False, True] if COMPRESSION_ENABLED else [False]

            compression_modes = [False, True] if COMPRESSION_ENABLED else [False]

            for idx, sample in enumerate(samples, start=1):
                # if sample["dataset"]!="hotpot_qa":
                #     continue
                sid = sample["id"]
                dataset_name = sample.get("dataset", "unknown")
                context = sample.get("context", "")
                question = sample["question"]
                gold = sample.get("answer", "")

                prompt = build_prompt(context, question)

                orig_prompt_tokens = len(tokenizer(prompt).input_ids)
                print(f"Prompt tokens: {orig_prompt_tokens} | dataset: {dataset_name}")

                # ==========================================
                # Run BOTH: uncompressed and compressed
                # ==========================================

                for apply_compression in compression_modes:

                    current_prompt = prompt

                    if apply_compression:
                        current_prompt, _, comp_tokens, compressed = maybe_compress_prompt(prompt)
                        orig_tokens=orig_prompt_tokens
                    else:
                        current_prompt=prompt
                        orig_tokens=orig_prompt_tokens
                        comp_tokens = orig_tokens
                        compressed = False

                    # Generate answers
                    responses = generate_answers(
                        llm,
                        tokenizer,
                        current_prompt,
                        num_return_sequences=SELF_CONSISTENCY_SAMPLES
                    )

                    main_answer = responses[0] if responses else ""

                    # Compute metrics
                    em = exact_match(main_answer, gold)
                    km = keyword_match_score(main_answer, gold)
                    sc = self_consistency_score(responses, emb_model)

                    premise = context if context.strip() else current_prompt
                    
                    # --------------------------------------------------
                    # Truncate premise for NLI model (max ~512 tokens)
                    # --------------------------------------------------
                    premise_tokens = nli_tokenizer.encode(premise)

                    if len(premise_tokens) > 400:
                        premise_tokens = premise_tokens[:400]
                        premise = nli_tokenizer.decode(premise_tokens)

                    # --------------------------------------------------

                    nli_score = nli_support_score(
                        nli_model,
                        nli_tokenizer,
                        premise,
                        main_answer
                    )

                    hallucination = float(nli_score < NLI_HALLUCINATION_THRESHOLD)

                    row = {
                        "id": sid,
                        "dataset": dataset_name,
                        "model_name": model_name,
                        "compressed": int(compressed),
                        "orig_tokens": orig_tokens,
                        "compressed_tokens": comp_tokens,
                        "prediction": main_answer,
                        "exact_match": em,
                        "keyword_match": km,
                        "self_consistency": sc,
                        "nli_support": nli_score,
                        "hallucination": hallucination
                    }

                    writer.writerow(row)
                    results.append(row)

                    print(
                        f"[{idx}/{len(samples)}] "
                        f"{dataset_name} | "
                        f"compressed={int(compressed)} | ",
                        f"tokens={orig_tokens}->{comp_tokens} | ",
                        f"EM={em:.1f} | SC={sc:.3f} | NLI={nli_score:.3f}"
                    )

    # ===============================
    # Aggregation & Summary
    # ===============================
    print("\nComputing aggregated metrics...")

    # ===============================
    # Delta Analysis (Compression Effect)
    # ===============================

    print("\nComputing delta analysis...")

    # Group by sample id
    from collections import defaultdict

    sample_groups = defaultdict(list)

    for r in results:
        sample_groups[r["id"]].append(r)

    delta_rows = []

    for sid, rows in sample_groups.items():
        if len(rows) != 2:
            continue  # skip if missing one mode

        uncompressed = next(r for r in rows if r["compressed"] == 0)
        compressed = next(r for r in rows if r["compressed"] == 1)

        delta_nli = compressed["nli_support"] - uncompressed["nli_support"]
        delta_hall = compressed["hallucination"] - uncompressed["hallucination"]

        delta_rows.append({
            "id": sid,
            "dataset": uncompressed["dataset"],
            "delta_nli": delta_nli,
            "delta_hallucination": delta_hall
        })

    # Compute statistics
    total = len(delta_rows)
    improved = sum(1 for d in delta_rows if d["delta_nli"] > 0)
    worsened = sum(1 for d in delta_rows if d["delta_nli"] < 0)
    unchanged = total - improved - worsened

    
    avg_delta_nli = (
    sum(d["delta_nli"] for d in delta_rows) / total
    if total > 0 else 0.0
    )

    print("\n===== Delta Analysis =====")
    print(f"Total samples: {total}")
    print(f"Improved (NLI ↑): {improved} ({improved/total:.2%})")
    print(f"Worsened (NLI ↓): {worsened} ({worsened/total:.2%})")
    print(f"Unchanged: {unchanged}")
    print(f"Average ΔNLI: {avg_delta_nli:.4f}")

    summary_rows = []

    def compute_group_stats(group_name, rows):
        if not rows:
            return None

        avg_em = sum(r["exact_match"] for r in rows) / len(rows)
        avg_sc = sum(r["self_consistency"] for r in rows) / len(rows)
        avg_nli = sum(r["nli_support"] for r in rows) / len(rows)
        avg_km = sum(r["keyword_match"] for r in rows) / len(rows)
        avg_hallucination = sum(r["hallucination"] for r in rows) / len(rows)

        return {
            "group": group_name,
            "count": len(rows),
            "avg_exact_match": round(avg_em, 4),
            "avg_keyword_match": round(avg_km, 4),
            "avg_self_consistency": round(avg_sc, 4),
            "avg_nli_support": round(avg_nli, 4),
            "hallucination_rate": round(avg_hallucination, 4)
        }

    # Overall
    overall_stats = compute_group_stats("overall", results)
    if overall_stats:
        summary_rows.append(overall_stats)

    # Per dataset
    datasets = set(r["dataset"] for r in results)
    for ds in datasets:
        ds_rows = [r for r in results if r["dataset"] == ds]
        stats = compute_group_stats(f"dataset={ds}", ds_rows)
        if stats:
            summary_rows.append(stats)

    # Per compression condition
    compression_values = set(r["compressed"] for r in results)
    for comp in compression_values:
        comp_rows = [r for r in results if r["compressed"] == comp]
        stats = compute_group_stats(f"compressed={comp}", comp_rows)
        if stats:
            summary_rows.append(stats)

    # Write summary CSV
    summary_path = os.path.join("results", "experiment_summary.csv")
    with open(summary_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "count",
                "avg_exact_match",
                "avg_keyword_match",
                "avg_self_consistency",
                "avg_nli_support",
                "hallucination_rate"
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # Print summary to terminal
    print("\n===== Experiment Summary =====")
    for row in summary_rows:
        print(
            f"{row['group']} | "
            f"n={row['count']} | "
            f"EM={row['avg_exact_match']} | "
            f"KM={row['avg_keyword_match']} | "
            f"SC={row['avg_self_consistency']} | "
            f"NLI={row['avg_nli_support']} | "
            f"HALL={row['hallucination_rate']}"
        )

    print(f"\nSummary saved to: {summary_path}")

    print("\nExperiment completed successfully.")
    print(f"Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

    