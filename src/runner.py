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
    HF_TOKEN,
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
    load_embedding_model,
    clear_gpu
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


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _checkpoint_key(sample_id, dataset_name: str, model_name: str, compressed) -> tuple:
    return (str(sample_id), str(dataset_name), str(model_name), _to_int(compressed))


def _is_oom_error(err: Exception) -> bool:
    msg = str(err).lower()
    oom_markers = [
        "out of memory",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cuda error",
    ]
    return any(marker in msg for marker in oom_markers)


def _hard_truncate_prompt(tokenizer, prompt: str, max_tokens: int = 1024) -> str:
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) <= max_tokens:
        return prompt
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=False)


def load_existing_results(path: str):
    """
    Load existing result rows for checkpoint resume and full-run aggregation.
    """
    checkpoint_keys = set()
    parsed_rows = []

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return checkpoint_keys, parsed_rows

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # If header is missing/corrupt, skip parsing and start fresh appends.
        if reader.fieldnames is None:
            return checkpoint_keys, parsed_rows

        for row in reader:
            sid = row.get("id", "")
            dataset_name = row.get("dataset", "unknown")
            model_name = row.get("model_name", "")
            compressed = _to_int(row.get("compressed", 0))

            checkpoint_keys.add(_checkpoint_key(sid, dataset_name, model_name, compressed))

            parsed_rows.append({
                "id": sid,
                "dataset": dataset_name,
                "model_name": model_name,
                "compressed": compressed,
                "orig_tokens": _to_int(row.get("orig_tokens", 0)),
                "compressed_tokens": _to_int(row.get("compressed_tokens", 0)),
                "prediction": row.get("prediction", ""),
                "exact_match": _to_float(row.get("exact_match", 0.0)),
                "keyword_match": _to_float(row.get("keyword_match", 0.0)),
                "self_consistency": _to_float(row.get("self_consistency", 0.0)),
                "nli_support": _to_float(row.get("nli_support", 0.0)),
                "hallucination": _to_float(row.get("hallucination", 0.0)),
                "run_id": row.get("run_id", ""),
                "timestamp": row.get("timestamp", ""),
            })

    return checkpoint_keys, parsed_rows


def main():
    from datetime import datetime

    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    TIMESTAMP = datetime.now().isoformat()

    if any("meta-llama" in m for m in LLM_MODELS) and not HF_TOKEN:
        print("[ERROR] HF_TOKEN not found. Gated models will fail.")

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
    print(f"Running {len(samples)} samples across {len(LLM_MODELS)} models...")

    checkpoint_keys, existing_results = load_existing_results(RESULTS_PATH)
    results = list(existing_results)
    skipped_count = 0
    written_count = 0
    error_count = 0

    results_dir = os.path.dirname(RESULTS_PATH) or "results"
    os.makedirs(results_dir, exist_ok=True)

    file_exists = os.path.exists(RESULTS_PATH)

    if file_exists:
        print("[INFO] Appending results to existing CSV file.")
        print(f"[INFO] Loaded {len(checkpoint_keys)} completed checkpoints.")
    else:
        print("[INFO] Creating new results CSV file.")

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
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
                "hallucination",
                "run_id",
                "timestamp",
            ]
        )
        if not file_exists or os.stat(RESULTS_PATH).st_size == 0:
            writer.writeheader()

        for model_name in LLM_MODELS:
            print(f"\nLoading LLM: {model_name}")
            try:
                llm, tokenizer = load_llm(model_name)
            except Exception as err:
                print(f"[WARN] Skipping model '{model_name}' due to load failure: {err}")
                clear_gpu()
                continue

            # Compression init only if enabled
            if COMPRESSION_ENABLED:
                init_compression(tokenizer)

            #compression_modes = [False, True] if COMPRESSION_ENABLED else [False]

            compression_modes = [False, True] if COMPRESSION_ENABLED else [False]

            seen_datasets = set()

            for idx, sample in enumerate(samples, start=1):
                # Run ALL datasets (small + large).
                sid = sample["id"]
                dataset_name = sample.get("dataset", "unknown")
                context = sample.get("context", "")
                question = sample["question"]
                gold = sample.get("answer", "")

                if idx == 1 or dataset_name not in seen_datasets:
                    print(f"Starting dataset: {dataset_name}")
                    seen_datasets.add(dataset_name)

                prompt = build_prompt(context, question)

                orig_prompt_tokens = len(tokenizer(prompt).input_ids)
                print(f"[INFO] Tokens: {orig_prompt_tokens} | Dataset: {dataset_name}")

                # ==========================================
                # Run BOTH: uncompressed and compressed
                # ==========================================

                for apply_compression in compression_modes:
                    checkpoint = _checkpoint_key(sid, dataset_name, model_name, int(apply_compression))
                    if checkpoint in checkpoint_keys:
                        skipped_count += 1
                        continue

                    current_prompt = prompt
                    orig_tokens = orig_prompt_tokens
                    comp_tokens = orig_tokens
                    compressed = bool(apply_compression)

                    try:
                        if apply_compression:
                            current_prompt, _, comp_tokens, compressed = maybe_compress_prompt(prompt)

                        # Generate answers
                        try:
                            responses = generate_answers(
                                llm,
                                tokenizer,
                                current_prompt,
                                num_return_sequences=SELF_CONSISTENCY_SAMPLES
                            )
                        except RuntimeError as gen_err:
                            if not _is_oom_error(gen_err):
                                raise

                            clear_gpu()
                            retry_prompt = _hard_truncate_prompt(tokenizer, current_prompt, max_tokens=1024)
                            if retry_prompt == current_prompt:
                                raise

                            current_prompt = retry_prompt
                            comp_tokens = len(tokenizer(current_prompt).input_ids)
                            print(
                                "[WARN] OOM during generation. Retrying with hard-truncated prompt "
                                f"({comp_tokens} tokens)."
                            )
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

                        if nli_tokenizer is None:
                            premise_tokens = []
                        else:
                            premise_tokens = nli_tokenizer.encode(premise)

                        if len(premise_tokens) > 400:
                            premise_tokens = premise_tokens[:400]
                            premise = nli_tokenizer.decode(premise_tokens)

                        if nli_tokenizer is None or nli_model is None:
                            nli_score = 0.0
                        else:
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
                            "hallucination": hallucination,
                            "run_id": RUN_ID,
                            "timestamp": TIMESTAMP,
                        }

                        writer.writerow(row)
                        f.flush()
                        checkpoint_keys.add(checkpoint)
                        results.append(row)
                        written_count += 1

                        print(
                            f"[{idx}/{len(samples)}] "
                            f"{dataset_name} | "
                            f"compressed={int(compressed)} | ",
                            f"tokens={orig_tokens}->{comp_tokens} | ",
                            f"EM={em:.1f} | SC={sc:.3f} | NLI={nli_score:.3f}"
                        )

                    except Exception as err:
                        error_count += 1
                        clear_gpu()

                        # Persist a sentinel row so resume logic will not repeatedly re-hit
                        # the same failing sample/mode across restarts.
                        err_prediction = f"[SKIPPED_ERROR] {type(err).__name__}: {str(err)[:180]}"
                        row = {
                            "id": sid,
                            "dataset": dataset_name,
                            "model_name": model_name,
                            "compressed": int(apply_compression),
                            "orig_tokens": orig_tokens,
                            "compressed_tokens": comp_tokens,
                            "prediction": err_prediction,
                            "exact_match": 0.0,
                            "keyword_match": 0.0,
                            "self_consistency": 0.0,
                            "nli_support": 0.0,
                            "hallucination": 1.0,
                            "run_id": RUN_ID,
                            "timestamp": TIMESTAMP,
                        }
                        writer.writerow(row)
                        f.flush()
                        checkpoint_keys.add(checkpoint)
                        results.append(row)
                        written_count += 1

                        print(
                            f"[WARN] Skipping sample after error: id={sid} | dataset={dataset_name} "
                            f"| model={model_name} | compressed={int(apply_compression)} | "
                            f"error={type(err).__name__}: {err}"
                        )

            # Free memory between model runs to avoid fragmentation/OOM when
            # running multiple large models sequentially in one process.
            del llm
            del tokenizer
            clear_gpu()

    # ===============================
    # Aggregation & Summary
    # ===============================
    print("\nComputing aggregated metrics...")
    print(f"[INFO] New rows written this run: {written_count}")
    print(f"[INFO] Rows skipped via checkpoint resume: {skipped_count}")
    print(f"[INFO] Rows marked as skipped due to runtime errors: {error_count}")

    # ===============================
    # Delta Analysis (Compression Effect)
    # ===============================

    print("\nComputing delta analysis...")

    # Group by sample id
    from collections import defaultdict

    sample_groups = defaultdict(list)

    for r in results:
        group_key = (r["id"], r["dataset"], r["model_name"])
        sample_groups[group_key].append(r)

    delta_rows = []

    for _, rows in sample_groups.items():
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
    # print(f"Improved (NLI ↑): {improved} ({improved/total:.2%})")
    # print(f"Worsened (NLI ↓): {worsened} ({worsened/total:.2%})")

    if total > 0:
        print(f"Improved (NLI ↑): {improved} ({improved/total:.2%})")
        print(f"Worsened (NLI ↓): {worsened} ({worsened/total:.2%})")
    else:
        print("No delta comparison available (compression disabled or insufficient data).")
        
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

    