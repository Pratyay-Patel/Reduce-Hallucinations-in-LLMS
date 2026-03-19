"""Lightweight pipeline execution test.

This test avoids GPU and large model downloads by temporarily monkey-patching
runner dependencies to CPU-safe/lightweight alternatives.

Run directly:
    python tests/test_pipeline.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer

import src.runner as runner


def main() -> None:
    print("[INFO] Running lightweight pipeline test (CPU-safe)")

    original_models = runner.LLM_MODELS
    original_load_all_datasets = runner.load_all_datasets
    original_load_llm = runner.load_llm
    original_load_nli_model = runner.load_nli_model
    original_load_embedding_model = runner.load_embedding_model
    original_self_consistency_score = runner.self_consistency_score
    original_nli_support_score = runner.nli_support_score
    original_compression_enabled = runner.COMPRESSION_ENABLED

    try:
        runner.LLM_MODELS = ["sshleifer/tiny-gpt2"]
        runner.COMPRESSION_ENABLED = False

        def limited_load_all_datasets(paths):
            for idx, sample in enumerate(original_load_all_datasets(paths)):
                if idx >= 3:
                    break
                yield sample

        def tiny_cpu_llm_loader(model_name):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            model.eval()
            return model, tokenizer

        runner.load_all_datasets = limited_load_all_datasets
        runner.load_llm = tiny_cpu_llm_loader

        # Avoid downloading large NLI/embedding models for smoke-test execution.
        runner.load_nli_model = lambda: (None, None)
        runner.load_embedding_model = lambda: None
        runner.self_consistency_score = lambda responses, emb_model: 1.0 if responses else 0.0
        runner.nli_support_score = lambda nli_model, nli_tokenizer, premise, answer: 1.0

        runner.main()
        print("✅ Lightweight pipeline test completed")

    except Exception as err:
        print(f"❌ Pipeline test failed: {err}")

    finally:
        runner.LLM_MODELS = original_models
        runner.load_all_datasets = original_load_all_datasets
        runner.load_llm = original_load_llm
        runner.load_nli_model = original_load_nli_model
        runner.load_embedding_model = original_load_embedding_model
        runner.self_consistency_score = original_self_consistency_score
        runner.nli_support_score = original_nli_support_score
        runner.COMPRESSION_ENABLED = original_compression_enabled


if __name__ == "__main__":
    main()
