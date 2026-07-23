"""CPU-safe test of the compression pipeline fix.

Unlike tests/test_pipeline.py (which disables compression entirely), this
test keeps COMPRESSION_ENABLED=True and exercises the real LLMLingua-2
compressor (src/compression.py) against a tiny CPU-friendly LLM, so the
compression fix can be validated without downloading/running any of the
large models in config.LLM_MODELS.

Checks:
    1) init_compression() succeeds (PromptCompressor loads without error).
    2) At least one compressed=1 row is written.
    3) For compressed=1 rows, compressed_tokens < orig_tokens and
       compressed_tokens != 0 (i.e. compression actually ran, rather than
       silently no-op'ing as it did before the fix).

Run directly:
    python tests/test_compression_pipeline.py
"""

import csv
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoModelForCausalLM, AutoTokenizer

import src.runner as runner

TEST_RESULTS_PATH = os.path.join("results", "test_compression_pipeline_output.csv")


def main() -> None:
    print("[INFO] Running CPU-safe compression pipeline test")

    original_models = runner.LLM_MODELS
    original_load_all_datasets = runner.load_all_datasets
    original_load_llm = runner.load_llm
    original_load_nli_model = runner.load_nli_model
    original_load_embedding_model = runner.load_embedding_model
    original_self_consistency_score = runner.self_consistency_score
    original_nli_support_score = runner.nli_support_score
    original_compression_enabled = runner.COMPRESSION_ENABLED
    original_results_path = runner.RESULTS_PATH

    if os.path.exists(TEST_RESULTS_PATH):
        os.remove(TEST_RESULTS_PATH)

    try:
        runner.LLM_MODELS = ["sshleifer/tiny-gpt2"]
        runner.COMPRESSION_ENABLED = True
        runner.RESULTS_PATH = TEST_RESULTS_PATH

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

        # Avoid downloading large NLI/embedding models; unrelated to compression.
        runner.load_nli_model = lambda: (None, None)
        runner.load_embedding_model = lambda: None
        runner.self_consistency_score = lambda responses, emb_model: 1.0 if responses else 0.0
        runner.nli_support_score = lambda nli_model, nli_tokenizer, premise, answer: 1.0

        runner.main()

        with open(TEST_RESULTS_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        compressed_rows = [r for r in rows if r["compressed"] == "1"]

        assert compressed_rows, (
            "No compressed=1 rows were written. init_compression() may have "
            "failed again (check console output above for the traceback)."
        )

        broken = [
            r for r in compressed_rows
            if int(r["compressed_tokens"]) == 0
            or int(r["compressed_tokens"]) >= int(r["orig_tokens"])
        ]

        assert not broken, (
            f"{len(broken)} compressed row(s) show no real reduction "
            f"(compressed_tokens==0 or >= orig_tokens) — compression is "
            f"still silently no-op'ing. Example: {broken[0]}"
        )

        print(f"✅ {len(compressed_rows)} compressed row(s) written, all with real token reduction")
        print("✅ Compression pipeline test passed")

    except Exception as err:
        print(f"❌ Compression pipeline test failed: {err}")
        raise

    finally:
        runner.LLM_MODELS = original_models
        runner.load_all_datasets = original_load_all_datasets
        runner.load_llm = original_load_llm
        runner.load_nli_model = original_load_nli_model
        runner.load_embedding_model = original_load_embedding_model
        runner.self_consistency_score = original_self_consistency_score
        runner.nli_support_score = original_nli_support_score
        runner.COMPRESSION_ENABLED = original_compression_enabled
        runner.RESULTS_PATH = original_results_path

        if os.path.exists(TEST_RESULTS_PATH):
            os.remove(TEST_RESULTS_PATH)


if __name__ == "__main__":
    main()
