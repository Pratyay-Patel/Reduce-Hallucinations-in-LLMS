# LLM Hallucination + Prompt Compression Pipeline

An experimental pipeline to measure the relationship between prompt compression and LLM hallucination rates. This system processes benchmark datasets (GSM8K, SQuAD v2), optionally compresses prompts using LLMLingua-2, generates answers using Phi-3-mini-4k-instruct, and computes three hallucination proxy metrics: exact match accuracy, self-consistency, and NLI entailment support.

## Requirements

- Python 3.8+
- CUDA-capable GPU with at least 16GB memory
- HuggingFace account and API token

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd llm-hallucination-compression-pipeline
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your HuggingFace token:
   - Create a `.env` file in the project root
   - Add your token: `HUGGINGFACE_TOKEN=your_token_here`
   - Or set the environment variable: `export HUGGINGFACE_TOKEN=your_token_here`

## Usage

Run the complete experiment pipeline:

```bash
python -m src.runner
```

This will:
1. Load the configured datasets from `data/`
2. Process each sample through the LLM with optional prompt compression
3. Generate multiple answers for self-consistency measurement
4. Compute all three hallucination metrics
5. Save results to `results/experiment_results.csv`

## Configuration

Edit `src/config.py` to adjust:
- Model names and paths
- Compression threshold
- Generation parameters (temperature, top_p, max_new_tokens)
- Number of samples for self-consistency
- Random seed for reproducibility

## Output

Results are saved to `results/experiment_results.csv` with the following columns:
- `id`: Sample identifier
- `dataset`: Dataset name (gsm8k, squad_v2)
- `model_name`: LLM model used
- `compressed`: Whether compression was applied (1 or 0)
- `orig_tokens`: Original prompt token count
- `compressed_tokens`: Final prompt token count
- `prediction`: Generated answer
- `exact_match`: Exact match score (0.0 or 1.0)
- `self_consistency`: Self-consistency score [0, 1]
- `nli_support`: NLI entailment score [0, 1]

## Sample Data

The `data/` directory includes sample datasets:
- `gsm8k_subset.jsonl`: 5 math reasoning questions
- `squad_v2_subset.jsonl`: 3 reading comprehension questions

## Project Structure

```
.
├── data/                    # Dataset files (JSONL format)
├── results/                 # Experiment results (CSV)
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py           # Configuration parameters
│   ├── dataset.py          # Dataset loading
│   ├── models.py           # Model loading and inference
│   ├── compression.py      # Prompt compression
│   ├── metrics.py          # Metric computation
│   └── runner.py           # Main pipeline orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Metrics

### Exact Match
Binary correctness indicator comparing generated answer to ground truth (1.0 = correct, 0.0 = incorrect).

### Self-Consistency
Average pairwise cosine similarity between multiple generated answers. Higher scores indicate more consistent responses, suggesting lower hallucination risk.

### NLI Support
Probability that the context entails the generated answer using a Natural Language Inference model. Higher scores indicate the answer is better supported by the input context.

## License

[Add your license here]
