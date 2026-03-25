# GPU Node Setup and Run Guide

## 1. Prerequisites on the GPU Node

- Linux GPU node with NVIDIA driver installed
- CUDA available to PyTorch
- Git installed
- Python 3.10+ (3.11 recommended)
- Internet access for Hugging Face model downloads

## 2. Clone the Repository from Main

```bash
git clone -b main https://github.com/Pratyay-Patel/Reduce-Hallucinations-in-LLMS.git
cd Reduce-Hallucinations-in-LLMS
```

If the repository already exists on the node:

```bash
cd Reduce-Hallucinations-in-LLMS
git fetch origin
git checkout main
git pull origin main
```

## 3. Create and Activate Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Configure Hugging Face Token (Required for LLaMA Models)

Create a `.env` file in the repository root:

```bash
cat > .env << 'EOF'
HF_TOKEN=your_token_here
EOF
```

Notes:
- Use a token that has access to gated models (Meta LLaMA family).
- No `huggingface-cli login` is required in this project flow.

## 6. Confirm GPU Visibility

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected:
- `CUDA: True`
- GPU name printed (for example, L40)

## 7. (Optional but Recommended) Verify Setup Before Full Run

```bash
python tests/test_all.py
```

This does three checks:
- Environment and token presence
- Model access check via Hugging Face API
- Lightweight CPU-safe pipeline smoke test

## 8. Run the Full Pipeline on GPU

Use either command below (both are valid):

```bash
python -m src.runner
```

or

```bash
python runner.py
```

For long runs, capture logs:

```bash
mkdir -p logs
python -m src.runner 2>&1 | tee logs/pipeline_$(date +%Y%m%d_%H%M%S).log
```

## 9. Output Files

Main outputs are written in `results/`:

- `results/experiment_results.csv`
- `results/experiment_summary.csv`

`experiment_results.csv` is configured to append across runs.
Each run now includes metadata columns such as `run_id` and `timestamp`.

## 10. Dataset Size Note

The prepared subset files under `data/` are expected to contain 1000 samples each.
If you need to regenerate them:

```bash
python prepare_datasets.py
```

## 11. Common Issues and Fixes

### A) Gated model access failure (LLaMA models)

Symptom:
- Errors mentioning access denied / gated repo.

Fix:
- Ensure `.env` exists at repo root.
- Ensure `HF_TOKEN` is valid and has accepted access for:
  - `meta-llama/Llama-2-7b-chat-hf`
  - `meta-llama/Meta-Llama-3-8B-Instruct`

### B) CUDA not available

Symptom:
- `CUDA: False`

Fix:
- Check NVIDIA driver installation.
- Ensure node has GPU resources allocated.
- Reinstall the correct CUDA-enabled PyTorch build if needed.

### C) Out-of-memory (OOM)

Fix options:
- Re-run after ensuring no other large GPU jobs are active.
- In `src/config.py`, set:
  - `USE_4BIT = True`
- Keep models running sequentially (already handled by pipeline).

## 12. Operator Checklist

- Pulled latest `main` branch
- Activated virtual environment
- Installed requirements
- Added `.env` with valid `HF_TOKEN`
- Confirmed CUDA is available
- Ran `python -m src.runner`
- Verified CSV outputs in `results/`
