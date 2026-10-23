# LoRA adapters: train on Colab, measure on the Mac

One base model, several small adapters (one per subject), evaluated with the existing pipeline.
The base is downloaded by each machine that runs it (Colab for training, the Mac for inference);
only the adapters (~30 MB each) travel, through a private Hugging Face repo.

```
Colab T4                                   Mac
---------                                  ---
base model (downloaded, discarded)         base model (HF cache, already there)
+ 2,000 subject examples                   + adapter from the Hub
-> adapter  --push-->  HF repo  --pull-->  scorer loads base, attaches adapter
   training energy logged                  inference/routing/energy measured as usual
```

## Files

- `train_lora.py` — the training script (plain `transformers.Trainer`, loss on the answer
  tokens only, CodeCarbon energy log, pushes the adapter + `training_summary.json`).
- `colab_train_lora.ipynb` — a thin Colab wrapper around it: parameters, install, HF login, run.
- Mac side: `common/scorer.py` accepts `adapter=`; a role in `config.py` can carry `"adapter"`.

## Step by step

### A. Before Colab (once)

1. Hugging Face account with a **write** token (Settings → Access Tokens → New → Write).
   Optional but convenient: save it in Colab as a secret named `HF_TOKEN` (key icon, left sidebar).
2. If the base is gated (Llama), accept its licence on its model page. Qwen is not gated.
3. Optional: put `mac_pipeline/questions/questions.csv` in `MyDrive/lora_runs/` so evaluation
   questions are excluded from training.

### B. Train one adapter on Colab (repeat per subject)

The notebook is self-contained: the training script is embedded in it, nothing is uploaded.

1. colab.research.google.com → File → Upload notebook → `colab_train_lora.ipynb`.
2. Runtime → Change runtime type → **T4 GPU**.
3. In **1. Parameters** set `BASE`, `SUBJECT`, `HF_USER`. Keep `FOUR_BIT` off for 1.5B/3B.
4. Runtime → **Run all**. Approve the Drive mount; paste the token if no `HF_TOKEN` secret exists.
5. Cell 7 streams the training output (trainable params → exclusion-filter count → loss every
   10 steps) and also writes it to `lora_runs/<repo>/train.log` on Drive.
   Rough times on a T4 for 2,000 examples × 2 epochs: 1.5B ≈ 40 min, 3B ≈ 1–2 h, 7B (QLoRA) ≈ 3–4 h.
   A disconnected session is rerun from cell 7 (it restarts from step 0).
6. Cell 8 prints `training_summary.json` (steps, loss, wall time, kWh, gCO₂e, library versions);
   the same file is inside the adapter repo. Cell 9 re-pushes the adapter from Drive if only the
   upload failed.
7. Confirm the adapter at `https://huggingface.co/<HF_USER>/<base>-lora-<subject>` (private).

Subjects, their training data, and where each adapter is evaluated on the Mac:

| subject | trains on | evaluated on |
|---|---|---|
| `math` | GSM8K train, reason-then-"Answer: N" (2,000 of 7,473) | AQuA-RAT with the reasoning protocol: `dev_cot_test.py --extra qwen3b_math=Qwen/Qwen2.5-3B-Instruct@<repo>` (the main pipeline's datasets are letter-scored; AQuA is not in the bank) |
| `code` | MBPP train (374) topped up to 2,000 from CodeAlpaca-20k Python tasks | `codemmlu` dataset in the bank (489 q, letter-scored), via `02_score_model.py` |
| `knowledge` | MMLU auxiliary_train in the pipeline's exact letter prompt (2,000 of 99,842) | `mmlu`, `csqa`, `arc` via `02_score_model.py` |

With `questions.csv` in `MyDrive/lora_runs/`, the trainer drops any training row whose question is
in the evaluation bank (the auxiliary set contains ARC training items; 0 matches against the
current bank). All three adapters are also scored on every dataset as the forgetting check.

### C. Evaluate on the Mac

**0. Setup, once**
```
git pull
pip install -r mac_pipeline/requirements-mac.txt
huggingface-cli login        # read token: the adapter repo is private
```
The 3B base is already in the Mac's cache from the earlier runs; the adapter (~120 MB) is pulled by
its repo id the first time a role uses it.

**1. Math check, reasoning protocol (~25 min).** Plain 3B against 3B + adapter on the same 30 AQuA
questions:
```
python mac_pipeline/dev_cot_test.py --n 30 --max-new-tokens 512 --out ~/cot_3b --models   --extra qwen3b=Qwen/Qwen2.5-3B-Instruct           qwen3b_math=Qwen/Qwen2.5-3B-Instruct@<HF_USER>/qwen2.5-3b-instruct-lora-math
```
`--models` with nothing after it switches off the default 1.5B entries. Read the `cot` rows of the
summary: `qwen3b_math` vs `qwen3b`. A gain of 10+ points on 30 questions is a real signal; under 5
is not shown either way at that size. If truncation is above 50 %, rerun with `--max-new-tokens 768`.

**2. Forgetting check, letter-scored (~10 min).** In `config.py` uncomment the `qwen3b_math` role
and set its `adapter` to the repo id (leave `POOL` alone for now). Then, on the same seeded 60
questions per dataset:
```
for ds in mmlu csqa arc codemmlu; do
  python mac_pipeline/02_score_model.py --role qwen3b_math --dataset $ds --limit 60 --no-energy
  python mac_pipeline/02_score_model.py --role qwen3b      --dataset $ds --limit 60 --no-energy
done
```
The adapter should be within ~1 point of the plain 3B on `mmlu`, `csqa`, `arc`. A drop of 3+
means it overwrote general knowledge: retrain with one epoch or a lower learning rate first.
(The math adapter was trained in the chat format and these datasets use the raw prompt, as the
`qwen3b` role does; that is the right comparison for forgetting, but do not expect a gain here.)

**3. Full run, only if 1 and 2 pass.** Add the role to `POOL`, then
```
bash mac_pipeline/run_all.sh
```
Roles and datasets are read from `config.py`, so the adapter role is scored, gated, routed and
energy-measured like every other model. Report: `mac_pipeline/results/report.md`.

The same three steps apply to the `code` adapter (check on `codemmlu` instead of AQuA, step 1 via
`02_score_model.py`) and the `knowledge` adapter (check on `mmlu`/`csqa`/`arc`; for it, step 2
*is* the check).

### D. Reporting

- Inference energy of an adapter role is measured like any model (the LoRA adds 1–3 %).
- Training energy is in `training_summary.json`; report it separately and amortised over the
  questions served (e.g. per 1,000 questions), the same way the router's own overhead is charged.

## If training goes wrong

- **Loss becomes NaN** on the T4 (fp16, no bf16): rerun with `FOUR_BIT=True`. The base is held in
  4-bit with fp16 compute, which is numerically tamer; the adapter still loads on the fp16 base.
- **`CUDA out of memory`**: lower `N_EXAMPLES` has no effect on memory; instead pass a smaller
  batch by editing the Train cell's command (`"--batch", "2", "--grad-accum", "8"` keeps the same
  effective batch of 16).
- **Session disconnected**: rerun the Train cell; it starts from step 0 (cheap at this size).

## Dev-only switches

`MACPIPE_ADAPTER_OVERRIDE='{"base": "/local/adapter/dir"}'` attaches an adapter to a role
without editing the config (used for the smoke test on the Windows box).

## Smoke test done on the RTX 3050 (2026-10-22)

`train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --subject math --n 64 --max-steps 8 --four-bit`
ran end to end locally (17 s, loss 0.86, energy logged, 74 MB adapter) and the adapter loaded
through the scorer on 5 ARC questions. The T4 path (fp16, no `--four-bit`) is the same code with a
different flag and is untested here. On Windows, pass `HF_HOME=C:/tmp/hf HF_HUB_CACHE=C:/tmp/hf`
explicitly when scoring with a local adapter; on the Mac the default cache is fine.
