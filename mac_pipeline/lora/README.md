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
2. If the base is gated (Llama), accept its licence on its model page. Qwen is not gated.
3. Push this repo's `mac_pipeline/lora/train_lora.py` somewhere Colab can fetch it, or plan to
   upload the file through Colab's Files panel (left sidebar, upload icon).

### B. Train one adapter on Colab (repeat per subject)

1. Open `colab_train_lora.ipynb` in Colab (File → Upload notebook).
2. Runtime → Change runtime type → **T4 GPU**.
3. In the **Parameters** cell set `BASE`, `SUBJECT`, `HF_USER`. Keep `FOUR_BIT=False` for
   1.5B/3B so the adapter matches the fp16 base the Mac serves; set it True only for the 7B.
4. Runtime → **Run all**. Paste the write token when the login cell asks.
5. If the fetch cell stops, upload `train_lora.py` through the Files panel and run the cell again.
6. Wait. Rough times on a T4 for 2,000 examples × 2 epochs: 1.5B ≈ 40 min, 3B ≈ 1–2 h,
   7B (QLoRA) ≈ 3–4 h. Checkpoints go to Drive every 200 steps if `USE_DRIVE` is on, so a
   disconnected session can be resumed by re-running (the trainer picks up the last checkpoint
   only if you pass `--resume_from_checkpoint`; simplest is to rerun from scratch for a 1.5B).
7. The last cell prints `training_summary.json`: steps, loss, wall time, kWh and gCO₂e of the
   training run, library versions. The same file is inside the adapter repo.
8. Confirm the adapter appears at `https://huggingface.co/<HF_USER>/<base>-lora-<subject>`.

Subjects, their training data, and where each adapter is evaluated on the Mac:

| subject | trains on | evaluated on |
|---|---|---|
| `math` | GSM8K train, reason-then-"Answer: N" (2,000 of 7,473) | AQuA-RAT with the reasoning protocol: `dev_cot_test.py --extra qwen3b_math=Qwen/Qwen2.5-3B-Instruct@<repo>` (the main pipeline's datasets are letter-scored; AQuA is not in the bank) |
| `code` | MBPP train (374) topped up to 2,000 from CodeAlpaca-20k Python tasks | `codemmlu` dataset in the bank (489 q, letter-scored), via `02_score_model.py` |
| `knowledge` | MMLU auxiliary_train in the pipeline's exact letter prompt (2,000 of 99,842) | `mmlu`, `csqa`, `arc` via `02_score_model.py` |

Upload `mac_pipeline/questions/questions.csv` next to the script in Colab and the trainer drops any
training row whose question is in the evaluation bank (the auxiliary set contains ARC training
items; 0 matches against the current bank). All three adapters are also scored on every dataset
as the forgetting check.

### C. Evaluate on the Mac

1. `pip install -r mac_pipeline/requirements-mac.txt` (adds `peft`), then `huggingface-cli login`
   with a **read** token so the private adapter repo is reachable.
2. In `mac_pipeline/config.py`, uncomment a LoRA role and set its `adapter` to the repo id.
   The `id` must be the same base the adapter was trained on. Add the role to `POOL` if it is a
   pool member.
3. Quick sanity check on 60 questions before anything long:
   ```
   python mac_pipeline/02_score_model.py --role qwen3b_math --dataset mmlu --limit 60 --no-energy
   ```
   and compare to the plain `qwen3b` on the same 60 (`--limit` is seeded, so the questions match).
4. Regression check: the adapter must not lose more than ~1 point on the datasets it was not
   trained for. If it does, lower the learning rate or the epochs and retrain.
5. Full run: `bash mac_pipeline/run_all.sh` as before. The adapter roles are scored, gated,
   routed and energy-measured exactly like any other role.

### D. Reporting

- Inference energy of an adapter role is measured like any model (the LoRA adds 1–3 %).
- Training energy is in `training_summary.json`; report it separately and amortised over the
  questions served (e.g. per 1,000 questions), the same way the router's own overhead is charged.

## Dev-only switches

`MACPIPE_ADAPTER_OVERRIDE='{"base": "/local/adapter/dir"}'` attaches an adapter to a role
without editing the config (used for the smoke test on the Windows box).

## Smoke test done on the RTX 3050 (2026-10-22)

`train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --subject math --n 64 --max-steps 8 --four-bit`
ran end to end locally (17 s, loss 0.86, energy logged, 74 MB adapter) and the adapter loaded
through the scorer on 5 ARC questions. The T4 path (fp16, no `--four-bit`) is the same code with a
different flag and is untested here. On Windows, pass `HF_HOME=C:/tmp/hf HF_HUB_CACHE=C:/tmp/hf`
explicitly when scoring with a local adapter; on the Mac the default cache is fine.
