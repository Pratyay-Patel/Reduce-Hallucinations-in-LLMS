# Plan: cascade + routing pipeline on the M4 MacBook Air

Scripts that produce the paper's result tables: per-model accuracy and emissions, end-to-end
system accuracy and emissions, all measured with CodeCarbon on Apple Silicon. Design follows
[`FINDINGS.md`](../FINDINGS.md) and [`METHODS.md`](../METHODS.md): gate first, option-likelihood
scoring, confidence for *whether*, embedding router for *where*, held-out test split,
cost-matched random baseline, router overhead charged.

**Status (2026-10-04): implemented and run once.** The first Mac run (`Results_Run1`, 2026-10-01)
produced clean accuracy results but exposed two measurement faults and one dataset fault, all
fixed since: CodeCarbon's CPU term was an 85 W-TDP model and its emissions omitted GPU energy
(§2 now describes the direct `powermetrics` sampler that replaced it); AQuA-RAT sat at chance for
three of four models under letter scoring and was replaced by CommonsenseQA; Llama-3B was never
scored on ARC. A full rerun is needed for the energy tables. See §7 for the unattended script.

---

## 1. Fixed choices

**Models** (fp16, torch MPS, loaded one at a time):

| role | HF id | fp16 | published MMLU |
|---|---|---:|---:|
| `base` | `Qwen/Qwen2.5-1.5B-Instruct` | 3.1 GB | ~60 |
| `qwen3b` | `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB | ~66 |
| `phi4mini` | `microsoft/Phi-4-mini-instruct` | 7.7 GB | ~67 |
| `llama3b` | `meta-llama/Llama-3.2-3B-Instruct` | 6.4 GB | ~63 |

Pool = {`qwen3b`, `phi4mini`, `llama3b`}. Llama needs the gated-repo licence accepted on HF.

**Datasets** (all fixed-option, scored by option likelihood, no generation):

| name | source | split used | n | options |
|---|---|---|---:|---|
| `mmlu` | `cais/mmlu`, 20 subjects × 50 | test | 1,000 | 4 |
| `csqa` | `tau/commonsense_qa` | validation, seeded sample | 500 | 5 |
| `arc` | `allenai/ai2_arc` ARC-Challenge | test, seeded sample | 500 | 3–5 |

MMLU subjects, 5 per group: STEM `abstract_algebra college_mathematics high_school_mathematics
college_physics high_school_chemistry`; humanities `philosophy world_religions moral_scenarios
prehistory high_school_world_history`; social `high_school_psychology sociology econometrics
high_school_macroeconomics us_foreign_policy`; other `clinical_knowledge nutrition management
marketing computer_security`.

Total 1,992 questions × 4 models ≈ 8,000 scored rows. (AQuA-RAT was the third set in the first
run; letter-likelihood scoring cannot do multi-step arithmetic, so it was replaced by CommonsenseQA.) ARC has a variable option count, so the
scorer must accept 3–5 options and the gate must use each question's own 1/k.

**Split.** 70 / 30 train / test, stratified on `dataset × subject`, one fixed seed, written once
to disk. Router and thresholds are fit on train only (5-fold out-of-fold inside train). **Every
number in the paper comes from test.**

**Chat template.** Decided per model by `00_check_env.py` on 200 questions: raw
`Question: … Answer:` vs the model's chat template, keep whichever scores higher, record the
choice in `config.py`.

---

## 2. Measuring energy on the Mac

**What is actually measurable.** On Apple Silicon the only source of power readings is
`powermetrics`, which **needs password-less sudo**. It reports CPU, GPU, ANE and their sum
("Combined Power") as Apple's own on-chip estimate, once per sample. It covers the SoC only:
DRAM, display and SSD are not included, so it is a lower bound on machine energy, consistent
across configurations on one machine and not comparable across devices.

**What CodeCarbon 3.3.1 does on this platform, found in the first run:** its CPU term is a
generic 85 W-TDP load model (floor 8.5 W, which appeared as a constant in every file), only its
GPU term comes from `powermetrics`, RAM is a fixed 3 W heuristic, and its emissions figure omits
the GPU energy entirely (every gCO₂e was ~30% low). So CodeCarbon is **not** the measurement.

**Design.** `common/energy.py` streams `sudo powermetrics --samplers cpu_power,gpu_power -i 1000`
for the whole of every tracked run and integrates each sample's power over its elapsed time.
The combined kWh is the **energy of record** (`soc_kwh`). CodeCarbon still runs alongside for
the country carbon factor (read directly from its tables: 713.4 gCO₂e/kWh for `IND`) and for
its standard report fields; its own kWh is kept in the summaries as `total_kwh` for comparison.
Emissions = `soc_kwh × factor`. `record_kwh()` and `emissions_g()` give every downstream script
one number each, and recompute old summaries the same way.

Setup, once:
```
sudo visudo
# add:  <username> ALL = (root) NOPASSWD: /usr/bin/powermetrics
pip install -U codecarbon
```
- `00_check_env.py` runs a 30 s load and **fails unless powermetrics delivered SoC samples**.
  Without them there is no measurement, only CodeCarbon's model.
- `COUNTRY` is set explicitly in `config.py` so the CO₂ factor is the one the paper states.

**Granularity.** One sample per second; a question takes 0.6–3 s, so per-question tracking
would be noise. Therefore:
- Energy is measured **per model per dataset run** (one tracked session, 500–1,000 questions,
  hundreds to thousands of samples).
- Per-question energy is *attributed*: `E_q = E_run × seconds_q / Σ seconds`. This assumes
  roughly constant power across the run; under throttling it slightly over-charges slow late
  questions. Exact for any group of questions, which is what every table uses. Stated as
  attribution in the paper.
- Per run we also record: idle baseline (60 s with nothing running, before each model), gross
  energy, net = gross − idle × duration, mean power, `powermetrics` thermal pressure and mean
  CPU clock. Report gross; keep net for the appendix.
- Router overhead (MiniLM embedding over the escalated questions) is its own tracked run and
  is **charged to every routed configuration**.

**Conditions.** Plugged in, lid open, brightness fixed, no other apps, models pre-downloaded,
`caffeinate` keeping machine and display awake, 5 min idle between models. The first run
showed the fanless Air at "Heavy" thermal pressure with seconds/question rising 30–50% within
a 3B run and idle at ~11.5 W against 17–21 W mean; report both.

**Repeats.** One full run of everything, plus **3 repeats of the 200-question MMLU subset per
model** to give an energy standard deviation for the tables.

---

## 3. Scripts

```
mac_pipeline/
  config.py              models, datasets, split seed, COUNTRY, chat-template choices, paths
  common/
    scorer.py            option-likelihood scorer, MPS fp16, 3–5 options (port of poc/score_options.py)
    energy.py            tracker context manager, idle baseline, attribution helper
    data.py              load/save questions, split
  00_check_env.py
  01_build_questions.py
  02_score_model.py
  03_gate.py
  04_train_router.py
  05_simulate.py
  06_run_end_to_end.py
  07_report.py
  questions/             questions.csv, split.csv
  scores/                <role>.csv
  energy/                <role>_<dataset>_r<k>.json, idle_*.json, emissions.csv (CodeCarbon's own log)
  router/                embeddings.npy, router.pkl, thresholds.json
  results/               tables (csv + md + one xlsx workbook), figures, end-to-end logs
```

### `00_check_env.py`
- Asserts torch MPS available, fp16 works, `sudo -n powermetrics` succeeds, CodeCarbon
  version, HF access to all four repos (Llama gate).
- Loads each model in turn, scores the first 200 MMLU questions raw and with chat template,
  prints accuracy vs the published number, peak memory, seconds/question, and the energy
  fields. Writes `config.py`'s template choice. Stop if any model is >15 points below
  published — that means a prompt-format problem, not a model problem.

### `01_build_questions.py`
- Pulls the three datasets, normalises to `id, dataset, subject, question, choices (list),
  gold (index)`, drops rows with missing gold or
  <3 options, dedups on question text.
- Writes `questions/questions.csv` with columns
  `id, dataset, subject, source_split, question, choices (JSON list), n_options, gold (index),
  gold_letter, prompt_text (the exact rendered prompt)`, and `questions/split.csv`
  (`id, fold ∈ {train, test}`).
- Prints counts per dataset × subject × fold.

### `02_score_model.py --role R [--limit N] [--dataset D]`
- Loads model `R` from `config.py`; for each dataset: idle baseline → tracked run over all
  questions (train and test) → writes `scores/R.csv`, one row per question:
  `id, dataset, subject, fold, model, chat_template, n_options, gold, pred, pred_letter,
  correct, graded, p_gold, logprobs (JSON list, one per option), prompt_tokens, seconds,
  rendered_prompt (exact text the model saw), peak_mem_gb, energy_kwh_attr, run_id, repeat`
  and `energy/R_D_<repeat>.json`:
  `role, model_id, dataset, repeat, run_id, n, duration_s,
  soc_kwh, soc_cpu_kwh, soc_gpu_kwh, soc_ane_kwh, soc_mean_w, soc_samples` (powermetrics, energy
  of record), `cpu_kwh, gpu_kwh, ram_kwh, total_kwh, codecarbon_gco2e, cpu_w, gpu_w, ram_w`
  (CodeCarbon, for comparison), `carbon_factor_g_per_kwh, gco2e` (= soc_kwh × factor), `country,
  mean_w, idle_w, idle_kwh_over_run, net_kwh, thermal_pressure, mean_cpu_ghz, powermetrics,
  codecarbon_version, timestamp`.
- Re-running a role replaces its rows (same as `poc/run_extra_model.py`).
- Order: `base` first (cheapest; its 200-question subset is the calibration check), then the
  three pool models. Rough time on the Air: base ~15 min, each 3B ~35–45 min.

### `03_gate.py`
- Reads all `scores/*.csv`. For every ordered pair (12) and per dataset: A's errors, B's
  accuracy on them, per-question 1/k chance, one-sided binomial test, chance-corrected
  headroom; PASS / one-way / STOP. Pool-level oracle vs each single model.
- Reuses the logic of `poc/analyze_poc.py`. Outputs `results/gate.csv` + `gate.md`.
- **Decision point.** If no pool member passes both ways against another, the hybrid step is
  skipped and the paper reports a plain cascade (dominance regime). If the base is at chance
  on some dataset, that dataset is flagged; confidence there is noise.

### `04_train_router.py`
- Train split only. Computes frozen `all-MiniLM-L6-v2` embeddings for every question once
  (`router/embeddings.npy`; this run is energy-tracked and its per-question cost saved as the
  router overhead).
- *Whether*: sweeps τ on the base's `graded` confidence with 5-fold OOF; stores the τ grid and
  the τ that maximises train accuracy at each escalation fraction.
- *Where*: for each pool model, logistic regression on `[confidence, PCA-32(embedding)]` →
  P(model correct), 5-fold OOF, regularisation chosen by CV (as in
  `poc/analyze_embedding_router_refine.py`). Also a subject-one-hot baseline router.
- Writes `router/router.pkl`, `router/thresholds.json`, and OOF AUCs to `results/router_train.md`.

### `05_simulate.py`
- Test split, stored scores, attributed energy. For every configuration, accuracy overall and
  per dataset, energy (kWh), emissions (gCO₂e), seconds:
  1. each single model (`base`, `qwen3b`, `phi4mini`, `llama3b`)
  2. oracle over the pool, oracle over base+pool
  3. plain cascade `base → X` for each X, τ grid
  4. hybrid `base → router-picked pool member`, τ grid, with router overhead charged
  5. random escalation at matched rate and cost-matched random line (`poc/analyze_cascade.py`)
  6. cheapest cascade / hybrid point matching each single model's accuracy, degenerate flag
- Charges the base's cost on every question and the escalated model's on escalated ones,
  exactly as `poc/analyze_hybrid.py` does.
- Outputs `results/simulated.csv`, `results/frontier.png` (accuracy vs kWh with the random line).

### `06_run_end_to_end.py --tau T [--config hybrid|cascade:X]`
The live measurement, on the test split, with the deployed decision rule:
- Pass 1 (tracked): `base` on every test question → confidence → escalation set.
- Router pass (tracked): embed escalated questions, pick pool member.
- Pass 2 (tracked, one model at a time): each pool model on the questions assigned to it.
- Reported end-to-end energy = sum of the tracked passes. Grouping by model instead of
  interleaving per query is a memory constraint (all four models resident would need ~23 GB);
  the per-question computation is identical, and the paper says so.
- Writes `results/end_to_end_<config>_<tau>.json` (accuracy, kWh, gCO₂e, escalation rate,
  per-dataset breakdown, per-pass energy) and the per-question decision log
  `results/end_to_end_<config>_<tau>_log.csv`:
  `id, dataset, subject, n_options, gold, base_pred, base_correct, base_conf, escalated,
  router_p_qwen3b, router_p_phi4mini, router_p_llama3b, chosen_model, final_pred,
  final_correct, base_kwh_attr, router_kwh_attr, pool_kwh_attr, total_kwh_attr`.
- Run for: the τ chosen on train for the hybrid; the same τ for the best plain cascade; and
  τ=0 (always base) and τ=1 (always best single) as measured anchors.

### `07_report.py`
Builds the paper tables from `results/`:
- **Table 1** per-model accuracy per dataset and overall; kWh, gCO₂e, seconds per 1,000
  questions; ± from the 3 repeats.
- **Table 2** gate matrix (12 ordered pairs).
- **Table 3** end-to-end: always-base, always-best-single, plain cascade, hybrid, oracle;
  accuracy, energy, emissions, escalation rate, Δ vs best single, Δ vs cost-matched random.
- **Table 4** simulated vs measured energy for the same configurations (validates attribution).
- **Figure 1** accuracy–energy frontier with singles, cascade, hybrid, random line.
- **Figure 2** per-dataset gains.
Markdown + CSV + PNG, all numbers traceable to a results file. Also writes
`results/results.xlsx` — one sheet per table, generated from the same CSVs (openpyxl), so the
workbook and the CSVs can never disagree. The per-question files stay CSV; the workbook holds
the tables only.

---

## 4. Run order and time

| step | script | wall time (est.) |
|---|---|---:|
| 1 | `00_check_env.py` | 20 min |
| 2 | `01_build_questions.py` | 2 min |
| 3 | `02_score_model.py` × 4 | ~2.5 h |
| 4 | `02_score_model.py --limit 200` × 4 × 3 repeats | ~40 min |
| 5 | `03_gate.py` | seconds |
| 6 | `04_train_router.py` | 5 min |
| 7 | `05_simulate.py` | seconds |
| 8 | `06_run_end_to_end.py` × 4 configs | ~1.5 h |
| 9 | `07_report.py` | seconds |

Everything after step 3 is re-runnable from the CSVs without touching a model.

---

## 5. Checks and stop conditions

- `00`: any model >15 points under published MMLU on the 200 subset → fix prompt format before
  scoring 2,000. No powermetrics SoC samples → fix sudoers before anything.
- `02`: `scores/R.csv` has 1,992 rows (993 MMLU + 500 CSQA + 499 ARC), no null `correct`,
  per-dataset accuracy above chance for every model (the base at chance on a dataset makes its
  confidence useless there), and no single predicted letter above ~40% on any dataset (a
  position habit, the AQuA failure mode). `03`/`04`/`05` print a coverage warning if any role is
  missing a dataset; do not proceed past it.
- `03`: at least one pool pair passes both ways → hybrid is worth training. None → report
  cascade only. That is a valid paper result, not a failure.
- `05` vs `06`: simulated and measured end-to-end energy within ~10 %. Larger gap → the
  attribution or the idle baseline is wrong; fix before reporting.
- Every routed configuration is compared to the cost-matched random line. Below the line means
  the router's decisions cost more than they earn; say so.

---

## 6. Known pitfalls

- CodeCarbon on Apple Silicon reports `powermetrics` **software estimates**; Apple says not to
  compare across devices. The paper compares configurations on one machine, which is fine, and
  states this.
- `tracking_mode="process"` does not isolate the GPU on Apple Silicon; use `"machine"` and the
  idle baseline.
- Wall-clock is overhead-dominated below ~2B parameters (FINDINGS §8.4). Energy, not seconds,
  is the cost axis in every table.
- Do not report a cascade point that escalates 0 % or 100 % as a routing result (degenerate
  flag from `poc/analyze_cascade.py`).
- Phi-4-mini's tokenizer adds no leading space before the answer letter under the chat
  template; the scorer's letter formatter handles both cases as `poc/score_options.py` does.

---

## 7. How to run on the Mac

Unattended: `bash mac_pipeline/run_all.sh` runs everything below in order with cooldowns,
logs to `mac_pipeline/logs/`, keeps the Mac awake, and notifies when done. `MODE=minimal`
scores only what Results_Run1 lacks; `resume` skips finished steps. The manual sequence:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r mac_pipeline/requirements-mac.txt
huggingface-cli login                      # Llama-3.2 is a gated repo; accept its licence on HF first
sudo visudo                                # add: <user> ALL = (root) NOPASSWD: /usr/bin/powermetrics

python mac_pipeline/01_build_questions.py                  # 1,992 questions + 70/30 split (already committed)
python mac_pipeline/00_check_env.py                        # env, energy, 200-q format check per model
python mac_pipeline/02_score_model.py --role base
python mac_pipeline/02_score_model.py --role qwen3b
python mac_pipeline/02_score_model.py --role phi4mini
python mac_pipeline/02_score_model.py --role llama3b
for k in 1 2 3; do for r in base qwen3b phi4mini llama3b; do
  python mac_pipeline/02_score_model.py --role $r --dataset mmlu --limit 200 --repeat $k; done; done
python mac_pipeline/03_gate.py                             # decision: hybrid or cascade-only
python mac_pipeline/04_train_router.py
python mac_pipeline/05_simulate.py
python mac_pipeline/06_run_end_to_end.py --config hybrid              # tau* from train
python mac_pipeline/06_run_end_to_end.py --config cascade:<best pool member from gate.md>
python mac_pipeline/06_run_end_to_end.py --config base
python mac_pipeline/06_run_end_to_end.py --config single:<best single from simulated.md>
python mac_pipeline/07_report.py                           # results/report.md + results.xlsx
```

Set `COUNTRY` in `config.py` to where the Mac runs before step 02 (it fixes the CO₂ factor).

Dev-only switches (never needed on the Mac): `MACPIPE_DATA=<dir>` redirects all outputs to a
scratch directory; `MACPIPE_MODEL_OVERRIDE='{"qwen3b": "Qwen/Qwen2.5-0.5B-Instruct"}'` swaps
model ids; `--no-energy` on 02/04/06 skips CodeCarbon.
