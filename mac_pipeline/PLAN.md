# Plan: route-then-cascade pipeline on the M4 MacBook Air

Scripts that produce the paper's result tables: per-model accuracy and emissions, end-to-end
system accuracy and emissions, all measured with `powermetrics` on Apple Silicon. Design follows
[`FINDINGS.md`](../FINDINGS.md) and [`METHODS.md`](../METHODS.md): gate first, option-likelihood
scoring, embedding router for *where*, confidence for *whether*, held-out test split,
cost-matched random baseline, router overhead charged.

**Decision rule (Oct 24, replaces the Run2 design).** Three models. A router reads the
question text and sends it to one of two 1.5B models, the generalist (Qwen2.5-1.5B-Instruct)
or the math specialist (Qwen2.5-Math-1.5B-Instruct). That model answers. If its confidence is
below the threshold for that route, the 3B model (Qwen2.5-3B-Instruct) answers instead.
Run2 did it the other way round (base answers first, then a router chose among three 3B
models); its results are in `RESULTS_RUN2.md`. Phi-4-mini and Llama-3.2-3B are no longer used.

**Status (2026-10-04): implemented and run once.** The first Mac run (`Results_Run1`, 2026-10-01)
produced clean accuracy results but exposed two measurement faults and one dataset fault, all
fixed since: CodeCarbon's CPU term was an 85 W-TDP model and its emissions omitted GPU energy
(§2 now describes the direct `powermetrics` sampler that replaced it); AQuA-RAT sat at chance for
three of four models under letter scoring and was replaced by CommonsenseQA; Llama-3B was never
scored on ARC. A full rerun is needed for the energy tables. See §7 for the unattended script.

---

## 1. Fixed choices

**Models** (fp16, torch MPS, loaded one at a time):

| role | HF id | fp16 | published MMLU | job |
|---|---|---:|---:|---|
| `generalist` | `Qwen/Qwen2.5-1.5B-Instruct` | 3.1 GB | ~60 | routed tier; also the plain cascade's first model |
| `math` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | 3.1 GB | not published | routed tier |
| `qwen3b` | `Qwen/Qwen2.5-3B-Instruct` | 6.2 GB | ~66 | escalation |

Routed tier = {`generalist`, `math`}; escalation = `qwen3b`. No gated repos. The math model
is at chance on non-maths questions in raw letter format (0.35 on a 60-question MMLU sample,
chance 0.25) and below that with the chat template, so the router has to keep it on maths;
that is the point of the router. (Run2 models: `base`=the generalist, pool `qwen3b`,
`phi4mini`, `llama3b`.)

**Datasets** (fixed-option; three by option likelihood, AQuA by reason-then-letter):

| name | source | split used | n | options | scoring |
|---|---|---|---:|---|---|
| `mmlu` | `cais/mmlu`, 20 subjects × 50 | test | 1,000 | 4 | letter |
| `csqa` | `tau/commonsense_qa` | validation, seeded sample | 500 | 5 | letter |
| `arc` | `allenai/ai2_arc` ARC-Challenge | test, seeded sample | 500 | 3–5 | letter |
| `aqua` | `deepmind/aqua_rat` | test (all) | 254 | 5 | reason, then letter |

**AQuA and the reasoning protocol.** Plain letter scoring cannot do multi-step arithmetic
(Run1: three of four models at chance with a position habit), so on AQuA every model first
writes its reasoning through the chat template (greedy, at most 512 new tokens), then the
text `Therefore, the answer is (` is appended and the option letters are scored exactly as on
the other datasets. Nothing is parsed from the reasoning; `graded` is the same kind of
confidence. Cost is the difference: ~25 s per question for a 1.5B model and ~50 s for the 3B
on the Air, against under a second for a letter score, so AQuA is most of the run time. The
scores file keeps the reasoning text, its token count, whether it hit the cap, and the letter
the model stated in text (for the record; it is not used).

Two code datasets were built and then dropped from the workflow (Oct 24): `codemmlu` (489
letter-scored knowledge questions) and `codegen` (HumanEval + MBPP, 421 problems scored by
running unit tests). A test showed the 1.5B coder specialist below the 1.5B generalist on
CodeMMLU, and code generation costs ~30× a letter score (about 16 h of the run). The builders
in `01` and the generate-and-test scorer (`common/execute.py`, `OptionScorer.score_code`) stay
in the code; listing the names in `config.DATASETS` and rerunning `01` brings them back.

MMLU subjects, 5 per group: STEM `abstract_algebra college_mathematics high_school_mathematics
college_physics high_school_chemistry`; humanities `philosophy world_religions moral_scenarios
prehistory high_school_world_history`; social `high_school_psychology sociology econometrics
high_school_macroeconomics us_foreign_policy`; other `clinical_knowledge nutrition management
marketing computer_security`.

Total 2,246 questions × 3 models ≈ 6,700 scored rows. ARC has a variable option count, so the
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

**Repeats.** Dropped (Oct 24). Run2 had 3 back-to-back repeats of 200 MMLU questions per
model; they agreed with each other to 1% but ran at ~2× the power of the main run, so the ±
they gave was not the real run-to-run spread (30–44% between runs hours apart). The spread is
now read from the end-to-end passes, which score the same models on the same test questions
hours after the scoring step (`07`, Table 4). `REPEATS=3` in `run_all.sh` brings them back.

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
  gold_letter, prompt_text (the exact rendered prompt), test_code (codegen only: JSON with
  the entry point and tests, or the asserts and imports)`, and `questions/split.csv`
  (`id, fold ∈ {train, test}`). Codegen rows have `choices = []`, `n_options = 0`, `gold = -1`.
- Prints counts per dataset × subject × fold.

### `02_score_model.py --role R [--limit N] [--dataset D]`
- Loads model `R` from `config.py`; for each dataset: idle baseline → tracked run over all
  questions (train and test) → writes `scores/R.csv`, one row per question:
  `id, dataset, subject, fold, model, chat_template, n_options, gold, pred, pred_letter,
  correct, graded, p_gold, logprobs (JSON list, one per option), prompt_tokens, seconds,
  rendered_prompt (exact text the model saw), peak_mem_gb, generated, gen_tokens, truncated,
  exec_status, exec_seconds (the last five are codegen only: the reply, its length, whether it
  hit the token cap, pass/fail/timeout/error from the test run, and the test run's own time,
  which is not counted in `seconds` or in the model's energy), energy_kwh_attr, run_id, repeat`
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
- **Decision point.** Routing gate: the generalist and the math model must be complementary
  (each fixes some of the other's errors above chance), otherwise a router cannot beat the
  dominant one and the route-only line in 05 should sit on it. Escalation gate: the 3B must fix
  the routed models' errors above chance. If a model is at chance on some dataset, that dataset
  is flagged; confidence there is noise.

### `04_train_router.py`
- Train split only. Computes frozen `all-MiniLM-L6-v2` embeddings for every question once
  (`router/embeddings.npy`; this run is energy-tracked and its per-question cost saved as the
  router overhead, charged on every question since the router runs before any model).
- *Where* (ex-ante, text only): for each routed model, a classifier on `PCA-32(embedding)` →
  P(model right), 5-fold OOF, regularisation by CV; subject-one-hot baseline; head ablation
  (lr, rf, xgb, svc, mlp, voting) selected by OOF routed accuracy. The router picks the argmax.
  Reports where the picks go per dataset (the math model should get AQuA and little else).
- *Whether*: for each route, on the questions the OOF router sent there, sweeps τ on that
  model's confidence and keeps τ[route] (best accuracy, ties to least escalation). Also one
  global τ (ablation) and the τ of the plain cascade generalist → 3B.
- Writes `router/router.pkl`, `router/thresholds.json` (`tau_per_route`, `tau_star` = global,
  `tau_cascade`), and the OOF tables to `results/router_train.md`.

### `05_simulate.py`
- Test split, stored scores, attributed energy. For every configuration, accuracy overall and
  per dataset, energy (kWh), emissions (gCO₂e):
  1. each single model (`generalist`, `math`, `qwen3b`)
  2. oracle over the routed tier, oracle over all three
  3. route-only: router pick, no escalation
  4. plain cascade `generalist → qwen3b` and `math → qwen3b`, τ grid
  5. route-cascade: router pick, then escalation at one global τ (grid) and at τ[route]
  6. cost-matched random line over the singles; cheapest routed point matching each single
- Charges the router on every question, the routed model on its questions, the 3B on the
  escalated ones.
- Outputs `results/simulated.csv`, `simulated.md`, `frontier.png`.

### `06_run_end_to_end.py --config route-cascade|route|cascade:X|base|single:X [--tau T]`
The live measurement, on the test split, with the deployed decision rule:
- Router pass (tracked, route* configs): embed every test question, pick the routed model.
- Pass 1 (tracked, one model at a time): each routed model on its questions → confidence.
- Pass 2 (tracked): the 3B on the questions whose confidence was below τ[route].
- End-to-end energy = sum of the tracked passes. Grouping by model instead of interleaving per
  query is a memory constraint; the per-question computation is identical, and the paper says so.
- Writes `results/end_to_end_<config>.json` (accuracy, first-pass accuracy, kWh, gCO₂e,
  escalation rate, per-dataset breakdown with routing split, per-pass energy) and the decision
  log `results/end_to_end_<config>_log.csv`:
  `id, dataset, subject, n_options, gold, first_model, router_p_generalist, router_p_math,
  first_pred, first_correct, first_conf, tau_used, escalated, chosen_model, final_pred,
  final_correct, router_kwh_attr, first_kwh_attr, pool_kwh_attr, total_kwh_attr`.
- Run for: `route-cascade` (τ per route from train), `cascade:qwen3b` (τ from train, no
  router), `base` (generalist alone) and `single:qwen3b` (always 3B) as measured anchors.

### `07_report.py`
Builds the paper tables from `results/`:
- **Table 1** per-model accuracy per dataset and overall; kWh, gCO₂e, seconds per 1,000
  questions (± from repeats only if `REPEATS` > 0).
- **Table 2** gate matrix (6 ordered pairs).
- **Table 3** end-to-end: generalist alone, always-3B, plain cascade, route-cascade;
  accuracy, energy, emissions, escalation rate, Δ vs best single.
- **Table 4** simulated vs measured energy for the same configurations (validates attribution).
- **Figure 1** accuracy–energy frontier with singles, cascades, route-cascade, random line.
- **Figure 2** per-dataset gains.
Markdown + CSV + PNG, all numbers traceable to a results file. Also writes
`results/results.xlsx` — one sheet per table, generated from the same CSVs (openpyxl), so the
workbook and the CSVs can never disagree. The per-question files stay CSV; the workbook holds
the tables only.

---

## 4. Run order and time

| step | script | wall time (est.) |
|---|---|---:|
| 1 | `00_check_env.py --skip-models` | seconds (1 h with `CHECK_MODELS=1`) |
| 2 | `01_build_questions.py` | 2 min |
| 3 | `02_score_model.py` × 4 | ~4 h (Run2 measured) |
| 4 | energy repeats | dropped (`REPEATS=3` to restore, ~1.3 h) |
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
  position habit, the AQuA failure mode). (If `codegen` is ever re-enabled: every row has an
  `exec_status`, under 5% executor errors, and a pass rate above 5%.) `03`/`04`/`05` print a coverage warning if any role is
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

python mac_pipeline/01_build_questions.py                  # 2,246 questions + 70/30 split (already committed)
python mac_pipeline/00_check_env.py --skip-models          # env + energy check; drop the flag to redo the prompt-format choice (~45 min)
python mac_pipeline/02_score_model.py --role generalist    # ~2.3 h (AQuA reasoning is most of it)
python mac_pipeline/02_score_model.py --role math          # ~2.3 h
python mac_pipeline/02_score_model.py --role qwen3b        # ~5 h
python mac_pipeline/03_gate.py                             # routing gate + escalation gate
python mac_pipeline/04_train_router.py
python mac_pipeline/05_simulate.py
python mac_pipeline/06_run_end_to_end.py --config base                 # generalist alone
python mac_pipeline/06_run_end_to_end.py --config single:qwen3b        # always 3B
python mac_pipeline/06_run_end_to_end.py --config cascade:qwen3b       # generalist -> 3B, tau from train
python mac_pipeline/06_run_end_to_end.py --config route-cascade        # the method, tau per route
python mac_pipeline/07_report.py                           # results/report.md + results.xlsx
```

Set `COUNTRY` in `config.py` to where the Mac runs before step 02 (it fixes the CO₂ factor).

Dev-only switches (never needed on the Mac): `MACPIPE_DATA=<dir>` redirects all outputs to a
scratch directory; `MACPIPE_MODEL_OVERRIDE='{"qwen3b": "Qwen/Qwen2.5-0.5B-Instruct"}'` swaps
model ids; `--no-energy` on 02/04/06 skips CodeCarbon.
