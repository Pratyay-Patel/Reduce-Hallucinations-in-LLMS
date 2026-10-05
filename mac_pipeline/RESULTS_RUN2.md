# Mac run 2 — findings (2026-10-05)

Full pipeline on the M4 MacBook Air: 4 models × 3 datasets × 1,992 questions, 3 energy repeats,
8 measured end-to-end configurations. Files in `Results_Run2/`. Companion to
[`PLAN.md`](PLAN.md) (method) and [`../FINDINGS.md`](../FINDINGS.md) (project record).

## 1. The run is sound

- Every model covers every dataset; no nulls; chat template on for Qwen-3B only (as the check chose).
- Energy of record is `powermetrics` SoC power, ~1 sample/s, 5.6–7.4 W while scoring, 0.6 W idle.
  CodeCarbon's figure is ~2× that because of its 85 W-TDP CPU model; kept for comparison only.
  Carbon factor 713 gCO₂e/kWh (India).
- CommonsenseQA replaced AQuA and works: accuracies 0.69–0.80, no predicted letter above 25%,
  base confidence AUC 0.79 for its own failures.

## 2. Single models (test, n = 598)

| model | acc | MMLU | CSQA | ARC | Wh / 1,000 q |
|---|---:|---:|---:|---:|---:|
| base Qwen2.5-1.5B | 0.686 | 0.607 | 0.713 | 0.813 | 1.58 |
| Qwen2.5-3B | **0.741** | 0.651 | 0.800 | 0.860 | 3.96 |
| Phi-4-mini | 0.724 | 0.685 | 0.687 | 0.840 | 3.42 |
| Llama-3.2-3B | 0.674 | 0.611 | 0.733 | 0.740 | 3.00 |

Qwen − base = +0.055 (95% CI +0.018 to +0.092). Qwen − Phi = +0.017 (CI crosses zero).

## 3. The gate: a fully complementary pool

All 12 ordered pairs pass (p < 0.001). Each model fixes 34–48% of every other model's errors
against 24% chance. Oracle over the pool: **0.878**, 13.7 points above the best single model.
This is the regime the hybrid was designed for.

## 4. Routing, measured end to end (test, n = 598)

| configuration | acc | Wh / 1,000 q | vs best single | vs random at cost |
|---|---:|---:|---:|---:|
| base alone | 0.686 | 1.51 | −0.055 | — |
| cascade base → Qwen, τ = 0.95 | 0.741 | 3.62 | **+0.000** (CI ±0.010) | +0.001 |
| hybrid (router over pool), τ = 0.95 | 0.736 | 3.82 | −0.005 | −0.005 |
| Qwen alone | 0.741 | 3.66 | — | — |

- The cascade equals the best model exactly, for 1% less energy.
- The hybrid is 0.5 pt below the plain cascade and uses 5% more energy.
- Nothing beats the best single model; nothing is more than half a point from cost-matched random.

**Why.** (a) Break-even for the cascade needs ≥ 41% of questions kept at the base (its cost is 41%
of Qwen's). At τ = 0.95 it keeps 46%, on which base and Qwen both score 0.924; below that Qwen is
better in every confidence bucket. Saving ≈ 1%. (b) The "where" signal is weak: for predicting
which pool model is right, confidence alone gives AUC 0.75 and adding the prompt embedding
lowers it to 0.72; subject one-hot gives 0.63–0.68. The router reaches 0.737 at full escalation
against an oracle of 0.879. In run 1 the embedding looked useful only because AQuA's rows were
a dataset-identity signal.

Per dataset the router chose well on MMLU (0.681 vs Phi's 0.685) and badly on CSQA and ARC,
where Qwen dominates and a third of escalations still went elsewhere.

Router-head ablation (lr, rf, xgb, svc, mlp, voting): routed accuracies 0.724–0.737, within noise;
auto-selection picked `mlp` on that noise despite its having the worst AUC (0.666). **Pin
`ROUTER_HEAD = "lr"` for the paper**; keep the table.

## 5. Energy noise

The same model's SoC power varied across runs on one night: base 5.7–12.8 W, Qwen 5.0–9.6 W,
Phi 4.3–7.9 W (±30–44%). The three back-to-back repeats agree to 1% with each other but ran at
~2× the power of the main runs. Table 4's simulated-vs-measured gaps (−12% to +10%) are the same
effect. Thermal pressure read Heavy or Moderate in almost every run.

Consequence: any energy difference under ~20% is inside the noise. The cascade's 1–6% saving
is not a claim. The base-alone −59% and the ~2.5× cost ratio between tiers are well outside it.

## 6. What the paper can say from this run

1. A confidence-gated cascade matches the best model's accuracy on 598 held-out questions
   (Δ = 0.000, CI ±0.010) at energy equal within measurement noise.
2. The small model alone saves 59% energy and loses 5.5 points (CI 1.8–9.2).
3. The gate certifies a complementary pool with 13.7 points of oracle headroom; frozen-embedding
   routing recovers under one point of it. The gate is necessary for routing to be possible and
   this run shows it is not sufficient.
4. On fanless hardware, energy must be reported as a spread across separate runs, not a
   within-session standard deviation.

## 7. Before a final run

- Pin the router head to logistic regression.
- Replace back-to-back repeats with the same configuration run three times at different points
  in the night, so the spread is real.
- An energy saving needs a bigger cost gap: the 1.5B base is 41% of a 3B's cost here, and no
  threshold rule beats break-even by much from there (→ §9, 7B pool).
- An accuracy gain from routing needs a pool whose strengths are topic-shaped (→ §8, specialists).

## 8. Why the router finds so little: what the embedding actually encodes

Same logistic head, trained on the 1,394 train questions, picking the pool model with the highest
predicted P(right). Test n = 598. References: always-Qwen 0.741, plain cascade → Qwen 0.741,
oracle over the pool 0.878.

| router input | AUC (train, OOF) | hybrid τ = 0.95 (test) | routed, all escalated (test) |
|---|---:|---:|---:|
| confidence only | 0.745 | 0.729 | 0.732 |
| confidence + 384-d MiniLM embedding (PCA 32) | 0.722 | 0.741 | 0.739 |
| confidence + 3-bit "which dataset" flag | 0.735 | 0.744 | **0.753** |
| embedding alone | 0.611 | 0.744 | 0.742 |
| dataset flag alone | 0.581 | 0.744 | 0.753 |
| confidence + the base's full option distribution | 0.755 | 0.727 | 0.736 |

- **Three bits beat the embedding.** Knowing only which dataset a question came from routes as
  well as or better than the full embedding. Within a dataset the embedding finds nothing usable.
- **The dataset-level policy is right and small.** Phi is best on MMLU, Qwen on CSQA and ARC; the
  router learns that. Phi's MMLU edge is 3.4 pts on half the questions, so the policy is worth
  ~1 pt over always-Qwen, inside the noise of 598 questions.
- **The base's own uncertainty is a *whether* signal, not a *where* signal.** Its option
  distribution predicts its own failures (AUC 0.79) and says nothing about which pool model to call.
- In run 1 the embedding looked stronger only because AQuA's rows were a dataset-identity signal
  with a 20-pt accuracy spread across the pool.

Reading: the three pool models differ by training recipe, not by domain, so their question-level
disagreements (13.7 pts of oracle headroom) are not predictable from prompt text with a frozen
encoder at n ≈ 1,400. A gate can certify complementarity; a prompt-side router cannot exploit it
unless the pool's strengths are topic-shaped. See §10 of PLAN.md / the chat record for the
specialist options that would make them topic-shaped.

## 9. Projection: a 7B pool in 8-bit (not run)

Scaled from run 2's measurements. The scorer's four forward passes over a 100–600-token prompt
are compute-bound on the M4 (≈ 1.5–2 TFLOPS sustained under throttling), so time scales with
parameters; 8-bit halves the weights but not the arithmetic.

| tier | s / question | SoC J / question | Wh / 1,000 q |
|---|---:|---:|---:|
| base 1.5B fp16 (measured) | 0.8–0.9 | 5.7 | 1.6 |
| 3B fp16 (measured) | 2.0–2.5 | 13–15 | 3.4–4.0 |
| 7B 8-bit (estimated) | 5–7 | 30–40 | 8–11 |

- Scoring three 7B models on 1,992 questions ≈ 10 h; the full script ≈ 17–20 h (two nights, or
  cut the end-to-end set to hybrid, best cascade, base, best single).
- Accuracy from published MMLU minus the 0–2 pts our zero-shot letter scoring costs:
  Qwen2.5-7B-Instruct ≈ 0.72–0.74 MMLU / 0.82–0.85 CSQA / ≈ 0.90 ARC; Llama-3.1-8B ≈ 0.68–0.71 /
  0.80 / 0.88. Best single ≈ 0.80 overall (from 0.741). 8-bit loss < 1 pt.
- **What changes for routing:** the base would cost ≈ 16% of a 7B question instead of 41%, so the
  cascade's break-even keep-fraction falls from 41% to 16%. It already keeps 46% at τ = 0.95 with
  the base scoring 0.924 there; against a 7B at ≈ 0.95 on that slice, the cascade loses ≈ 0.5 pt
  and saves ≈ 40% energy. That is a defensible claim; run 2's 1% is not. The hybrid would not
  improve (§8: the *where* signal is the problem, not pool strength).
- Requirements: `mlx-lm` with `mlx-community` 8-bit checkpoints (bitsandbytes does not run on
  Apple Silicon; torch/MPS has no fast int8 path) → a small MLX port of the scorer. 16 GB RAM is
  enough one model at a time. The ±30% run-to-run energy noise (§5) remains; a 40% saving clears
  it, a 10% one would not.
