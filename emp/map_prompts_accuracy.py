"""
map_prompts_accuracy.py
-----------------------
Maps Llama 3.2 and Phi-3 Mini prompt results by (dataset, sample_index) and
produces a single CSV with SEPARATE rows for each model — ONLY for prompts
that are common (intersection) to both models, restricted to:
  - gsm8k/main
  - glue/sst2

Input files (expected in the same directory as this script):
  - ecoprompt_results.xlsx              → Llama 3.2 | gsm8k/main + glue/mnli
  - ecoprompt_results_2026_04_04_1235.xlsx → Llama 3.2 | squad_v2
  - phi3mini_acc.csv                    → Phi-3 Mini | gsm8k/main + glue/sst2

Output:
  - prompt_accuracy_comparison.csv  (one row per model per unique prompt)
"""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LLAMA_FILE1  = os.path.join(SCRIPT_DIR, "ecoprompt_results.xlsx")
LLAMA_FILE2  = os.path.join(SCRIPT_DIR, "ecoprompt_results_2026_04_04_1235.xlsx")
PHI_FILE     = os.path.join(SCRIPT_DIR, "phi3mini_acc.csv")
OUTPUT_FILE  = os.path.join(SCRIPT_DIR, "prompt_accuracy_comparison.csv")

# ── Datasets to include in the comparison ─────────────────────────────────────
TARGET_DATASETS = {"gsm8k/main", "glue/sst2"}

# ── Load Llama 3.2 data ────────────────────────────────────────────────────────
print("Loading Llama 3.2 data...")

df_llama1 = pd.read_excel(LLAMA_FILE1, sheet_name=0)
df_llama2 = pd.read_excel(LLAMA_FILE2, sheet_name=0)

df_llama = pd.concat([df_llama1, df_llama2], ignore_index=True)
df_llama = df_llama[df_llama["dataset"].isin(TARGET_DATASETS)].reset_index(drop=True)
print(f"  Total Llama rows loaded : {len(df_llama)}")
print(f"  Datasets                : {sorted(df_llama['dataset'].dropna().unique())}")

# ── Load Phi-3 Mini data ───────────────────────────────────────────────────────
print("\nLoading Phi-3 Mini data...")

df_phi = pd.read_csv(PHI_FILE)
# Drop separator / blank rows (rows without a valid scenario_id)
df_phi = df_phi.dropna(subset=["scenario_id"]).reset_index(drop=True)
df_phi = df_phi[df_phi["dataset"].isin(TARGET_DATASETS)].reset_index(drop=True)
print(f"  Total Phi-3 Mini rows loaded : {len(df_phi)}")
print(f"  Datasets                     : {sorted(df_phi['dataset'].dropna().unique())}")

# ── Helper: deduplicate & aggregate per (dataset, sample_index) ────────────────
def aggregate_per_prompt(df, model_label, include_energy=True):
    """
    For each unique (dataset, sample_index) keep:
      - the first full_prompt      (identical across repeated runs)
      - the first full_output      (representative output)
      - mean accuracy_score        (averaged over repeated runs)
      - mean energy_consumed_kwh   (averaged over repeated runs, if available)
      - run_count                  (how many runs were recorded)
    Returns a DataFrame tagged with 'model' = model_label.
    """
    # Build the aggregation dict dynamically
    agg_dict = {
        "full_prompt":    ("full_prompt",    "first"),
        "output":         ("full_output",    "first"),
        "accuracy_score": ("accuracy_score", "mean"),
        "run_count":      ("accuracy_score", "count"),
    }

    if include_energy and "energy_consumed_kwh" in df.columns:
        agg_dict["energy_consumed_kwh"] = ("energy_consumed_kwh", "mean")

    agg = (
        df.groupby(["dataset", "sample_index"], as_index=False)
        .agg(**agg_dict)
    )

    # Ensure energy column exists even when not computed
    if "energy_consumed_kwh" not in agg.columns:
        agg["energy_consumed_kwh"] = None

    agg["model"] = model_label
    return agg


llama_agg = aggregate_per_prompt(df_llama, "llama3.2",  include_energy=True)
phi_agg   = aggregate_per_prompt(df_phi,   "phi3mini",  include_energy=False)

print(f"\nUnique prompts — Llama 3.2  : {len(llama_agg)}")
print(f"Unique prompts — Phi-3 Mini : {len(phi_agg)}")

# ── Keep ONLY prompts common to BOTH models (intersection) ───────────────────
llama_keys = set(zip(llama_agg["dataset"], llama_agg["sample_index"]))
phi_keys   = set(zip(phi_agg["dataset"],   phi_agg["sample_index"]))
common_keys = llama_keys & phi_keys

print(f"\nIntersection size (common prompts) : {len(common_keys)}")

def filter_to_common(df, keys):
    mask = [( row["dataset"], row["sample_index"]) in keys
            for _, row in df.iterrows()]
    return df[mask].reset_index(drop=True)

llama_agg = filter_to_common(llama_agg, common_keys)
phi_agg   = filter_to_common(phi_agg,   common_keys)

print(f"  Llama 3.2 rows after filter : {len(llama_agg)}")
print(f"  Phi-3 Mini rows after filter: {len(phi_agg)}")

# ── Stack rows (separate row per model, NOT merged columns) ────────────────────
print("\nCombining into separate rows per model …")

combined = pd.concat([llama_agg, phi_agg], ignore_index=True)

# Sort by dataset → sample_index → model so matching prompts appear together
combined = combined.sort_values(
    ["dataset", "sample_index", "model"]
).reset_index(drop=True)

# ── Final column ordering ──────────────────────────────────────────────────────
final_cols = [
    "model",
    "dataset",
    "sample_index",
    "full_prompt",
    "output",
    "accuracy_score",
    "energy_consumed_kwh",
    "run_count",
]
combined = combined[final_cols]

# ── Summary stats ──────────────────────────────────────────────────────────────
print(f"\n{'─'*60}")
print(f"{'FINAL TABLE SUMMARY':^60}")
print(f"{'─'*60}")
print(f"Total rows               : {len(combined)}")
print(f"  Llama 3.2 rows         : {(combined['model']=='llama3.2').sum()}")
print(f"  Phi-3 Mini rows        : {(combined['model']=='phi3mini').sum()}")
print()
print("Per-model / per-dataset accuracy (mean):")
for (model, ds), grp in combined.groupby(["model", "dataset"]):
    acc  = grp["accuracy_score"].mean()
    nrg  = grp["energy_consumed_kwh"].mean()
    nrg_str = f"{nrg:.6f} kWh" if pd.notna(nrg) else "N/A (blank)"
    print(f"  {model:<12}  {ds:<18}  acc={acc:.4f}   energy={nrg_str}")
print(f"{'─'*60}")

# ── Save ───────────────────────────────────────────────────────────────────────
combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅  Saved → {OUTPUT_FILE}")
