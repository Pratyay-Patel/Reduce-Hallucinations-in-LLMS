"""
run_parse_all.py
────────────────
Applies the parse_test.py logic to every CSV in results/ that:
  • starts with a dataset name (arc_, boolq, gsm8k_, sst2_)
  • is NOT an ecoprompt_* or emissions* file
  • is NOT a _per_prompt or _summary file

Outputs land in Classification_model/final_data/final_<original_name>.csv
Run from the project root:  python Classification_model/run_parse_all.py
"""

import os
import json
import pathlib
import pandas as pd

# ── Paths (relative to project root) ──────────────────────────────────────────
PROJECT_ROOT  = pathlib.Path(__file__).resolve().parent.parent   # Ecoprompt_eval/
RESULTS_DIR   = PROJECT_ROOT / "results"
REFERENCE_CSV = PROJECT_ROOT / "Classification_model" / "finalllama.csv"
OUTPUT_DIR    = PROJECT_ROOT / "Classification_model" / "final_data"

# ── Exclusion rules ────────────────────────────────────────────────────────────
EXCLUDE_PREFIXES = ("ecoprompt", "emissions")
EXCLUDE_SUFFIXES = ("_per_prompt.csv", "_summary.csv")

# ── Load reference columns once ───────────────────────────────────────────────
print(f"Reading reference columns from {REFERENCE_CSV} ...")
TARGET_COLUMNS = pd.read_csv(REFERENCE_CSV, nrows=0).columns.tolist()
print(f"Target columns ({len(TARGET_COLUMNS)}): {TARGET_COLUMNS}\n")

# ── Helper: parse a single nemo_raw_output cell ───────────────────────────────
def parse_nemo(row):
    try:
        if pd.isna(row):
            return {}
        data = json.loads(row) if isinstance(row, str) else row
        return {k: v[0] if isinstance(v, list) and len(v) > 0 else v
                for k, v in data.items()}
    except Exception:
        return {}

# ── Collect files to process ───────────────────────────────────────────────────
csv_files = sorted([
    f for f in RESULTS_DIR.glob("*.csv")
    if not f.name.startswith(EXCLUDE_PREFIXES)
    and not any(f.name.endswith(s) for s in EXCLUDE_SUFFIXES)
])

print(f"Found {len(csv_files)} file(s) to process:\n")
for f in csv_files:
    print(f"  {f.name}")
print()

# ── Ensure output directory exists ────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Process each file ─────────────────────────────────────────────────────────
success, failed = [], []

for input_csv in csv_files:
    output_csv = OUTPUT_DIR / f"final_{input_csv.name}"
    print(f"{'─'*60}")
    print(f"  Input : {input_csv.name}")
    print(f"  Output: {output_csv.name}")

    try:
        df = pd.read_csv(input_csv)

        if 'nemo_raw_output' in df.columns:
            print("  Parsing nemo_raw_output ...")
            parsed_df = pd.DataFrame(df['nemo_raw_output'].apply(parse_nemo).tolist())
            df = pd.concat(
                [df.drop(columns=['nemo_raw_output', 'nemo_raw_output.1'], errors='ignore'),
                 parsed_df],
                axis=1,
            )
        else:
            print("  (no nemo_raw_output column - skipping NeMo parse)")

        df = df.rename(columns={
            'accuracy_score': 'label',
            'sample_index':   'id',
        })

        present = [c for c in TARGET_COLUMNS if c in df.columns]
        df = df[present]

        df.to_csv(output_csv, index=False)
        print(f"  SAVED ({len(df)} rows, {len(present)} cols)")
        success.append(input_csv.name)

    except Exception as e:
        print(f"  ERROR: {e}")
        failed.append((input_csv.name, str(e)))

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Done.  {len(success)} succeeded   {len(failed)} failed")
if failed:
    print("\nFailed files:")
    for name, err in failed:
        print(f"  {name}: {err}")
