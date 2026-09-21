import pandas as pd
import json

REFERENCE_CSV = 'Classification_model/finalllama.csv'
INPUT_CSV     = 'results/ag_news_llama_all.csv'
OUTPUT_CSV    = 'Classification_model/final_data/final_ag_news_llama_all.csv'

# ── Derive target columns from reference file ─────────────────────────────────
print(f"Reading reference columns from {REFERENCE_CSV} …")
TARGET_COLUMNS = pd.read_csv(REFERENCE_CSV, nrows=0).columns.tolist()
print(f"Target columns: {TARGET_COLUMNS}")

# ── Parse a single nemo_raw_output cell ───────────────────────────────────────
def parse_nemo(row):
    try:
        if pd.isna(row):
            return {}
        data = json.loads(row) if isinstance(row, str) else row
        # Unwrap single-element lists produced by NeMo Curator
        return {k: v[0] if isinstance(v, list) and len(v) > 0 else v
                for k, v in data.items()}
    except Exception:
        return {}

print(f"Reading {INPUT_CSV} …")
df = pd.read_csv(INPUT_CSV)

print("Parsing nemo_raw_output …")
parsed_df = pd.DataFrame(df['nemo_raw_output'].apply(parse_nemo).tolist())

print("Expanding NeMo columns into main DataFrame …")
df = pd.concat(
    [df.drop(columns=['nemo_raw_output', 'nemo_raw_output.1'], errors='ignore'),
     parsed_df],
    axis=1,
)

# ── Rename to match target schema ─────────────────────────────────────────────
df = df.rename(columns={
    'accuracy_score': 'label',
    'sample_index':   'id',
})

# ── Keep only reference columns (in the same order), skip any missing ones ────
present = [c for c in TARGET_COLUMNS if c in df.columns]
df = df[present]

print(f"Saving to {OUTPUT_CSV} …")
df.to_csv(OUTPUT_CSV, index=False)
print("Done! Columns in output:")
print(df.columns.tolist())
