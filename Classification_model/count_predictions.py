"""
count_predictions.py
--------------------
Loads the test split of finalllama.csv filtered to gsm8k/main and glue/sst2
(same split logic used during training), runs the saved classification model,
and reports the percentage of times the model predicts label=1.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "finalllama.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "best_advanced_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "advanced_scaler.pkl")

# ── Load data & filter to target datasets ────────────────────────────────────
TARGET_DATASETS = ['gsm8k/main', 'glue/sst2']

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  Total rows (full): {len(df)}")

df = df[df['dataset'].isin(TARGET_DATASETS)].reset_index(drop=True)
print(f"  Rows after filtering to {TARGET_DATASETS}: {len(df)}")

# ── Reproduce the exact train/test split on filtered data ─────────────────────
FEATURES_TO_DROP = ['id', 'label', 'prompt_complexity_score', 'dataset']
X = df.drop(columns=FEATURES_TO_DROP)
y = df['label']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Test split size: {len(X_test)} rows")

# ── Load model & scaler ────────────────────────────────────────────────────────
print("\nLoading model and scaler...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Scale & predict ────────────────────────────────────────────────────────────
print("Running predictions on test split...")
X_test_scaled = scaler.transform(X_test)
predictions   = model.predict(X_test_scaled)

# ── Results ────────────────────────────────────────────────────────────────────
total      = len(predictions)
count_ones = int((predictions == 1).sum())
count_zero = total - count_ones
pct_ones   = (count_ones / total) * 100
pct_zeros  = (count_zero / total) * 100

print("\n" + "=" * 45)
print("        PREDICTION DISTRIBUTION (Test Split)")
print("=" * 45)
print(f"  Total predictions : {total}")
print(f"  Predicted 1       : {count_ones:>6}  ({pct_ones:.2f}%)")
print(f"  Predicted 0       : {count_zero:>6}  ({pct_zeros:.2f}%)")
print("=" * 45)
