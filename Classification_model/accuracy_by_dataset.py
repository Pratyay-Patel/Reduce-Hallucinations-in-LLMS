"""
accuracy_by_dataset.py
----------------------
Reproduces the exact train/test split used during training, runs the saved
model, and reports accuracy broken down by individual dataset and every
combination of datasets — to find where the model performs best.
"""

import os
import warnings
import itertools
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "finalllama.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "best_advanced_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "advanced_scaler.pkl")

FEATURES_TO_DROP = ['id', 'label', 'prompt_complexity_score', 'dataset']

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=FEATURES_TO_DROP)
y = df['label']
datasets_col = df['dataset']  # keep for grouping later

# ── Reproduce exact train/test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, datasets_col.values,
    test_size=0.2, stratify=y, random_state=42
)
print(f"  Test split size: {len(X_test)} rows\n")

# ── Load model & scaler ────────────────────────────────────────────────────────
print("Loading model and scaler...")
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ── Predict ────────────────────────────────────────────────────────────────────
print("Running predictions...")
X_test_scaled = scaler.transform(X_test)
preds = model.predict(X_test_scaled)

# Build a results DataFrame for easy slicing
results_df = pd.DataFrame({
    'dataset': idx_test,       # dataset name each test row came from
    'y_true':  y_test.values,
    'y_pred':  preds
})

# ── 1. Accuracy per individual dataset ────────────────────────────────────────
print("\n" + "=" * 60)
print("  ACCURACY PER INDIVIDUAL DATASET (test split rows only)")
print("=" * 60)
print(f"  {'Dataset':<20} {'Rows':>6} {'Acc':>8} {'F1':>8}")
print("-" * 60)

per_dataset = []
for ds, grp in results_df.groupby('dataset'):
    acc = accuracy_score(grp['y_true'], grp['y_pred'])
    f1  = f1_score(grp['y_true'], grp['y_pred'], average='macro', zero_division=0)
    per_dataset.append({'dataset': ds, 'rows': len(grp), 'accuracy': acc, 'f1': f1})
    print(f"  {ds:<20} {len(grp):>6} {acc:>8.4f} {f1:>8.4f}")

# Overall
overall_acc = accuracy_score(results_df['y_true'], results_df['y_pred'])
overall_f1  = f1_score(results_df['y_true'], results_df['y_pred'], average='macro', zero_division=0)
print("-" * 60)
print(f"  {'OVERALL':<20} {len(results_df):>6} {overall_acc:>8.4f} {overall_f1:>8.4f}")
print("=" * 60)

# ── 2. All combinations of datasets ───────────────────────────────────────────
all_datasets = sorted(results_df['dataset'].unique())

combo_results = []
for r in range(1, len(all_datasets) + 1):
    for combo in itertools.combinations(all_datasets, r):
        mask = results_df['dataset'].isin(combo)
        grp  = results_df[mask]
        acc  = accuracy_score(grp['y_true'], grp['y_pred'])
        f1   = f1_score(grp['y_true'], grp['y_pred'], average='macro', zero_division=0)
        combo_results.append({
            'combo':    ' + '.join(combo),
            'n_datasets': len(combo),
            'rows':     len(grp),
            'accuracy': acc,
            'f1_macro': f1
        })

combo_df = pd.DataFrame(combo_results).sort_values('accuracy', ascending=False)

print("\n" + "=" * 80)
print("  ALL DATASET COMBINATIONS — sorted by accuracy (descending)")
print("=" * 80)
print(f"  {'Combination':<45} {'Rows':>6} {'Acc':>8} {'F1':>8}")
print("-" * 80)
for _, row in combo_df.iterrows():
    print(f"  {row['combo']:<45} {int(row['rows']):>6} {row['accuracy']:>8.4f} {row['f1_macro']:>8.4f}")
print("=" * 80)

# ── 3. Best combination ────────────────────────────────────────────────────────
best = combo_df.iloc[0]
print(f"\n🏆 Best combination : {best['combo']}")
print(f"   Accuracy         : {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
print(f"   F1 Macro         : {best['f1_macro']:.4f}")
print(f"   Rows evaluated   : {int(best['rows'])}")
