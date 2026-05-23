import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

print("Loading finalllama.csv to generate scaler...")
df = pd.read_csv('finalllama.csv')

# Exact filters from train_advanced.ipynb
df = df[(df['dataset'] == 'glue/sst2') | (df['dataset'] == 'gsm8k/main')]
df_processed = df.copy()

features_to_drop = ['id', 'label', 'prompt_complexity_score', 'dataset']
X = df_processed.drop(features_to_drop, axis=1)
y = df_processed['label']

print("Splitting data to match training setup...")
X_train, _, _, _ = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Fitting scaler...")
scaler = StandardScaler()
scaler.fit(X_train)

print(f"Features fitted: {scaler.feature_names_in_}")
joblib.dump(scaler, 'advanced_scaler.pkl')
print("✅ Saved advanced_scaler.pkl")
