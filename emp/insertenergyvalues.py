import pandas as pd
import numpy as np
from scipy import stats

# ── 1. Load OLD data and fit models ─────────────────────────────────────────
old = pd.read_csv('complete_energy_comparison_with_outputs.csv')

old['output_len_S1'] = old['full_output_S1'].str.len()
old['output_len_S2'] = old['full_output_S2'].str.len()

models = {}
sigmas = {}

for ds in old['dataset'].unique():
    sub = old[old['dataset'] == ds]
    for system, ecol, outcol in [
        ('llama3.2', 'energy_consumed_kwh_S1', 'output_len_S1'),
        ('phi3',     'energy_consumed_kwh_S2', 'output_len_S2'),
    ]:
        x = sub[outcol].values.astype(float)
        y = sub[ecol].values.astype(float)
        n = len(x)

        if n <= 2:
            # ── Force through origin when n≤2 ────────────────────────────
            # With only 2 points linregress gives R²=1 and a potentially
            # large negative intercept that produces negative predictions.
            # Forcing intercept=0 (slope = Σ(xy)/Σ(x²)) avoids floor hits.
            slope     = np.dot(x, y) / np.dot(x, x)
            intercept = 0.0
            # Sigma from log-residuals (std, not IQR — n too small for IQR)
            y_pred = np.maximum(slope * x, 1e-12)
            sigma  = (np.log(y) - np.log(y_pred)).std()
            print(f"  [origin-forced] {ds} | {system}: slope={slope:.4e}  n={n}")
        else:
            # ── Normal linregress for n>2 ────────────────────────────────
            reg       = stats.linregress(x, y)
            slope, intercept = reg.slope, reg.intercept

            y_pred    = np.maximum(slope * x + intercept, 1e-12)
            log_resid = np.log(y) - np.log(y_pred)

            # IQR-based robust sigma (resistant to outliers)
            # IQR of N(0,σ²) = 1.349σ  →  σ_robust = IQR / 1.349
            q75, q25  = np.percentile(log_resid, [75, 25])
            sigma_iqr = (q75 - q25) / 1.349
            # Fall back to std if IQR-sigma is zero (rare edge case)
            sigma     = sigma_iqr if sigma_iqr > 0 else log_resid.std()

        models.setdefault(ds, {})[system] = (slope, intercept)
        sigmas.setdefault(ds, {})[system] = sigma

print("\n=== Fitted models ===")
for ds, m in models.items():
    for sys, (sl, ic) in m.items():
        print(f"  {ds} | {sys}: slope={sl:.4e}  intercept={ic:.4e}  σ={sigmas[ds][sys]:.4f}")

# ── 2. Model name aliases ────────────────────────────────────────────────────
MODEL_ALIASES = {
    'phi3mini':   'phi3',
    'phi-3-mini': 'phi3',
    'phi3.5':     'phi3',
    'llama3':     'llama3.2',
    'llama3.1':   'llama3.2',
}

def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)

# ── 3. Extrapolation function ─────────────────────────────────────────────────
def extrapolate_energy(dataset, output_len, model_name, seed=None):
    model_name = resolve_model_name(model_name)
    rng        = np.random.default_rng(seed)

    if dataset not in models or model_name not in models.get(dataset, {}):
        print(f"  [WARN] No model for ({dataset}, {model_name}) — using cross-dataset mean")
        all_slopes     = [v[model_name][0] for v in models.values() if model_name in v]
        all_intercepts = [v[model_name][1] for v in models.values() if model_name in v]
        all_sigmas     = [sigmas[d][model_name] for d in sigmas if model_name in sigmas[d]]
        if not all_slopes:
            raise ValueError(
                f"model_name '{model_name}' not found in any dataset. "
                f"Known: {set(k for v in models.values() for k in v)}"
            )
        slope     = np.mean(all_slopes)
        intercept = np.mean(all_intercepts)
        sigma     = np.mean(all_sigmas)
    else:
        slope, intercept = models[dataset][model_name]
        sigma            = sigmas[dataset][model_name]

    # Point prediction — guaranteed positive (intercept=0 for n≤2 cells)
    E_pred = max(slope * output_len + intercept, 1e-12)

    # Log-normal sample
    E_sample = E_pred * np.exp(rng.normal(0, sigma))

    # Analytical 95% CI
    lower = E_pred * np.exp(-1.96 * sigma)
    upper = E_pred * np.exp(+1.96 * sigma)

    # Flag cells where calibration had n≤2 (GSM8K) — estimates are rough
    n_calib = len(old[
        (old['dataset'] == dataset)
    ]) if dataset in old['dataset'].values else 0

    return {
        'energy_point':   E_pred,
        'energy_sample':  max(E_sample, 1e-12),
        'energy_lower':   lower,
        'energy_upper':   upper,
        'calib_warning':  'low_calibration_n' if n_calib <= 2 else None,
    }

# ── 4. Load new data ──────────────────────────────────────────────────────────
new = pd.read_csv('prompt_accuracy_comparison.csv')

print("\n=== Model names found in new data ===")
print(new['model'].unique())
print("=== Dataset names found in new data ===")
print(new['dataset'].unique())

# ── 5. Fill missing energy values (phi3mini only) ────────────────────────────
PHI3MINI_NAMES = {'phi3mini', 'phi-3-mini', 'phi3.5'}

filled_count  = 0
skipped_count = 0

for i, row in new.iterrows():
    model_raw = str(row['model']).strip()

    # Non-phi3mini rows: always leave untouched
    if model_raw not in PHI3MINI_NAMES:
        skipped_count += 1
        continue

    # phi3mini rows: only fill when energy is missing
    if pd.notna(row['energy_consumed_kwh']) and str(row['energy_consumed_kwh']).strip() not in ('', 'nan'):
        skipped_count += 1
        continue

    output_len = len(str(row['output'])) if pd.notna(row['output']) else 0

    est = extrapolate_energy(
        dataset    = row['dataset'],
        output_len = output_len,
        model_name = model_raw,
        seed       = int(row['sample_index']) if pd.notna(row['sample_index']) else i,
    )

    new.at[i, 'energy_consumed_kwh'] = est['energy_sample']
    new.at[i, 'energy_lower_95']     = est['energy_lower']
    new.at[i, 'energy_upper_95']     = est['energy_upper']
    new.at[i, 'energy_source']       = 'extrapolated'
    # Flag rows where calibration data was thin (n≤2)
    if est['calib_warning']:
        new.at[i, 'energy_note'] = est['calib_warning']
    filled_count += 1

# Tag measured phi3mini rows
new.loc[
    new['model'].isin(PHI3MINI_NAMES) & new['energy_source'].isna(),
    'energy_source'
] = 'measured'

print(f"\n=== Done: {filled_count} rows filled, {skipped_count} rows skipped ===")

# ── 6. Summary statistics — median + IQR (robust to skew) ────────────────────
new['energy_valid'] = new['energy_consumed_kwh'].where(
    (pd.to_numeric(new['energy_consumed_kwh'], errors='coerce') > 1e-11)
    & new['energy_consumed_kwh'].notna()
)
new['output_len'] = new['output'].astype(str).str.len()

print("\n=== Summary statistics (median + IQR) ===")
summary_rows = []

for (m, d), sub in new.groupby(['model', 'dataset']):
    valid    = sub.dropna(subset=['energy_valid'])
    valid    = valid[pd.to_numeric(valid['energy_valid'], errors='coerce') > 1e-11]
    e        = pd.to_numeric(valid['energy_valid'], errors='coerce').dropna().values
    acc      = sub['accuracy_score'].dropna().mean()
    n        = len(e)

    if n >= 3:
        median   = np.median(e)
        q1, q3   = np.percentile(e, [25, 75])
        iqr      = q3 - q1
        skewness = stats.skew(e)
        _, p_sw  = stats.shapiro(e)
        dist     = 'log-normal' if p_sw < 0.05 else 'approx. normal'
    else:
        median = iqr = skewness = p_sw = float('nan')
        dist   = 'insufficient data'

    summary_rows.append({
        'model':             m,
        'dataset':           d,
        'n_valid':           n,
        'accuracy':          round(acc, 3),
        'median_energy_kwh': median,
        'IQR_energy_kwh':    iqr,
        'skewness':          round(skewness, 3) if n >= 3 else float('nan'),
        'shapiro_p':         round(p_sw, 4)     if n >= 3 else float('nan'),
        'distribution':      dist,
    })

    print(f"\n{m} | {d}")
    print(f"  n={n}, accuracy={acc:.3f}")
    if n >= 3:
        print(f"  median={median:.4e} kWh,  IQR={iqr:.4e} kWh")
        print(f"  skewness={skewness:.3f},  Shapiro-Wilk p={p_sw:.4f}  →  {dist}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('summary_stats_median_iqr.csv', index=False)
print("\nSaved → summary_stats_median_iqr.csv")

# ── 7. Preview and save ───────────────────────────────────────────────────────
preview_cols = ['model', 'dataset', 'sample_index', 'accuracy_score',
                'energy_consumed_kwh', 'energy_lower_95', 'energy_upper_95',
                'energy_source', 'energy_note']
# Only show columns that exist
preview_cols = [c for c in preview_cols if c in new.columns]
print("\n=== Preview ===")
print(new[preview_cols].to_string())

new.drop(columns=['energy_valid', 'output_len'], inplace=True, errors='ignore')
new.to_csv('new_data_with_energy.csv', index=False)
print("\nSaved → new_data_with_energy.csv")