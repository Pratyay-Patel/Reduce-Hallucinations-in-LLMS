import pandas as pd
import numpy as np
from scipy import stats

# ── 1. Load the OLD data (with known energy) to fit the models ──────────────
old = pd.read_csv('complete_energy_comparison_with_outputs.csv')

old['output_len_S1'] = old['full_output_S1'].str.len()
old['output_len_S2'] = old['full_output_S2'].str.len()

# Fit per-dataset linear models + compute log-residual sigma
# We treat S1=llama3.2, S2=phi3 from the old file
models = {}
sigmas = {}

for ds in old['dataset'].unique():
    sub = old[old['dataset'] == ds]
    for system, ecol, outcol in [
        ('llama3.2', 'energy_consumed_kwh_S1', 'output_len_S1'),
        ('phi3',     'energy_consumed_kwh_S2', 'output_len_S2'),
    ]:
        x = sub[outcol].values
        y = sub[ecol].values
        reg = stats.linregress(x, y)
        slope, intercept = reg.slope, reg.intercept

        y_pred = slope * x + intercept
        # clamp predictions to avoid log(0) or log(negative)
        y_pred = np.maximum(y_pred, 1e-12)
        sigma  = (np.log(y) - np.log(y_pred)).std()

        models.setdefault(ds, {})[system] = (slope, intercept)
        sigmas.setdefault(ds, {})[system] = sigma

print("Fitted datasets:", list(models.keys()))
for ds, m in models.items():
    for sys, (sl, ic) in m.items():
        print(f"  {ds} | {sys}: slope={sl:.4e}, intercept={ic:.4e}, σ_log={sigmas[ds][sys]:.4f}")