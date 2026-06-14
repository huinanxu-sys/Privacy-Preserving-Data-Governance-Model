"""
HONEST data generation + cross-validation + D10 regeneration.
* No calibrated noise multipliers (k_fin, k_macro, k_eng removed).
* Proper stratified CV with per-fold standardization.
* All metrics reported without hand-tuning.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import warnings, os, copy, json
warnings.filterwarnings('ignore')

# ==========================================================
# 1. HONEST DATA GENERATION (theory-driven, not calibrated)
# ==========================================================
SEED = 20230101
N = 5000
N_RUNS = 20
N_FOLDS = 5

np.random.seed(SEED)
Theta  = np.random.normal(0, 1, N)
Zeta   = np.random.normal(0, 1, N)

Y_star   = 0.55 * Theta + np.sqrt(1 - 0.55**2) * Zeta
Default_Y = (Y_star < -1.405).astype(int)

# Financial baseline: Appendix A formulas, NO noise calibration
np.random.seed(SEED + 1)
CR  =  1.5 + 0.4 * Theta + np.random.normal(0, 0.40, N)
DE  = -2.0 + 0.5 * Theta + np.random.normal(0, 0.40, N)
ROA = 0.05 + 0.03 * Theta + np.random.normal(0, 0.40, N)

# Model A: Macro public data
np.random.seed(SEED + 2)
RRV = 0.5 * Theta + np.random.normal(0, 1.0, N)
IG  = 0.5 * Theta + np.random.normal(0, 1.0, N)
TB  = 0.5 * Theta + np.random.normal(0, 1.0, N)

# Model B: Domain-logic engines
np.random.seed(SEED + 3)
M            = 0.5 + 0.3 * Theta + np.random.normal(0, 0.15, N)
VI           = 0.3 - 0.1 * Theta + np.random.normal(0, 0.10, N)
Score_anomaly = 0.2 - 0.2 * Theta * (1 - Default_Y) + np.random.normal(0, 0.12, N)

# Model C: Raw micro-transaction data (pure noise, no signal)
np.random.seed(SEED + 4)
P = 5.0 * np.random.normal(0, 1, N)
F = 5.0 * np.random.normal(0, 1, N)
C = 5.0 * np.random.normal(0, 1, N)

# Save
df = pd.DataFrame({
    'Firm_ID': range(1, N+1), 'Default_Y': Default_Y,
    'CR': CR, 'DE': DE, 'ROA': ROA,
    'RRV': RRV, 'IG': IG, 'TB': TB,
    'M': M, 'VI': VI, 'Score_anomaly': Score_anomaly,
    'P': P, 'F': F, 'C': C,
})
df.to_csv('synthetic_sme_customs_data_honest.csv', index=False)
print("Saved: synthetic_sme_customs_data_honest.csv")

# ==========================================================
# 2. FEATURE GROUPS & CROSS-VALIDATION
# ==========================================================
y = df['Default_Y'].values
FEATURE_SETS = {
    'Baseline': ['CR','DE','ROA'],
    'Model A':  ['CR','DE','ROA','RRV','IG','TB'],
    'Model B':  ['CR','DE','ROA','RRV','IG','TB','M','VI','Score_anomaly'],
    'Model C':  ['CR','DE','ROA','RRV','IG','TB','M','VI','Score_anomaly','P','F','C'],
}

def honest_cv_auc(X, n_repeats=N_RUNS, n_folds=N_FOLDS):
    """Stratified CV: standardize per-fold from training data only."""
    aucs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rep)
        fold_aucs = []
        for tr, te in skf.split(X, y):
            mu, sig = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
            X_tr = (X[tr] - mu) / sig
            X_te = (X[te] - mu) / sig
            lr = LogisticRegression(max_iter=2000, solver='lbfgs')
            lr.fit(X_tr, y[tr])
            fold_aucs.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:,1]))
        aucs.append(np.mean(fold_aucs))
    arr = np.array(aucs)
    return float(arr.mean()), float(arr.std(ddof=1)), float(arr.std(ddof=1)/np.sqrt(len(arr)))

# ==========================================================
# 3. CORE AUC RESULTS
# ==========================================================
print("\n=== HONEST CROSS-VALIDATION (20 runs, 5-fold, per-fold std) ===")
results = {}
for name, flist in FEATURE_SETS.items():
    X = df[flist].values
    m, sd, se = honest_cv_auc(X)
    ci_lo, ci_hi = m - 1.96*se, m + 1.96*se
    results[name] = {'m': m, 'sd': sd, 'se': se, 'ci_lo': ci_lo, 'ci_hi': ci_hi}
    print(f"  {name:12s}: {m:.4f} +/- {se:.4f}  CI [{ci_lo:.4f}, {ci_hi:.4f}]")

# ==========================================================
# 4. MODEL C vs B SIGNIFICANCE TEST
# ==========================================================
X_b = df[FEATURE_SETS['Model B']].values
X_c = df[FEATURE_SETS['Model C']].values
b_aucs, c_aucs = [], []
for rep in range(N_RUNS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
    rb, rc = [], []
    for tr, te in skf.split(X_b, y):
        for X_arr, lst in [(X_b, rb), (X_c, rc)]:
            mu = X_arr[tr].mean(axis=0); sig = X_arr[tr].std(axis=0) + 1e-8
            X_tr = (X_arr[tr] - mu) / sig; X_te = (X_arr[te] - mu) / sig
            lr = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X_tr, y[tr])
            lst.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:,1]))
    b_aucs.append(np.mean(rb)); c_aucs.append(np.mean(rc))

t_bc, p_bc = stats.ttest_rel(np.array(c_aucs), np.array(b_aucs))
delta_bc = float(np.mean(np.array(c_aucs) - np.array(b_aucs)))
print(f"\nModel C vs B: delta={delta_bc:.4f}, t={t_bc:.4f}, p={p_bc:.4f}")

# Also B vs A, B vs Baseline
X_a = df[FEATURE_SETS['Model A']].values
X_base = df[FEATURE_SETS['Baseline']].values

# B vs A
a_aucs = []
for rep in range(N_RUNS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
    ro = []
    for tr, te in skf.split(X_a, y):
        mu = X_a[tr].mean(axis=0); sig = X_a[tr].std(axis=0) + 1e-8
        X_tr = (X_a[tr]-mu)/sig; X_te = (X_a[te]-mu)/sig
        lr = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X_tr, y[tr])
        ro.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:,1]))
    a_aucs.append(np.mean(ro))
t_ba, p_ba = stats.ttest_rel(np.array(b_aucs), np.array(a_aucs))
delta_ba = float(np.mean(np.array(b_aucs) - np.array(a_aucs)))

# B vs Baseline
base_aucs = []
for rep in range(N_RUNS):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=rep)
    ro = []
    for tr, te in skf.split(X_base, y):
        mu = X_base[tr].mean(axis=0); sig = X_base[tr].std(axis=0) + 1e-8
        X_tr = (X_base[tr]-mu)/sig; X_te = (X_base[te]-mu)/sig
        lr = LogisticRegression(max_iter=2000, solver='lbfgs').fit(X_tr, y[tr])
        ro.append(roc_auc_score(y[te], lr.predict_proba(X_te)[:,1]))
    base_aucs.append(np.mean(ro))
t_bb, p_bb = stats.ttest_rel(np.array(b_aucs), np.array(base_aucs))
delta_bb = float(np.mean(np.array(b_aucs) - np.array(base_aucs)))

print(f"Model B vs Model A:  delta={delta_ba:.4f}, t={t_ba:.4f}, p={p_ba:.4e}")
print(f"Model B vs Baseline: delta={delta_bb:.4f}, t={t_bb:.4f}, p={p_bb:.4e}")

# ==========================================================
# 5. ABLATION STUDY
# ==========================================================
print("\n=== HONEST ABLATION STUDY ===")
ablation_configs = {
    'Model B (Full)':             [0,1,2,3,4,5,6,7,8],
    'B - Trade Momentum (M_i)':   [0,1,2,3,4,5,  7,8],
    'B - Buyer HHI (VI_i)':       [0,1,2,3,4,5,6,  8],
    'B - Anomaly Detection':      [0,1,2,3,4,5,6,7 ],
    'B - All Three Engines':      [0,1,2,3,4,5      ],
}

ablation = {}
for name, cols in ablation_configs.items():
    X_sub = X_b[:, cols]
    m, sd, se = honest_cv_auc(X_sub)
    ablation[name] = {'m': m, 'se': se}
    delta = m - results['Model B']['m']
    print(f"  {name:28s}: {m:.4f} +/- {se:.4f}  delta={delta:+.4f}")

# ==========================================================
# 6. COST & NET UTILITY
# ==========================================================
print("\n=== NET UTILITY ===")
W = {'Baseline': 0.05, 'Model A': 0.10, 'Model B': 0.40, 'Model C': 1.00}
LAMBDA_DEFAULT = 4.0

for name in ['Baseline', 'Model A', 'Model B', 'Model C']:
    auc = results[name]['m']
    cost = 0.01 * np.exp(LAMBDA_DEFAULT * W[name])
    netu = auc - cost
    results[name]['cost'] = float(cost)
    results[name]['netu'] = float(netu)
    print(f"  {name}: AUC={auc:.4f}  Cost={cost:.4f}  NetU={netu:.4f}")

# ==========================================================
# 7. SENSITIVITY TABLE
# ==========================================================
print("\n=== SENSITIVITY ===")
LAMBDAS = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
sensitivity = []
for lam in LAMBDAS:
    netus = {}
    for name in ['Baseline', 'Model A', 'Model B', 'Model C']:
        cost = 0.01 * np.exp(lam * W[name])
        netus[name] = results[name]['m'] - cost
    peak = max(netus, key=netus.get)
    row = {'lambda': lam}
    for name in ['Baseline', 'Model A', 'Model B', 'Model C']:
        row[name] = netus[name]
    row['peak'] = peak
    sensitivity.append(row)
    print(f"  lam={lam:.1f}  Base={netus['Baseline']:.4f}  A={netus['Model A']:.4f}  B={netus['Model B']:.4f}  C={netus['Model C']:.4f}  peak={peak}")

# ==========================================================
# 8. SAVE RESULTS TO JSON
# ==========================================================
output = {
    'AUC': {name: {'mean': results[name]['m'], 'se': results[name]['se'],
                    'ci_lo': results[name]['ci_lo'], 'ci_hi': results[name]['ci_hi'],
                    'cost': round(results[name]['cost'], 4), 'netu': round(results[name]['netu'], 4)}
             for name in ['Baseline', 'Model A', 'Model B', 'Model C']},
    'significance': {
        'B_vs_A':     {'delta': delta_ba, 't': float(t_ba), 'p': float(p_ba)},
        'B_vs_Base':  {'delta': delta_bb, 't': float(t_bb), 'p': float(p_bb)},
        'C_vs_B':     {'delta': delta_bc, 't': float(t_bc), 'p': float(p_bc)},
    },
    'ablation': {name: {'mean': ablation[name]['m'], 'se': ablation[name]['se'],
                        'delta': ablation[name]['m'] - results['Model B']['m']}
                  for name in ablation_configs},
    'sensitivity': sensitivity,
}
with open('honest_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nSaved: honest_results.json")

# Print summary table for D10
print("\n" + "="*70)
print("VALUES FOR D10.docx")
print("="*70)
for name in ['Baseline', 'Model A', 'Model B', 'Model C']:
    r = results[name]
    print(f"  {name}: AUC={r['m']:.4f}±{r['se']:.4f}  CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  Cost={r['cost']:.4f}  NetU={r['netu']:.4f}")

print(f"\n  C vs B: delta={delta_bc:.4f}, p={p_bc:.4f}")
print(f"  B vs A: delta={delta_ba:.4f}, p={p_ba:.4e}")
print(f"  B vs Base: delta={delta_bb:.4f}, p={p_bb:.4e}")
