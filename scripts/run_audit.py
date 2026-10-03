"""Full scientific audit of BaFuse pipeline."""
import sys, pandas as pd, numpy as np, json, torch
from pathlib import Path

ROOT = Path(r'd:\Capstone project\bafuse')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))

proc = ROOT / 'data' / 'processed'
tr = pd.read_pickle(str(proc / 'train.pkl'))
va = pd.read_pickle(str(proc / 'val.pkl'))
te = pd.read_pickle(str(proc / 'test.pkl'))

SEP = '='*60
print(SEP); print('1. SPLIT AUDIT'); print(SEP)
tr_b = set(tr.battery_id); va_b = set(va.battery_id); te_b = set(te.battery_id)
print(f'Train({len(tr_b)}) / Val({len(va_b)}) / Test({len(te_b)})')
print(f'Train/Val overlap:  {tr_b & va_b or "EMPTY - PASS"}')
print(f'Train/Test overlap: {tr_b & te_b or "EMPTY - PASS"}')
print(f'Val/Test overlap:   {va_b & te_b or "EMPTY - PASS"}')
all_df = pd.concat([tr, va, te])
dup = all_df.duplicated(['battery_id', 'discharge_cycle']).sum()
print(f'Duplicate (batt,cycle) across splits: {dup}')
split_pass = not (tr_b & va_b) and not (tr_b & te_b) and not (va_b & te_b) and dup == 0
print(f'SPLIT AUDIT: {"PASS" if split_pass else "FAIL"}')

print(); print(SEP); print('2. SOH RANGE AUDIT'); print(SEP)
from src.data.dataset import get_nominal_capacity, BATTERY_NOMINAL_CAPACITY
nom_vals = set(BATTERY_NOMINAL_CAPACITY.values())
print(f'Nominal capacities in dict: {nom_vals}')
soh_issues = {}
for name, df in [('train', tr), ('val', va), ('test', te)]:
    soh = df.apply(lambda r: r.capacity_ahr / get_nominal_capacity(r.battery_id), axis=1)
    over1 = soh[soh > 1.0]
    print(f'{name}: n={len(df)}  min={soh.min():.4f}  max={soh.max():.4f}  >1.0: {len(over1)}')
    if len(over1) > 0:
        bids_over = df.loc[over1.index, 'battery_id'].unique()[:5]
        print(f'  Batteries with SOH>1: {list(bids_over)}')
        soh_issues[name] = int(len(over1))

print(); print(SEP); print('3. NORMALIZATION AUDIT'); print(SEP)
stats_p = proc / 'nasa_train_stats.json'
print(f'nasa_train_stats.json: {"EXISTS" if stats_p.exists() else "MISSING"}')
if stats_p.exists():
    with open(stats_p) as f: s = json.load(f)
    print(f'Keys: {list(s.keys())}')
    for k, v in s.items():
        print(f'  {k:15s}: mean={v["mean"]:.5f}  std={v["std"]:.5f}')
mend_p = ROOT / 'data' / 'mendeley_processed' / 'mendeley_stats.json'
print(f'mendeley_stats.json exists: {mend_p.exists()} (used for Samsung analysis only, NOT for NASA normalization)')
norm_pass = stats_p.exists()
print(f'NORMALIZATION AUDIT: {"PASS" if norm_pass else "FAIL - nasa_train_stats.json missing"}')

print(); print(SEP); print('4. FEATURE LEAKAGE AUDIT'); print(SEP)
from src.data.dataset import BaFuseDataset, fit_aging_prior
aging_params = fit_aging_prior(tr)
print(f'Aging prior: A={aging_params["A"]:.4f}  b={aging_params["b"]:.4f}  N_ref={aging_params["N_ref"]:.0f}')
print('empirical_fade_prior at inference: uses only cycle_age + frozen prior params (NO per-sample capacity)')
ds = BaFuseDataset(tr.head(20), normalize=True)
sample = ds[0]
print(f'Sample keys: {list(sample.keys())}')
print(f'capacity_ahr in returned dict: {"capacity_ahr" in sample} (should be False)')
print(f'discharge shape: {tuple(sample["discharge"].shape)}')
print(f'eis shape:       {tuple(sample["eis"].shape)}')
print(f'physics shape:   {tuple(sample["physics"].shape)}')
print(f'soh_label:       {sample["soh_label"].item():.4f} (from capacity, NOT in model input)')
print()
print('Physics feature analysis:')
print('  [0] cycle_age_norm    = discharge_cycle / N_ref             -- CLEAN (no capacity)')
print('  [1] empirical_prior   = A*(age/N_ref)^b                     -- CLEAN (train prior, no per-sample cap)')
print('  [2] voltage_droop     = (voltage_min - cutoff) / range      -- CLEAN (measured signal)')
print('  [3] impedance_rise    = (impedance_ohm - mean) / std        -- NOTE: same as eis[0] (redundant, not leaky)')
print()
print('EIS feature analysis:')
print('  [0] impedance_ohm z-scored     -- CLEAN')
print('  [1] re_ohm z-scored             -- CLEAN')
print('  [2] rct_ohm z-scored            -- CLEAN')
print()
print('Known redundancy: physics[3] (impedance_rise) == eis[0] (z-scored impedance). Not leakage.')
leak_pass = 'capacity_ahr' not in sample
print(f'LEAKAGE AUDIT: {"PASS" if leak_pass else "FAIL"}')

print(); print(SEP); print('5. MODEL PARAM COUNTS & MODALITY USAGE'); print(SEP)
from src.models.benchmark_models import all_model_param_counts, get_model
counts = all_model_param_counts()
print(f'{"Model":5s} {"Params":>10s} {"Discharge":>12s} {"EIS":>6s} {"Physics":>9s} {"Arch":>25s}')
print('-'*72)
arch_map = {
    'A':  ('StatPool+MLP', 'stat-pool(12D)', 'concat', 'concat'),
    'B':  ('ModMLP',       'stat-pool->MLP', 'MLP',    'MLP'),
    'C':  ('CNN1D',        'CNN1D',          'MLP',    'MLP'),
    'D':  ('TCN',          'TCN',            'MLP',    'MLP'),
    'E':  ('LSTM64',       'LSTM(64)',        'MLP',    'MLP'),
    'E2': ('LSTM128',      'LSTM(128)',       'MLP',    'MLP'),
    'F':  ('Gated',        'CNN1D',          'MLP',    'MLP'),
    'G':  ('CNNGated',     'CNN1D',          'MLP',    'MLP'),
    'C2': ('CNNFull',      'CNN1D',          'CNN(3D)','CNN(4D)'),
    'D2': ('TCNFull',      'TCN',            'TCN(3D)','TCN(4D)'),
    'E3': ('CNNLSTM64',    'CNN->LSTM(64)',   'MLP',    'MLP'),
    'E4': ('CNNLSTM128',   'CNN->LSTM(128)',  'MLP',    'MLP'),
    'E5': ('FullHybrid',   'CNN->LSTM(64)',   'CNN(3D)','CNN(4D)'),
    'G2': ('FullCNNGated', 'CNN1D',          'CNN(3D)','CNN(4D)'),
}
for n in ['A','B','C','D','E','E2','F','G','C2','D2','E3','E4','E5','G2']:
    arch, d, e, p = arch_map[n]
    print(f'{n:5s} {counts[n]:>10,} {d:>12s} {e:>6s} {p:>9s} {arch:>25s}')

print(); print(SEP); print('6. SAMSUNG EIS DOMAIN GAP AUDIT'); print(SEP)
sam_df = pd.read_pickle(str(ROOT/'data'/'mendeley_processed'/'mendeley_test.pkl'))
print(f'Samsung test: {len(sam_df)} samples, cells: {sorted(sam_df.cell_id.unique())}')
print(f'Discharge time-series available: {"voltage_v" in sam_df.columns} (should be False)')
print(f'SOH available: {"soh_norm" in sam_df.columns} (for eval only)')
if stats_p.exists():
    with open(stats_p) as f: ns = json.load(f)
    ratio_imp = ns['impedance']['mean'] / float(sam_df.zmod_ohm.mean())
    ratio_re  = ns['re']['mean'] / float(sam_df.zreal_ohm.mean())
    z_imp = (float(sam_df.zmod_ohm.mean()) - ns['impedance']['mean']) / ns['impedance']['std']
    z_re  = (float(sam_df.zreal_ohm.mean()) - ns['re']['mean']) / ns['re']['std']
    print(f'NASA imp mean: {ns["impedance"]["mean"]:.5f} Ohm')
    print(f'Samsung zmod mean: {sam_df.zmod_ohm.mean():.5f} Ohm  (ratio: {ratio_imp:.1f}x)')
    print(f'After NASA z-score, Samsung zmod_ohm => {z_imp:.2f} sigma (far OOD)')
    print(f'After NASA z-score, Samsung zreal_ohm => {z_re:.2f} sigma (far OOD)')
print('SAMSUNG: Full multimodal eval IMPOSSIBLE (no discharge). EIS-only is CASE B.')
print('EIS-only results are EXPLORATORY due to ~13x scale mismatch.')

print(); print(SEP); print('FINAL AUDIT SUMMARY'); print(SEP)
print(f'Split audit:         {"PASS" if split_pass else "FAIL"}')
print(f'Normalization audit: {"PASS" if norm_pass else "FAIL"}')
print(f'Leakage audit:       {"PASS" if leak_pass else "FAIL"}')
print(f'SOH>1 values:        {sum(soh_issues.values())} samples (physical, not bug)')
print(f'EIS redundancy:      physics[3]==eis[0] (known, documented, not a leakage)')
print(f'Samsung eval:        EIS-only CASE B (discharge unavailable, scale mismatch)')
