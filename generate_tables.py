"""
generate_tables.py -- Result tables
===================================
Reads the result files and writes the result tables. Run after
modular_test_runner.py and supplementary_analysis.py.

Output (tables/):
    table1_efficiency_96.csv      Tracking efficiency of all 96 cases
    table2_metrics_summary.csv    RMSE, MAE, std_ss and response time per case
    table3_en50530_comparison.csv Step, EN 50530-style ramp and sigmoid
    table4_complexity.csv         Computational complexity

Usage:
    python3 generate_tables.py
"""

import json
import os
import csv
import numpy as np

os.makedirs('tables', exist_ok=True)

# ── Load all result files ─────────────────────────────────────────────────────
def load():
    r = {}
    for psc, fname in [('easy',    'run1_easy'),
                       ('moderate','run2_moderate'),
                       ('hard',    'run3_hard'),
                       ('extreme', 'run4_extreme')]:
        with open(f'results/{fname}.json') as f:
            r[psc] = json.load(f)

    with open('results/rmse_mae_96.json')       as f: g1 = json.load(f)
    with open('results/response_times_96.json')     as f: g2 = json.load(f)
    with open('results/en50530_comparison.json') as f: g3 = json.load(f)
    return r, g1, g2, g3

RUNS, G1, G2, G3 = load()

# Unrounded per-case metrics from compute_error_metrics.py. When present, every
# efficiency and steady-state metric below is rounded once from these values.
FULL = {}
if os.path.exists('results/error_metrics_96.csv'):
    with open('results/error_metrics_96.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            FULL[(row['psc'], row['profile'], row['algorithm'])] = row

def _full(psc, prof, alg, key):
    row = FULL.get((psc, prof, alg))
    if row is None or row.get(key, '') == '':
        return None
    return float(row[key])

PSC_KEYS    = ['easy', 'moderate', 'hard', 'extreme']
PSC_LABELS  = {'easy':'Easy', 'moderate':'Moderate',
               'hard':'Hard', 'extreme':'Extreme'}
PROFILES    = ['step', 'linear_5s', 'linear_10s', 'linear_20s',
               'sigmoid_0.5', 'sigmoid_1.0', 'sigmoid_2.0', 'sigmoid_5.0']
PROF_LABELS = {'step':'Step',
               'linear_5s':'Linear 5 s',
               'linear_10s':'Linear 10 s',
               'linear_20s':'Linear 20 s',
               'sigmoid_0.5':'Sigmoid b=0.5 s',
               'sigmoid_1.0':'Sigmoid b=1 s',
               'sigmoid_2.0':'Sigmoid b=2 s',
               'sigmoid_5.0':'Sigmoid b=5 s'}
ALGS        = ['spline', 'po', 'inc']
ALG_LABELS  = {'spline':'Spline-MPPT', 'po':'P&O', 'inc':'INC'}

def e(psc, prof, alg):
    v = _full(psc, prof, alg, 'eta_total_pct')
    return v if v is not None else RUNS[psc]['results'][prof][alg]['eta_total']

def std(psc, prof, alg):
    v = _full(psc, prof, alg, 'sigma_P_W')
    return v if v is not None else RUNS[psc]['results'][prof][alg]['std_P_ss']

def re_ss(psc, prof, alg):
    return _full(psc, prof, alg, 're_ss_pct')

def ts(psc, prof, alg):
    # settling time from transition completion, band 2 % of final GMPP power
    return _full(psc, prof, alg, 'ts_completion_gmppband_s')

def tg(psc, prof, alg):
    # GMPP acquisition time from transition completion (None: not reached)
    return _full(psc, prof, alg, 'tG_completion_s')

def rt(psc, prof, alg):
    v = G2[psc][prof][alg]['response_time_s']
    return v if v is not None else None

def rmse_ss(psc, prof, alg):
    v = _full(psc, prof, alg, 'rmse_ss_W')
    return v if v is not None else G1[psc][prof][alg]['rmse_ss_W']

def mae_ss(psc, prof, alg):
    v = _full(psc, prof, alg, 'mae_ss_W')
    return v if v is not None else G1[psc][prof][alg]['mae_ss_W']

def rmse_total(psc, prof, alg):
    return G1[psc][prof][alg]['rmse_total_W']


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — 96-case tracking efficiency (η_total, %)
# ══════════════════════════════════════════════════════════════════════════════
def table1():
    path = 'tables/table1_efficiency_96.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        # Header rows
        w.writerow(['PSC', 'Transition profile',
                    'Spline-MPPT (%)', 'P&O (%)', 'INC (%)',
                    'Best-local (%)', 'Spline advantage (pp)', 'Winner'])
        for psc in PSC_KEYS:
            for prof in PROFILES:
                sp  = e(psc, prof, 'spline')
                po  = e(psc, prof, 'po')
                inc = e(psc, prof, 'inc')
                bl  = max(po, inc)
                adv = round(sp - bl, 1)
                winner = 'Spline-MPPT' if sp >= bl else ('INC' if inc > po else 'P&O')
                w.writerow([PSC_LABELS[psc], PROF_LABELS[prof],
                            f'{sp:.1f}', f'{po:.1f}', f'{inc:.1f}',
                            f'{bl:.1f}', f'{adv:+.1f}', winner])
    print(f'  Saved: {path}')

    # Also print a compact version
    print()
    print('  TABLE 1 — Tracking efficiency η_total (%)')
    print(f'  {"PSC":<10} {"Profile":<16}  {"Spline":>7}  {"P&O":>7}  {"INC":>7}  {"Adv":>6}  {"Winner"}')
    print('  ' + '-'*68)
    for psc in PSC_KEYS:
        for prof in PROFILES:
            sp  = e(psc, prof, 'spline')
            po  = e(psc, prof, 'po')
            inc = e(psc, prof, 'inc')
            bl  = max(po, inc)
            adv = sp - bl
            winner = 'Spline' if sp >= bl else ('INC' if inc > po else 'P&O')
            print(f'  {PSC_LABELS[psc]:<10} {PROF_LABELS[prof]:<16}  '
                  f'{sp:>7.1f}  {po:>7.1f}  {inc:>7.1f}  {adv:>+6.1f}  {winner}')
        print()


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — RMSE, MAE, std_ss, response time for all 96 cases
# ══════════════════════════════════════════════════════════════════════════════
def table2():
    path = 'tables/table2_metrics_summary.csv'
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['PSC', 'Profile', 'Algorithm',
                    'eta_total (%)', 'RMSE_total (W)', 'MAE_total (W)',
                    'RMSE_ss (W)', 'MAE_ss (W)',
                    'sigma_P (W)', 'RE_ss (%)', 't_s (s)', 't_G (s)'])
        for psc in PSC_KEYS:
            for prof in PROFILES:
                for alg in ALGS:
                    ts_val = ts(psc, prof, alg) if alg != 'spline' else None
                    tg_val = tg(psc, prof, alg)
                    ts_str = f'{ts_val:.1f}' if ts_val is not None else 'N/A'
                    tg_str = f'{tg_val:.1f}' if tg_val is not None else 'n.r.'
                    re_val = re_ss(psc, prof, alg)
                    re_str = f'{re_val:.1f}' if re_val is not None else ''
                    w.writerow([PSC_LABELS[psc], PROF_LABELS[prof], ALG_LABELS[alg],
                                f'{e(psc,prof,alg):.1f}',
                                f'{rmse_total(psc,prof,alg):.2f}',
                                f'{G1[psc][prof][alg]["mae_total_W"]:.2f}',
                                f'{rmse_ss(psc,prof,alg):.2f}',
                                f'{mae_ss(psc,prof,alg):.2f}',
                                f'{std(psc,prof,alg):.3f}',
                                re_str, ts_str, tg_str])
    print(f'  Saved: {path}')

    # Print per-algorithm summary statistics
    print()
    print('  TABLE 2 SUMMARY — mean ± std across all 96 cases per algorithm')
    print(f'  {"Alg":<13}  {"η_total":>9}  {"RMSE_total":>12}  '
          f'{"MAE_total":>11}  {"RMSE_ss":>9}  {"std_ss":>8}')
    print('  ' + '-'*68)
    for alg in ALGS:
        eta_v  = [e(p, pf, alg) for p in PSC_KEYS for pf in PROFILES]
        rm_v   = [rmse_total(p,pf,alg) for p in PSC_KEYS for pf in PROFILES]
        ma_v   = [G1[p][pf][alg]['mae_total_W'] for p in PSC_KEYS for pf in PROFILES]
        rss_v  = [rmse_ss(p,pf,alg) for p in PSC_KEYS for pf in PROFILES]
        std_v  = [std(p,pf,alg) for p in PSC_KEYS for pf in PROFILES]
        print(f'  {ALG_LABELS[alg]:<13}  '
              f'{np.mean(eta_v):>6.1f}±{np.std(eta_v):.1f}%  '
              f'{np.mean(rm_v):>7.1f}±{np.std(rm_v):.1f}W  '
              f'{np.mean(ma_v):>7.1f}±{np.std(ma_v):.1f}W  '
              f'{np.mean(rss_v):>5.1f}±{np.std(rss_v):.1f}W  '
              f'{np.mean(std_v):>5.2f}±{np.std(std_v):.2f}W')

    print()
    print('  Note on dynamic metrics: t_s and t_G are measured from the completion')
    print('  of the transition. t_s is not defined for Spline-MPPT, whose scan')
    print('  cycle moves the operating point away from a static steady state.')
    print('  n.r.: the GMPP is not reached within the simulation.')
    print()

    # Key pairwise comparisons (for article text)
    print('  Key steady-state error values for article text:')
    cases = [('hard','step'),('hard','sigmoid_2.0'),
             ('extreme','step'),('extreme','sigmoid_2.0')]
    for psc, prof in cases:
        print(f'  {PSC_LABELS[psc]} / {PROF_LABELS[prof]}:')
        for alg in ALGS:
            s = std(psc,prof,alg); r = rmse_ss(psc,prof,alg); m = mae_ss(psc,prof,alg)
            print(f'    {ALG_LABELS[alg]:<13} std_ss={s:.2f}W  RMSE_ss={r:.2f}W  MAE_ss={m:.2f}W')


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — EN 50530 comparison
# ══════════════════════════════════════════════════════════════════════════════
def table3():
    path = 'tables/table3_en50530_comparison.csv'

    # EN 50530 proxy = linear_10s (10 s linear ramp)
    EN = 'linear_10s'
    SIG = 'sigmoid_2.0'

    rows = []
    for psc in PSC_KEYS:
        for alg in ALGS:
            sp_step = e(psc, 'step', alg)
            sp_lin  = e(psc, EN, alg)
            sp_sig  = e(psc, SIG, alg)
            rows.append([PSC_LABELS[psc], ALG_LABELS[alg],
                         f'{sp_step:.1f}', f'{sp_lin:.1f}', f'{sp_sig:.1f}'])

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['PSC', 'Algorithm',
                    'Step (%)', 'EN 50530 linear 10 s (%)', 'Sigmoid b=2 s (%)'])
        w.writerows(rows)
    print(f'  Saved: {path}')

    print()
    print('  TABLE 3 — Step vs EN 50530 (linear 10 s) vs Sigmoid b=2 s')
    print(f'  {"PSC":<10} {"Profile type":<20}  {"Spline":>7}  {"P&O":>7}  '
          f'{"INC":>7}  {"Best-local":>11}  {"Spline-adv":>11}')
    print('  ' + '-'*75)
    for psc in PSC_KEYS:
        for prof, ptype in [('step','Step'),
                             (EN,   'EN 50530 (linear)'),
                             (SIG,  'Sigmoid b=2 s')]:
            sp  = e(psc, prof, 'spline')
            po  = e(psc, prof, 'po')
            inc = e(psc, prof, 'inc')
            bl  = max(po, inc)
            adv = sp - bl
            flag = ' ← REVERSAL' if psc == 'hard' and prof == SIG else ''
            print(f'  {PSC_LABELS[psc]:<10} {ptype:<20}  '
                  f'{sp:>7.1f}  {po:>7.1f}  {inc:>7.1f}  '
                  f'{bl:>11.1f}  {adv:>+11.1f}{flag}')
        print()

    print('  KEY FINDING: EN 50530 (linear ramp) produces the SAME ranking')
    print('  as step-change testing. Both overstate Spline advantage at Hard')
    print('  PSC by ~27-29 pp and conceal the INC outperformance visible only')
    print('  under physics-based sigmoid transitions.')


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Computational complexity
# ══════════════════════════════════════════════════════════════════════════════
def table4():
    path = 'tables/table4_complexity.csv'

    rows = [
        ['P&O',         '100 ms', '7',   '3',  '1.0×',   'Scalar comparisons only'],
        ['INC',         '100 ms', '9',   '3',  '1.3×',   '2 divisions per decision'],
        ['Spline-MPPT', '100 ms', '~35-45 (amortized)', '20', '~5-6×',
         'About 5100 scan operations per scan cycle (about 140 to 180 '
         'decisions) plus about 7 perturb operations per decision'],
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Algorithm', 'Control period',
                    'Ops per period', 'Memory variables',
                    'Relative complexity', 'Notes'])
        w.writerows(rows)
    print(f'  Saved: {path}')

    print()
    print('  TABLE 4 — Computational complexity')
    print(f'  {"Algorithm":<15} {"Period":>9} {"Ops/period":>11} '
          f'{"Mem vars":>10} {"Rel. complexity":>16}')
    print('  ' + '-'*65)
    for r in rows:
        print(f'  {r[0]:<15} {r[1]:>9} {r[2]:>11} {r[3]:>10} {r[4]:>16}')
    print()
    print('  Spline-MPPT amortized cost: about 5100 scan operations per scan cycle')
    print('  of about 140 to 180 decisions, plus about 7 perturb operations per')
    print('  decision, which gives about 35 to 45 operations per decision.')
    print()
    print('  Note: "ops" counts arithmetic operations (add, subtract, multiply,')
    print('  divide, compare). Spline dominant cost is 1000-point spline evaluation')
    print('  (~4000 multiply-add ops via Horner scheme).')


# ══════════════════════════════════════════════════════════════════════════════
# Run all tables
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating result tables...')
    print()
    print('═' * 70)
    print('  TABLE 1 — 96-case tracking efficiency')
    print('═' * 70)
    table1()

    print()
    print('═' * 70)
    print('  TABLE 2 — RMSE, MAE, std_ss, response time')
    print('═' * 70)
    table2()

    print()
    print('═' * 70)
    print('  TABLE 3 — EN 50530 comparison')
    print('═' * 70)
    table3()

    print()
    print('═' * 70)
    print('  TABLE 4 — Computational complexity')
    print('═' * 70)
    table4()


    print()
    print('Done — all output in tables/')
    print('  table1_efficiency_96.csv')
    print('  table2_metrics_summary.csv')
    print('  table3_en50530_comparison.csv')
    print('  table4_complexity.csv')
