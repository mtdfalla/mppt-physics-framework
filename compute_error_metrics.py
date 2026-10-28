#!/usr/bin/env python3
"""
compute_error_metrics.py -- Error and dynamic metrics for the 96-case MPPT test matrix
=====================================================================================

Regenerates, from the released simulation framework, the complete set of
steady-state error metrics and response-time metrics for the primary test
matrix (4 partial-shading conditions x 8 irradiance transition profiles x
3 MPPT algorithms, cell temperature 25 C).

Each case is simulated with the released modules (``modular_test_runner``,
``mppt_algorithms``, ``tct_eval``) without modification, and the full time
series t_k, P_tracked(t_k), P_GMPP(t_k) and V_op(t_k) are retained. All
metrics are then computed directly from these trajectories.

Definitions (article, Section 2.5)
----------------------------------
Let S be the set of N_ss samples in the final 20 % of a simulation
(indices k >= floor(0.8 N)), and e_k = P_GMPP(t_k) - P_tracked(t_k).

    RMSE_ss  = sqrt( (1/N_ss) * sum_{k in S} e_k^2 )                      [W]
    MAE_ss   = (1/N_ss) * sum_{k in S} |e_k|                              [W]
    sigma_P  = population standard deviation of P_tracked over S          [W]
    Pbar_GMPP,ss = (1/N_ss) * sum_{k in S} P_GMPP(t_k)                    [W]
    nRMSE_ss = 100 * RMSE_ss / Pbar_GMPP,ss                               [%]
    RE_ss    = 100 * MAE_ss  / Pbar_GMPP,ss                               [%]
    MRE_ss   = 100 * (1/N_ss) * sum_{k in S} |e_k| / P_GMPP(t_k)          [%]
               (pointwise mean relative error, reported for reference)
    eta_ss   = 100 * mean_S(P_tracked) / Pbar_GMPP,ss                     [%]

Final steady-state power:
    P_final = mean of P_tracked over the final 15 % of the simulation
              (indices k >= floor(0.85 N)).

Response time, onset-referenced (definition used in the article):
    t_r = t_k* - t_start, where k* is the first sample with t_k >= t_start
    such that |P_tracked - P_final| < 0.02 P_final holds for the 10
    consecutive samples k*, ..., k*+9 (1 s at the 0.1 s step).

Settling time, completion-referenced (alternative):
    t_s = t_k* - t_end, with the same band and hold criterion but the search
    started at the first sample with t_k >= t_end, the instant at which the
    irradiance transition is complete. t_s = 0 when the tracked power is
    already settled at t_end.

Final settling time, completion-referenced:
    t_s,final = time after t_end from which P_tracked remains inside the
    +-2 % band until the end of the simulation (0 if it never leaves it).

Post-transition error metrics (supplementary window):
    RMSE_post, MAE_post, sigma_P,post, nRMSE_post, RE_post are the same
    quantities evaluated over the complete 30 s hold that follows transition
    completion (t_k >= t_end). This window has the same length for every
    profile and spans at least one full scan-and-track cycle of Spline-MPPT.

Settling time with GMPP-referred band width:
    as t_s, but with the band half-width 0.02 * mean(P_GMPP) over the final
    15 % instead of 0.02 * P_final, so that operation at a low-power local
    maximum is not judged against a band narrower than the P&O ripple.

GMPP-referenced acquisition time:
    t_G = time after t_end until |P_tracked - P_GMPP| < 0.02 P_GMPP holds
    for 1 s; "not reached" if this never occurs.

Transition timing (from ``build_irradiance``):
    step         : t_start = t_end = 5.0 s
    linear_D     : ramp over [5, 5 + D) s, G = G_final from t = 5 + D s
    sigmoid_b    : G(t) = a / (1 + exp((t - t_mid)/|b|)) + c,
                   t_mid = 5 + 5|b|, applied from t = 5 s; t_end = 5 + 10|b|
                   (98.7 % of the change lies inside [5, 5 + 10|b|]).

Usage
-----
    python3 compute_error_metrics.py --code-dir /path/to/released/code
                                     [--out-dir results] [--traj-dir traj]
                                     [--jobs 2]

Outputs (in --out-dir):
    error_metrics_96.csv        all metrics for the 96 cases
    error_metrics_96_table.tex  longtable for an appendix
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

PSC_KEYS = ['easy', 'moderate', 'hard', 'extreme']
PROFILES = ['step', 'linear_5s', 'linear_10s', 'linear_20s',
            'sigmoid_0.5', 'sigmoid_1.0', 'sigmoid_2.0', 'sigmoid_5.0']
ALGS = ['spline', 'po', 'inc']

PSC_LABEL = {'easy': 'Easy', 'moderate': 'Moderate',
             'hard': 'Hard', 'extreme': 'Extreme'}
PROF_TEX = {'step': 'Step', 'linear_5s': 'Linear 5\\,s',
            'linear_10s': 'Linear 10\\,s', 'linear_20s': 'Linear 20\\,s',
            'sigmoid_0.5': 'Sigmoid 0.5\\,s', 'sigmoid_1.0': 'Sigmoid 1\\,s',
            'sigmoid_2.0': 'Sigmoid 2\\,s', 'sigmoid_5.0': 'Sigmoid 5\\,s'}
ALG_TEX = {'spline': 'Spline-MPPT', 'po': 'P\\&O', 'inc': 'INC'}

SS_FRACTION = 0.20       # error-metric window: final 20 % of the simulation
FINAL_FRACTION = 0.15    # final steady-state power: final 15 % of the simulation
BAND = 0.02              # +-2 % settling band
HOLD_S = 1.0             # band must hold for 1 s
T_START = 5.0            # transition onset [s]


# ---------------------------------------------------------------------------
# Transition timing
# ---------------------------------------------------------------------------
def transition_window(profile):
    """Return (t_start, t_end) of the irradiance transition in seconds."""
    if profile == 'step':
        return T_START, T_START
    kind, par = profile.split('_')
    if kind == 'linear':
        return T_START, T_START + float(par.rstrip('s'))
    if kind == 'sigmoid':
        return T_START, T_START + 10.0 * abs(float(par))
    raise ValueError(profile)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(case, traj_dir=None):
    """Run one case with the released code and return its time series."""
    psc, prof, alg = case
    if traj_dir:
        path = os.path.join(traj_dir, f'{psc}__{prof}__{alg}.npz')
        if os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            return {'t': d['t'], 'P': d['P'], 'Pg': d['Pg'], 'V': d['V'],
                    'G3': d['G3'], 'summary': json.loads(str(d['summary']))}
    from modular_test_runner import (build_irradiance, make_algorithm, MODULE,
                                     DT, T_CELSIUS, IV_POINTS)
    from mppt_algorithms import MPPTSimulator
    t, g1, g2, g3 = build_irradiance(psc, prof)
    sim = MPPTSimulator(MODULE, make_algorithm(alg), dt=DT,
                        T_celsius=T_CELSIUS, num_iv_points=IV_POINTS)
    res = sim.run(t, g1, g2, g3)
    summary = {k: res[k] for k in ('eta_total', 'eta_trans', 'std_P_ss',
                                   'mean_P_ss', 'mean_gmpp_ss')}
    out = {'t': t, 'P': res['P_op'], 'Pg': res['P_gmpp'], 'V': res['V_op'],
           'G3': g3, 'summary': summary}
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
        np.savez(path, t=t, P=out['P'], Pg=out['Pg'], V=out['V'], G3=g3,
                 summary=json.dumps(summary))
    return out


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def first_hold(inside, k0, hold):
    """First index k >= k0 with inside[k:k+hold] all True, else None."""
    run = 0
    for k in range(k0, len(inside)):
        run = run + 1 if inside[k] else 0
        if run == hold:
            return k - hold + 1
    return None


def final_settle(inside, k0):
    """First index k >= k0 after which inside[] stays True to the end."""
    outside = np.where(~inside[k0:])[0]
    if outside.size == 0:
        return k0
    k = k0 + outside[-1] + 1
    return k if k < len(inside) else None


def trapezoid(y, x):
    """Trapezoidal integral, compatible with NumPy 1.x and 2.x."""
    f = getattr(np, 'trapezoid', None) or np.trapz
    return float(f(y, x))


def metrics(case, d):
    """Compute all error and dynamic metrics for one simulated case.

    ``d`` holds the arrays t, P (tracked power) and Pg (GMPP power).
    Times that are never reached are returned as None.
    """
    psc, prof, alg = case
    t, P, Pg = d['t'], d['P'], d['Pg']
    n = len(t)
    dt = float(t[1] - t[0])
    hold = int(round(HOLD_S / dt))
    t_start, t_end = transition_window(prof)
    k_start = int(np.searchsorted(t, t_start - 1e-9))
    k_end = int(np.searchsorted(t, t_end - 1e-9))

    # Error metrics over the final 20 % of the simulation
    k_ss = int(math.floor((1.0 - SS_FRACTION) * n + 1e-9))
    e = Pg[k_ss:] - P[k_ss:]
    rmse = float(np.sqrt(np.mean(e ** 2)))
    mae = float(np.mean(np.abs(e)))
    pg_ss = float(np.mean(Pg[k_ss:]))
    p_ss = float(np.mean(P[k_ss:]))
    sigma = float(np.std(P[k_ss:]))
    mre = float(100.0 * np.mean(np.abs(e) / Pg[k_ss:]))

    # Energy-based efficiency (Eq. 1)
    eta = 100.0 * trapezoid(P, t) / trapezoid(Pg, t)

    # Final steady-state power over the final 15 %
    k_fin = int(math.floor((1.0 - FINAL_FRACTION) * n + 1e-9))
    p_final = float(np.mean(P[k_fin:]))

    # Same error metrics over the full post-transition hold (t >= t_end)
    e_post = Pg[k_end:] - P[k_end:]
    pg_post = float(np.mean(Pg[k_end:]))
    rmse_post = float(np.sqrt(np.mean(e_post ** 2)))
    mae_post = float(np.mean(np.abs(e_post)))

    inside_own = np.abs(P - p_final) < BAND * p_final
    inside_gmpp = np.abs(P - Pg) < BAND * Pg

    def as_time(k, ref):
        return None if k is None else round(float(t[k]) - ref, 2)

    rt_onset = as_time(first_hold(inside_own, k_start, hold), t_start)
    rt_compl = as_time(first_hold(inside_own, k_end, hold), t_end)
    rt_final = as_time(final_settle(inside_own, k_end), t_end)
    # Same criterion with the band width referred to the final GMPP power
    pg_final = float(np.mean(Pg[k_fin:]))
    inside_abs = np.abs(P - p_final) < BAND * pg_final
    rt_compl_g = as_time(first_hold(inside_abs, k_end, hold), t_end)
    tg_onset = as_time(first_hold(inside_gmpp, k_start, hold), t_start)
    tg_compl = as_time(first_hold(inside_gmpp, k_end, hold), t_end)
    frac_gmpp_post = float(100.0 * np.mean(inside_gmpp[k_end:]))

    return {
        'psc': psc, 'profile': prof, 'algorithm': alg,
        't_start_s': t_start, 't_end_s': t_end,
        't_sim_end_s': round(float(t[-1]), 2), 'n_samples': n,
        'n_ss': n - k_ss, 'ss_window_s': round(float(t[k_ss]), 2),
        'eta_total_pct': eta,
        'rmse_ss_W': rmse, 'mae_ss_W': mae, 'sigma_P_W': sigma,
        'mean_P_ss_W': p_ss, 'mean_Pgmpp_ss_W': pg_ss,
        'nrmse_ss_pct': 100.0 * rmse / pg_ss, 're_ss_pct': 100.0 * mae / pg_ss,
        'mre_ss_pct': mre, 'eta_ss_pct': 100.0 * p_ss / pg_ss,
        'rmse_minus_mae_W': rmse - mae,
        'rmse_post_W': rmse_post, 'mae_post_W': mae_post,
        'sigma_P_post_W': float(np.std(P[k_end:])),
        'nrmse_post_pct': 100.0 * rmse_post / pg_post,
        're_post_pct': 100.0 * mae_post / pg_post,
        'P_final_W': p_final,
        'rt_onset_s': rt_onset,
        'ts_completion_s': rt_compl,
        'ts_final_completion_s': rt_final,
        'ts_completion_gmppband_s': rt_compl_g,
        'tG_onset_s': tg_onset,
        'tG_completion_s': tg_compl,
        'pct_time_within_2pct_gmpp_post': frac_gmpp_post,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def fmt_time(v):
    """Format a time value; None means the criterion was never met."""
    if v is None:
        return 'n.r.'
    return f'{v:.1f}'


def _csv_value(key, v):
    """CSV cell: blank for 'not reached', 2 decimals for times, 6 otherwise."""
    if v is None:
        return ''
    if isinstance(v, float):
        return f'{v:.2f}' if key.endswith('_s') else f'{v:.6f}'
    return v


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_value(k, r[k]) for k in keys})


def write_latex(rows, path):
    """Appendix longtable with the metrics reported in the article."""
    head = (
        '% Requires \\usepackage{longtable,booktabs}\n'
        '\\begingroup\\scriptsize\\setlength{\\tabcolsep}{3.5pt}\n'
        '\\begin{longtable}{@{}lll rr rr r rr@{}}\n'
        '\\caption{Error and dynamic metrics for all 96 primary test cases '
        '(25$^\\circ$C). RMSE$_\\mathrm{ss}$, MAE$_\\mathrm{ss}$, '
        'nRMSE$_\\mathrm{ss}$, RE$_\\mathrm{ss}$, and $\\sigma_P$ are evaluated '
        'over the final 20\\% of each simulation. $t_s$ and $t_G$ are '
        'measured from the completion of the transition as defined in '
        'Section~\\ref{sec:framework}. N/A: the Spline-MPPT scan cycle precludes '
        'conventional settling. n.r.: not reached within the simulation. Under '
        'Hard and Extreme PSC, the Spline-MPPT rows of profiles with the same '
        'simulation length (linear 5, 10, and 20~s and sigmoid $b = 0.5$, 1, and '
        '2~s) agree to within 0.01~W, because the first scan starts at the '
        'transition onset in both cases and the two profiles have nearly the same '
        'irradiance when the second scan takes place, so the two runs follow '
        'almost the same trajectory in the evaluation window.}'
        '\\label{tab:app_metrics}\\\\\n'
    )
    cols = ('\\toprule\n'
            'PSC & Profile & Algorithm & '
            '\\shortstack{RMSE$_\\mathrm{ss}$\\\\(W)} & '
            '\\shortstack{MAE$_\\mathrm{ss}$\\\\(W)} & '
            '\\shortstack{nRMSE$_\\mathrm{ss}$\\\\(\\%)} & '
            '\\shortstack{RE$_\\mathrm{ss}$\\\\(\\%)} & '
            '\\shortstack{$\\sigma_P$\\\\(W)} & '
            '\\shortstack{$t_s$\\\\(s)} & \\shortstack{$t_G$\\\\(s)} '
            '\\\\\n\\midrule\n')
    lines = [head, cols, '\\endfirsthead\n', cols, '\\endhead\n',
             '\\bottomrule\n\\endlastfoot\n']
    prev = (None, None)
    for r in rows:
        psc = PSC_LABEL[r['psc']] if r['psc'] != prev[0] else ''
        prof = (PROF_TEX[r['profile']]
                if (r['psc'], r['profile']) != prev else '')
        if (r['psc'], r['profile']) != prev and prev[0] is not None:
            lines.append('\\addlinespace[2pt]\n')
        prev = (r['psc'], r['profile'])
        if r['algorithm'] == 'spline':
            ts = 'N/A'
        else:
            ts = fmt_time(r['ts_completion_gmppband_s'])
        tg = fmt_time(r['tG_completion_s'])
        lines.append(
            f"{psc} & {prof} & {ALG_TEX[r['algorithm']]} & "
            f"{r['rmse_ss_W']:.1f} & "
            f"{r['mae_ss_W']:.1f} & {r['nrmse_ss_pct']:.1f} & "
            f"{r['re_ss_pct']:.1f} & {r['sigma_P_W']:.2f} & "
            f"{ts} & {tg} \\\\\n")
    lines.append('\\end{longtable}\n\\endgroup\n')
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def check_against_released(rows, results_dir):
    """Compare recomputed efficiencies with the released run files."""
    files = {'easy': 'run1_easy.json', 'moderate': 'run2_moderate.json',
             'hard': 'run3_hard.json', 'extreme': 'run4_extreme.json'}
    worst = 0.0
    for r in rows:
        fn = os.path.join(results_dir, files[r['psc']])
        if not os.path.exists(fn):
            return None
        with open(fn) as f:
            ref = json.load(f)['results'][r['profile']][r['algorithm']]
        worst = max(worst, abs(round(r['eta_total_pct'], 2) - ref['eta_total']))
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--code-dir', required=True,
                    help='directory containing the released simulation code')
    ap.add_argument('--out-dir', default='results')
    ap.add_argument('--traj-dir', default=None,
                    help='optional cache directory for trajectories (.npz)')
    ap.add_argument('--jobs', type=int, default=1)
    args = ap.parse_args()

    code_dir = os.path.abspath(args.code_dir)
    sys.dont_write_bytecode = True      # leave the code directory untouched
    sys.path.insert(0, code_dir)
    traj_dir = os.path.abspath(args.traj_dir) if args.traj_dir else None
    cases = [(p, f, a) for p in PSC_KEYS for f in PROFILES for a in ALGS]

    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            data = pool.starmap(simulate, [(c, traj_dir) for c in cases])
    else:
        data = [simulate(c, traj_dir) for c in cases]

    rows = [metrics(c, d) for c, d in zip(cases, data)]

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'error_metrics_96.csv')
    tex_path = os.path.join(args.out_dir, 'error_metrics_96_table.tex')
    write_csv(rows, csv_path)
    write_latex(rows, tex_path)

    worst = check_against_released(rows, os.path.join(code_dir, 'results'))
    if worst is not None:
        print(f'Maximum |eta - released eta_total| = {worst:.4f} pp')
    print(f'Wrote {csv_path}')
    print(f'Wrote {tex_path}')


if __name__ == '__main__':
    main()
