"""
study_fast_transitions.py — Fast-transition study (round-2 revision)
====================================================================
Extends the 96-case matrix with transition profiles faster than the
b = 0.5 s lower bound of the main test set: sigmoid b = 0.1 s and
b = 0.25 s, and a 1 s linear ramp. For the Hard PSC 900-to-100 W/m2
module swing these correspond to peak rates of roughly 2000, 800 and
800 W/m2/s, all far above the 100 W/m2/s threshold the review raised.

Uses the exact generators, PSC patterns, module model, simulator and
algorithm factory of modular_test_runner.py. Nothing is re-implemented.

Output: results/study_fast_transitions.json
"""
import json, os, sys, time
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

import modular_test_runner as mtr
from mppt_algorithms import MPPTSimulator

FAST_PROFILES = {
    'sigmoid_0.1':  ('sigmoid', 0.1),
    'sigmoid_0.25': ('sigmoid', 0.25),
    'linear_1s':    ('linear', 1.0),
}

def build_fast(psc_key, kind, par):
    psc = mtr.PSC_PATTERNS[psc_key]
    Gi, Gf = psc['G_init'], psc['G_final']
    t0 = mtr.T_BEFORE
    if kind == 'sigmoid':
        T_total = mtr.T_BEFORE + 10.0 * par + mtr.T_AFTER
    else:
        T_total = mtr.T_BEFORE + par + mtr.T_AFTER
    t_vec = np.arange(0.0, T_total + mtr.DT, mtr.DT)
    G1 = np.full_like(t_vec, Gi[0])
    G2 = np.full_like(t_vec, Gi[1])
    if kind == 'sigmoid':
        G3 = mtr._sigmoid(t_vec, Gi[2], Gf[2], t0, par)
    else:
        G3 = mtr._linear(t_vec, Gi[2], Gf[2], t0, par)
    return t_vec, G1, G2, G3

def main():
    out = {'metadata': {'study': 'fast_transitions',
                        'profiles': list(FAST_PROFILES),
                        'dt': mtr.DT, 'T_celsius': mtr.T_CELSIUS,
                        'note': 'extends the 96-case matrix below b = 0.5 s'},
           'results': {}}
    os.makedirs(mtr.RESULTS_DIR, exist_ok=True)
    out_file = os.path.join(mtr.RESULTS_DIR, 'study_fast_transitions.json')
    for psc_key in mtr.PSC_PATTERNS:
        out['results'][psc_key] = {}
        for prof, (kind, par) in FAST_PROFILES.items():
            out['results'][psc_key][prof] = {}
            t_vec, G1, G2, G3 = build_fast(psc_key, kind, par)
            for alg_key in mtr.ALGORITHMS:
                t0 = time.time()
                alg = mtr.make_algorithm(alg_key)
                sim = MPPTSimulator(mtr.MODULE, alg, dt=mtr.DT,
                                    T_celsius=mtr.T_CELSIUS,
                                    num_iv_points=mtr.IV_POINTS)
                res = sim.run(t_vec, G1, G2, G3)
                out['results'][psc_key][prof][alg_key] = {
                    'eta_total': res['eta_total'],
                    'eta_trans': res['eta_trans'],
                    'std_P_ss': res['std_P_ss'],
                    'mean_P_ss': res['mean_P_ss'],
                    'mean_gmpp_ss': res['mean_gmpp_ss'],
                    'settling_time': res['settling_time'],
                    'runtime_s': round(time.time() - t0, 2)}
                print(f"{psc_key:9s} {prof:13s} {alg_key:6s} "
                      f"eta={res['eta_total']:5.1f}%  [{time.time()-t0:.1f}s]")
            with open(out_file, 'w') as f:
                json.dump(out, f, indent=2, default=str)
    print("saved", out_file)

if __name__ == '__main__':
    main()
