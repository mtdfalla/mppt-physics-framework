"""
modular_test_runner.py  —  Physics-Based MPPT Testing Framework
================================================================
Executes the 96-test matrix (3 algorithms × 4 PSC × 8 profiles).
Each PSC level runs independently and saves to JSON to prevent progress loss.

PSC Patterns (from manuscript Section 2.2)
-------------------------------------------
Easy    : G_init=(1000,900,700)  ->G_final=(1000,900,500)
Moderate: G_init=(1000,600,400)  ->G_final=(1000,600,300)
Hard    : G_init=(1000,400,200)  ->G_final=(1000,400,100)
Extreme : G_init=(1000,300,150)  ->G_final=(1000,300,50)

Transition Profiles (8 total)
------------------------------
step, linear_5s, linear_10s, linear_20s,
sigmoid_0.5, sigmoid_1.0, sigmoid_2.0, sigmoid_5.0

Algorithms (3 total)
---------------------
spline, po, inc

Usage
-----
python3 modular_test_runner.py --run 1   # Easy PSC
python3 modular_test_runner.py --run 2   # Moderate PSC
python3 modular_test_runner.py --run 3   # Hard PSC
python3 modular_test_runner.py --run 4   # Extreme PSC
python3 modular_test_runner.py --run all # All runs sequentially
"""

import numpy as np
import json
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import warnings

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

from tct_eval import default_kc200gt, evaluate_tct
from mppt_algorithms import (PandO, IncrementalConductance,
                              SplineMPPT, MPPTSimulator)

# ─────────────────────────────────────────────────────────────────────────────
# Global configuration
# ─────────────────────────────────────────────────────────────────────────────
DT          = 0.1        # simulation time step [s]
T_CELSIUS   = 25.0       # temperature [°C]
T_BEFORE    = 5.0        # steady-state time before transition [s]
T_AFTER     = 30.0       # hold time after transition [s]
IV_POINTS   = 200        # I–V curve resolution
RESULTS_DIR = 'results'

MODULE = default_kc200gt()

# ─────────────────────────────────────────────────────────────────────────────
# PSC definitions
# ─────────────────────────────────────────────────────────────────────────────
PSC_PATTERNS = {
    'easy': {
        'label':   'Easy (Mild Shading)',
        'G_init':  [1000.0, 900.0, 700.0],
        'G_final': [1000.0, 900.0, 500.0],
    },
    'moderate': {
        'label':   'Moderate Shading',
        'G_init':  [1000.0, 600.0, 400.0],
        'G_final': [1000.0, 600.0, 300.0],
    },
    'hard': {
        'label':   'Hard Shading',
        'G_init':  [1000.0, 400.0, 200.0],
        'G_final': [1000.0, 400.0, 100.0],
    },
    'extreme': {
        'label':   'Extreme Shading',
        'G_init':  [1000.0, 300.0, 150.0],
        'G_final': [1000.0, 300.0,  50.0],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Irradiance transition generators
# ─────────────────────────────────────────────────────────────────────────────

def _step(t: np.ndarray, Gi: float, Gf: float, t0: float) -> np.ndarray:
    return np.where(t < t0, Gi, Gf).astype(float)


def _linear(t: np.ndarray, Gi: float, Gf: float,
            t0: float, dur: float) -> np.ndarray:
    G = np.full_like(t, Gi, dtype=float)
    mask_ramp = (t >= t0) & (t < t0 + dur)
    mask_end  = t >= t0 + dur
    G[mask_ramp] = Gi + (Gf - Gi) * (t[mask_ramp] - t0) / dur
    G[mask_end]  = Gf
    return G


def _sigmoid(t: np.ndarray, Gi: float, Gf: float,
             t0_start: float, b: float) -> np.ndarray:
    """
    G(t) = a / (1 + exp((t − t_mid) / b)) + c
    t_mid = t0_start + 5|b|  so transition is ~1% complete at t0_start.
    """
    t_mid = t0_start + 5.0 * abs(b)
    a = Gi - Gf
    c = Gf
    G = a / (1.0 + np.exp((t - t_mid) / abs(b))) + c
    G[t < t0_start] = Gi
    return np.clip(G, min(Gi, Gf), max(Gi, Gf))


# ─────────────────────────────────────────────────────────────────────────────
# Build irradiance time series for one (psc, profile) combination
# ─────────────────────────────────────────────────────────────────────────────

def build_irradiance(psc_key: str, profile_key: str) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (t_vec, G1, G2, G3) arrays for the requested combination.
    Only the third module undergoes shading; modules 1 and 2 are held constant.
    """
    psc = PSC_PATTERNS[psc_key]
    Gi  = psc['G_init']
    Gf  = psc['G_final']
    t0  = T_BEFORE

    # Determine simulation end time based on profile
    if profile_key == 'step':
        T_total = T_BEFORE + T_AFTER
    elif profile_key == 'linear_5s':
        T_total = T_BEFORE + 5.0  + T_AFTER
    elif profile_key == 'linear_10s':
        T_total = T_BEFORE + 10.0 + T_AFTER
    elif profile_key == 'linear_20s':
        T_total = T_BEFORE + 20.0 + T_AFTER
    elif profile_key == 'sigmoid_0.5':
        T_total = T_BEFORE + 10.0 * 0.5 + T_AFTER   # 10|b|
    elif profile_key == 'sigmoid_1.0':
        T_total = T_BEFORE + 10.0 * 1.0 + T_AFTER
    elif profile_key == 'sigmoid_2.0':
        T_total = T_BEFORE + 10.0 * 2.0 + T_AFTER
    elif profile_key == 'sigmoid_5.0':
        T_total = T_BEFORE + 10.0 * 5.0 + T_AFTER
    else:
        raise ValueError(f"Unknown profile: {profile_key}")

    t_vec = np.arange(0.0, T_total, DT)

    # Module 1 and 2: held at their final (same as initial) irradiance
    # (only the third module changes in all PSC patterns)
    G1 = np.full_like(t_vec, Gi[0])
    G2 = np.full_like(t_vec, Gi[1])

    # Module 3: apply transition
    if profile_key == 'step':
        G3 = _step(t_vec, Gi[2], Gf[2], t0)
    elif profile_key == 'linear_5s':
        G3 = _linear(t_vec, Gi[2], Gf[2], t0, 5.0)
    elif profile_key == 'linear_10s':
        G3 = _linear(t_vec, Gi[2], Gf[2], t0, 10.0)
    elif profile_key == 'linear_20s':
        G3 = _linear(t_vec, Gi[2], Gf[2], t0, 20.0)
    elif profile_key == 'sigmoid_0.5':
        G3 = _sigmoid(t_vec, Gi[2], Gf[2], t0, 0.5)
    elif profile_key == 'sigmoid_1.0':
        G3 = _sigmoid(t_vec, Gi[2], Gf[2], t0, 1.0)
    elif profile_key == 'sigmoid_2.0':
        G3 = _sigmoid(t_vec, Gi[2], Gf[2], t0, 2.0)
    elif profile_key == 'sigmoid_5.0':
        G3 = _sigmoid(t_vec, Gi[2], Gf[2], t0, 5.0)

    return t_vec, G1, G2, G3


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm factory
# ─────────────────────────────────────────────────────────────────────────────

def make_algorithm(alg_key: str):
    if alg_key == 'spline':
        return SplineMPPT(T_scan=5.0, fine_step=0.5, T_fine=0.05)
    elif alg_key == 'po':
        return PandO(delta_V=1.0, T_perturb=0.05, V_init_frac=0.80)
    elif alg_key == 'inc':
        return IncrementalConductance(delta_V=1.0, T_sample=0.05,
                                      epsilon=0.01, V_init_frac=0.80)
    else:
        raise ValueError(f"Unknown algorithm: {alg_key}")


# ─────────────────────────────────────────────────────────────────────────────
# Run one PSC block
# ─────────────────────────────────────────────────────────────────────────────

PROFILES = ['step', 'linear_5s', 'linear_10s', 'linear_20s',
            'sigmoid_0.5', 'sigmoid_1.0', 'sigmoid_2.0', 'sigmoid_5.0']
ALGORITHMS = ['spline', 'po', 'inc']


def run_psc(psc_key: str, out_file: str) -> Dict:
    """
    Run all 8 profiles × 3 algorithms for one PSC level.
    Save incrementally to out_file (JSON) after each profile.
    """
    psc = PSC_PATTERNS[psc_key]
    print(f"\n{'='*60}")
    print(f"  PSC: {psc['label']}")
    print(f"  G_init  = {psc['G_init']}")
    print(f"  G_final = {psc['G_final']}")
    print(f"  Output  -> {out_file}")
    print(f"{'='*60}")

    output = {
        'metadata': {
            'psc_key':   psc_key,
            'psc_label': psc['label'],
            'G_init':    psc['G_init'],
            'G_final':   psc['G_final'],
            'timestamp': datetime.now().isoformat(),
            'dt':        DT,
            'T_celsius': T_CELSIUS,
        },
        'results': {}
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)

    for prof in PROFILES:
        print(f"\n  Profile: {prof}")
        output['results'][prof] = {}
        t_vec, G1, G2, G3 = build_irradiance(psc_key, prof)

        for alg_key in ALGORITHMS:
            t0_alg = time.time()
            alg  = make_algorithm(alg_key)
            sim  = MPPTSimulator(MODULE, alg, dt=DT,
                                 T_celsius=T_CELSIUS,
                                 num_iv_points=IV_POINTS)
            res  = sim.run(t_vec, G1, G2, G3, verbose=False)
            elapsed = time.time() - t0_alg

            output['results'][prof][alg_key] = {
                'eta_total':     res['eta_total'],
                'eta_trans':     res['eta_trans'],
                'E_tracked':     res['E_tracked'],
                'E_available':   res['E_available'],
                'std_P_ss':      res['std_P_ss'],
                'mean_P_ss':     res['mean_P_ss'],
                'mean_gmpp_ss':  res['mean_gmpp_ss'],
                'settling_time': res['settling_time'],
                'runtime_s':     round(elapsed, 2),
            }
            eta = res['eta_total']
            print(f"    [{alg_key:6s}]  η = {eta:5.1f}%   "
                  f"(std_ss={res['std_P_ss']:.2f}W, "
                  f"settle={res['settling_time']}s)  "
                  f"[{elapsed:.1f}s]")

        # Save after each profile (crash-safe)
        with open(out_file, 'w') as f:
            json.dump(output, f, indent=2, default=str)

    print(f"\n  ✓ Saved: {out_file}")
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

RUN_MAP = {
    '1': ('easy',    f'{RESULTS_DIR}/run1_easy.json'),
    '2': ('moderate',f'{RESULTS_DIR}/run2_moderate.json'),
    '3': ('hard',    f'{RESULTS_DIR}/run3_hard.json'),
    '4': ('extreme', f'{RESULTS_DIR}/run4_extreme.json'),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Physics-Based MPPT Test Runner')
    parser.add_argument('--run', default='1',
                        choices=['1','2','3','4','all'],
                        help='Which PSC run to execute')
    args = parser.parse_args()

    t_start = time.time()

    if args.run == 'all':
        for key in ['1','2','3','4']:
            psc_key, out_file = RUN_MAP[key]
            run_psc(psc_key, out_file)
    else:
        psc_key, out_file = RUN_MAP[args.run]
        run_psc(psc_key, out_file)

    print(f"\n{'='*60}")
    print(f"  DONE — total time: {(time.time()-t_start)/60:.1f} min")
    print(f"{'='*60}")
