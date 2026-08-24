"""
bench_decision_time.py — Measured per-decision execution time (round-2 revision)
================================================================================
Microbenchmarks the controller step() of each MPPT algorithm in isolation
from the PV plant model: a fixed Hard-PSC I-V landscape is precomputed and
each algorithm's step() is called 20,000 times at its native 50 ms control
grid. The Spline-MPPT figure therefore amortizes its periodic 5 s scans
(about 200 scans across the 1000 s of simulated control time).

Absolute microseconds are specific to the stated interpreter, NumPy build
and CPU. The operation counts in the manuscript's complexity table
characterize the intrinsic arithmetic independently of platform.

Output: results/bench_decision_time.json
"""
import time, json, os, platform, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from modular_test_runner import MODULE, IV_POINTS, make_algorithm
from tct_eval import evaluate_tct

N_CALLS = 20000
G = np.array([1000.0, 400.0, 100.0]).reshape(3, 1)   # Hard PSC final state
T = np.full((3, 1), 25.0)

def main():
    iv = evaluate_tct(G, T, MODULE, num_points=IV_POINTS)
    out = {'metadata': {
        'n_calls': N_CALLS,
        'landscape': 'hard PSC final state (1000/400/100 W/m2), 25 C',
        'python': platform.python_version(),
        'numpy': np.__version__,
        'machine': platform.machine(),
        'note': 'controller step() isolated from the plant model; '
                'spline figure amortizes periodic scans'}}
    us = {}
    for alg_key in ['po', 'inc', 'spline']:
        alg = make_algorithm(alg_key)
        alg.reset(iv['Voc'])
        for k in range(100):
            alg.step(k * 0.05, iv)
        t0 = time.perf_counter()
        for k in range(N_CALLS):
            alg.step((k + 100) * 0.05, iv)
        us[alg_key] = (time.perf_counter() - t0) / N_CALLS * 1e6
        print(f"{alg_key:7s} {us[alg_key]:8.2f} us per control step")
    out['us_per_step'] = {k: round(v, 2) for k, v in us.items()}
    out['relative_to_po'] = {k: round(v / us['po'], 2) for k, v in us.items()}
    os.makedirs('results', exist_ok=True)
    with open('results/bench_decision_time.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("saved results/bench_decision_time.json")

if __name__ == '__main__':
    main()
