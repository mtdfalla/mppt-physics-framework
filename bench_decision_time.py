"""
bench_decision_time.py — Measured per-decision execution time
=================================================
Microbenchmarks the controller step() of each MPPT algorithm in isolation
from the PV plant model: a fixed Hard-PSC I-V landscape is precomputed and
each algorithm's step() is called 20,000 times at the 100 ms decision
interval used in the simulations (2000 s of control time). The Spline-MPPT
figure therefore amortizes its scan cycles (scan, approach and 5 s tracking
interval). The measurement is repeated N_REPEAT times and the median and
range across repeats are reported.

Absolute microseconds are specific to the stated interpreter, NumPy build
and CPU. The operation counts in the article's complexity table
characterize the intrinsic arithmetic independently of platform.

Output: results/bench_decision_time.json
"""
import time, json, os, platform, sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from modular_test_runner import MODULE, IV_POINTS, make_algorithm
from tct_eval import evaluate_tct

N_CALLS = 20000
N_REPEAT = 9
DT = 0.1
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
                'spline figure amortizes its scan cycles'}}
    runs = {k: [] for k in ['po', 'inc', 'spline']}
    for rep in range(N_REPEAT):
        for alg_key in ['po', 'inc', 'spline']:
            alg = make_algorithm(alg_key)
            alg.reset(iv['Voc'])
            for k in range(100):
                alg.step(k * DT, iv)
            t0 = time.perf_counter()
            for k in range(N_CALLS):
                alg.step((k + 100) * DT, iv)
            runs[alg_key].append((time.perf_counter() - t0) / N_CALLS * 1e6)
    out['metadata']['n_repeat'] = N_REPEAT
    out['metadata']['decision_interval_s'] = DT
    med = {k: float(np.median(v)) for k, v in runs.items()}
    out['us_per_step_median'] = {k: round(v, 2) for k, v in med.items()}
    out['us_per_step_range'] = {k: [round(min(v), 2), round(max(v), 2)]
                                for k, v in runs.items()}
    out['relative_to_po_median'] = {k: round(v / med['po'], 2)
                                    for k, v in med.items()}
    for k in runs:
        print(f"{k:7s} median {med[k]:8.2f} us  range "
              f"{min(runs[k]):.2f}-{max(runs[k]):.2f} us")
    os.makedirs('results', exist_ok=True)
    with open('results/bench_decision_time.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("saved results/bench_decision_time.json")

if __name__ == '__main__':
    main()
