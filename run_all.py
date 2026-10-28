"""
run_all.py  —  Full pipeline for Physics-Based MPPT Testing
============================================================
Run from the project directory:
    python3 run_all.py

What this does (in order):
    Step 1  — 96-test simulation matrix (4 PSC × 8 profiles × 3 algorithms)
    Step 2  — Supplementary analyses (RMSE/MAE, response times, EN 50530-style
               comparison, operating-point trajectories)
    Step 3  — Error and dynamic metrics for all 96 cases from the simulated
               time series (results/error_metrics_96.csv and its table)
    Step 4  — Figures 1 to 10, graphical abstract and mechanism figure (figures/)
    Step 5  — Result tables
    Step 6  — Temperature study (45 and 65 C)
    Step 7  — Fast-transition study
    Step 8  — Per-decision timing benchmark

Total runtime: about 30 minutes on a single core.

Output files:
    results/run1_easy.json          — Easy PSC simulation results
    results/run2_moderate.json      — Moderate PSC simulation results
    results/run3_hard.json          — Hard PSC simulation results
    results/run4_extreme.json       — Extreme PSC simulation results
    results/rmse_mae_96.json        — RMSE and MAE for all 96 cases
    results/response_times_96.json  — Response and settling times
    results/en50530_comparison.json — Step, EN 50530-style and sigmoid comparison
    figures/*.png                   — article figures and graphical abstract

Requirements:
    pip install numpy scipy matplotlib
"""

import os
import sys
import time


def step(label, fn):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    print(f"  Completed in {(time.time()-t0)/60:.1f} min")


# ── Step 1: Simulations ───────────────────────────────────────────────────────
def run_simulations():
    from modular_test_runner import run_psc, RUN_MAP
    os.makedirs('results', exist_ok=True)
    for key in ['1', '2', '3', '4']:
        psc_key, out_file = RUN_MAP[key]
        if os.path.exists(out_file):
            print(f"  Skipping {out_file} (already exists)")
        else:
            run_psc(psc_key, out_file)


# ── Step 2: Supplementary analyses ───────────────────────────────────────────────────
def run_supplementary():
    from supplementary_analysis import (rmse_mae_all96, response_times_all96,
                                        en50530_comparison, trajectory_check)
    if not os.path.exists('results/rmse_mae_96.json'):
        rmse_mae_all96()
    else:
        print("  Skipping RMSE and MAE (already exists)")

    if not os.path.exists('results/response_times_96.json'):
        response_times_all96()
    else:
        print("  Skipping response times (already exists)")

    if not os.path.exists('results/en50530_comparison.json'):
        en50530_comparison()
    else:
        print("  Skipping EN 50530-style comparison (already exists)")

    trajectory_check()   # prints only, no output file


# ── Step 4: Tables ───────────────────────────────────────────────────────────
def run_tables():
    import generate_tables as gt
    os.makedirs('tables', exist_ok=True)
    gt.table1(); gt.table2(); gt.table3(); gt.table4()


# ── Step 3: Figures ───────────────────────────────────────────────────────────
def run_error_metrics():
    import subprocess
    subprocess.run([sys.executable, 'compute_error_metrics.py',
                    '--code-dir', '.', '--out-dir', 'results'], check=True)


def run_figures():
    import generate_figures as gf
    import generate_overview_figures as gof
    import generate_concept_figure as gcf
    os.makedirs('figures', exist_ok=True)
    gf.main()
    gof.main()
    gcf.main()


# ── Step 5: Temperature study ────────────────────────────────────────────────
def run_temperature_study():
    import subprocess
    for T in (45, 65):
        for psc in ('easy', 'moderate', 'hard', 'extreme'):
            out = f'results/study_temperature_T{T}_{psc}.json'
            if os.path.exists(out):
                print(f"  Skipping {out} (already exists)")
                continue
            subprocess.run([sys.executable, 'study_temperature.py',
                            '--T', str(T), '--psc', psc], check=True)


# ── Step 6: Fast-transition study ────────────────────────────────────────────
def run_fast_study():
    import subprocess
    out = 'results/study_fast_transitions.json'
    if os.path.exists(out):
        print(f"  Skipping {out} (already exists)")
        return
    subprocess.run([sys.executable, 'study_fast_transitions.py'], check=True)


# ── Step 7: Timing benchmark ─────────────────────────────────────────────────
def run_bench():
    import subprocess
    subprocess.run([sys.executable, 'bench_decision_time.py'], check=True)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()
    print("Physics-Based MPPT Testing — Full Pipeline")
    print(f"Working directory: {os.getcwd()}")

    step("Step 1/8 — Run 96-case simulation matrix", run_simulations)
    step("Step 2/8 — Supplementary analyses (RMSE, response time, EN 50530)", run_supplementary)
    step("Step 3/8 — Error and dynamic metrics for all 96 cases", run_error_metrics)
    step("Step 4/8 — Generate publication figures", run_figures)
    step("Step 5/8 — Generate result tables", run_tables)
    step("Step 6/8 — Temperature study (45 and 65 C, all PSC levels)", run_temperature_study)
    step("Step 7/8 — Fast-transition study (b = 0.1 s, 0.25 s, 1 s ramp)", run_fast_study)
    step("Step 8/8 — Per-decision timing benchmark", run_bench)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Results: results/")
    print(f"  Figures: figures/")
    print(f"{'='*60}")
