"""
run_all.py  —  Full pipeline for Physics-Based MPPT Testing
============================================================
Run from the project directory:
    python3 run_all.py

What this does (in order):
    Step 1  — 96-test simulation matrix (4 PSC × 8 profiles × 3 algorithms)
    Step 2  — Gap metrics (RMSE/MAE all 96, response times, EN 50530 comparison,
               INC divergence verification)
    Step 3  — Publication figures (7 PNG files → figures/)

Estimated total runtime: ~25-35 minutes on a modern laptop.

Output files:
    results/run1_easy.json          — Easy PSC simulation results
    results/run2_moderate.json      — Moderate PSC simulation results
    results/run3_hard.json          — Hard PSC simulation results
    results/run4_extreme.json       — Extreme PSC simulation results
    results/gap1_rmse_mae_96.json   — RMSE + MAE for all 96 cases
    results/gap2_response_times.json— Response/settling times
    results/gap3_en50530_comparison.json — EN 50530 vs sigmoid comparison
    figures/fig_pv_curves.png       + 6 more PNG files

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


# ── Step 2: Gap metrics ───────────────────────────────────────────────────────
def run_gaps():
    from gap_analysis import (gap1_rmse_mae_all96, gap2_response_time_all96,
                               gap3_en50530_comparison,
                               gap4_inc_divergence_verification)
    if not os.path.exists('results/gap1_rmse_mae_96.json'):
        gap1_rmse_mae_all96()
    else:
        print("  Skipping Gap 1 (already exists)")

    if not os.path.exists('results/gap2_response_times.json'):
        gap2_response_time_all96()
    else:
        print("  Skipping Gap 2 (already exists)")

    if not os.path.exists('results/gap3_en50530_comparison.json'):
        gap3_en50530_comparison()
    else:
        print("  Skipping Gap 3 (already exists)")

    gap4_inc_divergence_verification()   # Always run (prints only, no output file)


# ── Step 3: Figures ───────────────────────────────────────────────────────────
def run_figures():
    import generate_figures as gf
    os.makedirs('figures', exist_ok=True)
    gf.fig_pv_curves()
    gf.fig_transition_profiles()
    gf.fig_heatmap_3alg()
    gf.fig_spline_advantage()
    gf.fig_main_finding()
    gf.fig_winner_matrix()
    gf.fig_inc_po_divergence()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t_total = time.time()
    print("Physics-Based MPPT Testing — Full Pipeline")
    print(f"Working directory: {os.getcwd()}")

    step("Step 1/3 — Run 96-case simulation matrix", run_simulations)
    step("Step 2/3 — Compute gap metrics (RMSE, response time, EN50530)", run_gaps)
    step("Step 3/3 — Generate publication figures", run_figures)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")
    print(f"  Results: results/")
    print(f"  Figures: figures/")
    print(f"{'='*60}")
