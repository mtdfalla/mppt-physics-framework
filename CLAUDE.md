# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-based simulation framework for comparing three MPPT (Maximum Power Point Tracking) algorithms across 96 test scenarios. Supports the paper "Physics-Based Irradiance Transition Testing for MPPT Algorithms" (Solar Energy, Ref: SEJ-D-26-00031). The core finding is that step-change testing overstates Spline-MPPT's advantage — sigmoid transition profiles reveal INC can match or exceed Spline performance, a ranking reversal hidden by conventional testing.

## Dependencies

```bash
pip install numpy scipy matplotlib
# Requires Python 3.10+
```

## Running the Simulation

```bash
# Full pipeline (~25–35 min): runs all PSC levels, gap analysis, then figures
python run_all.py

# Individual PSC levels (saves to results/run{N}_*.json)
python modular_test_runner.py --run 1   # Easy PSC
python modular_test_runner.py --run 2   # Moderate PSC
python modular_test_runner.py --run 3   # Hard PSC
python modular_test_runner.py --run 4   # Extreme PSC

# Supplementary gap metrics (requires PSC JSON outputs)
python gap_analysis.py

# Publication figures (requires all JSON outputs)
python generate_figures.py
```

`run_all.py` skips PSC runs and gap analysis if output JSON files already exist, but always regenerates figures.

## Architecture

**Data flow:**
```
modular_test_runner.py  →  tct_eval.py  →  mppt_algorithms.py  →  results/*.json  →  generate_figures.py
```

### `tct_eval.py` — PV Physics Model
Single-diode model for a 3S1P string of KC200GT modules with bypass diodes. Key functions:
- `ModuleParams` dataclass: STC electrical and thermal parameters
- `module_iv_curve(G, T, m)`: Newton-Raphson I-V solver (50 iterations per module)
- `evaluate_tct(G_map, T_map, module)`: Series-string P-V curve with bypass diode modeling
- `find_local_mpps(V, P)`: Local maxima detection in the P-V landscape

### `mppt_algorithms.py` — Algorithm Implementations
Three stateful MPPT objects sharing a `.step(t, iv_curve)` interface:
- **PandO**: Classic perturb-and-observe, ΔV=1V, T=50ms, init at 80% Voc
- **IncrementalConductance (INC)**: Conductance-based steering (I/V + dI/dV ≈ 0), same step/timing
- **SplineMPPT**: Global MPPT via 4-phase cycle (TRACK → SCAN → FIT → CONVERGE); samples 6 voltages at [25%, 38%, 53%, 68%, 83%, 93%] Voc then fits a cubic spline; T_scan=5s

**`MPPTSimulator`** drives any algorithm through a time-varying irradiance sequence, caching I-V curve rebuilds (only recalculates when G changes >0.01%). Returns η_total, η_trans, settling_time, steady-state std_P.

### `modular_test_runner.py` — Test Matrix
96 cases = 4 PSC levels × 8 irradiance profiles × 3 algorithms. Only module 3's irradiance transitions; modules 1 and 2 are held constant.

**PSC levels** (G_init → G_final for module 3): Easy (700→500), Moderate (400→300), Hard (200→100), Extreme (150→50) W/m².

**Transition profiles**: `step`, `linear_5s/10s/20s`, `sigmoid_0.5/1.0/2.0/5.0` (σ parameter controls steepness).

JSON output is written incrementally after each profile (crash-safe).

### `gap_analysis.py` — Supplementary Metrics
Addresses reviewer feedback with four gap analyses: RMSE/MAE for all 96 cases, redefined response time (to algorithm's own steady-state), EN 50530 comparison, and INC divergence trajectory verification.

### `generate_figures.py` — Publication Figures
Outputs 7 PNG files (300 DPI, serif font) to `figures/`: P-V curves, transition profiles, efficiency heatmaps, spline-advantage map, main-finding bar chart, winner matrix, and INC/P&O divergence analysis.

Algorithm color scheme: Spline=#1a6faf, P&O=#e05c2a, INC=#2ca02c.
