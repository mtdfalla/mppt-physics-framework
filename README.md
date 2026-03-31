# Physics-Based MPPT Testing Framework

Simulation codebase for:
> **"Physics-Based Irradiance Transition Testing for MPPT Algorithms"**
> Submitted to *Solar Energy* (Ref: SEJ-D-26-00031)

---

## Requirements

```
Python 3.10 or later
numpy
scipy
matplotlib
```

Install with:
```bash
pip install numpy scipy matplotlib
```

---

## File structure

```
project/
├── tct_eval.py            # PV array model (single-diode, KC200GT, 3S1P)
├── mppt_algorithms.py     # P&O, INC, Spline-MPPT algorithms + simulator
├── modular_test_runner.py # 96-case test matrix (4 PSC × 8 profiles × 3 algs)
├── gap_analysis.py        # RMSE/MAE, response time, EN50530, INC verification
├── generate_figures.py    # 7 publication figures
├── run_all.py             # Master script — runs everything in order
└── README.md
```

---

## Quick start (run everything)

```bash
python3 run_all.py
```

This runs the full pipeline (~25-35 min). Already-completed steps are skipped
automatically, so you can safely re-run after an interruption.

---

## Step-by-step (if you prefer manual control)

### Step 1 — Run simulations (96 test cases)

```bash
python3 modular_test_runner.py --run 1   # Easy PSC    → results/run1_easy.json
python3 modular_test_runner.py --run 2   # Moderate    → results/run2_moderate.json
python3 modular_test_runner.py --run 3   # Hard        → results/run3_hard.json
python3 modular_test_runner.py --run 4   # Extreme     → results/run4_extreme.json
```

Each run saves its JSON immediately after each profile, so interruption
only costs the current profile (~1-2 min), not the entire run.

### Step 2 — Compute supplementary metrics

```bash
python3 gap_analysis.py
```

Produces:
- `results/gap1_rmse_mae_96.json`    — RMSE + MAE for all 96 cases
- `results/gap2_response_times.json` — settling/response times
- `results/gap3_en50530_comparison.json` — EN 50530 proxy vs sigmoid

### Step 3 — Generate figures

```bash
python3 generate_figures.py
```

Produces 7 PNG files in `figures/`.

---

## Module descriptions

### `tct_eval.py`
Single-diode PV model for a 3-module series string (KC200GT, 3S1P TCT topology).
- Newton-Raphson I-V solver with bypass diode modelling
- `evaluate_tct(G_map, T_map, module)` → P-V curve dict
- `find_local_mpps(V, P)` → list of local peaks

### `mppt_algorithms.py`
Three MPPT algorithms operating on a simulated boost converter:
- `PandO` — Perturb & Observe (ΔV=1V, T=50ms)
- `IncrementalConductance` — INC (ΔV=1V, T=50ms, ε=0.01)
- `SplineMPPT` — Cubic-spline global search (T_scan=5s, 6 sample voltages)
- `MPPTSimulator.run()` — drives any algorithm through a time-varying
  irradiance sequence and returns efficiency, RMSE, settling time, etc.

### `modular_test_runner.py`
Runs all 96 test cases and saves results to JSON.

PSC patterns (G_initial → G_final per module, W/m²):
| Level    | G_init              | G_final             |
|----------|---------------------|---------------------|
| Easy     | (1000, 900, 700)    | (1000, 900, 500)    |
| Moderate | (1000, 600, 400)    | (1000, 600, 300)    |
| Hard     | (1000, 400, 200)    | (1000, 400, 100)    |
| Extreme  | (1000, 300, 150)    | (1000, 300, 50)     |

Transition profiles: step, linear_5s, linear_10s, linear_20s,
sigmoid_0.5, sigmoid_1.0, sigmoid_2.0, sigmoid_5.0

### `gap_analysis.py`
Four supplementary analyses for reviewer responses:
1. RMSE + MAE for all 96 cases
2. Response time (time to own steady-state) — meaningful even when
   algorithm fails to reach GMPP
3. EN 50530 comparison (linear ramp proxy vs sigmoid)
4. INC divergence verification with trajectory trace and physical
   explanation

---

## Key results summary

| PSC / Profile       | Spline | P&O  | INC  | Winner       |
|---------------------|--------|------|------|--------------|
| Easy / Step         | 90.9%  | 91.7%| 91.8%| Local        |
| Hard / Step         | 63.0%  | 36.2%| 36.2%| Spline +26.8pp|
| Hard / Sigmoid b=2s | 68.4%  | 40.7%| 72.0%| **INC beats Spline** |
| Extreme / Step      | 50.9%  | 19.2%| 19.2%| Spline +31.6pp|
| Extreme / Sigmoid 2s| 57.5%  | 23.6%| 54.1%| Spline +3.4pp |

**Core finding**: Step-change testing overstates Spline's advantage by
27-32pp at severe shading. Physics-based sigmoid testing reveals INC
can match or exceed Spline — a ranking reversal completely concealed by
step testing. EN 50530 linear ramps produce the same misleading rankings
as step testing.
