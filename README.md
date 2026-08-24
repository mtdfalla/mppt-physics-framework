# Physics-Based Irradiance Transition Testing for MPPT Algorithms

A deterministic simulation framework for evaluating maximum power point
tracking (MPPT) algorithms under physically representative irradiance
transitions. The framework models a total-cross-tied photovoltaic string
with per-substring bypass diodes, applies step, linear-ramp, and
measurement-derived sigmoid irradiance transitions across four partial
shading severity levels, and evaluates three MPPT algorithms
(Perturb and Observe, Incremental Conductance, and Spline-MPPT) on
energy-based tracking efficiency, steady-state error, oscillation, and
response time.

The sigmoid transition profiles follow the cloud shadow transition model
of Lappalainen and Valkealahti (Solar Energy, 2015), with shape
parameters spanning the reported measured distribution. All simulations
are deterministic: identical inputs produce identical results on any
platform.

## Requirements

Python 3.10 or later with NumPy, SciPy, and Matplotlib:

```bash
pip install -r requirements.txt
```

## Reproducing all results

```bash
python run_all.py
```

This executes the complete pipeline in order and is safe to re-run after
an interruption (completed steps are skipped):

| Step | What it does | Output |
|------|--------------|--------|
| 1 | 96-case test matrix (4 PSC levels x 8 profiles x 3 algorithms) | `results/run{1-4}_*.json` |
| 2 | Error metrics, response times, EN 50530 comparison | `results/gap{1-3}_*.json` |
| 3 | Publication figures | `figures/*.png` |
| 4 | Result tables | `tables/*.csv` |
| 5 | Temperature study (full matrix at 45 and 65 C) | `results/study_temperature_*.json` |
| 6 | Fast-transition study (sigmoid b = 0.1 s, 0.25 s, 1 s ramp) | `results/study_fast_transitions.json` |
| 7 | Per-decision timing benchmark | `results/bench_decision_time.json` |

The full pipeline takes roughly one to two hours on a single core.
Steps can also be run individually, see the module docstrings.

## Repository layout

```
tct_eval.py               PV array model: single-diode, per-substring bypass
                          diodes, total-cross-tied 3S1P KC200GT string
mppt_algorithms.py        P&O, INC, and Spline-MPPT implementations plus the
                          time-domain MPPT simulator
modular_test_runner.py    Test matrix definition: PSC patterns, transition
                          generators, and the 96-case driver
gap_analysis.py           RMSE/MAE, response-time, and EN 50530 analyses
study_temperature.py      Elevated-temperature re-run of the full matrix
study_fast_transitions.py Sub-half-second transition study
bench_decision_time.py    Isolated controller timing microbenchmark
generate_figures.py       All publication figures
generate_tables.py        All result tables
run_all.py                End-to-end pipeline driver
results/, figures/,       Committed outputs of the pipeline, reproducible
tables/                   from the code above
```

## Model summary

The PV string is three series KC200GT modules (single-diode model with
temperature-dependent photocurrent, saturation current, and open-circuit
voltage), each with three bypass-diode substrings. Partial shading
severity levels place the global maximum power point in either the
high-voltage or the low-voltage region of the P-V characteristic.
Irradiance transitions are applied to one module while the electrical
model updates every 100 ms, and each algorithm executes one control
decision per update.

## License

Released under the MIT License. See `LICENSE`.

## Citing

If this framework contributes to your work, please cite the associated
article on physics-based irradiance transition testing for MPPT
algorithms. Citation details are provided in `CITATION.cff`.
