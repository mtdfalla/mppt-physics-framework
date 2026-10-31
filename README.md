# Physics-Based Irradiance Transition Testing for MPPT Algorithms

Simulation framework and results for the article *Physics-based irradiance
transition testing reveals ranking reversals in MPPT algorithms*
(M. Abdelmagid, T. H. M. El-Fouly and M. Maalouf, Solar Energy 321 (2027)
115292, https://doi.org/10.1016/j.solener.2026.115292).

The framework models a series string of three KC200GT photovoltaic modules,
each with a bypass diode, under four partial shading conditions (PSC). It
applies step, linear-ramp and measurement-derived sigmoid irradiance
transitions and evaluates three maximum power point tracking (MPPT)
algorithms: Perturb and Observe (P&O), Incremental Conductance (INC) and
Spline-MPPT. The sigmoid profiles follow the cloud-shadow transition model of
Lappalainen and Valkealahti (Solar Energy, 2015). All simulations are
deterministic, so identical inputs give identical results.

## Requirements

Python 3.10 or later with NumPy, SciPy and Matplotlib:

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
python run_all.py
```

The pipeline runs the steps below in order. Completed simulation steps are
skipped on a re-run.

| Step | Content | Output |
|------|---------|--------|
| 1 | 96-case test matrix (4 PSC levels x 8 profiles x 3 algorithms) | `results/run{1-4}_*.json` |
| 2 | Supplementary analyses: RMSE and MAE, response times, EN 50530-style comparison | `results/rmse_mae_96.json`, `results/response_times_96.json`, `results/en50530_comparison.json` |
| 3 | Error and dynamic metrics of all 96 cases | `results/error_metrics_96.csv` |
| 4 | Figures | `figures/` |
| 5 | Result tables | `tables/` |
| 6 | Temperature study, full matrix at 45 and 65 C | `results/study_temperature_*.json` |
| 7 | Fast-transition study (sigmoid b = 0.1 and 0.25 s, 1 s ramp) | `results/study_fast_transitions.json` |
| 8 | Per-decision timing benchmark | `results/bench_decision_time.json` |

The complete pipeline takes about 30 minutes on a single core.

## Figures

| Article | File | Script |
|---------|------|--------|
| Figure 1 | `fig_concept.pdf`, `fig_concept.png` | `generate_concept_figure.py` |
| Figure 2 | `shading_scenarios.png` | `generate_figures.py` |
| Figure 3 | `fig_pv_curves.png` | `generate_figures.py` |
| Figure 4 | `fig_physics_basis.png` | `generate_figures.py` |
| Figure 5 | `fig_transition_profiles.png` | `generate_figures.py` |
| Figure 6 | `fig_heatmap_3alg.png` | `generate_figures.py` |
| Figure 7 | `fig_winner_matrix.png` | `generate_figures.py` |
| Figure 8 | `fig_spline_advantage.png` | `generate_figures.py` |
| Figure 9 | `fig_inc_po_divergence.png` | `generate_figures.py` |
| Figure 10 | `fig_main_finding.png` | `generate_figures.py` |
| Graphical abstract | `graphical_abstract_3alg.png` | `generate_overview_figures.py` |

`generate_overview_figures.py` also writes `fig_mechanism_trajectories.png`,
the operating voltages of the three algorithms under Hard PSC after a step
change and during a sigmoid transition with b = 2 s.

## Code

```
tct_eval.py                  PV string model: single-diode modules with bypass
                             diodes, 3S1P KC200GT string
mppt_algorithms.py           P&O, INC and Spline-MPPT, and the time-domain
                             MPPT simulator
modular_test_runner.py       PSC patterns, transition profiles and the
                             96-case driver
supplementary_analysis.py    RMSE and MAE, response times, EN 50530-style
                             comparison, operating-point trajectories
compute_error_metrics.py     Error and dynamic metrics of all 96 cases
study_temperature.py         Full matrix at elevated cell temperature
study_fast_transitions.py    Transitions faster than b = 0.5 s
bench_decision_time.py       Per-decision execution time of each controller
generate_concept_figure.py   Figure 1
generate_figures.py          Figures 2 to 10
generate_overview_figures.py Graphical abstract and mechanism figure
generate_tables.py           Result tables
run_all.py                   Complete pipeline
```

## Model

The string consists of three series KC200GT modules, each represented by a
single-diode model with temperature-dependent photocurrent, saturation current
and open-circuit voltage, and one bypass diode. The four PSC levels place the
global maximum power point (GMPP) in either the high-voltage or the
low-voltage region of the P-V curve. The irradiance transition is applied to
one module, the electrical model is updated every 100 ms, and each algorithm
takes one control decision per update.

## Metrics in `results/error_metrics_96.csv`

All values are unrounded and computed from the simulated time series. The
columns reported in the article are:

- `eta_total_pct`: energy-based tracking efficiency over the whole simulation.
- `rmse_ss_W`, `mae_ss_W`, `sigma_P_W`, `re_ss_pct`, `nrmse_ss_pct`: steady-state
  metrics over the final 20 % of each simulation. sigma_P is the population
  standard deviation, and RE and nRMSE are relative to the mean GMPP power.
- `ts_completion_gmppband_s`: settling time t_s from the completion of the
  transition, with a band of 2 % of the final GMPP power around the
  algorithm's own final steady-state power, held for 1 s.
- `tG_completion_s`: GMPP acquisition time t_G from the completion of the
  transition, empty when the GMPP is not reached.

The remaining columns are supplementary: `rt_onset_s` (response time from the
transition onset), `ts_completion_s` and `ts_final_completion_s` (settling
times with a band of 2 % of the algorithm's own final power), `tG_onset_s`,
the post-transition window metrics (`*_post_*`), `mre_ss_pct` and
`eta_ss_pct`.

## License

MIT License, see `LICENSE`.

## Citation

Please cite the article when using this framework:

M. Abdelmagid, T. H. M. El-Fouly, M. Maalouf, Physics-based irradiance
transition testing reveals ranking reversals in MPPT algorithms, Solar Energy
321 (2027) 115292. https://doi.org/10.1016/j.solener.2026.115292

The archived software release is available at
https://doi.org/10.5281/zenodo.23737201. Citation metadata are in
`CITATION.cff`.
