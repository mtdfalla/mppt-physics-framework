"""
generate_overview_figures.py -- Graphical abstract and mechanism figure
=======================================================================
Run from the project root directory, after generate_figures.py:

    python3 generate_overview_figures.py              # both figures
    python3 generate_overview_figures.py --alternative # also a compact
                                                       # mechanism variant

The mechanism figure (fig_mechanism_trajectories.png) shows the operating
voltages of the three algorithms behind the ranking change under Hard PSC. The six Hard PSC runs it needs (step and sigmoid
b = 2 s, three algorithms) are simulated here with the released model, and
their efficiencies are checked against results/error_metrics_96.csv before
anything is drawn, so every trajectory, voltage, power and efficiency shown is
computed rather than drawn by hand.

Outputs: figures/fig_mechanism_trajectories.png
         figures/graphical_abstract_3alg.png
         figures/fig_mechanism_compact.png   (with --alternative)
"""
import os

os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import modular_test_runner as mtr
from mppt_algorithms import MPPTSimulator
import generate_figures as gf
from generate_figures import (ALG_ORDER, ALG_STYLE, FS_LABEL, FS_PANEL,
                              FS_SMALL, FS_TICK, eta, fmt, pv_curve, pv_peaks,
                              save, width_in)

PSC = 'hard'
PROFILES_SHOWN = [('step', 'Step change'),
                  ('sigmoid_2.0', 'Sigmoid, $b$ = 2 s')]
ETA_TOLERANCE = 1e-6     # percentage points


# ---------------------------------------------------------------------------
# Simulation of the six runs shown in the mechanism figure
# ---------------------------------------------------------------------------
def simulate(profile, alg):
    """Time-domain run of one algorithm under the Hard PSC transition."""
    t, G1, G2, G3 = mtr.build_irradiance(PSC, profile)
    sim = MPPTSimulator(mtr.MODULE, mtr.make_algorithm(alg), dt=mtr.DT,
                        T_celsius=mtr.T_CELSIUS, num_iv_points=mtr.IV_POINTS)
    r = sim.run(t, G1, G2, G3, verbose=False)
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    e = 100.0 * trapz(r['P_op'], r['t']) / trapz(r['P_gmpp'], r['t'])
    ref = eta(PSC, profile, alg)
    if abs(e - ref) > ETA_TOLERANCE:
        raise RuntimeError(f'{PSC}/{profile}/{alg}: simulated efficiency '
                           f'{e:.6f} differs from {ref:.6f}')
    return dict(t=np.asarray(r['t']) - mtr.T_BEFORE, G3=G3,
                V=np.asarray(r['V_op']), P=np.asarray(r['P_op']),
                Pg=np.asarray(r['P_gmpp']))


def all_runs():
    return {(p, a): simulate(p, a) for p, _ in PROFILES_SHOWN
            for a in ALG_ORDER}


def hard_landscape():
    """Initial and final Hard PSC P-V curves and the final-state maxima."""
    Vi, Pi = pv_curve(mtr.PSC_PATTERNS[PSC]['G_init'])
    Vf, Pf = pv_curve(mtr.PSC_PATTERNS[PSC]['G_final'])
    peaks = pv_peaks(Vf, Pf)
    if len(peaks) != 3:
        raise RuntimeError(f'expected three maxima, found {len(peaks)}')
    p_g = max(p for _, p in peaks)
    labels = []
    for v, p in peaks:
        labels.append('GMPP' if p == p_g else f'{fmt(100.0 * p / p_g, 0)}%')
    return (Vi, Pi), (Vf, Pf), peaks, labels


# ---------------------------------------------------------------------------
# Shared panels
# ---------------------------------------------------------------------------
def draw_pv_panel(ax, landscape, show_legend=True):
    (Vi, Pi), (Vf, Pf), peaks, labels = landscape
    ax.plot(Vi, Pi, color='0.6', ls=(0, (3.0, 1.6)), lw=1.1,
            label='Initial')
    ax.plot(Vf, Pf, color='black', lw=1.3, label='Final')
    for (v, p), lab in zip(peaks, labels):
        ax.plot(v, p, 'o', ms=4.2, color='black',
                mfc='black' if lab == 'GMPP' else 'white', mew=1.0, zorder=5)
        ax.text(v, p + 9, lab, ha='center', va='bottom', fontsize=FS_SMALL)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 240)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 50, 100, 150, 200])
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Power (W)')
    ax.grid(True)
    if show_legend:
        ax.legend(loc='upper right', handlelength=1.8, labelspacing=0.25,
                  borderaxespad=0.1, handletextpad=0.5)


def _spread(values, min_gap):
    """Vertical label positions at least min_gap apart, kept close to the
    data values (simple iterative relaxation, deterministic)."""
    order = np.argsort(values)
    pos = np.array(values, float)[order]
    for _ in range(200):
        moved = False
        for k in range(1, len(pos)):
            gap = pos[k] - pos[k - 1]
            if gap < min_gap:
                shift = (min_gap - gap) / 2.0
                pos[k - 1] -= shift
                pos[k] += shift
                moved = True
        if not moved:
            break
    out = np.empty_like(pos)
    out[order] = pos
    return out


def draw_slope_panel(ax, ylim=(28, 80), min_gap=6.0):
    """Efficiency of the three algorithms under the two profiles, joined by
    lines so that a change in order appears as a crossing."""
    x = [0.0, 1.0]
    vals = {a: [eta(PSC, 'step', a), eta(PSC, 'sigmoid_2.0', a)]
            for a in ALG_ORDER}
    for a in ['spline', 'inc', 'po']:
        st = ALG_STYLE[a]
        ax.plot(x, vals[a], color=st['color'], ls=st['ls'], lw=1.3,
                marker=st['marker'], ms=4.6, mew=1.0,
                mfc='none' if a == 'po' else st['color'],
                zorder=5 if a == 'po' else 4)
    # right-hand labels: value and algorithm
    right = [vals[a][1] for a in ALG_ORDER]
    ypos = _spread(right, min_gap)
    for a, y, v in zip(ALG_ORDER, ypos, right):
        ax.text(1.09, y, f'{fmt(v, 1)}  {ALG_STYLE[a]["label"]}',
                ha='left', va='center', fontsize=FS_SMALL,
                color=ALG_STYLE[a]['color'] if a != 'inc' else '#1e7a1e')
    # left-hand labels: tied values share one label
    left = {}
    for a in ALG_ORDER:
        left.setdefault(fmt(vals[a][0], 1), []).append(a)
    for txt, algs in left.items():
        y = float(np.mean([vals[a][0] for a in algs]))
        ax.text(-0.09, y, txt, ha='right', va='center', fontsize=FS_SMALL)
    ax.set_xticks(x)
    ax.set_xticklabels(['Step', 'Sigmoid\n$b$ = 2 s'])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.42, 2.05)
    ax.set_ylim(*ylim)
    ax.spines['bottom'].set_bounds(-0.42, 1.25)
    ax.set_ylabel('Tracking efficiency (%)')
    ax.grid(True, axis='y')
    for xv in x:
        ax.axvline(xv, color='0.85', lw=0.6, zorder=0)


def alg_line_handles():
    hs = []
    for a in ALG_ORDER:
        st = ALG_STYLE[a]
        hs.append(Line2D([], [], color=st['color'], ls=st['ls'], lw=1.3,
                         marker=st['marker'], ms=4.2, mew=1.0,
                         mfc='none' if a == 'po' else st['color'],
                         label=st['label']))
    return hs


# ---------------------------------------------------------------------------
# Mechanism figure
# ---------------------------------------------------------------------------
def fig_overview(runs=None):
    """(a, b) Operating voltage of the three algorithms under the step change
    and the sigmoid b = 2 s transition, against the voltages of the three
    maxima of the final P-V curve, with the shaded-module irradiance above.
    (c) Hard PSC P-V curves before and after the transition. (d) Tracking
    efficiency under the two profiles."""
    if runs is None:
        runs = all_runs()
    land = hard_landscape()
    peaks, labels = land[2], land[3]
    t_span = {p: runs[(p, 'spline')]['t'][[0, -1]] for p, _ in PROFILES_SHOWN}
    dur = [t_span[p][1] - t_span[p][0] + 0.1 for p, _ in PROFILES_SHOWN]

    fig = plt.figure(figsize=(width_in(1.0), 4.45), layout='constrained')
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1.0])
    top = outer[0].subgridspec(2, 2, height_ratios=[0.30, 1.0],
                               width_ratios=dur)
    bot = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0])

    for col, (prof, name) in enumerate(PROFILES_SHOWN):
        ax_g = fig.add_subplot(top[0, col])
        ax_v = fig.add_subplot(top[1, col], sharex=ax_g)
        r0 = runs[(prof, 'spline')]
        ax_g.plot(r0['t'], r0['G3'], color='black', lw=1.1)
        ax_g.set_ylim(80, 220)
        ax_g.set_yticks([100, 200])
        ax_g.tick_params(labelbottom=False)
        ax_g.grid(True)
        if col == 1:
            ax_g.text(0.98, 0.50, 'Module 3 irradiance (W/m$^2$)',
                      transform=ax_g.transAxes, ha='right', va='center',
                      fontsize=FS_SMALL)
        ax_g.set_title(f'({"ab"[col]}) {name}', loc='left', fontsize=FS_PANEL,
                       pad=3)

        for (v, _), lab in zip(peaks, labels):
            ax_v.axhline(v, color='0.55', lw=0.6, ls=(0, (1.0, 1.5)),
                         zorder=1)
            if col == 1:
                ax_v.text(1.015, v, f'{lab}\n{fmt(v, 1)} V',
                          transform=ax_v.get_yaxis_transform(), ha='left',
                          va='center', fontsize=FS_SMALL, linespacing=1.05)
        for a in ALG_ORDER:
            st = ALG_STYLE[a]
            r = runs[(prof, a)]
            lw = 0.8 if a == 'spline' else 1.3
            ax_v.plot(r['t'], r['V'], color=st['color'], ls=st['ls'], lw=lw,
                      zorder={'spline': 2, 'po': 3, 'inc': 4}[a])
            mk = np.arange({'spline': 25, 'po': 40, 'inc': 55}[a],
                           len(r['t']), 60)
            ax_v.plot(r['t'][mk], r['V'][mk], ls='none', marker=st['marker'],
                      ms=3.8, mew=0.9, color=st['color'],
                      mfc='none' if a == 'po' else st['color'],
                      zorder={'spline': 2, 'po': 5, 'inc': 4}[a])
        ax_v.set_xlim(t_span[prof][0], t_span[prof][1] + 0.1)
        ax_v.set_ylim(0, 100)
        ax_v.set_yticks([0, 20, 40, 60, 80, 100])
        ax_v.set_xticks(np.arange(0, t_span[prof][1] + 1, 10))
        ax_v.set_xlabel('Time from transition start (s)')
        if col == 0:
            ax_v.set_ylabel('Operating voltage (V)')
        else:
            ax_v.tick_params(labelleft=False)
            ax_g.tick_params(labelleft=False)

    ax_c = fig.add_subplot(bot[0, 0])
    draw_pv_panel(ax_c, land)
    ax_c.set_title('(c) Hard PSC, P-V curves', loc='left', fontsize=FS_PANEL,
                   pad=3)
    ax_d = fig.add_subplot(bot[0, 1])
    draw_slope_panel(ax_d)
    ax_d.set_title('(d) Tracking efficiency', loc='left', fontsize=FS_PANEL,
                   pad=3)

    fig.legend(handles=alg_line_handles(), loc='outside upper center',
               ncol=3, columnspacing=2.0)
    save(fig, 'fig_mechanism_trajectories.png')


# ---------------------------------------------------------------------------
# Mechanism figure, compact variant
# ---------------------------------------------------------------------------
def dwell_voltage(r, last_s=20.0):
    """Voltage at which a run spends most time in its final last_s seconds
    (1 V bins), used to place its operating point on the P-V curve."""
    m = r['t'] >= r['t'][-1] - last_s
    h, edges = np.histogram(r['V'][m], bins=np.arange(0, 101, 1.0))
    k = int(np.argmax(h))
    sel = (r['V'][m] >= edges[k]) & (r['V'][m] < edges[k + 1])
    return float(np.median(r['V'][m][sel]))


def fig_overview_compact(runs=None):
    """(a) Final Hard PSC P-V curve with the operating point at which each
    algorithm dwells longest at the end of the step run (open markers) and of
    the sigmoid b = 2 s run (filled markers). (b) Tracking efficiency."""
    if runs is None:
        runs = all_runs()
    land = hard_landscape()
    (Vf, Pf) = land[1]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(width_in(1.0), 2.45),
                                     layout='constrained',
                                     gridspec_kw={'width_ratios': [1.1, 1.0]})
    draw_pv_panel(ax_a, land, show_legend=False)
    ax_a.set_ylim(-52, 240)
    ax_a.spines['left'].set_bounds(0, 240)
    for k, (prof, _) in enumerate(PROFILES_SHOWN):
        for j, a in enumerate(ALG_ORDER):
            st = ALG_STYLE[a]
            v = dwell_voltage(runs[(prof, a)])
            y = -14 - 22 * k
            ax_a.plot(v, y, marker=st['marker'], ls='none', ms=4.6, mew=1.0,
                      color=st['color'],
                      mfc='none' if k == 0 else st['color'], clip_on=False)
        ax_a.text(1.5, -14 - 22 * k, ['Step', 'Sigmoid'][k], ha='left',
                  va='center', fontsize=FS_SMALL)
    ax_a.set_yticks([0, 50, 100, 150, 200])
    ax_a.set_title('(a) Final P-V curve, dwell voltages', loc='left',
                   fontsize=FS_PANEL, pad=3)
    draw_slope_panel(ax_b)
    ax_b.set_title('(b) Tracking efficiency', loc='left', fontsize=FS_PANEL,
                   pad=3)
    fig.legend(handles=alg_line_handles(), loc='outside upper center',
               ncol=3, columnspacing=2.0)
    save(fig, 'fig_mechanism_compact.png')


# ---------------------------------------------------------------------------
# Graphical abstract
# ---------------------------------------------------------------------------
GA_SIZE_CM = (13.28, 5.31)      # Elsevier minimum 1328 x 531 px, same aspect


def fig_graphical_abstract():
    """Hard PSC tracking efficiency under the step change and the sigmoid
    b = 2 s transition, with the transition shapes and the key numbers."""
    w_in, h_in = GA_SIZE_CM[0] / 2.54, GA_SIZE_CM[1] / 2.54
    fig = plt.figure(figsize=(w_in, h_in))
    # explicit layout in figure fractions (deterministic, no tight bbox)
    ax = fig.add_axes([0.085, 0.27, 0.40, 0.57])
    w = 0.25
    offs = {'spline': -w, 'po': 0.0, 'inc': w}
    for g, (prof, _) in enumerate(PROFILES_SHOWN):
        for a in ALG_ORDER:
            v = eta(PSC, prof, a)
            gf.alg_bar(ax, g + offs[a], v, a, w * 0.9)
            ax.text(g + offs[a], v + 1.5, fmt(v, 1), ha='center',
                    va='bottom', fontsize=FS_SMALL)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Step change', 'Sigmoid, $b$ = 2 s'])
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 82)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_ylabel('Tracking efficiency (%)', fontsize=FS_TICK)
    ax.grid(True, axis='y')
    ax.legend(handles=[gf.alg_patch(a) for a in ALG_ORDER], ncol=3,
              loc='lower center', bbox_to_anchor=(0.5, 1.0),
              handlelength=1.4, columnspacing=1.2, borderaxespad=0.2,
              fontsize=FS_SMALL)

    # transition shapes of the shaded module, drawn from the profiles
    for g, (prof, _) in enumerate(PROFILES_SHOWN):
        t, _, _, G3 = mtr.build_irradiance(PSC, prof)
        ins = fig.add_axes([0.085 + 0.40 * (0.10 + 0.5 * g), 0.05,
                            0.40 * 0.30, 0.075])
        ins.plot(t - mtr.T_BEFORE, G3, color='0.25', lw=0.9)
        ins.set_xlim(-3, 25)
        ins.set_ylim(90, 210)
        ins.axis('off')

    adv_step = eta(PSC, 'step', 'spline') - max(eta(PSC, 'step', 'po'),
                                                eta(PSC, 'step', 'inc'))
    adv_sig = eta(PSC, 'sigmoid_2.0', 'inc') - max(
        eta(PSC, 'sigmoid_2.0', 'spline'), eta(PSC, 'sigmoid_2.0', 'po'))
    div = max(abs(eta(p, q, 'inc') - eta(p, q, 'po'))
              for p in gf.PSC_KEYS for q in gf.PROFILES)
    tie = abs(eta(PSC, 'step', 'inc') - eta(PSC, 'step', 'po'))
    lines = [
        ('Same Hard partial shading,', 'head'),
        ('two irradiance transition models', 'head'),
        (None, None),
        ('Step change:', 'body'),
        (f'Spline-MPPT leads by {fmt(adv_step, 1)} pp', 'body'),
        (None, None),
        ('Measured-shape sigmoid, $b$ = 2 s:', 'body'),
        (f'INC leads by {fmt(adv_sig, 1)} pp', 'body'),
        (None, None),
        (f'P&O and INC: tied under the step ({fmt(tie, 2)} pp),',
         'body'),
        (f'up to {fmt(div, 1)} pp apart under gradual', 'body'),
        ('transitions', 'body'),
        (None, None),
        ('96 simulated cases: 3 algorithms, 4 shading', 'note'),
        ('levels, 8 transition profiles', 'note'),
    ]
    y = 0.93
    for txt, kind in lines:
        if txt is None:
            y -= 0.035
            continue
        fig.text(0.545, y, txt, ha='left', va='top',
                 fontsize={'head': FS_LABEL, 'body': FS_SMALL,
                           'note': 7.0}[kind],
                 fontweight='bold' if kind == 'head' else 'normal',
                 color={'head': 'black', 'body': 'black', 'note': '0.3'}[kind])
        y -= 0.072 if kind == 'head' else 0.064
    save(fig, 'graphical_abstract_3alg.png')


def main(alternative=False):
    runs = all_runs()
    fig_overview(runs)
    if alternative:
        fig_overview_compact(runs)
    fig_graphical_abstract()


if __name__ == '__main__':
    main(alternative='--alternative' in sys.argv[1:])
