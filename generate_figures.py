"""
generate_figures.py -- Figures 2 to 10 of the article
=====================================================
Run from the project root directory:

    python3 generate_figures.py

Inputs
------
results/error_metrics_96.csv   unrounded tracking efficiencies of the 96 cases
                               (written by compute_error_metrics.py)
tct_eval.py, modular_test_runner.py
                               array model, shading patterns and transition
                               profiles, used for every P-V curve and profile

Outputs (figures/)
------------------
shading_scenarios.png        shading patterns and final P-V characteristics
fig_pv_curves.png            final P-V characteristics of the four PSC levels
fig_physics_basis.png        reported shape-parameter statistics, sigmoid family
fig_transition_profiles.png  the eight tested transition profiles (Hard PSC)
fig_heatmap_3alg.png         tracking efficiency of all 96 cases
fig_winner_matrix.png        highest-efficiency algorithm per condition
fig_spline_advantage.png     Spline-MPPT minus best local algorithm
fig_inc_po_divergence.png    INC and P&O across profiles, Hard and Extreme PSC
fig_main_finding.png         step against sigmoid b = 2 s, all PSC levels

Every figure is drawn at the width at which the article includes it, so the
font sizes set below are the printed sizes. Each printed number is rounded once,
half up, from the unrounded data. Output files are byte-for-byte reproducible.
"""

import os

os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import csv
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from scipy.signal import find_peaks

import modular_test_runner as mtr
from tct_eval import evaluate_tct

FIG_DIR = 'figures'
CSV_PATH = os.path.join('results', 'error_metrics_96.csv')

# ---------------------------------------------------------------------------
# Page geometry (elsarticle, preprint, 12 pt): \textwidth = \columnwidth = 390 pt
# ---------------------------------------------------------------------------
TEXTWIDTH_PT = 390.0
TEX_PT_PER_IN = 72.27
DPI = 600


def width_in(fraction):
    """Printed width in inches of a figure included at fraction x \\textwidth."""
    return fraction * TEXTWIDTH_PT / TEX_PT_PER_IN


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
FS_SMALL = 7.5   # cell values, annotations
FS_TICK = 8.0
FS_LABEL = 8.5
FS_PANEL = 9.0

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': FS_LABEL,
    'axes.labelsize': FS_LABEL,
    'axes.titlesize': FS_LABEL,
    'xtick.labelsize': FS_TICK,
    'ytick.labelsize': FS_TICK,
    'legend.fontsize': FS_TICK,
    'legend.frameon': False,
    'legend.handlelength': 2.4,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.35,
    'lines.linewidth': 1.3,
    'hatch.linewidth': 0.6,
    'savefig.dpi': DPI,
    'figure.dpi': 100,
    'axes.unicode_minus': True,
})

# Algorithm identity: identical color, marker and line style in every figure
COL_SPLINE = '#1a6faf'
COL_PO = '#e05c2a'
COL_INC = '#2ca02c'
ALG_STYLE = {
    'spline': dict(label='Spline-MPPT', color=COL_SPLINE, marker='o', ls='-',
                   hatch=None),
    'po': dict(label='P&O', color=COL_PO, marker='^', ls=(0, (4.0, 1.6)),
               hatch='//////'),
    'inc': dict(label='INC', color=COL_INC, marker='s',
                ls=(0, (5.0, 1.4, 1.2, 1.4)), hatch='......'),
}
ALG_ORDER = ['spline', 'po', 'inc']

# Neutral palette for quantities that are not algorithms (shading level, b)
NEUTRAL_RAMP = ['#a39fc9', '#7a6fb0', '#54278f', '#2b0a4f']
LS_RAMP = [(0, (1.0, 1.2)), (0, (4.0, 1.6)), '-', (0, (5.0, 1.4, 1.2, 1.4))]

PSC_KEYS = ['easy', 'moderate', 'hard', 'extreme']
PSC_LABELS = ['Easy', 'Moderate', 'Hard', 'Extreme']
PROFILES = ['step', 'linear_5s', 'linear_10s', 'linear_20s',
            'sigmoid_0.5', 'sigmoid_1.0', 'sigmoid_2.0', 'sigmoid_5.0']
PROFILE_NAMES = ['Step', 'Linear 5 s', 'Linear 10 s', 'Linear 20 s',
                 'Sigmoid 0.5 s', 'Sigmoid 1.0 s', 'Sigmoid 2.0 s',
                 'Sigmoid 5.0 s']
# Two-level profile axis: parameter values, grouped by profile type
PROFILE_TICKS = ['Step', '5', '10', '20', '0.5', '1', '2', '5']
PROFILE_GROUPS = [(1, 3, 'Linear (s)'),
                  (4, 7, 'Sigmoid $b$ (s)')]

MINUS = '−'


# ---------------------------------------------------------------------------
# Data access and number formatting
# ---------------------------------------------------------------------------
def load_eta(path=CSV_PATH):
    """Unrounded total tracking efficiency (%) keyed by (psc, profile, alg)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'{path} not found, run compute_error_metrics.py first')
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[(row['psc'], row['profile'], row['algorithm'])] = \
                float(row['eta_total_pct'])
    if len(out) != 96:
        raise ValueError(f'expected 96 cases in {path}, found {len(out)}')
    return out


_ETA = None


def eta(psc, prof, alg):
    global _ETA
    if _ETA is None:
        _ETA = load_eta()
    return _ETA[(psc, prof, alg)]


def fmt(x, nd=1, sign=False):
    """Round half up once from the unrounded value, typographic minus sign."""
    q = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd),
                                         rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    s = f'{q:.{nd}f}'
    if s.startswith('-'):
        s = MINUS + s[1:]
    elif sign:
        s = '+' + s
    return s


def pv_curve(G, num_points=3000, T=25.0):
    """String P-V characteristic for the module irradiance vector G."""
    iv = evaluate_tct(np.asarray(G, float).reshape(3, 1),
                      np.full((3, 1), T), mtr.MODULE, num_points=num_points)
    return np.asarray(iv['V']), np.asarray(iv['P'])


def pv_peaks(V, P, min_prominence=5.0):
    """Local maxima of a P-V curve with a power prominence of at least 5 W,
    returned in order of increasing voltage as (V, P) pairs."""
    idx, _ = find_peaks(P, prominence=min_prominence)
    if P.argmax() not in idx:
        idx = np.sort(np.append(idx, P.argmax()))
    return [(float(V[i]), float(P[i])) for i in idx]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, metadata={'Software': None})
    plt.close(fig)
    print(f'  Saved: {path}')


def panel_letter(ax, letter, x=-0.02, y=1.02, ha='right'):
    ax.text(x, y, f'({letter})', transform=ax.transAxes, fontsize=FS_PANEL,
            ha=ha, va='bottom')


def profile_axis(ax, show_labels=True, drop_pt=15.0):
    """Label the eight profiles on the x axis with their parameter values and
    a line under each profile type naming it. The line is placed drop_pt
    points below the axis, independent of the axes height."""
    ax.set_xticks(np.arange(len(PROFILES)))
    ax.tick_params(axis='x', length=0)
    if not show_labels:
        ax.set_xticklabels([])
        return
    ax.set_xticklabels(PROFILE_TICKS)
    y0 = 0.0                                    # bottom edge of the axes
    trans = ax.get_xaxis_transform() + mtransforms.ScaledTranslation(
        0, -drop_pt / 72.0, ax.figure.dpi_scale_trans)
    for i0, i1, txt in PROFILE_GROUPS:
        ax.plot([i0 - 0.4, i1 + 0.4], [y0, y0], color='0.35', lw=0.6,
                transform=trans, clip_on=False)
        ax.text((i0 + i1) / 2, y0, txt, transform=trans + mtransforms.
                ScaledTranslation(0, -1.5 / 72.0, ax.figure.dpi_scale_trans),
                ha='center', va='top', fontsize=FS_TICK)


def text_color(rgba):
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return 'black' if lum > 0.5 else 'white'


# ---------------------------------------------------------------------------
# Shading scenarios
# ---------------------------------------------------------------------------
def fig_shading_scenarios():
    """Module irradiance before and after the transition (upper row) and the
    model-computed P-V characteristic of the final distribution (lower row)."""
    cmap = plt.get_cmap('YlGnBu_r')
    fig = plt.figure(figsize=(width_in(0.9), 3.35), layout='constrained')
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25])
    ax_pv0 = None
    for col, (psc, name) in enumerate(zip(PSC_KEYS, PSC_LABELS)):
        Gi = mtr.PSC_PATTERNS[psc]['G_init']
        Gf = mtr.PSC_PATTERNS[psc]['G_final']
        ax = fig.add_subplot(gs[0, col])
        for row, (gi, gf) in enumerate(zip(Gi, Gf)):
            y = 2 - row
            for x0, g in [(0.02, gi), (0.60, gf)]:
                ax.add_patch(mpatches.Rectangle(
                    (x0, y * 1.1), 0.38, 0.92, facecolor=cmap(g / 1000.0),
                    edgecolor='0.3', lw=0.4))
                ax.text(x0 + 0.19, y * 1.1 + 0.46, f'{g:g}', ha='center',
                        va='center', fontsize=FS_SMALL,
                        color=text_color(cmap(g / 1000.0)))
            ax.annotate('', xy=(0.59, y * 1.1 + 0.46),
                        xytext=(0.41, y * 1.1 + 0.46),
                        arrowprops=dict(arrowstyle='->', lw=0.7, color='0.2'))
        ax.text(0.21, 3.36, 'Initial', ha='center', va='bottom',
                fontsize=FS_SMALL, color='0.25')
        ax.text(0.79, 3.36, 'Final', ha='center', va='bottom',
                fontsize=FS_SMALL, color='0.25')
        ax.text(0.5, 3.85, name, ha='center', va='bottom', fontsize=FS_PANEL)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 4.3)
        ax.axis('off')
        if col == 0:
            for row in range(3):
                ax.text(-0.04, (2 - row) * 1.1 + 0.46, f'Module {row + 1}',
                        ha='right', va='center', fontsize=FS_SMALL)
            ax.text(-0.04, 3.36, '(W/m$^2$)', ha='right', va='bottom',
                    fontsize=FS_SMALL, color='0.25')

        ax = fig.add_subplot(gs[1, col], sharey=ax_pv0)
        if ax_pv0 is None:
            ax_pv0 = ax
        V, P = pv_curve(Gf)
        ax.plot(V, P, color='black', lw=1.1)
        i = int(P.argmax())
        ax.plot(V[i], P[i], 'o', color='black', ms=4, zorder=3)
        ax.text(0.98, 1.0, f'GMPP {fmt(P[i], 0)} W\nat {fmt(V[i], 1)} V',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=FS_SMALL, linespacing=1.15)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 500)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 100, 200, 300, 400, 500])
        if col == 0:
            ax.set_ylabel('Power (W)')
        else:
            ax.tick_params(labelleft=False)
        ax.grid(True)
    fig.supxlabel('Voltage (V)', fontsize=FS_LABEL)
    save(fig, 'shading_scenarios.png')


# ---------------------------------------------------------------------------
# P-V curves of the final state
# ---------------------------------------------------------------------------
def fig_pv_curves():
    """Final-state P-V curves of the four PSC levels with all maxima marked."""
    fig, ax = plt.subplots(figsize=(width_in(0.9), 2.75), layout='constrained')
    handles = []
    for i, psc in enumerate(PSC_KEYS):
        Gf = mtr.PSC_PATTERNS[psc]['G_final']
        V, P = pv_curve(Gf)
        c = NEUTRAL_RAMP[i]
        h, = ax.plot(V, P, color=c, ls=LS_RAMP[i], lw=1.4,
                     label=f'{PSC_LABELS[i]} ({"/".join(f"{g:g}" for g in Gf)}'
                           ' W/m$^2$)')
        handles.append(h)
        pk = pv_peaks(V, P)
        g = int(np.argmax([p for _, p in pk]))
        for j, (vp, pp) in enumerate(pk):
            ax.plot(vp, pp, 'o', ms=4.2, zorder=5, color=c,
                    mfc=c if j == g else 'white', mew=1.0)
    ax.set_xlabel('String voltage (V)')
    ax.set_ylabel('Power (W)')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 400)
    ax.grid(True)
    marker_h = [Line2D([], [], ls='none', marker='o', ms=4.2, color='0.2',
                       label='GMPP'),
                Line2D([], [], ls='none', marker='o', ms=4.2, color='0.2',
                       mfc='white', mew=1.0, label='Local maximum')]
    fig.legend(handles=handles + marker_h, loc='outside upper center',
               ncol=2, columnspacing=1.4, handletextpad=0.6,
               labelspacing=0.3)
    save(fig, 'fig_pv_curves.png')


# ---------------------------------------------------------------------------
# Physical basis of the sigmoid model
# ---------------------------------------------------------------------------
B_VALUES = [0.5, 1.0, 2.0, 5.0]


def fig_physics_basis():
    """(a) Statistics of |b| reported by Lappalainen and Valkealahti (2015),
    drawn as the two reported fractions and the mean, with the four tested
    values. (b) Sigmoid profiles of the four tested b for a 700 to 500 W/m2
    fall."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in(0.9), 2.35),
                                   layout='constrained')
    # Reported means of |b|: 1.79 s for falls and 1.98 s for rises
    for mean_b in (1.79, 1.98):
        ax1.axvline(mean_b, color='0.35', ls=(0, (4, 2)), lw=0.9)
    ax1.text(2.10, 0.27, 'mean $|b|$: 1.79 s (falls),\n1.98 s (rises)',
             fontsize=FS_SMALL, va='center', ha='left', color='0.15')
    ax1.plot([1.5, 5.0], [0.54, 0.96], 'o', color='black', ms=4.5, zorder=4,
             ls='none')
    ax1.text(1.78, 0.54, '54% below 1.5 s', fontsize=FS_SMALL,
             ha='left', va='center',
             bbox=dict(boxstyle='square,pad=0.1', fc='white', ec='none'))
    ax1.text(5.0 - 0.15, 0.80, '96% below\n5.0 s', fontsize=FS_SMALL,
             ha='center', va='top')
    for b, c in zip(B_VALUES, NEUTRAL_RAMP):
        ax1.plot(b, 0.0, marker='^', color=c, ms=5.5, clip_on=False, zorder=5,
                 ls='none')
    ax1.text(3.4, 0.07, 'tested $b$', fontsize=FS_SMALL, ha='center',
             va='bottom', color='0.15')
    ax1.set_xlim(0, 6.0)
    ax1.set_ylim(0, 1.04)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax1.set_xlabel('Shape parameter $|b|$ (s)')
    ax1.set_ylabel('Reported fraction of\ntransitions below $|b|$')
    ax1.grid(True)
    panel_letter(ax1, 'a', x=-0.30, ha='left')

    tt = np.linspace(0.0, 60.0, 2401)
    for b, c, ls in zip(B_VALUES, NEUTRAL_RAMP, LS_RAMP):
        G = mtr._sigmoid(tt, 700.0, 500.0, 5.0, b)
        ax2.plot(tt, G, color=c, ls=ls, lw=1.4, label=f'$b$ = {b:.1f} s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Irradiance (W/m$^2$)')
    ax2.set_xlim(0, 60)
    ax2.set_ylim(490, 720)
    ax2.legend(loc='upper right', labelspacing=0.25, handlelength=1.8,
               borderaxespad=0.1)
    ax2.grid(True)
    panel_letter(ax2, 'b', x=-0.30, ha='left')
    save(fig, 'fig_physics_basis.png')


# ---------------------------------------------------------------------------
# Transition profiles
# ---------------------------------------------------------------------------
def fig_transition_profiles():
    """Shaded-module irradiance under Hard PSC for all eight tested profiles:
    (a) step and linear ramps, (b) sigmoid profiles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(width_in(1.0), 2.3),
                                   sharey=True, layout='constrained')

    def series(prof):
        t, _, _, G3 = mtr.build_irradiance('hard', prof)
        return t - mtr.T_BEFORE, G3

    t, G = series('step')
    ax1.plot(t, G, color='black', lw=1.4, label='Step')
    for prof, lab, c, ls in zip(['linear_5s', 'linear_10s', 'linear_20s'],
                                ['Linear 5 s', 'Linear 10 s', 'Linear 20 s'],
                                NEUTRAL_RAMP[:3], LS_RAMP[:3]):
        t, G = series(prof)
        ax1.plot(t, G, color=c, ls=ls, lw=1.4, label=lab)
    for b, c, ls in zip(B_VALUES, NEUTRAL_RAMP, LS_RAMP):
        t, G = series(f'sigmoid_{b:.1f}')
        ax2.plot(t, G, color=c, ls=ls, lw=1.4, label=f'$b$ = {b:.1f} s')
    for ax, letter in [(ax1, 'a'), (ax2, 'b')]:
        ax.set_xlim(-3, 55)
        ax.set_ylim(90, 210)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([100, 125, 150, 175, 200])
        ax.grid(True)
        ax.legend(loc='upper right', labelspacing=0.25, handlelength=1.8,
                  borderaxespad=0.1,
                  title=None if ax is ax1 else 'Sigmoid',
                  title_fontsize=FS_TICK)
        ax.set_xlabel('Time from transition start (s)')
        panel_letter(ax, letter, x=-0.02, ha='right')
    ax1.set_ylabel('Module 3 irradiance (W/m$^2$)')
    save(fig, 'fig_transition_profiles.png')


# ---------------------------------------------------------------------------
# Efficiency matrices
# ---------------------------------------------------------------------------
def _matrix(fn):
    return np.array([[fn(psc, prof) for prof in PROFILES] for psc in PSC_KEYS])


def _matrix_axes(ax, show_x=True):
    ax.set_yticks(range(4))
    ax.set_yticklabels(PSC_LABELS)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.tick_params(axis='x', pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    profile_axis(ax, show_labels=show_x)


def fig_heatmap_3alg():
    """Tracking efficiency of all 96 cases, one panel per algorithm."""
    fig = plt.figure(figsize=(width_in(0.95), 4.15), layout='constrained')
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 0.035])
    cmap = plt.get_cmap('cividis')
    vmin, vmax = 15.0, 100.0
    for k, alg in enumerate(ALG_ORDER):
        ax = fig.add_subplot(gs[k, 0])
        data = _matrix(lambda p, q: eta(p, q, alg))
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        for i in range(4):
            for j in range(8):
                v = data[i, j]
                ax.text(j, i, fmt(v, 0), ha='center', va='center',
                        fontsize=FS_SMALL,
                        color=text_color(cmap((v - vmin) / (vmax - vmin))))
        _matrix_axes(ax, show_x=(k == 2))
        ax.set_title(f'({"abc"[k]}) {ALG_STYLE[alg]["label"]}', loc='left',
                     fontsize=FS_PANEL, pad=3)
    cax = fig.add_subplot(gs[:, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Tracking efficiency (%)')
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)
    save(fig, 'fig_heatmap_3alg.png')


def fig_winner_matrix():
    """Highest-efficiency algorithm per condition and its margin (pp) over the
    second-best algorithm."""
    code = {'spline': 0, 'po': 1, 'inc': 2}
    win = np.zeros((4, 8), int)
    margin = np.zeros((4, 8))
    for i, psc in enumerate(PSC_KEYS):
        for j, prof in enumerate(PROFILES):
            vals = sorted(((eta(psc, prof, a), a) for a in ALG_ORDER),
                          reverse=True)
            win[i, j] = code[vals[0][1]]
            margin[i, j] = vals[0][0] - vals[1][0]
    cmap = ListedColormap([COL_SPLINE, COL_PO, COL_INC])
    fig, ax = plt.subplots(figsize=(width_in(0.9), 2.2), layout='constrained')
    ax.imshow(win, aspect='auto', cmap=cmap, vmin=-0.5, vmax=2.5)
    for i in range(4):
        for j in range(8):
            if win[i, j] == code['inc']:
                ax.add_patch(mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False, hatch='///',
                    edgecolor=(1, 1, 1, 0.30), lw=0))
            ax.text(j, i, f'+{fmt(margin[i, j], 1)}', ha='center',
                    va='center', fontsize=FS_SMALL, color='white')
    for x in np.arange(-0.5, 8.5, 1.0):
        ax.axvline(x, color='white', lw=1.0)
    for y in np.arange(-0.5, 4.5, 1.0):
        ax.axhline(y, color='white', lw=1.0)
    _matrix_axes(ax)
    present = sorted(set(win.ravel().tolist()))
    labels = {0: 'Spline-MPPT highest', 1: 'P&O highest', 2: 'INC highest'}
    handles = []
    for c in present:
        handles.append(mpatches.Patch(
            facecolor=cmap(c), edgecolor='white', lw=0,
            hatch='///' if c == code['inc'] else None, label=labels[c]))
    fig.legend(handles=handles, loc='outside upper center', ncol=len(handles),
               handlelength=1.6, columnspacing=1.6)
    save(fig, 'fig_winner_matrix.png')


def fig_spline_advantage():
    """Spline-MPPT efficiency minus that of the better local algorithm (pp)."""
    data = _matrix(lambda p, q: eta(p, q, 'spline') -
                   max(eta(p, q, 'po'), eta(p, q, 'inc')))
    absmax = float(np.abs(data).max())
    cmap = LinearSegmentedColormap.from_list(
        'adv', ['#8c510a', '#f5f5f5', COL_SPLINE], N=256)
    fig, ax = plt.subplots(figsize=(width_in(0.7), 1.95), layout='constrained')
    ax.imshow(data, aspect='auto', cmap=cmap, vmin=-absmax, vmax=absmax)
    for i in range(4):
        for j in range(8):
            v = data[i, j]
            ax.text(j, i, fmt(v, 1, sign=True), ha='center', va='center',
                    fontsize=FS_SMALL,
                    color='white' if abs(v) >= 15.0 else 'black')
    for x in np.arange(-0.5, 8.5, 1.0):
        ax.axvline(x, color='white', lw=0.8)
    for y in np.arange(-0.5, 4.5, 1.0):
        ax.axhline(y, color='white', lw=0.8)
    _matrix_axes(ax)
    save(fig, 'fig_spline_advantage.png')


# ---------------------------------------------------------------------------
# INC and P&O divergence
# ---------------------------------------------------------------------------
PROFILE_SEGMENTS = [(0, 1), (1, 4), (4, 8)]   # step, linear, sigmoid groups


def alg_handle(a, **kw):
    st = ALG_STYLE[a]
    return Line2D([], [], color=st['color'], ls=st['ls'], marker=st['marker'],
                  ms=4.2, lw=1.3, mew=1.0,
                  mfc='none' if a == 'po' else st['color'],
                  label=st['label'], **kw)


def fig_inc_po_divergence():
    """Efficiency of the three algorithms across the eight profiles under Hard
    and Extreme PSC. Lines join profiles of the same type only."""
    fig, axes = plt.subplots(1, 2, figsize=(width_in(0.95), 2.55),
                             sharey=True, layout='constrained')
    x = np.arange(len(PROFILES))
    for ax, psc, letter in zip(axes, ['hard', 'extreme'], 'ab'):
        vals = {a: np.array([eta(psc, p, a) for p in PROFILES])
                for a in ALG_ORDER}
        for s0, s1 in PROFILE_SEGMENTS:
            seg = slice(s0, s1)
            gap = vals['inc'][seg] > vals['po'][seg]
            ax.fill_between(x[seg], vals['po'][seg], vals['inc'][seg],
                            where=gap, color=COL_INC, alpha=0.15, lw=0,
                            interpolate=True)
            for a in ['po', 'inc', 'spline']:
                st = ALG_STYLE[a]
                ax.plot(x[seg], vals[a][seg], color=st['color'], ls=st['ls'],
                        marker=st['marker'], ms=4.2, lw=1.3,
                        mfc='none' if a == 'po' else st['color'],
                        mew=1.0, zorder=5 if a == 'po' else 4)
        for xb in [0.5, 3.5]:
            ax.axvline(xb, color='0.75', lw=0.5)
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(10, 80)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])
        ax.grid(True, axis='y')
        profile_axis(ax)
        ax.set_title(f'({letter}) {"Hard" if psc == "hard" else "Extreme"} PSC',
                     loc='left', fontsize=FS_PANEL, pad=3)
    axes[0].set_ylabel('Tracking efficiency (%)')
    handles = [alg_handle(a) for a in ALG_ORDER]
    handles.append(mpatches.Patch(color=COL_INC, alpha=0.15, lw=0,
                                  label='INC above P&O'))
    fig.legend(handles=handles, loc='outside upper center', ncol=4,
               columnspacing=1.5)
    save(fig, 'fig_inc_po_divergence.png')


# ---------------------------------------------------------------------------
# Step against sigmoid b = 2 s
# ---------------------------------------------------------------------------
def alg_patch(a):
    st = ALG_STYLE[a]
    return mpatches.Patch(
        facecolor=st['color'] if st['hatch'] is None else 'white',
        edgecolor=st['color'], hatch=st['hatch'], lw=0.9, label=st['label'])


def alg_bar(ax, x, h, alg, width):
    st = ALG_STYLE[alg]
    ax.bar(x, h, width, color=st['color'] if st['hatch'] is None else 'white',
           edgecolor=st['color'], hatch=st['hatch'], lw=0.9, zorder=3)


def fig_main_finding():
    """Tracking efficiency under the step change and the sigmoid b = 2 s
    profile for every PSC level. Bars are grouped by profile."""
    fig, axes = plt.subplots(1, 4, figsize=(width_in(0.95), 2.45),
                             sharey=True, layout='constrained')
    w = 0.26
    offs = {'spline': -w, 'po': 0.0, 'inc': w}
    for ax, psc, name in zip(axes, PSC_KEYS, PSC_LABELS):
        for g, prof in enumerate(['step', 'sigmoid_2.0']):
            for a in ALG_ORDER:
                v = eta(psc, prof, a)
                alg_bar(ax, g + offs[a], v, a, w * 0.92)
                ax.text(g + offs[a], v + 1.5, fmt(v, 1), ha='center',
                        va='bottom', fontsize=FS_SMALL, rotation=90)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Step', 'Sigmoid\n$b$ = 2 s'])
        ax.tick_params(axis='x', length=0)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, axis='y')
        ax.set_title(f'{name} PSC', fontsize=FS_PANEL, pad=2)
    axes[0].set_ylabel('Tracking efficiency (%)')
    fig.legend(handles=[alg_patch(a) for a in ALG_ORDER],
               loc='outside upper center', ncol=3, handlelength=1.6,
               columnspacing=1.8)
    save(fig, 'fig_main_finding.png')


ALL_FIGURES = [fig_shading_scenarios, fig_pv_curves, fig_physics_basis,
               fig_transition_profiles, fig_heatmap_3alg, fig_winner_matrix,
               fig_spline_advantage, fig_inc_po_divergence, fig_main_finding]


def main():
    print('Generating publication figures...')
    for fn in ALL_FIGURES:
        fn()
    print(f'Done, figures written to {FIG_DIR}/')


if __name__ == '__main__':
    main()
