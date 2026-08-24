"""
generate_figures.py  —  Publication Figures (Revised Manuscript)
=================================================================
Run from the project root directory:  python3 generate_figures.py
Requires results/run1_easy.json ... run4_extreme.json to exist.
Saves figures to:  figures/

Figures produced
----------------
fig_pv_curves.png        — P-V curves for 4 PSC levels
fig_transition_profiles.png — 8 irradiance profiles
fig_heatmap_3alg.png     — 96-case efficiency heatmap
fig_spline_advantage.png — Spline advantage over best-local
fig_main_finding.png     — Step vs Sigmoid bar chart (core result)
fig_winner_matrix.png    — Algorithm winner matrix
fig_inc_po_divergence.png— INC vs P&O divergence (new finding)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from tct_eval import default_kc200gt, evaluate_tct
from modular_test_runner import build_irradiance

os.makedirs('figures', exist_ok=True)

# ── Style (no LaTeX renderer needed) ─────────────────────────────────────────
plt.rcParams.update({
    'font.family':     'serif',
    'font.size':       10,
    'axes.labelsize':  11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'savefig.dpi':     300,
    'savefig.bbox':    'tight',
    'lines.linewidth': 1.5,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

COL_SPLINE = '#1a6faf'
COL_PO     = '#e05c2a'
COL_INC    = '#2ca02c'

PSC_KEYS   = ['easy', 'moderate', 'hard', 'extreme']
PROFILES   = ['step', 'linear_5s', 'linear_10s', 'linear_20s',
              'sigmoid_0.5', 'sigmoid_1.0', 'sigmoid_2.0', 'sigmoid_5.0']
PROF_LABELS = ['Step', 'Lin 5s', 'Lin 10s', 'Lin 20s',
               'Sig 0.5s', 'Sig 1.0s', 'Sig 2.0s', 'Sig 5.0s']

# Load simulation results
runs = {}
for psc, fname in [('easy','run1_easy'), ('moderate','run2_moderate'),
                   ('hard','run3_hard'), ('extreme','run4_extreme')]:
    with open(f'results/{fname}.json') as f:
        runs[psc] = json.load(f)

def eta(psc, prof, alg):
    return runs[psc]['results'][prof][alg]['eta_total']


# ── Fig 1: P-V curves ────────────────────────────────────────────────────────
def fig_pv_curves():
    m = default_kc200gt()
    T = np.full((3, 1), 25.0)
    G_finals = {'easy':[1000,900,500], 'moderate':[1000,600,300],
                'hard':[1000,400,100], 'extreme':[1000,300,50]}
    colors = [COL_SPLINE, COL_PO, COL_INC, '#9467bd']
    labels = ['Easy (1000/900/500 W/m²)', 'Moderate (1000/600/300)',
              'Hard (1000/400/100)', 'Extreme (1000/300/50)']

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, psc in enumerate(PSC_KEYS):
        G = np.array(G_finals[psc]).reshape(3, 1)
        iv = evaluate_tct(G, T, m, num_points=400)
        ax.plot(iv['V'], iv['P'], color=colors[i], label=labels[i], lw=1.8)
        ax.plot(iv['Vmpp'], iv['Pmpp'], 'o', color=colors[i], ms=6, zorder=5)
    ax.set_xlabel('String voltage (V)')
    ax.set_ylabel('Power (W)')
    ax.legend(frameon=False, loc='upper right', fontsize=8.5)
    ax.set_xlim(0, 105); ax.set_ylim(0, 660)
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()
    fig.savefig('figures/fig_pv_curves.png'); plt.close()
    print('  Saved: figures/fig_pv_curves.png')


# ── Fig 2: Transition profiles ───────────────────────────────────────────────
def fig_transition_profiles():
    profiles_to_show = [
        ('step',       'Step'),
        ('linear_5s',  'Linear 5 s'),
        ('linear_20s', 'Linear 20 s'),
        ('sigmoid_0.5','Sigmoid b = 0.5 s'),
        ('sigmoid_2.0','Sigmoid b = 2 s'),
        ('sigmoid_5.0','Sigmoid b = 5 s'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.7), sharey=True)
    for ax, (prof, lbl) in zip(axes.flatten(), profiles_to_show):
        t_vec, _, _, G3 = build_irradiance('hard', prof)
        t_plot = t_vec - 5.0
        mask = (t_plot >= -2) & (t_plot <= 30)
        ax.plot(t_plot[mask], G3[mask], color=COL_SPLINE, lw=1.8)
        ax.axhline(200, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(100, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.set_ylim(50, 250); ax.set_xlim(-2, 25)
        ax.text(0.97, 0.95, lbl, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none'))
        ax.grid(True, alpha=0.3, lw=0.5)
    for ax in axes[1]: ax.set_xlabel('Time from transition start (s)')
    for ax in axes[:, 0]: ax.set_ylabel('Module 3 irradiance (W/m²)')
    fig.tight_layout(h_pad=1.0, w_pad=0.5)
    fig.savefig('figures/fig_transition_profiles.png'); plt.close()
    print('  Saved: figures/fig_transition_profiles.png')


# ── Fig 3: 96-case heatmap ───────────────────────────────────────────────────
def fig_heatmap_3alg():
    algs       = ['spline', 'po', 'inc']
    alg_labels = ['Spline-MPPT', 'P&O', 'INC']
    cmap = LinearSegmentedColormap.from_list('rg', ['#d73027','#fee08b','#1a9850'])
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9), sharey=True, layout='constrained')
    for ax, alg, label in zip(axes, algs, alg_labels):
        data = np.array([[eta(psc, prof, alg)
                          for prof in PROFILES] for psc in PSC_KEYS])
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=19, vmax=96, origin='upper')
        ax.set_xticks(range(8)); ax.set_xticklabels(PROF_LABELS, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(4)); ax.set_yticklabels(['Easy','Moderate','Hard','Extreme'], fontsize=9)
        ax.set_xlabel(label, fontsize=10, fontweight='bold')
        for i in range(4):
            for j in range(8):
                v = data[i, j]
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=7, color='white' if (v < 55 or v > 85) else 'black')
    fig.colorbar(im, ax=axes, shrink=0.8, pad=0.015).set_label('Tracking efficiency (%)', fontsize=9)
    fig.savefig('figures/fig_heatmap_3alg.png'); plt.close()
    print('  Saved: figures/fig_heatmap_3alg.png')


# ── Fig 4: Spline advantage heatmap ─────────────────────────────────────────
def fig_spline_advantage():
    data = np.array([[eta(psc, prof, 'spline') - max(eta(psc, prof, 'po'), eta(psc, prof, 'inc'))
                      for prof in PROFILES] for psc in PSC_KEYS])
    absmax = max(abs(data.min()), abs(data.max()))
    cmap = LinearSegmentedColormap.from_list('bwr', ['#d73027','#f7f7f7','#1a6faf'], N=256)
    fig, ax = plt.subplots(figsize=(6.5, 3.1))
    im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=-absmax, vmax=absmax, origin='upper')
    ax.set_xticks(range(8)); ax.set_xticklabels(PROF_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(['Easy','Moderate','Hard','Extreme'], fontsize=10)
    ax.set_xlabel('Transition profile', fontsize=10)
    for i in range(4):
        for j in range(8):
            v = data[i, j]
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                    fontsize=7.5, color='white' if abs(v) > 15 else 'black')
    fig.colorbar(im, ax=ax, pad=0.01).set_label('Spline advantage over best-local (pp)', fontsize=9)
    fig.tight_layout()
    fig.savefig('figures/fig_spline_advantage.png'); plt.close()
    print('  Saved: figures/fig_spline_advantage.png')


# ── Fig 5: Main finding bar chart ────────────────────────────────────────────
def fig_main_finding():
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.6))
    x = np.arange(3); w = 0.32
    alg_keys = ['spline', 'po', 'inc']
    alg_labels = ['Spline-MPPT', 'P&O', 'INC']
    colors = [COL_SPLINE, COL_PO, COL_INC]

    for ax, psc in zip(axes, PSC_KEYS):
        vals_step = [eta(psc, 'step',       a) for a in alg_keys]
        vals_sig  = [eta(psc, 'sigmoid_2.0', a) for a in alg_keys]
        bars1 = ax.bar(x - w/2, vals_step, w, color=colors, alpha=0.95, zorder=3)
        bars2 = ax.bar(x + w/2, vals_sig,  w, color=colors, alpha=0.40,
                       hatch='///', zorder=3, edgecolor=colors, linewidth=0.8)
        for bar, dx in [(b, -0.04) for b in bars1] + [(b, 0.04) for b in bars2]:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2 + dx, h + 1.0, f'{h:.0f}',
                    ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(['Spline', 'P&O', 'INC'], fontsize=8.5)
        ax.set_ylim(0, 118)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        if psc == 'easy': ax.set_ylabel('Tracking efficiency (%)')
        ax.set_xlabel(['Easy PSC','Moderate PSC','Hard PSC','Extreme PSC'][PSC_KEYS.index(psc)],
                      fontsize=9, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3, lw=0.5, zorder=0)
        ax.set_axisbelow(True)

    legend_els = [
        mpatches.Patch(facecolor='gray', alpha=0.95, label='Step change'),
        mpatches.Patch(facecolor='gray', alpha=0.40, hatch='///', edgecolor='gray',
                       label='Sigmoid b = 2 s'),
    ]
    fig.legend(handles=legend_els, frameon=False, loc='upper center',
               ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.97))
    fig.savefig('figures/fig_main_finding.png'); plt.close()
    print('  Saved: figures/fig_main_finding.png')


# ── Fig 6: Winner matrix ─────────────────────────────────────────────────────
def fig_winner_matrix():
    winner_grid = np.zeros((4, 8))
    winner_labels = []
    for i, psc in enumerate(PSC_KEYS):
        row = []
        for j, prof in enumerate(PROFILES):
            sp = eta(psc, prof, 'spline')
            po = eta(psc, prof, 'po')
            inc = eta(psc, prof, 'inc')
            if inc > sp and inc > po:
                winner_grid[i, j] = 2; row.append('INC')
            elif sp >= max(po, inc):
                winner_grid[i, j] = 1; row.append('Spl')
            else:
                winner_grid[i, j] = 0; row.append('P&O')
        winner_labels.append(row)
    cmap = matplotlib.colors.ListedColormap([COL_PO, COL_SPLINE, COL_INC])
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.imshow(winner_grid, aspect='auto', cmap=cmap, vmin=0, vmax=2, origin='upper')
    ax.set_xticks(range(8)); ax.set_xticklabels(PROF_LABELS, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(['Easy','Moderate','Hard','Extreme'], fontsize=10)
    ax.set_xlabel('Transition profile', fontsize=10)
    for i in range(4):
        for j in range(8):
            ax.text(j, i, winner_labels[i][j], ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')
    legend_els = [mpatches.Patch(color=COL_PO,     label='P&O wins'),
                  mpatches.Patch(color=COL_SPLINE,  label='Spline-MPPT wins'),
                  mpatches.Patch(color=COL_INC,     label='INC wins')]
    ax.legend(handles=legend_els, frameon=False, loc='lower right', fontsize=9,
              bbox_to_anchor=(1.0, 1.01), ncol=3)
    fig.tight_layout()
    fig.savefig('figures/fig_winner_matrix.png'); plt.close()
    print('  Saved: figures/fig_winner_matrix.png')


# ── Fig 7: INC vs P&O divergence ────────────────────────────────────────────
def fig_inc_po_divergence():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
    x = np.arange(len(PROFILES))
    for ax, psc in zip(axes, ['hard', 'extreme']):
        sp_vals  = [eta(psc, p, 'spline') for p in PROFILES]
        inc_vals = [eta(psc, p, 'inc')    for p in PROFILES]
        po_vals  = [eta(psc, p, 'po')     for p in PROFILES]
        ax.plot(x, sp_vals,  'o-', color=COL_SPLINE, lw=2,   ms=6, label='Spline-MPPT', zorder=4)
        ax.plot(x, inc_vals, 's-', color=COL_INC,    lw=2,   ms=6, label='INC',         zorder=4)
        ax.plot(x, po_vals,  '^--',color=COL_PO,     lw=1.5, ms=5, label='P&O',         zorder=4, alpha=0.85)
        ax.fill_between(x, po_vals, inc_vals,
                        where=[i > p for i, p in zip(inc_vals, po_vals)],
                        color=COL_INC, alpha=0.12)
        ax.set_xticks(x); ax.set_xticklabels(PROF_LABELS, rotation=45, ha='right', fontsize=8.5)
        ax.set_ylabel('Tracking efficiency (%)')
        ax.set_xlabel('Hard PSC' if psc == 'hard' else 'Extreme PSC',
                      fontsize=10, fontweight='bold')
        ax.set_ylim(10, 85)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.legend(frameon=False, loc='upper left', fontsize=8.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig('figures/fig_inc_po_divergence.png'); plt.close()
    print('  Saved: figures/fig_inc_po_divergence.png')


# ── Fig 8: physics basis (reported statistics + model family) ────────────────
def fig_physics_basis():
    from modular_test_runner import _sigmoid
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))

    # (a) reported distribution statistics, drawn as cited quantities
    ax1.axvline(1.88, color='#c44e52', ls='--', lw=1.4)
    ax1.text(1.98, 0.30, 'reported mean\n$|b| = 1.88$ s', fontsize=8, color='#c44e52')
    pts_b   = [1.5, 5.0]
    pts_cum = [0.54, 0.96]
    ax1.plot(pts_b, pts_cum, 'o', color='#1a6faf', ms=6, zorder=3)
    ax1.annotate('54% of transitions\nbelow 1.5 s', xy=(1.5, 0.54), xytext=(2.6, 0.52),
                 fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.7))
    ax1.annotate('96% below 5.0 s', xy=(5.0, 0.96), xytext=(3.4, 0.84),
                 fontsize=8, arrowprops=dict(arrowstyle='-', lw=0.7))
    for b in [0.5, 1.0, 2.0, 5.0]:
        ax1.plot(b, 0.02, marker='^', color='#2ca02c', ms=7, clip_on=False)
    ax1.text(0.42, 0.08, 'tested values of $b$', fontsize=8, color='#2ca02c')
    ax1.set_xlim(0, 6.2); ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Shape parameter $|b|$ (s)')
    ax1.set_ylabel('Cumulative fraction of\nmeasured transitions')
    ax1.grid(True, alpha=0.3, lw=0.5)

    # (b) sigmoid model family for the tested shape parameters
    tt = np.linspace(0, 60, 1200)
    for b, c in zip([0.5, 1.0, 2.0, 5.0],
                    ['#1a6faf', '#2ca02c', '#e05c2a', '#7f4fc9']):
        G = _sigmoid(tt, 700.0, 500.0, 5.0, b)
        ax2.plot(tt, G, color=c, lw=1.6, label=f'$b={b}$ s')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Irradiance (W/m$^2$)')
    ax2.set_xlim(0, 60); ax2.set_ylim(480, 720)
    ax2.legend(frameon=False, fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, lw=0.5)

    fig.tight_layout(w_pad=1.2)
    fig.savefig('figures/fig_physics_basis.png'); plt.close()
    print('  Saved: figures/fig_physics_basis.png')


# ── Fig 9: shading scenarios with model-computed final P-V curves ────────────
def fig_shading_scenarios():
    from tct_eval import default_kc200gt, evaluate_tct
    M = default_kc200gt()
    scen = [
        ('Easy',     [1000, 900, 700], [1000, 900, 500]),
        ('Moderate', [1000, 600, 400], [1000, 600, 300]),
        ('Hard',     [1000, 400, 200], [1000, 400, 100]),
        ('Extreme',  [1000, 300, 150], [1000, 300,  50]),
    ]
    cmap = plt.get_cmap('YlGnBu_r')
    fig, axes = plt.subplots(2, 4, figsize=(6.5, 3.9),
                             gridspec_kw={'height_ratios': [1.0, 1.15]})
    for col, (name, Gi, Gf) in enumerate(scen):
        ax = axes[0, col]
        for row3, (gi, gf) in enumerate(zip(Gi, Gf)):
            y = 2 - row3
            for xx, g in [(0.05, gi), (0.60, gf)]:
                ax.add_patch(plt.Rectangle((xx, y*1.12), 0.34, 0.95,
                             facecolor=cmap(g/1000.0), edgecolor='black', lw=0.6))
                ax.text(xx+0.17, y*1.12+0.48, f'{g:g}', ha='center', va='center',
                        fontsize=7.2,
                        color='black' if g > 350 else 'white')
            ax.annotate('', xy=(0.585, y*1.12+0.48), xytext=(0.415, y*1.12+0.48),
                        arrowprops=dict(arrowstyle='->', lw=0.9))
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 3.45)
        ax.set_title(name, fontsize=9.5, fontweight='bold')
        ax.axis('off')
        if col == 0:
            ax.text(-0.13, 1.7, 'Module irradiance\n(W/m$^2$), initial to final',
                    rotation=90, va='center', ha='center', fontsize=8,
                    transform=ax.transData)
        ax = axes[1, col]
        iv = evaluate_tct(np.array(Gf, dtype=float).reshape(3, 1),
                          np.full((3, 1), 25.0), M, num_points=1500)
        V, P = np.array(iv['V']), np.array(iv['P'])
        ax.plot(V, P, color='#1a6faf', lw=1.4)
        i = P.argmax()
        ax.plot(V[i], P[i], 'o', color='#c44e52', ms=5, zorder=3)
        ax.annotate(f'GMPP {P[i]:.0f} W\nat {V[i]:.1f} V', xy=(V[i], P[i]),
                    xytext=(0.35, 0.93), textcoords='axes fraction',
                    fontsize=7.2, ha='left', va='top',
                    arrowprops=dict(arrowstyle='-', lw=0.6))
        ax.set_xlim(0, 100); ax.set_ylim(0, 400)
        ax.set_xlabel('Voltage (V)', fontsize=8)
        if col == 0:
            ax.set_ylabel('Power (W)', fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7.5)
        ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout(h_pad=0.8, w_pad=0.5)
    fig.savefig('figures/shading_scenarios.png'); plt.close()
    print('  Saved: figures/shading_scenarios.png')


if __name__ == '__main__':
    print('Generating publication figures...')
    fig_pv_curves()
    fig_transition_profiles()
    fig_heatmap_3alg()
    fig_spline_advantage()
    fig_main_finding()
    fig_winner_matrix()
    fig_inc_po_divergence()
    fig_physics_basis()
    fig_shading_scenarios()
    print('\nDone — all figures in figures/')
