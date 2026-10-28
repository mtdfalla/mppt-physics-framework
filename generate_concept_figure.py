"""
generate_concept_figure.py -- Figure 1, conceptual overview
===========================================================
Run from the project root directory, after generate_figures.py:

    python3 generate_concept_figure.py

Figure 1 introduces the study in three panels: (a) the physical event, a
cloud edge moving over the three-module string, and the two ways its
irradiance change is modeled, (b) the Hard PSC power-voltage landscape that
the trackers face after the change and where each algorithm operates, and
(c) the test matrix of four shading levels, eight transition profiles and
three algorithms. Every curve and voltage is computed with the released
model: the P-V curves come from the array model, the operating paths and
dwell voltages from time-domain runs of the three algorithms (whose
efficiencies are checked against results/error_metrics_96.csv first), and
the profile sketches from the profile generator.

Outputs: figures/fig_concept.pdf  (vector)
         figures/fig_concept.png  (600 dpi)
         figures/fig_concept.tif  (600 dpi, LZW, with --tiff)
"""
import os
import sys

os.environ.setdefault('SOURCE_DATE_EPOCH', '0')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import (Circle, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Rectangle)

import modular_test_runner as mtr
import generate_figures as gf
from generate_figures import (COL_INC, COL_PO, COL_SPLINE, eta, fmt,
                              pv_curve, pv_peaks)
import generate_overview_figures as go

MM = 1.0 / 25.4
W, H = 190.0, 112.0
INK, MUTED, GRID, CARD = '#1a1a1a', '#5f5f5f', '#e3e3e3', '#fbfaf7'
SKY, SUN, SUNE = '#eaf2fa', '#f6c85f', '#d99a1e'
NAVY, PVB, PVL = '#1f3b57', '#2f5b86', '#8fb3d6'
INC_DARK = '#1e7a1e'
DPI = 600
PSC = 'hard'
LW = 0.6

FS_TITLE, FS_TEXT, FS_NOTE, FS_TICK = 7.6, 6.9, 6.3, 6.3


# ---------------------------------------------------------------------------
# Drawing helpers on the millimetre canvas
# ---------------------------------------------------------------------------
def T(ax, x, y, s, fs=FS_TEXT, ha='center', va='center', c=INK, w='normal',
      z=10, **k):
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, color=c, fontweight=w,
            zorder=z, linespacing=1.2, **k)


def card(ax, x, y, w, h, fc=CARD, ec='#c9c4ba'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0,rounding_size=1.6',
                                fc=fc, ec=ec, lw=LW, zorder=1))


def arrow(ax, p0, p1, color=INK, lw=0.8, ms=6, z=8, cs='arc3,rad=0',
          style='-|>'):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                                 lw=lw, color=color, zorder=z,
                                 connectionstyle=cs, shrinkA=0, shrinkB=0))


def sun(ax, x, y, r):
    for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot([x + 1.4 * r * np.cos(a), x + 1.85 * r * np.cos(a)],
                [y + 1.4 * r * np.sin(a), y + 1.85 * r * np.sin(a)],
                color=SUNE, lw=0.8, solid_capstyle='round', zorder=3)
    ax.add_patch(Circle((x, y), r, fc=SUN, ec=SUNE, lw=LW, zorder=3.1))


def cloud(ax, x, y, s, fc='white', ec='#8a98a8', z=5):
    parts = [(-1.0, 0.0, 0.72), (-0.2, 0.42, 0.95), (0.75, 0.18, 0.78),
             (1.35, -0.12, 0.55), (0.1, -0.12, 0.8)]
    for dx, dy, r in parts:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, fc=fc, ec=ec,
                            lw=LW, zorder=z))
    for dx, dy, r in parts:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s - 0.28, fc=fc,
                            ec='none', zorder=z + 0.1))
    ax.add_patch(Rectangle((x - 1.0 * s, y - 0.72 * s), 2.35 * s, 0.6 * s,
                           fc=fc, ec='none', zorder=z + 0.1))
    ax.plot([x - 1.0 * s, x + 1.35 * s], [y - 0.76 * s] * 2, color=ec,
            lw=LW, zorder=z + 0.2)


def module(ax, x, y, w, h, shade=0.0, z=3):
    """Tilted PV module, (x, y) the lower-left corner; shade in [0, 1]."""
    sk = 0.32 * w
    pts = [(x, y), (x + w, y), (x + w - sk + 0.12 * w, y + h),
           (x - sk + 0.12 * w, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=PVB, ec=NAVY, lw=LW, zorder=z,
                         joinstyle='round'))
    for k in range(1, 6):
        f = k / 6
        ax.plot([x + f * w, x + f * w - sk + 0.12 * w], [y, y + h],
                color=PVL, lw=0.3, zorder=z + 0.1)
    for k in range(1, 4):
        f = k / 4
        ax.plot([x + f * (0.12 * w - sk), x + w + f * (0.12 * w - sk)],
                [y + f * h] * 2, color=PVL, lw=0.3, zorder=z + 0.1)
    if shade > 0:
        ax.add_patch(Polygon(pts, closed=True, fc='#0b1a2a', ec='none',
                             alpha=shade, zorder=z + 0.2))
    for lx in (x + 0.18 * w, x + 0.82 * w):
        ax.plot([lx, lx], [y - 2.2, y], color='#6b6b6b', lw=0.7, zorder=z - 0.1)


def inset(fig, x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H])
    a.set_facecolor('none')
    for s in ('top', 'right'):
        a.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        a.spines[s].set_linewidth(0.5)
    a.tick_params(length=1.8, width=0.5, labelsize=FS_TICK, pad=1.5)
    return a


# ---------------------------------------------------------------------------
# Data, all from the released model
# ---------------------------------------------------------------------------
def collect():
    runs = go.all_runs()                     # checks efficiencies vs CSV
    pat = mtr.PSC_PATTERNS[PSC]
    Vi, Pi = pv_curve(pat['G_init'])
    Vf, Pf = pv_curve(pat['G_final'])
    peaks = pv_peaks(Vf, Pf)
    if len(peaks) != 3:
        raise RuntimeError('expected three maxima on the final curve')
    t, _, _, G3s = mtr.build_irradiance(PSC, 'sigmoid_2.0')
    tt, _, _, G3t = mtr.build_irradiance(PSC, 'step')
    dwell = {k: go.dwell_voltage(r) for k, r in runs.items()}
    eff = {(p, a): eta(PSC, p, a) for p in ('step', 'sigmoid_2.0')
           for a in ('spline', 'po', 'inc')}
    # intermediate curves during the sigmoid fall of module 3
    g_lo, g_hi = pat['G_final'][2], pat['G_init'][2]
    family = []
    for g in np.linspace(g_hi, g_lo, 6)[1:-1]:
        family.append(pv_curve([pat['G_init'][0], pat['G_init'][1], g]))
    thumbs = []
    for key, lab in (('easy', 'Easy'), ('moderate', 'Moderate'),
                     ('hard', 'Hard'), ('extreme', 'Extreme')):
        V, P = pv_curve(mtr.PSC_PATTERNS[key]['G_final'], num_points=600)
        thumbs.append((key, lab, np.asarray(V), np.asarray(P)))
    glyphs = []
    labels = {'step': 'step', 'linear_5s': 'ramp 5 s',
              'linear_10s': 'ramp 10 s', 'linear_20s': 'ramp 20 s',
              'sigmoid_0.5': '$b$ = 0.5 s', 'sigmoid_1.0': '$b$ = 1 s',
              'sigmoid_2.0': '$b$ = 2 s', 'sigmoid_5.0': '$b$ = 5 s'}
    for key in mtr.PROFILES:
        tk, _, _, gk = mtr.build_irradiance(PSC, key)
        tk = np.asarray(tk) - mtr.T_BEFORE
        m = (tk >= -4.0) & (tk <= 36.0)
        gk = np.asarray(gk)[m]
        gn = (gk - gk.min()) / (gk.max() - gk.min())
        glyphs.append((key, labels[key], tk[m], gn))
    return dict(runs=runs, Vi=Vi, Pi=Pi, Vf=Vf, Pf=Pf, peaks=peaks,
                thumbs=thumbs, glyphs=glyphs,
                t=np.asarray(t) - mtr.T_BEFORE, G3s=np.asarray(G3s),
                G3t=np.asarray(G3t), tt=np.asarray(tt) - mtr.T_BEFORE,
                dwell=dwell, eff=eff, family=family,
                pat=pat)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build(d):
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['DejaVu Serif'],
                         'mathtext.fontset': 'dejavuserif',
                         'pdf.fonttype': 42, 'axes.unicode_minus': True})
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect('equal'); ax.axis('off')

    pat = d['pat']
    (vg, pg), (vm, pm), (vh, ph) = d['peaks']
    pct = lambda p: fmt(100.0 * p / pg, 0)
    Vf, Pf = d['Vf'], d['Pf']
    on_curve = lambda v: float(np.interp(v, Vf, Pf))

    # ===================== (a) the event =====================
    xa, wa = 1.5, 54.0
    card(ax, xa, 1.5, wa, H - 3.0)
    T(ax, xa + 2.5, H - 5.0, '(a)', fs=FS_TITLE, ha='left', w='bold')
    T(ax, xa + 8.5, H - 5.0, 'A cloud edge crosses the string',
      fs=FS_TITLE, ha='left')
    ax.add_patch(FancyBboxPatch((xa + 2.5, 59.0), wa - 5.0, 43.0,
                                boxstyle='round,pad=0,rounding_size=1.2',
                                fc=SKY, ec='none', zorder=1.5))
    sun(ax, xa + 9.0, 94.5, 3.0)
    cloud(ax, xa + 40.0, 94.0, 3.6)
    arrow(ax, (xa + 33.0, 87.2), (xa + 25.5, 87.2), color='#8a98a8', lw=0.8,
          ms=5)
    T(ax, xa + 29.2, 85.0, 'cloud motion', fs=FS_NOTE - 0.3, c=MUTED)
    mx = [xa + 6.0, xa + 20.5, xa + 35.0]
    # cloud-edge shadow: penumbra over module 2, deeper over module 3
    for k in range(7):                       # soft-edged penumbra, layered
        lft = 20.5 + 2.2 * k
        ax.add_patch(Polygon([(xa + 31.5 + 1.0 * k, 90.8), (xa + 47.5, 90.8),
                              (xa + 51.0, 69.0), (xa + lft, 69.0)],
                             closed=True, fc='#0b1a2a', ec='none',
                             alpha=0.018, zorder=2))
    for x0, sh in zip(mx, (0.0, 0.30, 0.55)):
        module(ax, x0, 70.5, 11.5, 7.0, shade=sh)
    for x0, x1 in ((mx[0] + 11.5, mx[1]), (mx[1] + 11.5, mx[2])):
        ax.plot([x0 - 0.3, x1 + 0.3], [71.2, 71.2], color='#6b6b6b', lw=0.6,
                zorder=2.5)
    labs = [('Module 1', '%d' % pat['G_init'][0], MUTED),
            ('Module 2', '%d' % pat['G_init'][1], MUTED),
            ('Module 3', '%d$\\rightarrow$%d' % (pat['G_init'][2],
                                                  pat['G_final'][2]), COL_PO)]
    for x0, (nm, g, c) in zip(mx, labs):
        T(ax, x0 + 4.6, 65.2, nm, fs=FS_NOTE, w='bold')
        T(ax, x0 + 4.6, 62.2, g, fs=FS_NOTE, c=c)
    T(ax, xa + wa / 2, 55.8, 'Three KC200GT modules in series, each',
      fs=FS_NOTE, c=MUTED)
    T(ax, xa + wa / 2, 53.0, 'with a bypass diode. Irradiance in W/m$^2$',
      fs=FS_NOTE, c=MUTED)

    T(ax, xa + wa / 2, 47.0, 'One fall, two transition models', fs=FS_TEXT,
      w='bold')
    a1 = inset(fig, xa + 10.0, 9.5, wa - 13.0, 33.0)
    a1.plot(d['tt'], d['G3t'], color='0.5', lw=1.0, ls=(0, (3.0, 1.6)))
    a1.plot(d['t'], d['G3s'], color=INK, lw=1.4)
    a1.set_xlim(-5, 35); a1.set_ylim(62, 262)
    a1.set_xticks([0, 10, 20, 30]); a1.set_yticks([100, 150, 200])
    a1.set_xlabel('Time (s)', fontsize=FS_NOTE, labelpad=1)
    a1.set_ylabel('Module 3 (W/m$^2$)', fontsize=FS_NOTE, labelpad=1)
    a1.grid(True, color=GRID, lw=0.4)
    a1.text(1.2, 84, 'Step: standard test', fontsize=FS_NOTE, color='0.4',
            ha='left', va='center')
    a1.text(34.5, 237, 'Sigmoid, $b$ = 2 s: measured shape,\n'
            'near the mean of 39,976\nrecorded transitions',
            fontsize=FS_NOTE, color=INK, ha='right', va='center')
    i_mid = int(np.argmin(np.abs(d['G3s'] - 165.0)))
    a1.annotate('', xy=(d['t'][i_mid] + 0.3, 166), xytext=(13.5, 213),
                arrowprops=dict(arrowstyle='-', lw=0.5, color=INK))

    # ===================== (b) the landscape =====================
    xb, wb = 57.5, 76.0
    card(ax, xb, 1.5, wb, H - 3.0)
    T(ax, xb + 2.5, H - 5.0, '(b)', fs=FS_TITLE, ha='left', w='bold')
    T(ax, xb + 8.5, H - 5.0, 'Same P$\\endash$V landscape, different paths',
      fs=FS_TITLE, ha='left')
    # marker and curve key
    ky = 103.0
    kx = xb + 9.5
    ax.plot(kx, ky, 'o', ms=3.6, color=COL_SPLINE)
    T(ax, kx + 1.6, ky, 'Spline-MPPT', fs=FS_NOTE, ha='left', c=COL_SPLINE)
    ax.plot(kx + 18.5, ky, 'v', ms=4.0, mfc='white', mec=COL_PO, mew=1.0)
    T(ax, kx + 20.1, ky, 'P&O', fs=FS_NOTE, ha='left', c=COL_PO)
    ax.plot(kx + 28.0, ky, 's', ms=3.4, color=COL_INC)
    T(ax, kx + 29.6, ky, 'INC', fs=FS_NOTE, ha='left', c=INC_DARK)
    ax.plot([kx + 38.5, kx + 42.5], [ky, ky], color=INK, lw=1.2)
    T(ax, kx + 43.4, ky, 'after', fs=FS_NOTE, ha='left', c=MUTED)
    ax.plot([kx + 51.5, kx + 55.5], [ky, ky], color='0.6', lw=0.9,
            ls=(0, (3.0, 1.6)))
    T(ax, kx + 56.4, ky, 'before', fs=FS_NOTE, ha='left', c=MUTED)

    def landscape(a):
        a.plot(d['Vi'], d['Pi'], color='0.6', lw=0.9, ls=(0, (3.0, 1.6)),
               zorder=2)
        a.plot(Vf, Pf, color=INK, lw=1.3, zorder=3)
        a.set_xlim(0, 100); a.set_ylim(0, 292)
        a.set_xticks([0, 25, 50, 75, 100]); a.set_yticks([0, 100, 200])
        a.grid(True, color=GRID, lw=0.4)
        a.set_ylabel('Power (W)', fontsize=FS_NOTE, labelpad=1)
        for v, p in d['peaks']:
            a.plot(v, p, 'o', ms=3.4, mfc=INK if p == pg else 'white',
                   mec=INK, mew=0.8, zorder=6)

    def tracker_marks(a, prof, lift=22.0):
        """P&O and INC markers on short leaders to their dwell points."""
        for alg, mk, dx in (('po', 'v', -3.0), ('inc', 's', -7.5)):
            v = d['dwell'][(prof, alg)]
            p = on_curve(v)
            if prof == 'step':
                xm, ym = vh + dx, p + lift
            elif alg == 'inc':
                xm, ym = v, p + lift
            else:
                xm, ym = v, p - lift - 6
            if ym > p:
                a.plot([v, xm], [p + 3, ym - 6], color=MUTED, lw=0.45,
                       zorder=7)
            else:
                a.plot([v, xm], [p - 3, ym + 6], color=MUTED, lw=0.45,
                       zorder=7)
            if alg == 'po':
                a.plot(xm, ym, 'v', ms=4.4, mfc='white', mec=COL_PO,
                       mew=1.1, zorder=8)
            else:
                a.plot(xm, ym, 's', ms=3.8, color=COL_INC, zorder=8)

    # step panel
    b1 = inset(fig, xb + 10.0, 59.5, wb - 13.5, 35.5)
    landscape(b1)
    tracker_marks(b1, 'step')
    b1.tick_params(labelbottom=False)
    T(ax, xb + 10.0, 97.8, 'After a step change', fs=FS_TEXT, ha='left',
      w='bold')
    b1.text(vg - 2.5, pg + 8, 'GMPP\n%s W\n%s V' % (fmt(pg, 0), fmt(vg, 1)),
            fontsize=FS_NOTE, ha='right', va='bottom', color=INK)
    b1.text(vm - 5.0, pm + 10, '%s%%, %s V' % (pct(pm), fmt(vm, 1)),
            fontsize=FS_NOTE, ha='center', va='bottom', color=MUTED)
    voc = float(Vf[-1])
    samples = [f * voc for f in (0.25, 0.38, 0.53, 0.68, 0.83, 0.93)]
    for v in samples:
        b1.plot(v, on_curve(v), 'o', ms=2.6, mfc=COL_SPLINE, mec='white',
                mew=0.4, zorder=7)
    b1.annotate('', xy=(vg + 2.5, pg + 5),
                xytext=(samples[-1] + 1.0, on_curve(samples[-1]) + 10),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color=COL_SPLINE,
                                connectionstyle='arc3,rad=0.55',
                                mutation_scale=7), zorder=7)
    b1.text(76.0, 268, 'Spline-MPPT samples six\nvoltages, moves to the GMPP',
            fontsize=FS_NOTE, color=COL_SPLINE, ha='center', va='center')
    b1.text(60.0, 22, 'P&O and INC remain at the\n%s%% maximum, %s V'
            % (pct(ph), fmt(vh, 1)), fontsize=FS_NOTE, color=INK,
            ha='center', va='center')

    # sigmoid panel
    b2 = inset(fig, xb + 10.0, 10.0, wb - 13.5, 35.5)
    for k, (V, P) in enumerate(d['family']):
        b2.plot(V, P, color='0.6', lw=0.8, alpha=0.45 + 0.1 * k, zorder=2)
    landscape(b2)
    T(ax, xb + 10.0, 48.2, 'During a sigmoid fade, $b$ = 2 s', fs=FS_TEXT,
      ha='left', w='bold')
    ax.plot([xb + 58.0, xb + 62.0], [48.2, 48.2], color='0.6', lw=0.9)
    T(ax, xb + 62.9, 48.2, 'during', fs=FS_NOTE, ha='left', c=MUTED)
    b2.set_xlabel('Voltage (V)', fontsize=FS_NOTE, labelpad=1)
    for alg, col, lw in (('po', COL_PO, 0.7), ('inc', COL_INC, 0.9)):
        r = d['runs'][('sigmoid_2.0', alg)]
        m = r['t'] >= 0.0
        b2.plot(r['V'][m], r['P'][m], color=col, lw=lw, alpha=0.9,
                zorder=6.5, solid_capstyle='round', solid_joinstyle='round')
    b2.annotate('', xy=(vm + 2.0, pm - 4), xytext=(vm + 5.5, 140),
                arrowprops=dict(arrowstyle='-|>', lw=1.1, color=COL_INC,
                                mutation_scale=7), zorder=8)
    tracker_marks(b2, 'sigmoid_2.0')
    b2.plot(vg, pg, 'o', ms=3.6, mfc=COL_SPLINE, mec=INK, mew=0.6, zorder=7)
    b2.text(77.0, 262, 'INC follows the drifting\npeak to %s V' % fmt(vm, 1),
            fontsize=FS_NOTE, color=INC_DARK, ha='center', va='center')
    b2.text(24.0, 256, 'Spline-MPPT reaches the\nGMPP in every scan cycle\n'
            'but spends much of each\ncycle approaching it',
            fontsize=FS_NOTE, color=COL_SPLINE, ha='center', va='center')
    b2.text(66.0, 14, 'P&O remains at %s V' % fmt(vh, 1),
            fontsize=FS_NOTE, color=COL_PO, ha='center', va='center')

    # ===================== (c) the test matrix =====================
    xc, wc = 135.5, 53.0
    card(ax, xc, 1.5, wc, H - 3.0)
    T(ax, xc + 2.5, H - 5.0, '(c)', fs=FS_TITLE, ha='left', w='bold')
    T(ax, xc + 8.5, H - 5.0, 'The test matrix', fs=FS_TITLE, ha='left')
    cx = xc + wc / 2

    def section(y, n, title, sub):
        ax.add_patch(Circle((xc + 4.6, y), 1.6, fc=INK, ec='none', zorder=6))
        T(ax, xc + 4.6, y - 0.05, n, fs=6.0, c='white', w='bold', z=7)
        T(ax, xc + 7.4, y, title, fs=FS_TEXT, ha='left', w='bold')
        T(ax, xc + wc - 3.0, y, sub, fs=FS_NOTE, ha='right', c=MUTED)

    def times(y):
        T(ax, cx, y, '$\\times$', fs=8.5, c=MUTED)

    # 1. four shading levels, P-V thumbnails of the final distribution
    section(100.0, '4', 'Shading levels', 'P$\\endash$V after the fall')
    tw, th, gap = 10.6, 10.0, 1.6
    x0 = xc + (wc - 4 * tw - 3 * gap) / 2
    for k, (key, lab, V, P) in enumerate(d['thumbs']):
        xx = x0 + k * (tw + gap)
        hard = key == 'hard'
        ax.add_patch(FancyBboxPatch((xx, 83.0), tw, th,
                                    boxstyle='round,pad=0,rounding_size=0.8',
                                    fc='white', ec=INK if hard else '#c9c4ba',
                                    lw=0.8 if hard else 0.5, zorder=2))
        vs = xx + 0.9 + (tw - 1.8) * V / 100.0
        ps = 84.0 + (th - 2.6) * P / 600.0
        ax.plot(vs, ps, color=INK, lw=0.7, zorder=3)
        ig = int(np.argmax(P))
        ax.plot(vs[ig], ps[ig], 'o', ms=2.2, color=INK, zorder=4)
        T(ax, xx + tw / 2, 80.6, lab, fs=FS_NOTE - 0.2, w='bold' if hard
          else 'normal', c=INK if hard else MUTED)
    T(ax, x0 + 2 * (tw + gap) + tw / 2, 95.2, 'in (b)', fs=FS_NOTE - 0.4,
      c=INK)
    times(77.2)

    # 2. eight transition profiles, drawn from the profile generator
    section(73.0, '8', 'Transition profiles', '')
    gw, gh, gx_gap = 10.6, 6.0, 1.6
    names = d['glyphs']
    for k, (key, lab, tt, gg) in enumerate(names):
        row, col = divmod(k, 4)
        xx = x0 + col * (gw + gx_gap)
        yy = 61.5 - row * 12.0
        sig = key.startswith('sigmoid')
        ax.add_patch(FancyBboxPatch((xx, yy), gw, gh + 1.6,
                                    boxstyle='round,pad=0,rounding_size=0.8',
                                    fc='white', ec='#c9c4ba', lw=0.5, zorder=2))
        xs = xx + 0.8 + (gw - 1.6) * (tt - tt[0]) / (tt[-1] - tt[0])
        ys = yy + 1.2 + (gh - 1.0) * gg
        col_ = INK if sig else ('0.45' if key != 'step' else '0.45')
        ls = '-' if sig else ((0, (2.4, 1.2)) if key != 'step' else '-')
        ax.plot(xs, ys, color=col_, lw=0.8, ls=ls, zorder=3)
        T(ax, xx + gw / 2, yy - 1.7, lab, fs=FS_NOTE - 0.4, c=MUTED)
    T(ax, cx, 45.0, 'Top: step and EN 50530-style ramps.',
      fs=FS_NOTE - 0.3, c=MUTED)
    T(ax, cx, 42.6, 'Bottom: sigmoids of measured shape.',
      fs=FS_NOTE - 0.3, c=INK)

    # 3. three trackers
    section(38.4, '3', 'MPPT algorithms', '')
    cwid, cgap = 22.0, 2.0
    xl = xc + (wc - 2 * cwid - cgap) / 2
    xr = xl + cwid + cgap
    ax.add_patch(FancyBboxPatch((xl, 29.5), cwid, 6.2,
                                boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec=COL_SPLINE, lw=0.9, zorder=2))
    T(ax, xl + cwid / 2, 33.6, 'Spline-MPPT', fs=FS_NOTE, c=COL_SPLINE,
      w='bold')
    T(ax, xl + cwid / 2, 31.2, 'global search', fs=FS_NOTE - 0.3, c=MUTED)
    ax.add_patch(FancyBboxPatch((xr, 29.5), cwid, 6.2,
                                boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec='#8a8a8a', lw=0.9, zorder=2))
    T(ax, xr + cwid / 2 - 1.2, 33.6, 'and', fs=FS_NOTE, c=MUTED)
    T(ax, xr + cwid / 2 - 3.6, 33.6, 'P&O', fs=FS_NOTE, c=COL_PO, w='bold',
      ha='right')
    T(ax, xr + cwid / 2 + 1.2, 33.6, 'INC', fs=FS_NOTE, c=INC_DARK,
      w='bold', ha='left')
    T(ax, xr + cwid / 2, 31.2, 'local hill climbing', fs=FS_NOTE - 0.3,
      c=MUTED)

    # 4. outcome of the design
    ax.add_patch(FancyBboxPatch((xc + 3.0, 3.0), wc - 6.0, 23.0,
                                boxstyle='round,pad=0,rounding_size=1.2',
                                fc='white', ec=INK, lw=0.8, zorder=2))
    T(ax, cx, 22.4, '= 96 test cases at 25\u00b0C', fs=FS_TITLE,
      c=INK, w='bold')
    T(ax, cx, 18.4, 'Tracking efficiency, power ripple,', fs=FS_NOTE,
      c=INK)
    T(ax, cx, 15.6, 'steady-state error (RMSE, MAE,', fs=FS_NOTE,
      c=INK)
    T(ax, cx, 12.8, 'nRMSE, RE), settling and GMPP times', fs=FS_NOTE,
      c=INK)
    ax.plot([xc + 7.0, xc + wc - 7.0], [10.2, 10.2], color='#c9c4ba', lw=0.5,
            zorder=3)
    T(ax, cx, 7.8, '+ 192 cases at 45 and 65\u00b0C', fs=FS_NOTE,
      c=MUTED)
    T(ax, cx, 5.2, '+ 36 cases with faster transitions', fs=FS_NOTE,
      c=MUTED)
    return fig


def main():
    d = collect()
    fig = build(d)
    os.makedirs(gf.FIG_DIR, exist_ok=True)
    base = os.path.join(gf.FIG_DIR, 'fig_concept')
    fig.savefig(base + '.pdf', metadata={'Creator': None, 'Producer': None,
                                         'CreationDate': None})
    fig.savefig(base + '.png', dpi=DPI, metadata={'Software': None})
    if '--tiff' in sys.argv[1:]:
        fig.savefig(base + '.tif', dpi=DPI,
                    pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    print('  Saved: %s.pdf / .png' % base)


if __name__ == '__main__':
    main()
