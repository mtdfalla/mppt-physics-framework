"""
mppt_algorithms.py  —  MPPT Algorithm Implementations
=======================================================
Implements three MPPT algorithms operating on a simulated boost converter
connected to a 3S1P KC200GT PV string.

Algorithms
----------
PandO          : Perturb & Observe  (local, hill-climbing)
IncrementalConductance : INC         (local, hill-climbing)
SplineMPPT     : Spline-based GMPPT  (global, periodic scan)

Simulator
---------
MPPTSimulator  : Drives any algorithm through a time-varying P–V landscape
                 and records tracking efficiency.

Parameters (from manuscript Section 2.4)
-----------------------------------------
P&O   : ΔV = 1 V,  T_perturb = 0.05 s  (50 ms)
INC   : ΔV = 1 V,  T_sample  = 0.05 s  (50 ms)
Spline: sample at [20, 40, 60, 80, 95] % Voc,
        T_scan = 5 s scan interval,
        P&O fine-tune step = 0.5 V

Author : Mohamed Abdelmagid (Physics-Based MPPT Testing Research)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from tct_eval import ModuleParams, evaluate_tct


# ─────────────────────────────────────────────────────────────────────────────
# Boost converter model  (duty cycle ↔ voltage mapping)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoostConverter:
    """
    Ideal boost converter: V_out = V_in / (1 - D),  D ∈ (0, 1).
    Used to map duty cycle to PV operating voltage.

    Attributes
    ----------
    V_out    : DC bus voltage [V]  (fixed load)
    D_min    : Minimum duty cycle  (prevents V_in > V_out)
    D_max    : Maximum duty cycle  (prevents inductor saturation)
    """
    V_out: float = 150.0    # Fixed output / bus voltage
    D_min: float = 0.10
    D_max: float = 0.90

    def duty_to_vpv(self, D: float) -> float:
        D = np.clip(D, self.D_min, self.D_max)
        return self.V_out * (1.0 - D)

    def vpv_to_duty(self, Vpv: float) -> float:
        if Vpv <= 0.0:
            return self.D_max
        D = 1.0 - Vpv / self.V_out
        return float(np.clip(D, self.D_min, self.D_max))

    def clamp_vpv(self, Vpv: float, Voc: float) -> float:
        """Clamp Vpv to [0, Voc]."""
        return float(np.clip(Vpv, 0.0, Voc))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: interpolate P and I at a given operating voltage
# ─────────────────────────────────────────────────────────────────────────────

def _pv_operating_point(iv: Dict, V_op: float) -> Tuple[float, float, float]:
    """
    Return (V, I, P) at operating voltage V_op by linear interpolation.
    """
    V_op = float(np.clip(V_op, iv['V'][0], iv['V'][-1]))
    I_op = float(np.interp(V_op, iv['V'], iv['I']))
    P_op = V_op * I_op
    return V_op, I_op, P_op


# ─────────────────────────────────────────────────────────────────────────────
# Perturb & Observe
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PandO:
    """
    Perturb & Observe MPPT algorithm.

    Parameters
    ----------
    delta_V      : Voltage perturbation step [V]  (1 V)
    T_perturb    : Perturbation period [s]          (0.05 s = 50 ms)
    V_init_frac  : Initial voltage as fraction of Voc  (0.80)
    """
    delta_V:     float = 1.0
    T_perturb:   float = 0.05
    V_init_frac: float = 0.80

    # Internal state
    _V_op:        float = field(default=0.0,   init=False, repr=False)
    _P_prev:      float = field(default=-1.0,  init=False, repr=False)
    _V_prev:      float = field(default=0.0,   init=False, repr=False)
    _direction:   int   = field(default=1,     init=False, repr=False)
    _t_last:      float = field(default=-999.0,init=False, repr=False)
    _initialized: bool  = field(default=False, init=False, repr=False)

    def reset(self, Voc: float) -> None:
        self._V_op        = self.V_init_frac * Voc
        self._P_prev      = -1.0
        self._V_prev      = self._V_op
        self._direction   = 1
        self._t_last      = -999.0
        self._initialized = True

    def step(self, t: float, iv: Dict) -> float:
        """Return operating voltage for current time step."""
        if not self._initialized:
            self.reset(iv['Voc'])

        Voc = iv['Voc']
        _, _, P_now = _pv_operating_point(iv, self._V_op)

        if (t - self._t_last) >= self.T_perturb:
            # Perturb decision
            dP = P_now - self._P_prev
            dV = self._V_op - self._V_prev

            if self._P_prev < 0:
                # First step — just record
                pass
            else:
                if dP >= 0:
                    # Keep direction
                    pass
                else:
                    self._direction *= -1  # Reverse

            # Update operating point
            self._V_prev  = self._V_op
            self._P_prev  = P_now
            self._V_op   += self._direction * self.delta_V
            self._V_op    = float(np.clip(self._V_op, 0.0, Voc))
            self._t_last  = t

        return self._V_op


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Conductance
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IncrementalConductance:
    """
    Incremental Conductance MPPT algorithm.

    At MPP: dI/dV = -I/V  ⟺  I/V + dI/dV = 0
    Left of MPP: I/V + dI/dV > 0  → increase V
    Right of MPP: I/V + dI/dV < 0 → decrease V

    Parameters
    ----------
    delta_V   : Voltage step [V]         (1 V)
    T_sample  : Sampling period [s]      (0.05 s)
    epsilon   : Conductance tolerance [-] (0.01)
    """
    delta_V:     float = 1.0
    T_sample:    float = 0.05
    epsilon:     float = 0.01
    V_init_frac: float = 0.80

    # Internal state
    _V_op:        float = field(default=0.0,   init=False, repr=False)
    _I_prev:      float = field(default=0.0,   init=False, repr=False)
    _V_prev:      float = field(default=0.0,   init=False, repr=False)
    _t_last:      float = field(default=-999.0,init=False, repr=False)
    _initialized: bool  = field(default=False, init=False, repr=False)

    def reset(self, Voc: float) -> None:
        self._V_op        = self.V_init_frac * Voc
        self._V_prev      = self._V_op
        self._I_prev      = 0.0
        self._t_last      = -999.0
        self._initialized = True

    def step(self, t: float, iv: Dict) -> float:
        """Return operating voltage for current time step."""
        if not self._initialized:
            self.reset(iv['Voc'])

        Voc = iv['Voc']
        V_now, I_now, _ = _pv_operating_point(iv, self._V_op)

        if (t - self._t_last) >= self.T_sample:
            dV = V_now - self._V_prev
            dI = I_now - self._I_prev

            if abs(dV) < 1e-6:
                # No voltage change — use instantaneous conductance
                if abs(I_now) < 1e-6:
                    pass  # At origin, no action
                elif dI > 0:
                    self._V_op += self.delta_V
                elif dI < 0:
                    self._V_op -= self.delta_V
            else:
                conductance_sum = I_now / V_now + dI / dV if V_now > 1e-6 else dI / dV
                if conductance_sum > self.epsilon:
                    self._V_op += self.delta_V      # Left of MPP
                elif conductance_sum < -self.epsilon:
                    self._V_op -= self.delta_V      # Right of MPP
                # else: at MPP, hold

            self._V_prev = V_now
            self._I_prev = I_now
            self._t_last = t
            self._V_op   = float(np.clip(self._V_op, 0.0, Voc))

        return self._V_op


# ─────────────────────────────────────────────────────────────────────────────
# Spline-MPPT  (global search)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SplineMPPT:
    """
    Spline-based Global MPPT algorithm (Padmanaban et al. style).

    Operation cycle (period = T_scan):
    1. TRACK: P&O fine-tuning around current operating point (between scans)
    2. SCAN:  visit sample voltages spread across P–V range
    3. FIT:   cubic spline through samples → locate global maximum
    4. CONVERGE: move to identified GMPP location
    5. Return to TRACK

    Key implementation details
    --------------------------
    - Initialises in TRACK mode at V_init_frac*Voc (same as P&O)
    - First scan triggers T_scan seconds after start
    - Sample voltages are placed near bypass-diode activation points so the
      global maximum is always bracketed regardless of shading severity
    - Spline fitting uses actual sampled (V, P) data; fallback to best sample

    Parameters
    ----------
    T_scan       : Scan interval [s]               (5 s)
    V_init_frac  : Initial voltage fraction of Voc  (0.80)
    fine_step    : P&O step during track/converge   (0.5 V)
    T_fine       : Period of fine-tuning steps [s]  (0.05 s)
    """
    T_scan:      float = 5.0
    V_init_frac: float = 0.80
    fine_step:   float = 0.5
    T_fine:      float = 0.05

    # Internal state
    _phase:        str   = field(default='track', init=False, repr=False)
    _V_op:         float = field(default=0.0,     init=False, repr=False)
    _V_target:     float = field(default=0.0,     init=False, repr=False)
    _t_scan_start: float = field(default=0.0,     init=False, repr=False)
    _t_fine_last:  float = field(default=-999.0,  init=False, repr=False)
    _scan_idx:     int   = field(default=0,       init=False, repr=False)
    _scan_V:       List  = field(default_factory=list, init=False, repr=False)
    _scan_P:       List  = field(default_factory=list, init=False, repr=False)
    _initialized:  bool  = field(default=False,   init=False, repr=False)
    _P_prev:       float = field(default=-1.0,    init=False, repr=False)
    _direction:    int   = field(default=1,        init=False, repr=False)

    def _sample_voltages(self, Voc: float) -> List[float]:
        """
        Return scan sample voltages based on bypass-diode activation points.
        For a 3S1P KC200GT string (Voc≈98.7V, Vmpp_module≈26.3V):
          Activation near: ~25V (1 mod), ~40V, ~53V (2 mod), ~68V, ~83V (3 mod)
        Uses 6 samples covering the full P–V range.
        Equivalent to the original D=[75,65,55,45,35,25]% with V_out=150V.
        """
        # Absolute sample voltages; clip each to [1, Voc-1]
        V_abs = [0.25*Voc, 0.38*Voc, 0.53*Voc, 0.68*Voc, 0.83*Voc, 0.93*Voc]
        return [float(np.clip(v, 1.0, Voc - 1.0)) for v in V_abs]

    def reset(self, Voc: float) -> None:
        """Start in TRACK mode near Voc (same as P&O) — no startup scan loss."""
        self._V_op         = self.V_init_frac * Voc
        self._V_target     = self._V_op
        self._phase        = 'track'
        self._t_scan_start = 0.0      # first scan at t = T_scan
        self._t_fine_last  = -999.0
        self._scan_idx     = 0
        self._scan_V       = []
        self._scan_P       = []
        self._P_prev       = -1.0
        self._direction    = 1
        self._initialized  = True

    def _fit_spline_max(self, V_samples: np.ndarray,
                        P_samples: np.ndarray) -> float:
        """
        Fit cubic spline and return V at global maximum.
        Falls back to best-sample if scipy unavailable or spline fails.
        """
        try:
            from scipy.interpolate import CubicSpline
            sort_i = np.argsort(V_samples)
            cs     = CubicSpline(V_samples[sort_i], P_samples[sort_i])
            V_fine = np.linspace(V_samples.min(), V_samples.max(), 1000)
            P_fine = cs(V_fine)
            return float(V_fine[np.argmax(P_fine)])
        except Exception:
            return float(V_samples[np.argmax(P_samples)])

    def step(self, t: float, iv: Dict) -> float:
        """Return operating voltage for current time step."""
        if not self._initialized:
            self.reset(iv['Voc'])

        Voc = iv['Voc']

        # ── Trigger scan every T_scan seconds ──
        if self._phase == 'track' and (t - self._t_scan_start) >= self.T_scan:
            self._phase     = 'scan'
            self._scan_idx  = 0
            self._scan_V    = []
            self._scan_P    = []
            self._t_scan_start = t

        # ── SCAN PHASE: visit each sample voltage sequentially ──
        if self._phase == 'scan':
            samples = self._sample_voltages(Voc)
            if self._scan_idx < len(samples):
                V_probe = samples[self._scan_idx]
                _, _, P_probe = _pv_operating_point(iv, V_probe)
                self._scan_V.append(V_probe)
                self._scan_P.append(P_probe)
                self._V_op    = V_probe
                self._scan_idx += 1
            else:
                # All samples collected — fit spline and identify GMPP
                V_arr = np.array(self._scan_V)
                P_arr = np.array(self._scan_P)
                self._V_target    = self._fit_spline_max(V_arr, P_arr)
                self._V_target    = float(np.clip(self._V_target, 1.0, Voc - 1.0))
                self._phase       = 'converge'
                self._t_fine_last = t

        # ── CONVERGE PHASE: move steadily toward identified GMPP ──
        if self._phase == 'converge':
            if (t - self._t_fine_last) >= self.T_fine:
                diff = self._V_target - self._V_op
                if abs(diff) > self.fine_step:
                    self._V_op += np.sign(diff) * self.fine_step
                else:
                    self._V_op        = self._V_target
                    self._phase       = 'track'
                    self._P_prev      = -1.0   # reset P&O memory
                    self._direction   = 1       # reset direction
                    # Reset scan timer from convergence completion
                    # so algorithm gets a full T_scan of tracking time
                    self._t_scan_start = t
                self._t_fine_last = t

        # ── TRACK PHASE: P&O fine-tuning around current point ──
        if self._phase == 'track':
            if (t - self._t_fine_last) >= self.T_fine:
                _, _, P_now = _pv_operating_point(iv, self._V_op)
                if self._P_prev >= 0:
                    dP = P_now - self._P_prev
                    if dP < 0:
                        self._direction *= -1
                self._P_prev      = P_now
                self._V_op       += self._direction * self.fine_step
                self._V_op        = float(np.clip(self._V_op, 1.0, Voc - 1.0))
                self._t_fine_last = t

        return float(np.clip(self._V_op, 1.0, Voc - 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# MPPT Simulator
# ─────────────────────────────────────────────────────────────────────────────

class MPPTSimulator:
    """
    Drive an MPPT algorithm through a time-varying P–V landscape and
    record energy-based tracking efficiency.

    Parameters
    ----------
    module    : ModuleParams
    algorithm : one of PandO | IncrementalConductance | SplineMPPT
    dt        : simulation time step [s]
    T_celsius : operating temperature [°C]
    """

    def __init__(self, module: ModuleParams, algorithm, dt: float = 0.1,
                 T_celsius: float = 25.0, num_iv_points: int = 200):
        self.module       = module
        self.algorithm    = algorithm
        self.dt           = dt
        self.T_celsius    = T_celsius
        self.num_iv_points = num_iv_points

    def _build_iv(self, G_modules: np.ndarray) -> Dict:
        """Evaluate P–V curve for given 3-module irradiance vector."""
        G_map = G_modules.reshape(3, 1)
        T_map = np.full((3, 1), self.T_celsius)
        return evaluate_tct(G_map, T_map, self.module,
                            num_points=self.num_iv_points)

    def run(self, t_vec: np.ndarray,
            G1_vec: np.ndarray, G2_vec: np.ndarray, G3_vec: np.ndarray,
            verbose: bool = False) -> Dict:
        """
        Simulate MPPT over time series of irradiance vectors.

        Parameters
        ----------
        t_vec              : time vector [s]
        G1_vec, G2_vec, G3_vec : per-module irradiance arrays [W/m²]

        Returns
        -------
        dict with keys:
            't'         : time vector
            'V_op'      : operating voltage
            'P_op'      : operating (tracked) power
            'P_gmpp'    : available GMPP power
            'eta_total' : overall tracking efficiency [%]
            'eta_trans' : transition-window efficiency [%]
            'E_tracked' : total energy tracked [J]
            'E_available': total available energy [J]
            'steady_state_after': dict with Vmpp, Pmpp, P_tracked, std_P
        """
        n = len(t_vec)
        V_op_vec   = np.zeros(n)
        P_op_vec   = np.zeros(n)
        P_gmpp_vec = np.zeros(n)

        # Reset algorithm
        iv0 = self._build_iv(np.array([G1_vec[0], G2_vec[0], G3_vec[0]]))
        self.algorithm.reset(iv0['Voc'])

        prev_G = np.array([G1_vec[0], G2_vec[0], G3_vec[0]])
        prev_iv = iv0

        for k, t in enumerate(t_vec):
            G_now = np.array([G1_vec[k], G2_vec[k], G3_vec[k]])

            # Rebuild I–V only if irradiance changed (saves compute time)
            if np.any(np.abs(G_now - prev_G) > 0.01):
                iv = self._build_iv(G_now)
                prev_G = G_now.copy()
                prev_iv = iv
            else:
                iv = prev_iv

            V_op = self.algorithm.step(t, iv)
            _, _, P_op = _pv_operating_point(iv, V_op)

            V_op_vec[k]   = V_op
            P_op_vec[k]   = P_op
            P_gmpp_vec[k] = iv['Pmpp']

            if verbose and k % 100 == 0:
                print(f"  t={t:.1f}s  V={V_op:.1f}V  P={P_op:.1f}W  "
                      f"GMPP={iv['Pmpp']:.1f}W")

        # ── Energy-based tracking efficiency ──
        trapz = getattr(np, 'trapezoid', None) or np.trapz
        E_tracked   = float(trapz(P_op_vec,   t_vec))
        E_available = float(trapz(P_gmpp_vec, t_vec))
        eta_total   = 100.0 * E_tracked / E_available if E_available > 0 else 0.0

        # ── Transition-window efficiency ──
        # Transition starts when G changes, ends when G stabilizes
        # Detect transition window
        G_total = G1_vec + G2_vec + G3_vec
        dG = np.abs(np.diff(G_total))
        trans_mask = dG > 0.5
        if np.any(trans_mask):
            t_trans_start_idx = np.argmax(trans_mask)
            t_trans_end_idx   = len(trans_mask) - np.argmax(trans_mask[::-1]) - 1
            # Add buffer: 2s after stabilization
            buf = int(2.0 / self.dt)
            t_trans_end_idx = min(t_trans_end_idx + buf, n - 1)
            trans_slice = slice(t_trans_start_idx, t_trans_end_idx + 1)
            E_tr = float(trapz(P_op_vec[trans_slice],   t_vec[trans_slice]))
            E_av = float(trapz(P_gmpp_vec[trans_slice], t_vec[trans_slice]))
            eta_trans = 100.0 * E_tr / E_av if E_av > 0 else 0.0
        else:
            eta_trans = eta_total

        # ── Steady-state after transition ──
        # Last 20% of simulation
        ss_start = int(0.80 * n)
        P_ss = P_op_vec[ss_start:]
        P_gmpp_ss = P_gmpp_vec[ss_start:]
        std_P_ss   = float(np.std(P_ss))
        mean_P_ss  = float(np.mean(P_ss))
        mean_gmpp_ss = float(np.mean(P_gmpp_ss))

        # Settling time: first index where |P_op - P_gmpp| < 2% of P_gmpp
        settling_time = None
        for k in range(n):
            if P_gmpp_vec[k] > 1.0:
                if abs(P_op_vec[k] - P_gmpp_vec[k]) / P_gmpp_vec[k] < 0.02:
                    settling_time = t_vec[k]
                    break

        return {
            't':            t_vec,
            'V_op':         V_op_vec,
            'P_op':         P_op_vec,
            'P_gmpp':       P_gmpp_vec,
            'eta_total':    round(eta_total, 2),
            'eta_trans':    round(eta_trans, 2),
            'E_tracked':    round(E_tracked, 2),
            'E_available':  round(E_available, 2),
            'std_P_ss':     round(std_P_ss, 3),
            'mean_P_ss':    round(mean_P_ss, 2),
            'mean_gmpp_ss': round(mean_gmpp_ss, 2),
            'settling_time': settling_time,
        }
