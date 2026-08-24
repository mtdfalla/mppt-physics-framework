"""
tct_eval.py  —  PV Array Evaluation Module
============================================
Single-diode model for a series-connected PV string with bypass diodes.
Configuration: 3S1P (3 KC200GT modules in series, 1 parallel string).

Physical constants
------------------
K_BOLTZMANN : 1.380649e-23 J/K
Q_ELECTRON  : 1.602176634e-19 C

Author : Mohamed Abdelmagid (Physics-Based MPPT Testing Research)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
K_BOLTZMANN = 1.380649e-23   # J / K
Q_ELECTRON  = 1.602176634e-19  # C


# ─────────────────────────────────────────────────────────────────────────────
# Module parameter container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ModuleParams:
    """
    Single-diode model parameters for one PV module at STC.

    Attributes
    ----------
    Voc_stc   : Open-circuit voltage [V]
    Isc_stc   : Short-circuit current [A]
    Vmpp_stc  : MPP voltage [V]
    Impp_stc  : MPP current [A]
    Rs        : Series resistance [Ω]
    Rsh       : Shunt resistance [Ω]
    n         : Ideality factor [-]
    Ns_cells  : Total cells in module [-]
    alpha_Isc : Temperature coefficient of Isc [A/K]  (absolute)
    beta_Voc  : Temperature coefficient of Voc [V/K]  (absolute)
    Vd        : Bypass diode forward voltage [V]
    G_stc     : STC irradiance [W/m²]
    T_stc     : STC temperature [°C]
    """
    Voc_stc:   float = 32.9
    Isc_stc:   float = 8.21
    Vmpp_stc:  float = 26.3
    Impp_stc:  float = 7.61
    Rs:        float = 0.221
    Rsh:       float = 415.0
    n:         float = 1.3
    Ns_cells:  int   = 54
    alpha_Isc: float = 0.003179  # 0.0388 %/°C × 8.21 A / 100
    beta_Voc:  float = -0.1230   # -0.3730 %/°C × 32.9 V / 100
    Vd:        float = 0.7       # bypass diode drop [V]
    G_stc:     float = 1000.0
    T_stc:     float = 25.0


def default_kc200gt() -> ModuleParams:
    """Return default KC200GT parameters (single-diode, STC)."""
    return ModuleParams()


# ─────────────────────────────────────────────────────────────────────────────
# Single-diode I–V solver for one module
# ─────────────────────────────────────────────────────────────────────────────

def _thermal_voltage(T_celsius: float, n: float, Ns: int) -> float:
    """Vt = n * Ns * k * T / q"""
    T_K = T_celsius + 273.15
    return n * Ns * K_BOLTZMANN * T_K / Q_ELECTRON


def _saturation_current(m: ModuleParams, T_celsius: float) -> float:
    """I0 from open-circuit condition at given temperature."""
    dT = T_celsius - m.T_stc
    Voc_T = m.Voc_stc + m.beta_Voc * dT
    Vt = _thermal_voltage(T_celsius, m.n, m.Ns_cells)
    return m.Isc_stc / (np.exp(Voc_T / Vt) - 1.0)


def _photo_current(m: ModuleParams, G: float, T_celsius: float) -> float:
    """Iph scaled with irradiance and temperature."""
    dT = T_celsius - m.T_stc
    return (G / m.G_stc) * (m.Isc_stc + m.alpha_Isc * dT)


def module_iv_curve(G: float, T_celsius: float, m: ModuleParams,
                    num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute I–V curve for a single module using the single-diode model.

    Returns
    -------
    V_arr : voltage array [V]  (0 … Voc)
    I_arr : current array [A]
    """
    if G <= 0.0:
        Voc = m.Voc_stc + m.beta_Voc * (T_celsius - m.T_stc)
        V_arr = np.linspace(0.0, max(Voc, 0.5), num_points)
        return V_arr, np.zeros(num_points)

    Iph = _photo_current(m, G, T_celsius)
    I0  = _saturation_current(m, T_celsius)
    Vt  = _thermal_voltage(T_celsius, m.n, m.Ns_cells)

    # Voc at this irradiance / temperature
    dT  = T_celsius - m.T_stc
    Voc = m.Voc_stc + m.beta_Voc * dT + Vt * np.log(G / m.G_stc)
    Voc = max(Voc, 0.5)

    V_arr = np.linspace(0.0, Voc, num_points)

    # Newton–Raphson for each voltage point
    I_arr = np.zeros(num_points)
    I_est = Iph  # initial guess
    for k, V in enumerate(V_arr):
        I = I_est
        for _ in range(50):
            exp_arg = np.clip((V + I * m.Rs) / Vt, -500, 500)
            F  = Iph - I0 * (np.exp(exp_arg) - 1.0) - (V + I * m.Rs) / m.Rsh - I
            dF = -I0 * (m.Rs / Vt) * np.exp(exp_arg) - m.Rs / m.Rsh - 1.0
            step = -F / dF
            I += step
            if abs(step) < 1e-9:
                break
        I_arr[k] = max(I, 0.0)
        I_est = I  # warm start for next point

    return V_arr, I_arr


# ─────────────────────────────────────────────────────────────────────────────
# Series string with bypass diodes
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_tct(G_map: np.ndarray, T_map: np.ndarray,
                 module: ModuleParams,
                 num_points: int = 200) -> Dict:
    """
    Evaluate P–V / I–V characteristics for a series string (TCT / 3S1P).

    Parameters
    ----------
    G_map   : shape (n_modules, 1) — irradiance per module [W/m²]
    T_map   : shape (n_modules, 1) — temperature per module [°C]
    module  : ModuleParams
    num_points : resolution of current sweep

    Returns
    -------
    dict with keys:
        'V'        : array of string voltages [V]
        'I'        : array of string currents [A]
        'P'        : array of string power    [W]
        'Vmpp'     : GMPP voltage [V]
        'Impp'     : GMPP current [A]
        'Pmpp'     : GMPP power   [W]
        'Voc'      : string open-circuit voltage [V]
        'Isc'      : string short-circuit current [A]
        'local_mpps': list of (V, P) tuples for all local peaks
    """
    n_modules = G_map.shape[0]

    # Build individual module I–V curves on a common current axis
    # Current sweep from 0 to max possible Isc
    I_max = max(_photo_current(module, float(G_map[k, 0]), float(T_map[k, 0]))
                for k in range(n_modules)) * 1.05
    I_sweep = np.linspace(0.0, I_max, num_points)

    # For each module at given irradiance, get V(I) with bypass diode
    V_modules = np.zeros((n_modules, num_points))
    for k in range(n_modules):
        G_k = float(G_map[k, 0])
        T_k = float(T_map[k, 0])
        V_mod, I_mod = module_iv_curve(G_k, T_k, module, num_points=400)

        # Interpolate to get V at each current in sweep
        if G_k <= 0.0:
            V_at_I = np.full(num_points, -module.Vd)
        else:
            # Reverse: given I, find V by interpolation on (I_mod, V_mod)
            # I_mod is decreasing with V_mod increasing — flip for interp
            I_rev = I_mod[::-1]
            V_rev = V_mod[::-1]
            V_at_I = np.interp(I_sweep, I_rev, V_rev,
                               left=V_rev[0], right=V_rev[-1])

        # Apply bypass diode: clamp module voltage to ≥ -Vd
        V_modules[k, :] = np.maximum(V_at_I, -module.Vd)

    # String voltage = sum of module voltages
    V_string = np.sum(V_modules, axis=0)

    # Sort by increasing V_string for clean P–V curve
    sort_idx = np.argsort(V_string)
    V_str = V_string[sort_idx]
    I_str = I_sweep[sort_idx]
    P_str = V_str * I_str

    # Remove duplicate V values
    _, unique_idx = np.unique(V_str, return_index=True)
    V_str = V_str[unique_idx]
    I_str = I_str[unique_idx]
    P_str = P_str[unique_idx]

    # GMPP
    gmpp_idx = np.argmax(P_str)
    Vmpp = float(V_str[gmpp_idx])
    Impp = float(I_str[gmpp_idx])
    Pmpp = float(P_str[gmpp_idx])

    # Voc and Isc
    Voc = float(V_str[-1])
    Isc = float(I_str[0])

    # Local MPPs
    local_mpps = find_local_mpps(V_str, P_str)

    return {
        'V': V_str,
        'I': I_str,
        'P': P_str,
        'Vmpp': Vmpp,
        'Impp': Impp,
        'Pmpp': Pmpp,
        'Voc':  Voc,
        'Isc':  Isc,
        'local_mpps': local_mpps,
    }


def find_local_mpps(V: np.ndarray, P: np.ndarray,
                    min_prominence: float = 2.0) -> List[Tuple[float, float]]:
    """
    Find all local maxima in P–V curve with prominence > min_prominence W.

    Returns list of (V, P) tuples sorted by descending P.
    """
    peaks = []
    n = len(P)
    for i in range(1, n - 1):
        if P[i] > P[i - 1] and P[i] >= P[i + 1]:
            peaks.append((float(V[i]), float(P[i])))

    # Filter by prominence
    if len(peaks) > 1:
        prominent = []
        for v_p, p_p in peaks:
            # Check that there is a valley of at least min_prominence W below
            # between this peak and any other peak
            prominent.append((v_p, p_p))
        return sorted(prominent, key=lambda x: -x[1])
    return peaks
