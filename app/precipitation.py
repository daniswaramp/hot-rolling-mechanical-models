"""
Precipitation & Solubility tab: microalloy precipitate thermodynamics for the current schedule.

Published austenite solubility products (log10 of the wt%-product), a 1:1-molar precipitation
balance, a sequential nitride-then-carbide engineering model for the undissolved Nb/Ti/V/Al
fraction, closed-form JMAK/Avrami precipitation KINETICS (how much has actually precipitated
after a hold time, and along a full rolling path), a C-N-Me ternary stability map and Zener
pinning / pancaked-austenite screening indicators.

Caveat carried through to the UI: these are paraequilibrium screening estimates from published
wt%-product fits (literature spread of roughly +-200 K on solution temperatures), not
Thermo-Calc calls. The kinetics are an engineering calibration on the same equilibrium model,
evaluated in microseconds in-process: NO Modal dispatch and NO cloud calls, per the app rule
that only button-gated heavy calculations leave the machine.
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------- constants
ATOMIC_MASS = {"Nb": 92.906, "Ti": 47.867, "V": 50.942, "Al": 26.982,
               "C": 12.011, "N": 14.007}

# Austenite solubility products, log10([Me][X]) = A - B/T with T in Kelvin and both
# concentrations in wt%. Sources as cited per entry.
SOLUBILITY = {
    "NbC": {"A": 3.42, "B": 7900.0, "metal": "Chem_Nb", "inter": "Chem_C",
            "label": "NbC", "source": "Irvine et al., 1967"},
    "NbN": {"A": 3.70, "B": 10800.0, "metal": "Chem_Nb", "inter": "Chem_N",
            "label": "NbN", "source": "Irvine et al., 1967"},
    "TiC": {"A": 4.40, "B": 10475.0, "metal": "Chem_Ti", "inter": "Chem_C",
            "label": "TiC", "source": "austenite fit, cf. Wada & Pehlke family"},
    "TiN": {"A": 0.32, "B": 7780.0, "metal": "Chem_Ti", "inter": "Chem_N",
            "label": "TiN", "source": "Wada & Pehlke, 1977"},
    "VC": {"A": 6.72, "B": 9500.0, "metal": "Chem_V", "inter": "Chem_C",
           "label": "VC", "source": "Irvine et al., 1967"},
    "VN": {"A": 3.63, "B": 8250.0, "metal": "Chem_V", "inter": "Chem_N",
           "label": "VN", "source": "Irvine et al., 1967"},
    "AlN": {"A": 1.03, "B": 6770.0, "metal": "Chem_Al", "inter": "Chem_N",
            "label": "AlN", "source": "austenite fit, cf. Davern & Edmonds family"},
}

# Per metal, its precipitate systems in precipitation order: nitrides (esp. TiN) are far
# stabler and precipitate first, so the sequential nitride-then-carbide order is the standard
# engineering approximation for the coupled equilibrium.
METAL_SYSTEMS = {
    "Nb": ("NbN", "NbC"),
    "Ti": ("TiN", "TiC"),
    "V": ("VN", "VC"),
    "Al": ("AlN",),  # aluminium carbide is not stable in austenite; AlN only
}

# Closed-form JMAK kinetics calibration (engineering constants, NOT a fitted TTT dataset).
#
# Why no Arrhenius term and no ODE solver: the characteristic time is expressed as a pure
# power law in the supersaturation driving force D = max(0, log10(P_actual / K(T))). The
# temperature dependence of K(T) already carries the dominant Arrhenius behaviour through the
# solubility product, so a separate exp(Q/R * (1/T - 1/T_ref)) factor was deliberately
# SIMPLIFIED away -- it would add two more unfitted constants without changing the screening
# answer. The whole calculation is a handful of closed-form numpy expressions (microseconds,
# megabyte-free), so it runs on every rerun in-process: NO Modal call and NO cloud compute.
#
# Coupling to equilibrium: kinetics only modulate HOW FAST the equilibrium-precipitated
# fraction (sequential nitride-then-carbide model) forms. At infinite hold time the kinetic
# fraction converges to the equilibrium fraction, never beyond it.
KINETICS_CONSTANTS = {
    # Reference driving force D_REF = log10(S) at which t_50 equals T50_REF_S:
    # moderate supersaturation (S = 10^0.5 ~ 3.2) precipitates half the equilibrium
    # deficit in one hour.
    "D_REF": 0.5,
    "T50_REF_S": 3600.0,   # 1.0 h characteristic time at D = D_REF
    # Power-law steepness: t_50 = T50_REF_S * ((D_REF + EPS) / (D + EPS))**M_K.
    # M_K = 3 sits in the classic "interface-reaction-controlled" range (2-4) of
    # transformation-time literature and gives t_50 ~ 500 s at D = 1.0 and
    # t_50 ~ 10 h at D = 0.2 -- the calibrated screening behaviour below.
    "M_K": 3.0,
    # Floor keeps the power law finite as D -> 0 (t_50 saturates near the 30-day cap
    # instead of diverging to infinity).
    "EPS_FLOOR": 0.05,
    # JMAK/Avrami exponent: 1.5 approximates site-saturated nucleation on austenite
    # grain boundaries plus one-dimensional growth (n = 1.5 in the standard tables).
    "N_K": 1.5,
    # Cap: past 30 days the fraction is frozen on any rolling / coiling timescale.
    "T50_CAP_S": 2.6e6,
    # Dissolution floor: the dissolved product used in the reverse (dissolution) driving
    # force is floored at this fraction of P_actual so a fully precipitated state (f = 1,
    # dissolved product ~ 0) still yields a finite, fast-dissolving driving force.
    "DISS_FLOOR_REL": 1e-3,
    # When the matrix is thermodynamically undersaturated (equilibrium below the current
    # fraction) but the simple dissolved-product proxy cannot resolve a positive driving
    # force, assume the reference moderate driving force D_REF instead of stalling.
    "DISS_FALLBACK_D": 0.5,
}

RHO_MATRIX = 7.87   # g/cm3, austenite
RHO_PPT = {"NbC": 7.83, "NbN": 8.40, "TiN": 5.40, "TiC": 4.90,
           "VC": 5.70, "VN": 6.10, "AlN": 3.26}  # g/cm3
GAMMA_GB = 0.5      # J/m2, austenite grain-boundary energy

# wt% axis maxima used to scale the ternary simplex
TERNARY_MAX = {"C": 0.35, "N": 0.02, "Nb": 0.09, "Ti": 0.06, "V": 0.12}

INK, HAIR, GRID = "#0b0d10", "#e5e7eb", "#f1f2f4"
ACCENT = "#1746c8"   # GRP brand blue, same token as the app stylesheet


def _shade(hex_color, factor):
    """Lighter (factor > 0, toward white) or darker (factor < 0, toward black) step of a hex."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + factor)) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


ACCENT_DARK = _shade(ACCENT, -0.45)
# Semantic state tokens, matching the pill colours used across the app.
SEMANTIC_OK = "#065f46"
SEMANTIC_WARN = "#92400e"
SEMANTIC_BAD = "#991b1b"
WARN_LINE = "#b45309"
# Categorical series palette anchored on the accent plus the semantic counter-hues
# (the former violet/pink picks are gone); order matters for the solubility chart.
SERIES = [ACCENT, SEMANTIC_BAD, SEMANTIC_OK, WARN_LINE, ACCENT_DARK, SEMANTIC_WARN,
          _shade(ACCENT, 0.35)]
METAL_COLOR = {"Nb": SERIES[0], "Ti": SERIES[1], "V": SERIES[2], "Al": SERIES[3]}


# ----------------------------------------------------------------------------- pure model
def _stoich_ratio(sysd):
    """Interstitial-to-metal mass ratio r = X_atmass / M_atmass for a precipitate system."""
    return ATOMIC_MASS[sysd["inter"][len("Chem_"):]] / ATOMIC_MASS[sysd["metal"][len("Chem_"):]]


def solution_temperature(M0, X0, A, B):
    """Temperature (K) at which the system just fully dissolves; None when unsolvable.

    T_sol = B / (A - log10(M0*X0)). Empty chemistry (P <= 0) or a product so large that
    log10(P) >= A (no finite positive solution) returns None.
    """
    try:
        P = float(M0) * float(X0)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(P) or P <= 0.0:
        return None
    denom = float(A) - np.log10(P)
    if not np.isfinite(denom) or denom <= 0.0:
        return None
    return float(B) / denom


def _precipitated(M0, X0, K, r):
    """Precipitated metal s (wt%) of the 1:1-molar balance; broadcasts over arrays.

    With totals (M0, X0), stoichiometric ratio r and solubility product K, the dissolved
    product must satisfy (M0 - s)(X0 - r*s) = K, i.e.
        r*s^2 - (r*M0 + X0)*s + (M0*X0 - K) = 0.
    The physical root lies in [0, min(M0, X0/r)] (the dissolved product decreases
    monotonically from M0*X0 to 0 over that interval, so the root is unique). It is taken in
    the numerically stable rationalised form s = 2c / (a + sqrt(a^2 - 4rc)) to avoid the
    catastrophic cancellation of (a - sqrt(D)) near T_sol where K -> M0*X0.
    """
    M0 = np.asarray(M0, dtype=float)
    X0 = np.asarray(X0, dtype=float)
    K = np.asarray(K, dtype=float)
    r = float(r)
    s_max = np.minimum(M0, X0 / r)
    a = r * M0 + X0
    c = M0 * X0 - K  # <= 0 means the steel can hold everything dissolved
    # (rM - X)^2 >= 0 guarantees a non-negative discriminant whenever c > 0
    disc = a * a - 4.0 * r * np.maximum(c, 0.0)
    denom = a + np.sqrt(np.maximum(disc, 0.0))
    s = np.where(c > 0.0, 2.0 * np.maximum(c, 0.0) / np.maximum(denom, 1e-300), 0.0)
    s = np.clip(s, 0.0, s_max)
    if np.ndim(s) == 0:
        return float(s)
    return s


def system_undissolved(label, M0, X0, T_C):
    """Undissolved metal fraction [0, 1] for one precipitate system's 1:1 balance."""
    sysd = SOLUBILITY[label]
    T = np.asarray(T_C, dtype=float) + 273.15
    K = 10.0 ** (sysd["A"] - sysd["B"] / T)
    s = _precipitated(M0, X0, K, _stoich_ratio(sysd))
    M0a = np.asarray(M0, dtype=float)
    frac = np.where(M0a > 0.0, s / np.where(M0a > 0.0, M0a, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    if np.ndim(frac) == 0:
        return float(frac)
    return frac


def sequential_precipitated(metal, T_C, M0, C0, N0):
    """wt% of metal precipitated per system, sequential nitride -> carbide (broadcasts).

    Engineering approximation: the far stabler nitride takes its share of the metal first,
    then the remainder is offered to the carbide. K <= 0 / empty chemistry degrade to zero
    precipitation through _precipitated's guards.
    """
    M0 = np.asarray(M0, dtype=float)
    C0 = np.asarray(C0, dtype=float)
    N0 = np.asarray(N0, dtype=float)
    T = np.asarray(T_C, dtype=float) + 273.15
    out = {}
    remaining = M0
    for label in METAL_SYSTEMS.get(metal, ()):
        sysd = SOLUBILITY[label]
        X0 = N0 if sysd["inter"].endswith("_N") else C0
        K = 10.0 ** (sysd["A"] - sysd["B"] / T)
        s = _precipitated(remaining, X0, K, _stoich_ratio(sysd))
        out[label] = s
        remaining = remaining - s
    return out


def undissolved_fraction(metal, T_C, M0, C0, N0):
    """Undissolved metal fraction [0, 1] under the sequential coupled model."""
    s_map = sequential_precipitated(metal, T_C, M0, C0, N0)
    total = sum(s_map.values()) if s_map else 0.0
    M0a = np.asarray(M0, dtype=float)
    frac = np.where(M0a > 0.0, np.asarray(total, dtype=float) / np.where(M0a > 0.0, M0a, 1.0), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    if np.ndim(frac) == 0:
        return float(frac)
    return frac


def precipitate_mass_fractions(metal, T_C, M0, C0, N0):
    """Steel mass fraction tied up in undissolved precipitate, per system (scalar input)."""
    s_map = sequential_precipitated(metal, T_C, M0, C0, N0)
    out = {}
    for label, s in s_map.items():
        sysd = SOLUBILITY[label]
        out[label] = float(np.asarray(s, dtype=float)) * (1.0 + _stoich_ratio(sysd)) / 100.0
    return out


def volume_fraction(f_mass, rho_ppt):
    """Precipitate volume fraction from its mass fraction and density."""
    return float(f_mass) * RHO_MATRIX / float(rho_ppt)


def zener_pinning_mpa(f_v, r_nm, gamma=GAMMA_GB):
    """Zener pinning pressure P_z = 3 f_v gamma / (2 r) in MPa.

    Clamps f_v into [0, 1] and the radius to at least 1 nm so degenerate slider input can
    never divide by zero or go negative.
    """
    f_v = min(max(float(f_v), 0.0), 1.0)
    r_m = max(float(r_nm), 1.0) * 1e-9
    return 3.0 * f_v * float(gamma) / (2.0 * r_m) / 1.0e6


# ----------------------------------------------------------------------------- kinetics
def _primary_system(metal_or_system):
    """Resolve a metal name ("Nb") or a system label ("NbN") to one SOLUBILITY label.

    A metal resolves to its FIRST system in the sequential order -- the far stabler nitride,
    which nucleates first and therefore sets the kinetic clock for the whole metal.
    """
    if metal_or_system in SOLUBILITY:
        return metal_or_system
    systems = METAL_SYSTEMS.get(metal_or_system)
    return systems[0] if systems else None


def _system_chemistry(label, M0, C0, N0):
    """(M0, X0) wt% pair for a system label: nitrides consume N, carbides consume C."""
    sysd = SOLUBILITY[label]
    return float(sysd["A"]), float(sysd["B"]), \
        (float(N0) if sysd["inter"].endswith("_N") else float(C0))


def _t50_from_drive(D):
    """Characteristic time t_50 (s) from the driving force D = log10(S), or None.

    Engineering power law t_50 = T50_REF_S * ((D_REF + eps) / (D + eps))^M_K, capped at
    T50_CAP_S (30 days). Normalised so t_50 = T50_REF_S exactly at D = D_REF. Returns None
    when D <= 0: no supersaturation means no kinetic clock at all.
    """
    if D is None or not np.isfinite(D) or D <= 0.0:
        return None
    cfg = KINETICS_CONSTANTS
    t50 = cfg["T50_REF_S"] * ((cfg["D_REF"] + cfg["EPS_FLOOR"])
                              / (D + cfg["EPS_FLOOR"])) ** cfg["M_K"]
    return float(min(t50, cfg["T50_CAP_S"]))


def kinetics_t50(metal_or_system, T_C, M0, C0, N0):
    """Characteristic JMAK time t_50 (s) for one precipitate system at temperature T.

    `metal_or_system` is either a metal name ("Nb", "Ti", "V", "Al") -- resolved nitride-first
    through the sequential model order -- or a direct system label ("NbC", "TiN", ...).
    Driving force D = max(0, log10(S)) with supersaturation ratio S = P_actual / K(T),
    P_actual = M0*X0 in wt%-product and K(T) = 10^(A - B/T) from the published fit.
    Returns None when there is no chemistry, or no driving force (S <= 1: the system is
    (under)saturated and nothing can precipitate on any timescale).
    """
    label = _primary_system(metal_or_system)
    if label is None:
        return None
    try:
        M0v = float(M0)
    except (TypeError, ValueError):
        return None
    _A, _B, X0 = _system_chemistry(label, M0v, C0, N0)
    P = M0v * X0
    if not np.isfinite(P) or not np.isfinite(X0) or P <= 0.0:
        return None
    T = np.asarray(T_C, dtype=float) + 273.15
    K = 10.0 ** (SOLUBILITY[label]["A"] - SOLUBILITY[label]["B"] / T)
    D = float(np.log10(P / K))
    return _t50_from_drive(D)


def jmak_fraction(time_s, t50_s):
    """Closed-form JMAK/Avrami transformed fraction X = 1 - exp(-ln2 * (t/t_50)^n_k).

    X(0) = 0, X(t_50) = 0.5 exactly, X -> 1 as t -> infinity. Monotone in t. No ODE solve:
    two floating-point operations, evaluated per keystroke in-process (no Modal, no cloud).
    """
    if t50_s is None or not np.isfinite(t50_s) or t50_s <= 0.0:
        return 0.0
    t = np.maximum(np.asarray(time_s, dtype=float), 0.0)
    x = 1.0 - np.exp(-np.log(2.0) * (t / float(t50_s)) ** KINETICS_CONSTANTS["N_K"])
    if np.ndim(x) == 0:
        return float(x)
    return x


def precipitated_fraction_at(metal, T_C, time_s, M0, C0, N0):
    """Actually-precipitated metal fraction [0, 1] after holding time_s at T_C.

    Kinetics-eq coupling: the JMAK transformed fraction (how far the transformation has run)
    scales the EQUILIBRIUM-precipitated fraction at T — which is exactly the sequential
    model's "undissolved" (i.e. tied-up-in-precipitate) fraction — so kinetics only modulate
    how fast the equilibrium amount forms:
        fraction(t) = X_jmak(t; t_50(T)) * undissolved_fraction(T).
    Bounded in [0, equilibrium fraction]; monotone non-decreasing in time. `T_C` is a scalar
    temperature; `time_s` may be scalar or a numpy array (broadcast). Returns 0.0 when there
    is no driving force (t_50 is None).
    """
    eq = float(undissolved_fraction(metal, T_C, M0, C0, N0))
    t50 = kinetics_t50(metal, T_C, M0, C0, N0)
    if t50 is None:
        if np.ndim(time_s) == 0:
            return 0.0
        return np.zeros_like(np.asarray(time_s, dtype=float))
    x = np.clip(np.asarray(jmak_fraction(time_s, t50), dtype=float) * eq, 0.0, eq)
    if np.ndim(x) == 0:
        return float(x)
    return x


def _dissolution_t50(metal_or_system, T_C, M0, C0, N0, f_state):
    """Characteristic dissolution time (s) on reheating above the solution temperature.

    Symmetry assumption (documented engineering simplification): dissolution follows the SAME
    JMAK law and the SAME t_50 power law as precipitation, driven by the mirror
    (reverse) supersaturation of the DISSOLVED product: D_rev = max(0, log10(K(T)/P_diss))
    with P_diss = M0*(1 - f_state)*X0 of the primary system, floored at
    DISS_FLOOR_REL * P_actual so a fully precipitated state still dissolves fast. When the
    proxy cannot resolve a positive driving force but equilibrium demands dissolution
    (eq < f_state), the reference driving force D_REF is assumed rather than stalling.
    """
    label = _primary_system(metal_or_system)
    if label is None:
        return None
    try:
        M0v = float(M0)
    except (TypeError, ValueError):
        return None
    _A, _B, X0 = _system_chemistry(label, M0v, C0, N0)
    P_actual = M0v * X0
    if not np.isfinite(P_actual) or P_actual <= 0.0:
        return None
    T = np.asarray(T_C, dtype=float) + 273.15
    K = 10.0 ** (SOLUBILITY[label]["A"] - SOLUBILITY[label]["B"] / T)
    cfg = KINETICS_CONSTANTS
    P_diss = max(M0v * (1.0 - float(f_state)) * X0,
                 cfg["DISS_FLOOR_REL"] * P_actual, 1e-12)
    D_rev = float(np.log10(K / P_diss))
    if D_rev <= 0.0:
        D_rev = cfg["DISS_FALLBACK_D"]
    return _t50_from_drive(D_rev)


def stage_path(metal, stages, M0, C0, N0):
    """Kinetic evolution through a rolling path: stages = [(T_C, time_s), ...] (pure).

    State evolution, one closed-form relaxation per stage (no time-discretised loops):
      - state starts fully dissolved (fraction 0.0);
      - at each stage the EQUILIBRIUM precipitated fraction eq(T) comes from the sequential
        model, and the actual fraction relaxes toward it with the JMAK fraction of that
        stage's time: f_new = f_old + (eq - f_old) * X_jmak(dt; t_50(T));
      - precipitation (eq > f): t_50 from the forward supersaturation (kinetics_t50);
      - dissolution on reheating (eq < f): the same t_50 form driven by the reverse
        supersaturation of the dissolved product (_dissolution_t50, symmetry assumption
        documented there);
      - eq == f: hold, state unchanged.
    Returns a list of per-stage dicts with keys T_C, time_s, fraction (state AFTER the
    stage), equilibrium, t50_s and mode ("precipitate" / "dissolve" / "hold").
    """
    label = _primary_system(metal)
    out = []
    f = 0.0
    for stage in stages:
        try:
            T_C, dt = float(stage[0]), max(float(stage[1]), 0.0)
        except (TypeError, ValueError, IndexError):
            T_C, dt = float("nan"), 0.0
        if label is None:
            out.append({"T_C": T_C, "time_s": dt, "fraction": 0.0, "equilibrium": 0.0,
                        "t50_s": None, "mode": "hold"})
            continue
        eq = float(undissolved_fraction(metal, T_C, M0, C0, N0))
        if eq > f:
            t50 = kinetics_t50(metal, T_C, M0, C0, N0)
            mode = "precipitate" if dt > 0.0 else "hold"
        elif eq < f:
            t50 = _dissolution_t50(metal, T_C, M0, C0, N0, f)
            mode = "dissolve" if dt > 0.0 else "hold"
        else:
            t50, mode = None, "hold"
        x = jmak_fraction(dt, t50) if (dt > 0.0 and t50 is not None) else 0.0
        f = float(np.clip(f + (eq - f) * x, 0.0, 1.0))
        out.append({"T_C": T_C, "time_s": dt, "fraction": f, "equilibrium": eq,
                    "t50_s": t50, "mode": mode})
    return out


def rolling_stages(mill, raw):
    """Fixed rolling-path stages derived from the CURRENT sidebar schedule.

    Returns [(stage_label, T_C, default_time_s), ...] -- the UI lets the operator edit the
    times and replay through a form gate.
    """
    if mill == "Plate":
        acc = _num(raw, "ACC_Temp_Post_Laminar", 744.0)
        return [
            ("Roughing exit", _num(raw, "FM_Temp_Before_Rolling", 985.0), 30.0),
            ("Finishing", _num(raw, "FM_Finish_Temp", 832.0), 10.0),
            ("ACC", acc, 5.0),
            ("Coil/stack hold", max(400.0, acc - 100.0), 600.0),
        ]
    ct = (_num(raw, "Temp_CT_Top_C", 595.0) + _num(raw, "Temp_CT_Bot_C", 558.0)) / 2.0
    return [
        ("Entry", _num(raw, "Temp_Transfer_Bar_C", 937.0), 20.0),
        ("Finishing", _num(raw, "Temp_FT_Avg_C", 835.0), 8.0),
        ("Runout", ct, 10.0),
        ("Coil hold", ct, 900.0),
    ]


# ----------------------------------------------------------------------------- helpers
def _num(raw, key, default):
    """Defensive float read: missing, non-numeric or NaN input falls back, never raises."""
    try:
        v = float(raw.get(key, default))
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def _chem(raw):
    return {el: _num(raw, f"Chem_{el}", 0.0) for el in ("C", "N", "Nb", "Ti", "V", "Al")}


def _route_temps(mill, raw):
    """Entry / FRT / exit temperatures of the current schedule, per route."""
    if mill == "Plate":
        entry = _num(raw, "FM_Temp_Before_Rolling", 985.0)
        frt = _num(raw, "FM_Finish_Temp", 832.0)
        exit_t = _num(raw, "ACC_Temp_Post_Laminar", 744.0)
        exit_lbl = "ACC exit"
    else:
        entry = _num(raw, "Temp_Transfer_Bar_C", 937.0)
        frt = _num(raw, "Temp_FT_Avg_C", 835.0)
        exit_t = (_num(raw, "Temp_CT_Top_C", 595.0) + _num(raw, "Temp_CT_Bot_C", 558.0)) / 2.0
        exit_lbl = "coiling"
    return {"entry": entry, "frt": frt, "exit": exit_t, "exit_label": exit_lbl}


def _section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def _base_fig(title, x_title, y_title, height=420):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=12.5), x=0, xanchor="left"),
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        margin=dict(l=52, r=18, t=44, b=44),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        height=height,
    )
    fig.update_xaxes(title_text=x_title, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    return fig


# ----------------------------------------------------------------------------- charts
def _solubility_fig(chem, frt):
    """Chart 1: log K vs reciprocal temperature, chemistry levels and the FRT marker."""
    T = np.linspace(1100.0, 1550.0, 220)
    x = 1000.0 / T
    fig = _base_fig("Solubility products: log₁₀[M][X] vs 1000/T",
                    "1000/T (1/K)", "log₁₀ ([Me]·[X], wt%²)")
    for i, (label, sysd) in enumerate(SOLUBILITY.items()):
        fig.add_scatter(x=x, y=sysd["A"] - sysd["B"] / T, mode="lines", name=label,
                        line=dict(color=SERIES[i], width=2.0))
    x_frt = 1000.0 / (frt + 273.15)
    fig.add_vline(x=x_frt, line=dict(color=INK, width=1.3, dash="dot"))
    fig.add_annotation(x=x_frt, y=1.0, yref="paper", showarrow=False,
                       text=f"FRT {frt:.0f} °C", font=dict(size=10, color=INK),
                       xanchor="right", yanchor="top")
    for i, (label, sysd) in enumerate(SOLUBILITY.items()):
        m_el, x_el = sysd["metal"][5:], sysd["inter"][5:]
        M0, X0 = chem[m_el], chem[x_el]
        if M0 > 0.0 and X0 > 0.0:
            y0 = float(np.log10(M0 * X0))
            fig.add_scatter(x=[x[0], x[-1]], y=[y0, y0], mode="lines",
                            name=f"{label} level log₁₀({m_el}·{x_el})",
                            line=dict(color=SERIES[i], width=1.2, dash="dash"))
    fig.update_xaxes(range=[min(float(x.min()), x_frt) - 0.02,
                            max(float(x.max()), x_frt) + 0.02])
    fig.update_layout(legend=dict(orientation="h", y=-0.32, x=0, font=dict(size=9)))
    return fig


def _undissolved_fig(chem, temps):
    """Chart 2: sequential-model undissolved fraction per metal over the rolling range."""
    t_lo = max(400.0, min(800.0, temps["entry"], temps["frt"], temps["exit"]) - 15.0)
    Tc = np.linspace(t_lo, 1400.0, 301)
    fig = _base_fig("Undissolved Nb / Ti / V / Al vs temperature",
                    "Temperature (°C)", "Undissolved fraction")
    for metal in ("Nb", "Ti", "V", "Al"):
        frac = undissolved_fraction(metal, Tc, chem[metal], chem["C"], chem["N"])
        fig.add_scatter(x=Tc, y=frac, mode="lines", name=metal,
                        line=dict(color=METAL_COLOR[metal], width=2.0))
        f_frt = undissolved_fraction(metal, temps["frt"], chem[metal], chem["C"], chem["N"])
        fig.add_scatter(x=[temps["frt"]], y=[f_frt], mode="markers", showlegend=False,
                        marker=dict(color=METAL_COLOR[metal], size=8, line=dict(width=1, color="#ffffff")))
    for t_val, lbl, dash in ((temps["entry"], "entry", "dot"),
                             (temps["frt"], "FRT", "dash"),
                             (temps["exit"], temps["exit_label"], "dot")):
        if Tc[0] <= t_val <= Tc[-1]:
            fig.add_vline(x=t_val, line=dict(color="#9ca3af", width=1.1, dash=dash),
                          annotation_text=lbl, annotation_position="top",
                          annotation_font_size=9)
    fig.update_yaxes(range=[0.0, 1.02])
    return fig


def _ternary_fig(metal, T_C, chem):
    """Chart 4: C-N-Me ternary stability map, colour = undissolved Me fraction."""
    m_max = TERNARY_MAX[metal]
    a = np.linspace(0.0, 1.0, 41)
    A, B = np.meshgrid(a, a)
    keep = A + B <= 1.0 + 1e-9
    a_n, b_n = A[keep], B[keep]
    c_n = 1.0 - a_n - b_n
    me_wt = a_n * m_max
    c_wt = b_n * TERNARY_MAX["C"]
    n_wt = c_n * TERNARY_MAX["N"]
    frac = undissolved_fraction(metal, float(T_C), me_wt, c_wt, n_wt)

    fig = go.Figure(go.Scatterternary(
        a=a_n, b=b_n, c=c_n, mode="markers",
        customdata=np.stack([me_wt, c_wt, n_wt, frac], axis=-1),
        marker=dict(color=frac, colorscale="Viridis", cmin=0.0, cmax=1.0, size=5,
                    line=dict(width=0),
                    colorbar=dict(title=dict(text="undissolved", font=dict(size=10)),
                                  tickformat=".0%", tickfont=dict(size=9))),
        hovertemplate=("Me %{customdata[0]:.4f} wt% · C %{customdata[1]:.4f} wt% · "
                       "N %{customdata[2]:.4f} wt%<br>undissolved %{customdata[3]:.1%}"
                       "<extra></extra>"),
    ))
    fig.update_layout(
        title=dict(text=f"{metal}–C–N stability map at {T_C:.0f} °C "
                        "(colour: undissolved metal fraction)",
                   font=dict(size=12.5), x=0, xanchor="left"),
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=20, t=48, b=52),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        height=520,
    )
    def _axis(title, w_max, color):
        vals = [0.0, 0.25, 0.5, 0.75, 1.0]
        return dict(title=title, min=0.0, color=color, showgrid=True, gridcolor=GRID,
                    tickvals=vals, ticktext=[f"{v * w_max:.3f}" for v in vals])
    fig.update_ternaries(
        bgcolor="#ffffff",
        aaxis=_axis(f"{metal} (wt%)", m_max, SERIES[0]),
        baxis=_axis("C (wt%)", TERNARY_MAX["C"], "#6b7280"),
        caxis=_axis("N (wt%)", TERNARY_MAX["N"], "#6b7280"),
    )
    return fig


def _zener_fig(f_v, drive_mpa):
    """Chart 5b: Zener pinning pressure vs mean precipitate radius, threshold line."""
    r = np.linspace(5.0, 100.0, 200)
    pz = 3.0 * f_v * GAMMA_GB / (2.0 * r * 1e-9) / 1.0e6
    fig = _base_fig("Zener pinning pressure vs precipitate radius",
                    "Mean precipitate radius (nm)", "P_z (MPa)", height=360)
    fig.add_scatter(x=r, y=pz, mode="lines", name="P_z",
                    line=dict(color=SERIES[0], width=2.0))
    fig.add_hline(y=float(drive_mpa), line=dict(color=SERIES[1], width=1.3, dash="dash"),
                  annotation_text=f"driving pressure {drive_mpa:.1f} MPa",
                  annotation_position="top right", annotation_font_size=9)
    fig.update_yaxes(type="log")
    return fig


def _path_fig(stages, paths):
    """Rolling-path chart: kinetic precipitated fraction per metal, stage by stage.

    stages = [(label, T_C, time_s), ...]; paths = {metal: stage_path(...) output}.
    Solid line + markers: actual kinetic state. Dashed horizontal line: equilibrium
    fraction at the FINAL stage temperature (where each metal would end up given
    infinite hold time there).
    """
    labels = [lbl for lbl, _T, _t in stages]
    x = list(range(len(labels)))
    fig = _base_fig("Precipitated fraction along the rolling path (JMAK kinetics)",
                    "Stage", "Precipitated metal fraction", height=430)
    for i, metal in enumerate(("Nb", "Ti", "V", "Al")):
        frac = [s["fraction"] for s in paths[metal]]
        fig.add_scatter(x=x, y=frac, mode="lines+markers", name=metal,
                        line=dict(color=METAL_COLOR[metal], width=2.2),
                        marker=dict(size=7, line=dict(width=1, color="#ffffff")))
        eq_final = paths[metal][-1]["equilibrium"] if paths[metal] else 0.0
        fig.add_scatter(x=x, y=[eq_final] * len(x), mode="lines",
                        name="equilibrium at final stage T" if i == 0 else None,
                        showlegend=(i == 0),
                        line=dict(color=METAL_COLOR[metal], width=1.0, dash="dash"),
                        hoverinfo="skip")
    fig.update_xaxes(tickvals=x, ticktext=labels)
    fig.update_yaxes(range=[0.0, 1.02])
    fig.update_layout(legend=dict(orientation="h", y=-0.34, x=0, font=dict(size=9)))
    return fig


# ----------------------------------------------------------------------------- view
def view_precipitation(mill, raw):
    """Precipitation & Solubility tab: module-owned view, called from main()."""
    chem = _chem(raw)
    temps = _route_temps(mill, raw)

    _section("precipitation & solubility")
    st.markdown(
        '<div class="note">Paraequilibrium screening of Nb, Ti, V and Al precipitates at this '
        'schedule\'s temperatures: not Thermo-Calc. Chemistry comes from the sidebar; edit it '
        'there.</div>', unsafe_allow_html=True)

    # --- facts card: per-system solution temperature and status at FRT -------------------
    cells = []
    for label, sysd in SOLUBILITY.items():
        M0 = chem[sysd["metal"][5:]]
        X0 = chem[sysd["inter"][5:]]
        T_sol = solution_temperature(M0, X0, sysd["A"], sysd["B"])
        if T_sol is None:
            cells.append(
                f'<div><span>{label} · solution temp</span><b>—</b>'
                f'<em>no chemistry</em><br><span class="pill neutral">no chemistry</span></div>')
            continue
        und = float(system_undissolved(label, M0, X0, temps["frt"]))
        if und <= 1e-6:
            pill = '<span class="pill ok">fully dissolved at FRT</span>'
            sub = f"dissolved at {temps['frt']:.0f} °C"
        else:
            cls = "warn" if und < 0.5 else "bad"
            pill = f'<span class="pill {cls}">undissolved {und * 100:.1f}% at FRT</span>'
            sub = f"T_sol {T_sol - 273.15:.0f} °C"
        cells.append(
            f'<div><span>{label} · solution temp</span><b>{T_sol - 273.15:.0f} °C</b>'
            f'<em>{sub}</em><br>{pill}</div>')
    st.markdown('<div class="facts">' + "".join(cells) + '</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)

    # --- charts 1 & 2 --------------------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(_solubility_fig(chem, temps["frt"]), use_container_width=True)
        st.markdown(
            '<div class="note">Each solid curve is log₁₀[M][X] = A − B/T for one precipitate '
            'system; the dashed horizontal line is this steel\'s own log₁₀(Me·X) level. Where the '
            'level crosses its curve the precipitate is just fully dissolved: that intersection '
            'is the solution temperature. Right of the dotted FRT marker (colder) the level sits '
            'above the curve, so the excess precipitates out.</div>', unsafe_allow_html=True)
    with col_b:
        st.plotly_chart(_undissolved_fig(chem, temps), use_container_width=True)
        st.markdown(
            '<div class="note">Sequential coupled model, one metal at a time: the far stabler '
            'nitride (especially TiN) precipitates first, then the remaining metal is offered to '
            'the carbide. That nitride-then-carbide order is the standard engineering '
            'approximation of the coupled equilibrium. Markers sit on each curve at the '
            'schedule\'s finish temperature.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # --- ternary stability map ------------------------------------------------------------
    _section("c–n–me stability map")
    ctl_a, ctl_b = st.columns([1, 2])
    metal = ctl_a.selectbox("Metal", ["Nb", "Ti", "V"], key=f"ppt_metal_{mill}",
                            help="Axis maxima scale the simplex: Nb 0.09, Ti 0.06, V 0.12 wt%; "
                                 "C 0.35 wt%; N 0.02 wt%.")
    t_def = min(max(temps["frt"], 800.0), 1300.0)
    t_tern = ctl_b.slider("Map temperature (°C)", 800.0, 1300.0, float(t_def), 5.0,
                          key=f"ppt_ttemp_{mill}")
    st.plotly_chart(_ternary_fig(metal, t_tern, chem), use_container_width=True)
    st.markdown(
        '<div class="note">Every grid point is a chemistry normalised onto the ternary simplex '
        '(axes labelled in true wt%); the colour is the sequential-model undissolved metal '
        'fraction at the chosen temperature. Deep purple corners are compositions that would '
        'leave microalloy in solution; yellow interiors pin it out as carbonitride.</div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # --- pancaking indicator + Zener pinning ----------------------------------------------
    _section("pancaking & zener pinning")
    f_entry = float(undissolved_fraction(metal, temps["entry"], chem[metal], chem["C"], chem["N"]))
    f_frt = float(undissolved_fraction(metal, temps["frt"], chem[metal], chem["C"], chem["N"]))
    f_exit = float(undissolved_fraction(metal, temps["exit"], chem[metal], chem["C"], chem["N"]))
    st.markdown(
        '<div class="facts">'
        f'<div><span>{metal} undissolved at entry ({temps["entry"]:.0f} °C)</span>'
        f'<b>{f_entry * 100:.1f}</b><em>%</em></div>'
        f'<div><span>{metal} undissolved at FRT ({temps["frt"]:.0f} °C)</span>'
        f'<b>{f_frt * 100:.1f}</b><em>%</em></div>'
        f'<div><span>{metal} undissolved at {temps["exit_label"]} ({temps["exit"]:.0f} °C)</span>'
        f'<b>{f_exit * 100:.1f}</b><em>%</em></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="note">Pancaked-austenite indicator: undissolved {metal} surviving into the '
        f'finishing passes pins the recrystallisation front. A large entry-to-FRT fraction means '
        f'the strain-induced precipitates can hold the elongated, unrecrystallised grain shape '
        f'through to the transformation.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    zc1, zc2 = st.columns(2)
    r_nm = zc1.slider("Mean precipitate radius (nm)", 5.0, 100.0, 20.0, 1.0,
                      key=f"ppt_rnm_{mill}")
    drive_mpa = zc2.slider("Recrystallization driving pressure (MPa)", 0.1, 5.0, 1.0, 0.1,
                           key=f"ppt_drv_{mill}",
                           help="At high reduction the stored-energy driving pressure is of "
                                "order 0.1–10 MPa; this slider sets the screening threshold.")

    f_mass_map = precipitate_mass_fractions(metal, temps["frt"], chem[metal],
                                            chem["C"], chem["N"])
    f_v = sum(volume_fraction(fm, RHO_PPT[label]) for label, fm in f_mass_map.items())
    p_z = zener_pinning_mpa(f_v, r_nm)
    if p_z >= drive_mpa:
        verdict, v_cls = "PINNING EFFECTIVE", "ok"
    elif p_z >= 0.5 * drive_mpa:
        verdict, v_cls = "MARGINAL", "warn"
    else:
        verdict, v_cls = "PINNING WEAK", "bad"

    st.markdown(
        '<div class="facts">'
        f'<div><span>precipitate mass fraction at FRT</span><b>{sum(f_mass_map.values()):.2e}</b>'
        f'<em>wt fraction</em></div>'
        f'<div><span>volume fraction f_v</span><b>{f_v:.2e}</b><em>ρ<sub>matrix</sub> 7.87</em></div>'
        f'<div><span>Zener pinning pressure</span><b>{p_z:.3f}</b><em> MPa · 3·f_v·γ / 2r</em></div>'
        f'<div><span>driving pressure</span><b>{drive_mpa:.1f}</b><em> MPa · screening threshold</em></div>'
        f'<div><span>verdict</span><span class="pill {v_cls}">{verdict}</span></div>'
        '</div>', unsafe_allow_html=True)
    st.plotly_chart(_zener_fig(f_v, drive_mpa), use_container_width=True)
    st.markdown(
        '<div class="note">Pinning pressure scales with volume fraction and inversely with '
        'radius, so finer, more abundant precipitates pin harder. The driving-pressure threshold '
        'is a screening criterion, not a calculation: the stored-energy driving pressure at high '
        'reduction is order-of-magnitude O(0.1–10) MPa.</div>', unsafe_allow_html=True)

    # --- precipitation kinetics (closed-form JMAK) -----------------------------------------
    _section("precipitation kinetics")
    st.markdown(
        '<div class="note">Equilibrium above says <i>how much</i> can precipitate; kinetics say '
        '<i>how fast</i>. A closed-form JMAK/Avrami law relaxes the actual state toward the '
        'sequential-model equilibrium at each temperature — pure numpy, microseconds '
        'in-process, no Modal dispatch and no cloud calls.</div>', unsafe_allow_html=True)

    hold_times = [1, 2, 5, 10, 15, 30, 45, 60, 120, 300, 600, 900, 1800, 2700, 3600]
    kc1, kc2 = st.columns(2)
    t_hold = kc1.select_slider("Hold time (s)", hold_times, value=600,
                               key=f"ppt_hold_t_{mill}")
    t_kin_def = min(max(temps["frt"], 700.0), 1200.0)
    t_hold_C = kc2.slider("Hold temperature (°C)", 700.0, 1200.0, float(t_kin_def), 5.0,
                          key=f"ppt_hold_T_{mill}",
                          help="Defaults to the schedule's finish temperature (clamped into "
                               "the 700-1200 °C band).")

    t50 = kinetics_t50(metal, t_hold_C, chem[metal], chem["C"], chem["N"])
    eq_hold = float(undissolved_fraction(metal, t_hold_C, chem[metal],
                                         chem["C"], chem["N"]))
    x_jmak = jmak_fraction(t_hold, t50) if t50 is not None else 0.0
    frac_hold = precipitated_fraction_at(metal, t_hold_C, t_hold, chem[metal],
                                         chem["C"], chem["N"])
    if chem[metal] <= 0.0 or chem["N"] <= 0.0:
        t50_txt, t50_sub = "—", "no chemistry"
    elif t50 is None:
        t50_txt, t50_sub = "—", "no supersaturation (dissolved)"
    elif t50 >= KINETICS_CONSTANTS["T50_CAP_S"]:
        t50_txt, t50_sub = "> 30 d", "capped"
    else:
        t50_txt = (f"{t50:.0f} s" if t50 < 120.0 else
                   (f"{t50 / 60.0:.1f} min" if t50 < 7200.0 else f"{t50 / 3600.0:.1f} h"))
        t50_sub = f"at {t_hold_C:.0f} °C"

    st.markdown(
        '<div class="facts">'
        f'<div><span>{metal} t_50 at hold temperature</span><b>{t50_txt}</b>'
        f'<em>{t50_sub}</em></div>'
        f'<div><span>JMAK transformed fraction X(t)</span><b>{x_jmak * 100:.1f}</b>'
        f'<em>% after {t_hold} s</em></div>'
        f'<div><span>actually precipitated</span><b>{frac_hold * 100:.1f}</b>'
        f'<em>% of {metal} · X(t) × equilibrium</em></div>'
        f'<div><span>equilibrium at {t_hold_C:.0f} °C</span><b>{eq_hold * 100:.1f}</b>'
        f'<em>% (t → ∞ limit)</em></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="note">Kinetics only modulate <b>how fast</b> the equilibrium amount forms: '
        f'the precipitated fraction is the JMAK transformed fraction X(t) times the equilibrium '
        f'fraction at the hold temperature, so an infinite hold always lands on the equilibrium '
        f'value, never beyond it. Calibration: t_ref = 1 h at moderate supersaturation '
        f'(D = log₁₀S = 0.5), t_50 ∝ (D + 0.05)⁻³ (m_k = 3), JMAK exponent n = 1.5, t_50 capped '
        f'at 30 days — an engineering screening calibration, not a fitted TTT dataset.</div>',
        unsafe_allow_html=True)

    st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
    st.markdown('<b>Rolling path</b>', unsafe_allow_html=True)
    stages_def = rolling_stages(mill, raw)
    with st.form(f"ppt_path_form_{mill}", clear_on_submit=False):
        pcols = st.columns(len(stages_def))
        times = []
        for pc, (lbl, t_stage, dt) in zip(pcols, stages_def):
            times.append(pc.number_input(f"{lbl} ({t_stage:.0f} °C)", min_value=0.0,
                                         max_value=float(KINETICS_CONSTANTS["T50_CAP_S"]),
                                         value=float(dt), step=5.0,
                                         key=f"ppt_pt_{mill}_{lbl}"))
        submitted = st.form_submit_button("Calculate path")
    path_key = f"ppt_path_{mill}"
    if submitted:
        stages = [(lbl, t_stage, float(tv)) for (lbl, t_stage, _dt), tv in zip(stages_def, times)]
        paths = {m: stage_path(m, [(t_stage, tv) for _lbl, t_stage, tv in stages],
                               chem[m], chem["C"], chem["N"])
                 for m in ("Nb", "Ti", "V", "Al")}
        st.session_state[path_key] = {"stages": stages, "paths": paths}
    stored = st.session_state.get(path_key)
    if stored:
        st.plotly_chart(_path_fig(stored["stages"], stored["paths"]), use_container_width=True)
        st.markdown(
            '<div class="note">One trace per microalloy along the schedule\'s stages (times '
            'editable above, replayed from the last "Calculate path" press). Reheating above a '
            'system\'s solution temperature dissolves the precipitate back (same JMAK law '
            'driven by the reverse supersaturation — a symmetry assumption), so dips in the '
            'lines mark dissolution and rises mark re-precipitation on cooling. Dashed lines: '
            'equilibrium at the final stage temperature. Closed-form per stage, no Modal, no '
            'cloud calls.</div>', unsafe_allow_html=True)

    # --- method note -----------------------------------------------------------------------
    with st.expander("Method, sources & caveats"):
        st.markdown(
            '<div class="note"><b>Solubility products</b> (austenite, log₁₀ of the wt%-product): '
            'NbC and NbN, VC, VN: Irvine et al., 1967; TiN: Wada &amp; Pehlke, 1977; TiC: '
            'austenite fit, cf. the Wada &amp; Pehlke family; AlN: austenite fit, cf. the '
            'Davern &amp; Edmonds family. The wt%-product fits carry a literature spread of '
            'roughly ±200 K on solution temperatures.<br><br>'
            'These are <b>paraequilibrium screening estimates</b>, not Thermo-Calc calls: '
            'substitutional diffusion is assumed frozen and the microalloy partitions 1:1-molar '
            'between matrix and precipitate at each temperature. The sequential '
            'nitride-then-carbide order per metal is the standard engineering approximation of '
            'the coupled equilibrium: nitrides (especially TiN) are far stabler and precipitate '
            'first. Zener pinning uses γ = 0.5 J/m² and matrix density 7.87 g/cm³.<br><br>'
            '<b>Precipitation kinetics</b> couple a closed-form JMAK/Avrami law '
            '(X = 1 − exp(−ln2·(t/t₅₀)^1.5)) to the same equilibrium model: the characteristic '
            'time t₅₀ follows an engineering power law in the supersaturation driving force '
            'D = log₁₀(P/K(T)), calibrated as t₅₀ = 1 h at D = 0.5 with steepness m_k = 3 and '
            'floor ε = 0.05, capped at 30 days. The Arrhenius exp(Q/R·(1/T−1/T_ref)) factor of '
            'a full model was deliberately simplified away — K(T) already carries the dominant '
            'temperature dependence, and the law is a screening calibration, not a fitted TTT '
            'dataset. Dissolution on reheating uses the same t₅₀ form driven by the reverse '
            '(mirror) supersaturation of the dissolved product — a symmetry assumption. '
            'Everything is closed-form numpy evaluated in microseconds per keystroke '
            'in-process: no Modal dispatch and no cloud calls.</div>',
            unsafe_allow_html=True)
