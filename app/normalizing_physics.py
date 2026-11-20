"""Pure-physics metallurgy & thermal engine for plate furnace normalizing.

Implements closed-form 1D through-thickness heating/cooling thermal profiles, coupled
sequential nitride-then-carbide precipitate dissolution/re-precipitation kinetics (reusing
``app/precipitation.py``), Zener-pinned austenite grain growth with Ostwald coarsening,
Sellars-Beynon air-cooled ferrite transformation, calibrated standalone constitutive property
predictions, and nominal-soak ML-anchored hybrid predictions.

Design invariants:
- ZERO ``streamlit`` imports and ZERO ``modal`` imports.
- Imports ``physics`` directly (NumPy-only) and resolves ``precipitation`` lazily on first
  precipitate/solubility/kinetics call so module import stays sub-millisecond locally and
  inside Modal containers.
"""

import json
import os
import sys

import numpy as np

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_APP_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import physics as _physics  # noqa: E402 - NumPy-only read-only helper module

_PPT_MOD = None
_CALIB_CACHE = {}

DEFAULT_CALIBRATION_PATH = os.path.join(_REPO_ROOT, "models", "normalizing_calibration.json")

# Physical constants for Ostwald ripening and Burke-Turnbull Zener-pinned grain growth
_R_GAS = 8.314                 # J / (mol * K)
_R0_PPT_NM = 12.0              # initial mean microalloy precipitate radius, nm
_K_COARSEN_NM3_S = 7.6e11      # Lifshitz-Slyozov-Wagner coarsening prefactor, nm^3 / s
_Q_COARSEN_J_MOL = 270_000.0   # substitutional diffusion activation energy in austenite, J/mol
_K_GRAIN_UM2_S = 3.5e7         # austenite grain boundary mobility prefactor, um^2 / s
_Q_GRAIN_J_MOL = 190_000.0     # grain boundary migration activation energy, J/mol
_D_GAMMA_0_UM = 14.0           # newly nucleated austenite grain diameter at Ac3, um

# Thermodynamic stability order for coupled interstitial N / C consumption across metals
_NITRIDE_ORDER = ("Ti", "Nb", "Al", "V")
_DISPLAY_METALS = ("Al", "Nb", "Ti", "V")

DEFAULT_CALIBRATION = {
    "source": "default_literature_baseline",
    "t_soak_ref_c": 910.0,
    "soak_min_per_mm_ref": 0.6,
    "YS_AN": {
        "sigma_0": 35.0,
        "k_ss": 0.90,
        "k_hp": 14.0,
        "k_ce": 220.0,
        "k_c": 80.0,
        "k_v": 1200.0,
        "k_nb": 1000.0,
        "k_cr": 12.0,
    },
    "TS_AN": {
        "sigma_0": 155.0,
        "k_ss": 0.95,
        "k_hp": 7.5,
        "k_ce": 360.0,
        "k_c": 420.0,
        "k_v": 950.0,
        "k_nb": 800.0,
        "k_cr": 10.0,
    },
    "EL_AN": {
        "sigma_0": 36.0,
        "k_ss": -0.04,
        "k_hp": 0.45,
        "k_ce": -16.0,
        "k_c": -18.0,
        "k_v": -25.0,
        "k_nb": -20.0,
        "k_cr": -0.8,
    },
}

_REGIME_META = {
    "SUBCRITICAL": (
        "Subcritical Stress Relief / Tempering",
        "Soaking below Ac1 does not re-austenitize the plate; existing ferrite/bainite recovers "
        "and partially tempers without austenite-to-ferrite grain refinement.",
    ),
    "INTERCRITICAL": (
        "Intercritical (α + γ) Two-Phase",
        "Soaking between Ac1 and Ac3+15 °C yields partial austenitization, leaving mixed "
        "untransformed ferrite and transformed pockets with lower yield uniformity.",
    ),
    "UNDER_SOAKED": (
        "Under-Soaked Core (< 5 min)",
        "Furnace residence barely exceeds through-thickness heating time, leaving insufficient "
        "effective core soak for full carbide dissolution and thermal homogenization.",
    ),
    "OVERHEATED": (
        "Overheated / Zener Pinning Breakdown",
        "Excessive soaking temperature or hold time coarsens precipitates and drives austenite "
        "grain growth near or beyond the Zener pinning limit, coarsening d_α and raising DBTT.",
    ),
    "FULL_NORMALIZING": (
        "Full Austenitizing & Pinning",
        "Core reaches full austenitic soak above Ac3+15 °C while fine precipitates pin austenite "
        "grains, transforming to fine uniform polygonal ferrite + pearlite on air cooling.",
    ),
}


def _ppt():
    """Lazy loader for ``app/precipitation.py`` (defers plotly/streamlit import until needed)."""
    global _PPT_MOD
    if _PPT_MOD is not None:
        return _PPT_MOD
    existing = sys.modules.get("precipitation") or sys.modules.get("app.precipitation")
    if existing is not None:
        _PPT_MOD = existing
        return _PPT_MOD
    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)
    import precipitation as ppt  # noqa: PLC0415
    _PPT_MOD = ppt
    return _PPT_MOD


def _chem(chem, el, default=0.0):
    """Read element wt% from either ``Chem_<el>`` or ``<el>`` keys safely."""
    if not chem:
        return float(default)
    for key in (f"Chem_{el}", el):
        if key in chem:
            val = chem[key]
            if val is not None:
                try:
                    num = float(val)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(num):
                    return num
    return float(default)


def _raw_num(raw, keys, default):
    """Read first finite numeric value from ``raw`` across candidate keys."""
    if not raw:
        return float(default)
    for key in keys:
        if key in raw and raw[key] is not None:
            try:
                num = float(raw[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(num):
                return num
    return float(default)


def _extract_thickness_mm(raw):
    """Extract plate/strip gauge (mm) from Plate, HRC, or generic raw dict."""
    t_mm = _raw_num(raw, ("Plate_T_mm", "plate_t_mm", "Coil_T_mm", "coil_t_mm", "thickness"), 16.0)
    return max(t_mm, 1.0)


def _ce_iiw_from_chem(chem):
    """IIW carbon equivalent via ``app/physics.py``."""
    return float(_physics.ce_iiw(
        max(_chem(chem, "C"), 0.0),
        max(_chem(chem, "Mn"), 0.0),
        max(_chem(chem, "Cr"), 0.0),
        max(_chem(chem, "Mo"), 0.0),
        max(_chem(chem, "V"), 0.0),
        max(_chem(chem, "Ni"), 0.0),
        max(_chem(chem, "Cu"), 0.0),
    ))


# ----------------------------------------------------------------------------- 1. Ac1 & Ac3
def ac1_temp(chem):
    """Andrews empirical Ac1 lower critical temperature (°C)."""
    mn = max(_chem(chem, "Mn"), 0.0)
    ni = max(_chem(chem, "Ni"), 0.0)
    si = max(_chem(chem, "Si"), 0.0)
    cr = max(_chem(chem, "Cr"), 0.0)
    return float(_physics.ar1(mn, ni, si, cr))


def ac3_temp(chem):
    """Andrews empirical Ac3 upper critical temperature (°C), clipped to [760, 960] °C."""
    c = max(_chem(chem, "C"), 0.0)
    mn = max(_chem(chem, "Mn"), 0.0)
    si = max(_chem(chem, "Si"), 0.0)
    ni = max(_chem(chem, "Ni"), 0.0)
    cr = max(_chem(chem, "Cr"), 0.0)
    mo = max(_chem(chem, "Mo"), 0.0)
    v = max(_chem(chem, "V"), 0.0)
    cu = max(_chem(chem, "Cu"), 0.0)
    p = max(_chem(chem, "P"), 0.0)
    al = max(_chem(chem, "Al"), 0.0)
    ti = max(_chem(chem, "Ti"), 0.0)
    val = (
        910.0
        - 203.0 * np.sqrt(c)
        - 15.2 * ni
        + 44.7 * si
        + 104.0 * v
        + 31.5 * mo
        - 30.0 * mn
        - 11.0 * cr
        - 20.0 * cu
        + 700.0 * p
        + 400.0 * al
        + 400.0 * ti
    )
    return float(np.clip(val, 760.0, 960.0))


# ----------------------------------------------------------------------------- 2. Thermal profile
def _normalize_cooling_mode(cooling_mode):
    mode = str(cooling_mode or "still_air").strip().lower().replace(" ", "_")
    if mode.startswith("forced") or "fan" in mode:
        return "forced_air"
    return "still_air"


def thermal_profile(
    plate_t_mm,
    t_soak_c,
    residence_min,
    heating_rate_min_per_mm=1.2,
    cooling_mode="still_air",
    t_ambient=25.0,
    n_pts=120,
):
    """1D through-thickness normalizing furnace thermal cycle + 20-minute air-cooling tail."""
    plate_t = max(float(plate_t_mm), 1.0)
    t_soak = float(t_soak_c)
    res_min = max(float(residence_min), 0.0)
    rate = max(float(heating_rate_min_per_mm), 0.1)
    t_amb = float(t_ambient)
    n_pts_int = max(int(n_pts), 12)

    t_heat_min = rate * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
    t_eff_soak_min = max(0.0, res_min - t_heat_min)

    tau_surf = max(1.5, 0.18 * t_heat_min)
    tau_core = max(2.5, t_heat_min / 2.2)

    still_air = 18.0 / (max(plate_t, 2.0) ** 1.05)
    mode_canon = _normalize_cooling_mode(cooling_mode)
    cr_air_k_s = (2.2 * still_air) if mode_canon == "forced_air" else still_air

    hj_param = (t_soak + 273.15) * (20.0 + float(np.log10(max(t_eff_soak_min / 60.0, 1e-3)))) / 1000.0

    t_surf_peak_c = t_amb + (t_soak - t_amb) * (1.0 - float(np.exp(-res_min / tau_surf)))
    t_core_peak_c = t_amb + (t_soak - t_amb) * (1.0 - float(np.exp(-res_min / tau_core)))

    # Time series covering [0 .. residence_min] in furnace and [residence_min .. +20] air cool
    n_furn = max(6, int(round(0.75 * n_pts_int)))
    n_cool = max(6, n_pts_int - n_furn + 1)
    t_furn = np.linspace(0.0, res_min, n_furn)
    t_cool_rel = np.linspace(0.0, 20.0, n_cool)

    surf_furn = t_amb + (t_soak - t_amb) * (1.0 - np.exp(-t_furn / tau_surf))
    core_furn = t_amb + (t_soak - t_amb) * (1.0 - np.exp(-t_furn / tau_core))

    # Newton cooling matched to initial core cooling rate cr_air_k_s (K/s)
    cr_min = max(cr_air_k_s * 60.0, 1e-3)
    tau_cool_core = max(abs(t_core_peak_c - t_amb), 50.0) / cr_min
    tau_cool_surf = 0.82 * tau_cool_core

    surf_cool = t_amb + (t_surf_peak_c - t_amb) * np.exp(-t_cool_rel / tau_cool_surf)
    core_cool = t_amb + (t_core_peak_c - t_amb) * np.exp(-t_cool_rel / tau_cool_core)

    time_min = np.concatenate([t_furn, res_min + t_cool_rel[1:]])
    t_surface_c = np.concatenate([surf_furn, surf_cool[1:]])
    t_core_c = np.concatenate([core_furn, core_cool[1:]])

    return {
        "t_heat_min": float(t_heat_min),
        "t_eff_soak_min": float(t_eff_soak_min),
        "t_core_peak_c": float(t_core_peak_c),
        "t_surf_peak_c": float(t_surf_peak_c),
        "tau_surf_min": float(tau_surf),
        "tau_core_min": float(tau_core),
        "cr_air_k_s": float(cr_air_k_s),
        "hj_param": float(hj_param),
        "heating_rate_min_per_mm": float(rate),
        "cooling_mode": mode_canon,
        "time_min": time_min,
        "t_surface_c": t_surface_c,
        "t_core_c": t_core_c,
    }


# ----------------------------------------------------------------------------- 3. Precipitates
def _stoich_ratio(ppt, label):
    sysd = ppt.SOLUBILITY[label]
    return ppt.ATOMIC_MASS[sysd["inter"][5:]] / ppt.ATOMIC_MASS[sysd["metal"][5:]]


def _metal_t_sol_c(ppt, metal, m0, c0, n0):
    """Highest solution temperature (°C) across a metal's systems, or None."""
    temps = []
    for label in ppt.METAL_SYSTEMS.get(metal, ()):
        sysd = ppt.SOLUBILITY[label]
        x0 = n0 if sysd["inter"].endswith("_N") else c0
        t_k = ppt.solution_temperature(m0, x0, sysd["A"], sysd["B"])
        if t_k is not None:
            temps.append(t_k - 273.15)
    return float(max(temps)) if temps else None


def _system_t_sol_c(ppt, label, m0, c0, n0):
    sysd = ppt.SOLUBILITY[label]
    x0 = n0 if sysd["inter"].endswith("_N") else c0
    t_k = ppt.solution_temperature(m0, x0, sysd["A"], sysd["B"])
    return float(t_k - 273.15) if t_k is not None else None


def _effective_kinetics_t50(ppt, metal, t_c, m0, c_avail, n_avail):
    """Fastest valid precipitation t_50 (s) across a metal's nitride and carbide systems.

    Uses ``precipitation.kinetics_t50`` directly; falling back across ``METAL_SYSTEMS[metal]``
    ensures NbC/VC still nucleate when free N has already been consumed by TiN/AlN.
    """
    candidates = []
    for label in ppt.METAL_SYSTEMS.get(metal, ()):
        t50 = ppt.kinetics_t50(label, t_c, m0, c_avail, n_avail)
        if t50 is not None and np.isfinite(t50) and t50 > 0.0:
            candidates.append(float(t50))
    return min(candidates) if candidates else None


def _effective_dissolution_t50(ppt, metal, t_c, m0, c_avail, n_avail, f_state):
    """Characteristic dissolution t_50 (s) mirroring ``precipitation._dissolution_t50``."""
    if m0 <= 0.0:
        return None
    t_k = float(t_c) + 273.15
    cfg = ppt.KINETICS_CONSTANTS
    candidates = []
    for label in ppt.METAL_SYSTEMS.get(metal, ()):
        sysd = ppt.SOLUBILITY[label]
        x0 = float(n_avail) if sysd["inter"].endswith("_N") else float(c_avail)
        p_actual = float(m0) * x0
        if not np.isfinite(p_actual) or p_actual <= 0.0:
            continue
        k_val = 10.0 ** (sysd["A"] - sysd["B"] / t_k)
        p_diss = max(float(m0) * (1.0 - float(f_state)) * x0, cfg["DISS_FLOOR_REL"] * p_actual, 1e-12)
        d_rev = float(np.log10(k_val / p_diss))
        if d_rev <= 0.0:
            d_rev = cfg["DISS_FALLBACK_D"]
        t50 = cfg["T50_REF_S"] * ((cfg["D_REF"] + cfg["EPS_FLOOR"]) / (d_rev + cfg["EPS_FLOOR"])) ** cfg["M_K"]
        candidates.append(float(min(t50, cfg["T50_CAP_S"])))
    if candidates:
        return min(candidates)
    # Fully depleted N: dissolution still proceeds at reference driving force
    return float(cfg["T50_REF_S"])


def _run_metal_stages(ppt, metal, stages, m0, c_avail, n_avail):
    """Run 3-stage JMAK relaxation using ``precipitation`` equilibrium & kinetics primitives."""
    if m0 <= 0.0:
        return [
            {"T_C": float(tc), "time_s": max(float(dt), 0.0), "fraction": 0.0,
             "equilibrium": 0.0, "t50_s": None, "mode": "hold"}
            for tc, dt in stages
        ]
    # When N is available and nitride drives kinetics, delegate directly to precipitation.stage_path
    if n_avail > 1e-7 and ppt.kinetics_t50(metal, stages[0][0], m0, c_avail, n_avail) is not None:
        all_ok = all(
            ppt.kinetics_t50(metal, tc, m0, c_avail, n_avail) is not None
            or float(ppt.undissolved_fraction(metal, tc, m0, c_avail, n_avail)) == 0.0
            for tc, _ in stages
        )
        if all_ok:
            return ppt.stage_path(metal, stages, m0, c_avail, n_avail)

    out = []
    f = 0.0
    for tc, dt in stages:
        tc_f = float(tc)
        dt_f = max(float(dt), 0.0)
        eq = float(ppt.undissolved_fraction(metal, tc_f, m0, c_avail, n_avail))
        if eq > f:
            t50 = _effective_kinetics_t50(ppt, metal, tc_f, m0, c_avail, n_avail)
            mode = "precipitate" if dt_f > 0.0 else "hold"
        elif eq < f:
            t50 = _effective_dissolution_t50(ppt, metal, tc_f, m0, c_avail, n_avail, f)
            mode = "dissolve" if dt_f > 0.0 else "hold"
        else:
            t50, mode = None, "hold"
        x = ppt.jmak_fraction(dt_f, t50) if (dt_f > 0.0 and t50 is not None) else 0.0
        f = float(np.clip(f + (eq - f) * x, 0.0, 1.0))
        out.append({
            "T_C": tc_f,
            "time_s": dt_f,
            "fraction": f,
            "equilibrium": eq,
            "t50_s": t50,
            "mode": mode,
        })
    return out


def _apportion_bound_systems(ppt, metal, target_bound_wt, t_c, m0, c_avail, n_avail, fallback_t_c):
    """Split a metal's bound wt% nitride-first across its systems while strictly conserving C & N."""
    labels = ppt.METAL_SYSTEMS.get(metal, ())
    systems = {lbl: 0.0 for lbl in labels}
    if target_bound_wt <= 0.0 or m0 <= 0.0 or not labels:
        return systems, 0.0, 0.0, 0.0

    eq_map = ppt.sequential_precipitated(metal, t_c, m0, c_avail, n_avail)
    eq_tot = sum(float(v) for v in eq_map.values())
    if eq_tot <= 1e-12:
        eq_map = ppt.sequential_precipitated(metal, min(t_c, fallback_t_c), m0, c_avail, n_avail)
        eq_tot = sum(float(v) for v in eq_map.values())

    rem_m = min(float(target_bound_wt), float(m0))
    n_used = 0.0
    c_used = 0.0
    if eq_tot > 1e-12:
        # Scale up to target_bound_wt, nitride first, capped by available interstitials
        for lbl in labels:
            if rem_m <= 0.0:
                break
            sysd = ppt.SOLUBILITY[lbl]
            r = _stoich_ratio(ppt, lbl)
            is_n = sysd["inter"].endswith("_N")
            share = float(eq_map[lbl]) / eq_tot
            want = min(rem_m, float(target_bound_wt) * share)
            cap = (max(0.0, n_avail - n_used) / r) if is_n else (max(0.0, c_avail - c_used) / r)
            take = min(want, cap, rem_m)
            systems[lbl] += take
            rem_m -= take
            if is_n:
                n_used += take * r
            else:
                c_used += take * r
        # Any leftover if one interstitial hit its cap goes to the remaining system if available
        if rem_m > 1e-12:
            for lbl in labels:
                sysd = ppt.SOLUBILITY[lbl]
                r = _stoich_ratio(ppt, lbl)
                is_n = sysd["inter"].endswith("_N")
                cap = (max(0.0, n_avail - n_used) / r) if is_n else (max(0.0, c_avail - c_used) / r)
                take = min(rem_m, cap)
                systems[lbl] += take
                rem_m -= take
                if is_n:
                    n_used += take * r
                else:
                    c_used += take * r
    else:
        for lbl in labels:
            sysd = ppt.SOLUBILITY[lbl]
            r = _stoich_ratio(ppt, lbl)
            is_n = sysd["inter"].endswith("_N")
            cap = (max(0.0, n_avail - n_used) / r) if is_n else (max(0.0, c_avail - c_used) / r)
            take = min(rem_m, cap)
            systems[lbl] = take
            rem_m -= take
            if is_n:
                n_used += take * r
            else:
                c_used += take * r

    actual_bound = sum(systems.values())
    return systems, actual_bound, n_used, c_used


def ostwald_radius_nm(t_soak_c, t_eff_min, r0_nm=_R0_PPT_NM):
    """Ostwald-ripened mean precipitate radius (nm) at soak temperature and effective hold time."""
    t_k = max(float(t_soak_c) + 273.15, 300.0)
    hold_s = max(float(t_eff_min), 0.0) * 60.0
    r3 = (float(r0_nm) ** 3) + _K_COARSEN_NM3_S * float(np.exp(-_Q_COARSEN_J_MOL / (_R_GAS * t_k))) * hold_s
    return float(r3 ** (1.0 / 3.0))


def normalizing_precipitates(chem, t_soak_c, t_eff_min, t_initial_c=820.0):
    """3-stage normalizing precipitation state (incoming -> soak -> post-discharge 700 °C).

    Tracks soak-end bound/dissolved fractions, Zener pinning, and Ostwald coarsening at
    ``t_soak_c`` as well as post-cool re-precipitation (especially V(C,N) at 700 °C), with
    strict stoichiometric conservation of N and C across ``Ti -> Nb -> Al -> V``.
    """
    ppt = _ppt()
    t_soak = float(t_soak_c)
    t_init = float(t_initial_c)
    t_eff = max(float(t_eff_min), 0.0)
    soak_s = max(t_eff, 0.5) * 60.0
    stages = [(t_init, 60.0), (t_soak, soak_s), (700.0, 300.0)]

    c0 = max(_chem(chem, "C"), 0.0)
    n0 = max(_chem(chem, "N"), 0.0)
    totals = {m: max(_chem(chem, m), 0.0) for m in _NITRIDE_ORDER}

    n_rem_soak = n0
    c_rem_soak = c0
    n_rem_cool = n0
    c_rem_cool = c0
    bound_n_soak = 0.0
    bound_c_soak = 0.0
    bound_n_cool = 0.0
    bound_c_cool = 0.0

    elements_raw = {}
    f_v_total = 0.0

    for metal in _NITRIDE_ORDER:
        m0 = totals[metal]
        t_sol_c = _metal_t_sol_c(ppt, metal, m0, c0, n0)
        sys_t_sol = {lbl: _system_t_sol_c(ppt, lbl, m0, c0, n0) for lbl in ppt.METAL_SYSTEMS[metal]}

        # Evaluate 3-stage trajectory with remaining interstitial N / C
        stage_out = _run_metal_stages(ppt, metal, stages, m0, c_rem_soak, n_rem_soak)

        # Soak-end state (stage index 1) drives Zener pinning & soak dissolution
        f_soak = float(stage_out[1]["fraction"])
        eq_soak = float(stage_out[1]["equilibrium"])
        systems_soak, bound_soak_wt, dn_s, dc_s = _apportion_bound_systems(
            ppt, metal, m0 * f_soak, t_soak, m0, c_rem_soak, n_rem_soak, t_init
        )
        n_rem_soak = max(0.0, n_rem_soak - dn_s)
        c_rem_soak = max(0.0, c_rem_soak - dc_s)
        bound_n_soak += dn_s
        bound_c_soak += dc_s

        bound_frac_soak = (bound_soak_wt / m0) if m0 > 0.0 else 0.0
        dissolved_wt_soak = max(0.0, m0 - bound_soak_wt)
        dissolved_frac_soak = (1.0 - bound_frac_soak) if m0 > 0.0 else 0.0

        for lbl, s_wt in systems_soak.items():
            f_mass = float(s_wt) * (1.0 + _stoich_ratio(ppt, lbl)) / 100.0
            f_v_total += ppt.volume_fraction(f_mass, ppt.RHO_PPT[lbl])

        # Post-cool state (stage index 2 at 700 °C) tracks re-precipitation on air cooling
        f_cool_target = float(stage_out[2]["fraction"])
        # Re-evaluate cool stage if available N at cool differs significantly
        if n_rem_cool < n0 * 0.99 and m0 > 0.0:
            eq_cool = float(ppt.undissolved_fraction(metal, 700.0, m0, c_rem_cool, n_rem_cool))
            if eq_cool > f_soak:
                t50_c = _effective_kinetics_t50(ppt, metal, 700.0, m0, c_rem_cool, n_rem_cool)
                x_c = ppt.jmak_fraction(300.0, t50_c) if t50_c is not None else 0.0
                f_cool_target = float(np.clip(f_soak + (eq_cool - f_soak) * x_c, 0.0, 1.0))
            elif eq_cool < f_soak:
                t50_d = _effective_dissolution_t50(ppt, metal, 700.0, m0, c_rem_cool, n_rem_cool, f_soak)
                x_d = ppt.jmak_fraction(300.0, t50_d) if t50_d is not None else 0.0
                f_cool_target = float(np.clip(f_soak + (eq_cool - f_soak) * x_d, 0.0, 1.0))
            else:
                f_cool_target = f_soak

        systems_cool, bound_cool_wt, dn_c, dc_c = _apportion_bound_systems(
            ppt, metal, m0 * f_cool_target, 700.0, m0, c_rem_cool, n_rem_cool, t_soak
        )
        n_rem_cool = max(0.0, n_rem_cool - dn_c)
        c_rem_cool = max(0.0, c_rem_cool - dc_c)
        bound_n_cool += dn_c
        bound_c_cool += dc_c

        bound_frac_cool = (bound_cool_wt / m0) if m0 > 0.0 else 0.0
        reprecip_wt = max(0.0, bound_cool_wt - bound_soak_wt)

        elements_raw[metal] = {
            "total_wt": float(m0),
            "bound_wt": float(bound_soak_wt),
            "dissolved_wt": float(dissolved_wt_soak),
            "bound_frac": float(bound_frac_soak),
            "dissolved_frac": float(dissolved_frac_soak),
            "eq_bound_frac": float(eq_soak),
            "systems": systems_soak,
            "t_sol_c": t_sol_c,
            "system_t_sol_c": sys_t_sol,
            "post_cool_bound_frac": float(bound_frac_cool),
            "post_cool_bound_wt": float(bound_cool_wt),
            "post_cool_dissolved_wt": float(max(0.0, m0 - bound_cool_wt)),
            "post_cool_systems": systems_cool,
            "reprecipitated_wt": float(reprecip_wt),
            "stages": stage_out,
        }

    elements = {m: elements_raw[m] for m in _DISPLAY_METALS}
    r_ppt_nm = ostwald_radius_nm(t_soak, t_eff)
    pz_mpa = float(ppt.zener_pinning_mpa(f_v_total, r_ppt_nm))
    d_lim_um = float(np.clip(4.0 * r_ppt_nm / (3.0 * max(f_v_total, 1e-6)) / 1000.0, 8.0, 250.0))

    free_n_wt = max(0.0, n0 - bound_n_soak)
    free_c_wt = max(0.0, c0 - bound_c_soak)

    return {
        "elements": elements,
        "free_n_wt": float(free_n_wt),
        "bound_n_wt": float(bound_n_soak),
        "total_n_wt": float(n0),
        "free_c_wt": float(free_c_wt),
        "bound_c_wt": float(bound_c_soak),
        "total_c_wt": float(c0),
        "post_cool_free_n_wt": float(max(0.0, n0 - bound_n_cool)),
        "post_cool_bound_n_wt": float(bound_n_cool),
        "v_reprecip_wt": float(elements["V"]["reprecipitated_wt"]),
        "r_ppt_nm": float(r_ppt_nm),
        "f_v_total": float(f_v_total),
        "zener_pinning_mpa": float(pz_mpa),
        "d_lim_um": float(d_lim_um),
    }


def solubility_curves(chem, t_min=700.0, t_max=1150.0, n_pts=91):
    """Equilibrium dissolved & bound wt%/fractions per element and compound over [t_min, t_max]."""
    ppt = _ppt()
    temps_c = np.linspace(float(t_min), float(t_max), max(int(n_pts), 5))
    c0 = max(_chem(chem, "C"), 0.0)
    n0 = max(_chem(chem, "N"), 0.0)
    totals = {m: max(_chem(chem, m), 0.0) for m in _NITRIDE_ORDER}

    n_rem = np.full_like(temps_c, n0, dtype=float)
    c_rem = np.full_like(temps_c, c0, dtype=float)
    bound_n_tot = np.zeros_like(temps_c, dtype=float)
    bound_c_tot = np.zeros_like(temps_c, dtype=float)

    elements_raw = {}
    systems_out = {}
    t_k_arr = temps_c + 273.15

    for metal in _NITRIDE_ORDER:
        m0 = totals[metal]
        s_map = ppt.sequential_precipitated(metal, temps_c, m0, c_rem, n_rem)
        sys_wt = {}
        bound_wt = np.zeros_like(temps_c, dtype=float)
        for lbl, val in s_map.items():
            arr = np.asarray(val, dtype=float)
            sys_wt[lbl] = arr
            bound_wt = bound_wt + arr
            r = _stoich_ratio(ppt, lbl)
            sysd = ppt.SOLUBILITY[lbl]
            if sysd["inter"].endswith("_N"):
                dn = np.minimum(arr * r, n_rem)
                n_rem = np.maximum(0.0, n_rem - dn)
                bound_n_tot = bound_n_tot + dn
            else:
                dc = np.minimum(arr * r, c_rem)
                c_rem = np.maximum(0.0, c_rem - dc)
                bound_c_tot = bound_c_tot + dc

            x_tot = n0 if sysd["inter"].endswith("_N") else c0
            prod = m0 * x_tot
            systems_out[lbl] = {
                "bound_wt": arr,
                "bound_frac": (arr / m0) if m0 > 0.0 else np.zeros_like(temps_c),
                "log10_k": sysd["A"] - sysd["B"] / t_k_arr,
                "log10_product": float(np.log10(prod)) if prod > 0.0 else None,
                "t_sol_c": _system_t_sol_c(ppt, lbl, m0, c0, n0),
            }

        bound_wt = np.minimum(bound_wt, m0)
        dissolved_wt = np.maximum(0.0, m0 - bound_wt)
        bound_frac = (bound_wt / m0) if m0 > 0.0 else np.zeros_like(temps_c)
        dissolved_frac = (1.0 - bound_frac) if m0 > 0.0 else np.zeros_like(temps_c)

        elements_raw[metal] = {
            "total_wt": float(m0),
            "bound_wt": bound_wt,
            "dissolved_wt": dissolved_wt,
            "bound_frac": bound_frac,
            "dissolved_frac": dissolved_frac,
            "systems": sys_wt,
            "t_sol_c": _metal_t_sol_c(ppt, metal, m0, c0, n0),
        }

    return {
        "temps_c": temps_c,
        "elements": {m: elements_raw[m] for m in _DISPLAY_METALS},
        "systems": systems_out,
        "free_n_wt": np.maximum(0.0, n0 - bound_n_tot),
        "bound_n_wt": bound_n_tot,
        "free_c_wt": np.maximum(0.0, c0 - bound_c_tot),
        "bound_c_wt": bound_c_tot,
    }


def kinetics_time_series(chem, t_soak_c, residence_min, t_initial_c=820.0, n_pts=60):
    """Kinetic bound fractions per microalloy across [0 .. max(residence_min, 10.0)] min."""
    ppt = _ppt()
    t_soak = float(t_soak_c)
    t_init = float(t_initial_c)
    t_end = max(float(residence_min), 10.0)
    time_min = np.linspace(0.0, t_end, max(int(n_pts), 10))
    time_s = time_min * 60.0

    c0 = max(_chem(chem, "C"), 0.0)
    n0 = max(_chem(chem, "N"), 0.0)
    totals = {m: max(_chem(chem, m), 0.0) for m in _NITRIDE_ORDER}

    n_rem = n0
    c_rem = c0
    elements_raw = {}

    for metal in _NITRIDE_ORDER:
        m0 = totals[metal]
        if m0 <= 0.0:
            zeros = np.zeros_like(time_min)
            elements_raw[metal] = {
                "total_wt": 0.0,
                "bound_frac": zeros,
                "dissolved_frac": zeros,
                "bound_wt": zeros,
                "eq_bound_frac": 0.0,
            }
            continue

        # Initial incoming plate state after 60 s at t_init
        st1 = _run_metal_stages(ppt, metal, [(t_init, 60.0)], m0, c_rem, n_rem)[0]
        f0 = float(st1["fraction"])
        eq_soak = float(ppt.undissolved_fraction(metal, t_soak, m0, c_rem, n_rem))

        if eq_soak > f0:
            t50 = _effective_kinetics_t50(ppt, metal, t_soak, m0, c_rem, n_rem)
        elif eq_soak < f0:
            t50 = _effective_dissolution_t50(ppt, metal, t_soak, m0, c_rem, n_rem, f0)
        else:
            t50 = None

        x_arr = np.asarray(ppt.jmak_fraction(time_s, t50), dtype=float) if t50 is not None else np.zeros_like(time_s)
        f_arr = np.clip(f0 + (eq_soak - f0) * x_arr, 0.0, 1.0)

        # Deduct steady-soak N / C consumption for downstream metals
        _, _, dn, dc = _apportion_bound_systems(
            ppt, metal, m0 * float(f_arr[-1]), t_soak, m0, c_rem, n_rem, t_init
        )
        n_rem = max(0.0, n_rem - dn)
        c_rem = max(0.0, c_rem - dc)

        elements_raw[metal] = {
            "total_wt": float(m0),
            "bound_frac": f_arr,
            "dissolved_frac": 1.0 - f_arr,
            "bound_wt": m0 * f_arr,
            "eq_bound_frac": float(eq_soak),
        }

    elements = {m: elements_raw[m] for m in _DISPLAY_METALS}
    return {
        "time_min": time_min,
        "t_soak_c": t_soak,
        "t_initial_c": t_init,
        "elements": elements,
        "bound_frac": {m: elements[m]["bound_frac"] for m in _DISPLAY_METALS},
        "eq_bound_frac": {m: elements[m]["eq_bound_frac"] for m in _DISPLAY_METALS},
    }


# ----------------------------------------------------------------------------- 4. Grain evolution
def _integrate_grain_series(t_soak_c, ac1_c, ac3_c, time_min, f_v_total, d_lim_0):
    """Integrate Zener-pinned austenite grain growth alongside Ostwald precipitate coarsening."""
    t_k = max(float(t_soak_c) + 273.15, 300.0)
    coarsen_rate = _K_COARSEN_NM3_S * float(np.exp(-_Q_COARSEN_J_MOL / (_R_GAS * t_k)))
    r_series = ((_R0_PPT_NM ** 3) + coarsen_rate * (np.asarray(time_min, dtype=float) * 60.0)) ** (1.0 / 3.0)
    d_lim_series = np.clip(4.0 * r_series / (3.0 * max(f_v_total, 1e-6)) / 1000.0, 8.0, 250.0)

    d_gamma_0 = min(_D_GAMMA_0_UM, max(8.0, 0.65 * d_lim_0))
    d_series = np.empty_like(time_min, dtype=float)
    d_series[0] = d_gamma_0

    if t_soak_c < ac1_c:
        d_series[:] = d_gamma_0
        return d_series, d_lim_series, r_series

    aust_frac = float(np.clip((t_soak_c - ac1_c) / max(ac3_c + 15.0 - ac1_c, 1.0), 0.05, 1.0))
    mobility = _K_GRAIN_UM2_S * float(np.exp(-_Q_GRAIN_J_MOL / (_R_GAS * t_k))) * aust_frac

    d_cur = d_gamma_0
    for idx in range(1, len(time_min)):
        dt_s = max(float(time_min[idx] - time_min[idx - 1]), 0.0) * 60.0
        d_lim_i = max(float(d_lim_series[idx]), 1.0)
        drive = max(0.0, 1.0 - d_cur / d_lim_i)
        if drive > 0.0 and dt_s > 0.0:
            d_next = float(np.sqrt(d_cur * d_cur + 2.0 * mobility * drive * dt_s))
            d_cur = min(d_next, d_lim_i)
        d_series[idx] = d_cur
    return d_series, d_lim_series, r_series


def grain_evolution(plate_t_mm, t_soak_c, t_eff_min, ac1_c, ac3_c, precip_state, cr_air_k_s, ce_iiw=0.38):
    """Austenite grain growth under Zener pinning, regime classification, and ferrite grain size."""
    t_soak = float(t_soak_c)
    t_eff = max(float(t_eff_min), 0.0)
    ac1 = float(ac1_c)
    ac3 = float(ac3_c)
    f_v = max(float(precip_state.get("f_v_total", 0.0)), 0.0)
    d_lim_final = float(precip_state.get("d_lim_um", 250.0))
    d_lim_0 = float(np.clip(4.0 * _R0_PPT_NM / (3.0 * max(f_v, 1e-6)) / 1000.0, 8.0, 250.0))

    max_plot_min = max(t_eff, 5.0)
    soak_time_min = np.linspace(0.0, max_plot_min, 61)
    d_gamma_um_series, d_lim_um_series, r_ppt_nm_series = _integrate_grain_series(
        t_soak, ac1, ac3, soak_time_min, f_v, d_lim_0
    )

    if t_eff <= 0.0:
        d_gamma_final_um = float(d_gamma_um_series[0])
    elif t_eff < max_plot_min:
        d_gamma_final_um = float(np.interp(t_eff, soak_time_min, d_gamma_um_series))
    else:
        d_gamma_final_um = float(d_gamma_um_series[-1])

    # Regime classification (strict priority order)
    if t_soak < ac1:
        regime = "SUBCRITICAL"
    elif t_soak < ac3 + 15.0:
        regime = "INTERCRITICAL"
    elif t_eff < 5.0:
        regime = "UNDER_SOAKED"
    elif d_gamma_final_um > 0.85 * d_lim_final or d_gamma_final_um > 45.0:
        regime = "OVERHEATED"
    else:
        regime = "FULL_NORMALIZING"

    # Transformed normalized ferrite grain size (Sellars-Beynon air-cooling law, zero retained strain)
    cr = max(float(cr_air_k_s), 0.15)
    ce_factor = 1.0 - 0.35 * float(np.clip(float(ce_iiw) - 0.35, -0.25, 0.35))
    d_alpha_raw = (1.8 + 3.2 * (cr ** -0.35)) * (max(d_gamma_final_um, 5.0) ** 0.33) * ce_factor
    if regime == "INTERCRITICAL":
        f_gamma = float(np.clip((t_soak - ac1) / max(ac3 + 15.0 - ac1, 1.0), 0.0, 1.0))
        d_alpha_raw = d_alpha_raw + 2.5 * (1.0 - f_gamma)
    d_alpha_um = float(np.clip(d_alpha_raw, 4.0, 45.0))

    pinning_ratio = float(d_gamma_final_um / max(d_lim_final, 1e-6))
    regime_label, regime_note = _REGIME_META[regime]

    return {
        "regime": regime,
        "regime_label": regime_label,
        "regime_note": regime_note,
        "d_gamma_final_um": float(d_gamma_final_um),
        "d_lim_um": float(d_lim_final),
        "d_alpha_um": float(d_alpha_um),
        "pinning_ratio": pinning_ratio,
        "plate_t_mm": float(plate_t_mm),
        "soak_time_min": soak_time_min,
        "d_gamma_um_series": d_gamma_um_series,
        "d_lim_um_series": d_lim_um_series,
        "r_ppt_nm_series": r_ppt_nm_series,
    }


# ----------------------------------------------------------------------------- 5. Calibration loader
def _copy_calibration(src):
    out = {}
    for key, val in src.items():
        if isinstance(val, dict):
            out[key] = dict(val)
        elif isinstance(val, list):
            out[key] = list(val)
        else:
            out[key] = val
    return out


def load_calibration(path=None):
    """Load Modal-calibrated normalizing weights if present on disk, cached by (path, mtime)."""
    target_path = os.path.abspath(path or DEFAULT_CALIBRATION_PATH)
    try:
        mtime = os.path.getmtime(target_path)
    except OSError:
        return _copy_calibration(DEFAULT_CALIBRATION)

    cached = _CALIB_CACHE.get(target_path)
    if cached is not None and cached[0] == mtime:
        return _copy_calibration(cached[1])

    try:
        with open(target_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
    except (OSError, ValueError):
        return _copy_calibration(DEFAULT_CALIBRATION)

    merged = _copy_calibration(DEFAULT_CALIBRATION)
    for scalar_key in ("source", "t_soak_ref_c", "soak_min_per_mm_ref", "timestamp",
                       "n_schedules", "metrics", "parity_samples", "per_grade", "plant_audit"):
        if scalar_key in loaded:
            merged[scalar_key] = loaded[scalar_key]
    for tgt in ("YS_AN", "TS_AN", "EL_AN"):
        if isinstance(loaded.get(tgt), dict):
            w = dict(DEFAULT_CALIBRATION[tgt])
            for k, v in loaded[tgt].items():
                if v is not None and np.isfinite(float(v)):
                    w[k] = float(v)
            merged[tgt] = w
    if "source" not in loaded:
        merged["source"] = os.path.basename(target_path)

    _CALIB_CACHE[target_path] = (mtime, _copy_calibration(merged))
    return merged


# ----------------------------------------------------------------------------- 6. Property prediction
def _regime_adjustments(regime, t_soak_c, t_eff_min, ac1_c, ac3_c, d_gamma_um, d_lim_um):
    """Metallurgical property corrections outside full austenitic normalizing."""
    if regime == "SUBCRITICAL":
        return {"YS_AN": -10.0, "TS_AN": -6.0, "EL_AN": 1.0}
    if regime == "INTERCRITICAL":
        deficit = float(np.clip((ac3_c + 15.0 - t_soak_c) / max(ac3_c + 15.0 - ac1_c, 20.0), 0.0, 1.0))
        return {"YS_AN": -18.0 * deficit, "TS_AN": -10.0 * deficit, "EL_AN": 0.5 * deficit}
    if regime == "UNDER_SOAKED":
        deficit = float(np.clip((5.0 - max(t_eff_min, 0.0)) / 5.0, 0.0, 1.0))
        return {"YS_AN": -12.0 * deficit, "TS_AN": -8.0 * deficit, "EL_AN": -1.2 * deficit}
    if regime == "OVERHEATED":
        excess = float(np.clip(
            max((d_gamma_um - 40.0) / 40.0, (d_gamma_um / max(d_lim_um, 1.0)) - 0.85, 0.0),
            0.0,
            1.0,
        ))
        return {"YS_AN": -8.0 * excess, "TS_AN": -3.0 * excess, "EL_AN": -1.5 * excess}
    return {"YS_AN": 0.0, "TS_AN": 0.0, "EL_AN": 0.0}


def _constitutive_components(features, weights, regime_shift):
    """Decompose constitutive prediction into 6 waterfall buckets summing to the total."""
    lattice_friction = float(weights["sigma_0"])
    solid_solution = float(weights["k_ss"]) * features["sigma_ss"]
    hall_petch_grain = float(weights["k_hp"]) * features["inv_sqrt_d_mm"]
    pearlite_ce = (
        float(weights["k_ce"]) * features["ce_iiw"]
        + float(weights["k_c"]) * features["c_wt"]
        + float(weights["k_cr"]) * features["cr_air_k_s"]
    )
    microalloy_ppt = (
        float(weights["k_v"]) * features["v_reprecip_wt"]
        + float(weights["k_nb"]) * features["nb_dissolved_wt"]
    )
    soak_regime_shift = float(regime_shift)
    comps = {
        "lattice_friction": lattice_friction,
        "solid_solution": solid_solution,
        "hall_petch_grain": hall_petch_grain,
        "pearlite_ce": pearlite_ce,
        "microalloy_ppt": microalloy_ppt,
        "soak_regime_shift": soak_regime_shift,
    }
    total = float(sum(comps.values()))
    return total, comps


def _evaluate_physics_only(
    chem,
    plate_t_mm,
    t_soak_c,
    residence_min,
    calib,
    heating_rate_min_per_mm,
    cooling_mode,
    ac1_c,
    ac3_c,
    ce_iiw,
):
    """Run thermal -> precipitation -> grain -> constitutive equations for one (T_soak, res_min)."""
    therm = thermal_profile(
        plate_t_mm,
        t_soak_c,
        residence_min,
        heating_rate_min_per_mm=heating_rate_min_per_mm,
        cooling_mode=cooling_mode,
    )
    precip = normalizing_precipitates(chem, t_soak_c, therm["t_eff_soak_min"])
    grain = grain_evolution(
        plate_t_mm,
        t_soak_c,
        therm["t_eff_soak_min"],
        ac1_c,
        ac3_c,
        precip,
        therm["cr_air_k_s"],
        ce_iiw=ce_iiw,
    )

    mn = max(_chem(chem, "Mn"), 0.0)
    si = max(_chem(chem, "Si"), 0.0)
    ni = max(_chem(chem, "Ni"), 0.0)
    cr = max(_chem(chem, "Cr"), 0.0)
    cu = max(_chem(chem, "Cu"), 0.0)
    mo = max(_chem(chem, "Mo"), 0.0)
    ti_sol = float(precip["elements"]["Ti"]["dissolved_wt"])
    n_free = float(precip["free_n_wt"])

    sigma_ss = float(_physics.sigma_ss(mn, si, ni, cr, cu, ti_sol, mo, n_free))
    d_alpha_um = float(grain["d_alpha_um"])
    inv_sqrt_d_mm = float((d_alpha_um / 1000.0) ** -0.5)

    features = {
        "sigma_ss": sigma_ss,
        "inv_sqrt_d_mm": inv_sqrt_d_mm,
        "ce_iiw": float(ce_iiw),
        "c_wt": max(_chem(chem, "C"), 0.0),
        "v_reprecip_wt": float(precip["v_reprecip_wt"]),
        "nb_dissolved_wt": float(precip["elements"]["Nb"]["dissolved_wt"]),
        "cr_air_k_s": float(therm["cr_air_k_s"]),
        "d_alpha_um": d_alpha_um,
        "d_gamma_final_um": float(grain["d_gamma_final_um"]),
    }

    adj = _regime_adjustments(
        grain["regime"],
        float(t_soak_c),
        therm["t_eff_soak_min"],
        ac1_c,
        ac3_c,
        grain["d_gamma_final_um"],
        grain["d_lim_um"],
    )
    ys_phys, ys_comps = _constitutive_components(features, calib["YS_AN"], adj["YS_AN"])
    ts_phys, ts_comps = _constitutive_components(features, calib["TS_AN"], adj["TS_AN"])
    el_raw, _ = _constitutive_components(features, calib["EL_AN"], adj["EL_AN"])
    el_phys = float(np.clip(el_raw, 5.0, 55.0))

    return {
        "thermal": therm,
        "precipitates": precip,
        "grain": grain,
        "features": features,
        "ys_phys_mpa": ys_phys,
        "ts_phys_mpa": ts_phys,
        "el_phys_pct": el_phys,
        "ys_components": ys_comps,
        "ts_components": ts_comps,
    }


def extract_constitutive_features(
    raw,
    t_soak_c=None,
    residence_min=None,
    calib=None,
    heating_rate_min_per_mm=1.2,
    cooling_mode="still_air",
):
    """Extract constitutive predictor features for calibration scripts (e.g. Modal tuning)."""
    cal = calib if calib is not None else load_calibration()
    plate_t = _extract_thickness_mm(raw)
    t_soak = float(cal.get("t_soak_ref_c", 910.0) if t_soak_c is None else t_soak_c)
    t_heat_approx = float(heating_rate_min_per_mm) * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
    res_min = (
        t_heat_approx + float(cal.get("soak_min_per_mm_ref", 0.6)) * plate_t
        if residence_min is None
        else float(residence_min)
    )
    ac1 = ac1_temp(raw)
    ac3 = ac3_temp(raw)
    ce = _ce_iiw_from_chem(raw)
    ev = _evaluate_physics_only(
        raw, plate_t, t_soak, res_min, cal, heating_rate_min_per_mm, cooling_mode, ac1, ac3, ce
    )
    return ev["features"]


def _extract_ml_entry(ml_preds, key):
    """Read (pred, ci_low, ci_high) cleanly from either float or prediction-dict entries."""
    if not ml_preds or key not in ml_preds:
        return None, None, None
    entry = ml_preds[key]
    if isinstance(entry, dict):
        val = entry.get("pred")
        lo = entry.get("ci_low")
        hi = entry.get("ci_high")
    else:
        val, lo, hi = entry, None, None
    try:
        pred_f = float(val) if val is not None else float("nan")
    except (TypeError, ValueError):
        return None, None, None
    if not np.isfinite(pred_f):
        return None, None, None
    lo_f = float(lo) if lo is not None and np.isfinite(lo) else None
    hi_f = float(hi) if hi is not None and np.isfinite(hi) else None
    return pred_f, lo_f, hi_f


def _predict_from_precomputed(
    raw,
    plate_t,
    ac1,
    ac3,
    ce,
    t_soak,
    res_min,
    phys_ref,
    ml_preds,
    cal,
    heating_rate_min_per_mm,
    cooling_mode,
):
    """Shared assembly for single prediction and 2D process-window grid."""
    ev = _evaluate_physics_only(
        raw, plate_t, t_soak, res_min, cal, heating_rate_min_per_mm, cooling_mode, ac1, ac3, ce
    )

    ys_phys = ev["ys_phys_mpa"]
    ts_phys = ev["ts_phys_mpa"]
    el_phys = ev["el_phys_pct"]
    ys_ref = phys_ref["ys_phys_mpa"]
    ts_ref = phys_ref["ts_phys_mpa"]
    el_ref = phys_ref["el_phys_pct"]

    ys_an, ys_an_lo, ys_an_hi = _extract_ml_entry(ml_preds, "YS_AN")
    ts_an, ts_an_lo, ts_an_hi = _extract_ml_entry(ml_preds, "TS_AN")
    el_an, el_an_lo, el_an_hi = _extract_ml_entry(ml_preds, "EL_AN")
    ys_bn, _, _ = _extract_ml_entry(ml_preds, "YS_BN")
    ts_bn, _, _ = _extract_ml_entry(ml_preds, "TS_BN")
    el_bn, _, _ = _extract_ml_entry(ml_preds, "EL_BN")
    imp0, imp0_lo, imp0_hi = _extract_ml_entry(ml_preds, "Impact_0C")

    anchored = any(v is not None for v in (ys_an, ts_an, el_an))
    subcritical_tempered = False

    ys_delta = ys_phys - ys_ref
    ts_delta = ts_phys - ts_ref
    el_delta = el_phys - el_ref

    if t_soak < ac1 and anchored:
        # Subcritical tempering only partially relieves as-rolled excess strength above AN
        t_eff = ev["thermal"]["t_eff_soak_min"]
        temp_frac = float(np.clip(0.25 + 0.35 * max(t_soak - 550.0, 0.0) / max(ac1 - 550.0, 50.0), 0.15, 0.65))
        time_mod = float(np.clip(0.75 + 0.25 * np.log10(max(t_eff, 1.0) / 10.0 + 1.0), 0.70, 1.10))
        temper_frac = float(np.clip(temp_frac * time_mod, 0.15, 0.70))

        if ys_an is not None and ys_bn is not None and ys_bn > ys_an:
            ys_hybrid = float(ys_bn - temper_frac * (ys_bn - ys_an))
            subcritical_tempered = True
        else:
            ys_hybrid = float(ys_an + ys_delta) if ys_an is not None else ys_phys

        if ts_an is not None and ts_bn is not None and ts_bn > ts_an:
            ts_hybrid = float(ts_bn - temper_frac * (ts_bn - ts_an))
            subcritical_tempered = True
        else:
            ts_hybrid = float(ts_an + ts_delta) if ts_an is not None else ts_phys

        if el_an is not None and el_bn is not None and el_an > el_bn:
            el_hybrid = float(el_bn + temper_frac * (el_an - el_bn))
            subcritical_tempered = True
        else:
            el_hybrid = float(el_an + el_delta) if el_an is not None else el_phys
    else:
        ys_hybrid = float(ys_an + ys_delta) if ys_an is not None else ys_phys
        ts_hybrid = float(ts_an + ts_delta) if ts_an is not None else ts_phys
        el_hybrid = float(el_an + el_delta) if el_an is not None else el_phys

    el_hybrid = float(np.clip(el_hybrid, 4.0, 60.0))

    # Propagate conformal half-width around hybrid estimate (widened 20% outside full normalizing)
    ci_scale = 1.0 if ev["grain"]["regime"] == "FULL_NORMALIZING" else 1.2
    def _band(val, ml_val, ml_lo, ml_hi, floor_zero=False):
        if ml_val is None or ml_lo is None or ml_hi is None:
            return None, None
        hw = 0.5 * max(0.0, ml_hi - ml_lo) * ci_scale
        lo = val - hw
        if floor_zero:
            lo = max(0.0, lo)
        return float(round(lo, 2)), float(round(val + hw, 2))

    ys_ci_lo, ys_ci_hi = _band(ys_hybrid, ys_an, ys_an_lo, ys_an_hi)
    ts_ci_lo, ts_ci_hi = _band(ts_hybrid, ts_an, ts_an_lo, ts_an_hi)
    el_ci_lo, el_ci_hi = _band(el_hybrid, el_an, el_an_lo, el_an_hi, floor_zero=True)

    d_alpha = ev["grain"]["d_alpha_um"]
    d_alpha_ref = phys_ref["grain"]["d_alpha_um"]
    dbtt_shift_c = float(-11.5 * ((d_alpha / 1000.0) ** -0.5 - (d_alpha_ref / 1000.0) ** -0.5))
    impact_0c_est = (
        float(np.clip(imp0 * np.exp(-0.012 * dbtt_shift_c), 5.0, 400.0))
        if imp0 is not None
        else None
    )

    return {
        "ac1_c": ac1,
        "ac3_c": ac3,
        "ce_iiw": ce,
        "plate_t_mm": plate_t,
        "regime": ev["grain"]["regime"],
        "thermal": ev["thermal"],
        "precipitates": ev["precipitates"],
        "grain": ev["grain"],
        "physics": {
            "ys_phys_mpa": float(ys_phys),
            "ts_phys_mpa": float(ts_phys),
            "el_phys_pct": float(el_phys),
            "ys_ref_mpa": float(ys_ref),
            "ts_ref_mpa": float(ts_ref),
            "el_ref_pct": float(el_ref),
            "features": ev["features"],
        },
        "hybrid": {
            "ys_hybrid_mpa": float(ys_hybrid),
            "ts_hybrid_mpa": float(ts_hybrid),
            "el_hybrid_pct": float(el_hybrid),
            "ys_delta_mpa": float(ys_hybrid - ys_an) if ys_an is not None else 0.0,
            "ts_delta_mpa": float(ts_hybrid - ts_an) if ts_an is not None else 0.0,
            "el_delta_pct": float(el_hybrid - el_an) if el_an is not None else 0.0,
            "anchored_to_ml": anchored,
            "subcritical_tempered": subcritical_tempered,
            "ys_ci_low": ys_ci_lo,
            "ys_ci_high": ys_ci_hi,
            "ts_ci_low": ts_ci_lo,
            "ts_ci_high": ts_ci_hi,
            "el_ci_low": el_ci_lo,
            "el_ci_high": el_ci_hi,
        },
        "ml_baseline": {
            "YS_BN": ys_bn,
            "TS_BN": ts_bn,
            "EL_BN": el_bn,
            "YS_AN": ys_an,
            "TS_AN": ts_an,
            "EL_AN": el_an,
            "Impact_0C": imp0,
            "ci": {
                "YS_AN": (ys_an_lo, ys_an_hi),
                "TS_AN": (ts_an_lo, ts_an_hi),
                "EL_AN": (el_an_lo, el_an_hi),
                "Impact_0C": (imp0_lo, imp0_hi),
            },
        },
        "strength_components": ev["ys_components"],
        "ts_strength_components": ev["ts_components"],
        "dbtt_shift_c": dbtt_shift_c,
        "impact_0c_est_j": impact_0c_est,
        "calibration_meta": {
            "source": cal.get("source", "default_literature_baseline"),
            "t_soak_ref_c": float(cal.get("t_soak_ref_c", 910.0)),
            "soak_min_per_mm_ref": float(cal.get("soak_min_per_mm_ref", 0.6)),
            "timestamp": cal.get("timestamp"),
            "n_schedules": cal.get("n_schedules"),
            "metrics": cal.get("metrics"),
        },
    }


def predict_normalizing_properties(
    raw,
    t_soak_c,
    residence_min,
    ml_preds=None,
    calib=None,
    heating_rate_min_per_mm=1.2,
    cooling_mode="still_air",
):
    """Predict normalizing microstructure, standalone physics properties, and ML-anchored hybrid."""
    cal = calib if calib is not None else load_calibration()
    plate_t = _extract_thickness_mm(raw)
    ac1 = ac1_temp(raw)
    ac3 = ac3_temp(raw)
    ce = _ce_iiw_from_chem(raw)

    t_heat = float(heating_rate_min_per_mm) * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
    t_soak_ref = float(cal.get("t_soak_ref_c", 910.0))
    t_res_ref = t_heat + float(cal.get("soak_min_per_mm_ref", 0.6)) * plate_t

    phys_ref = _evaluate_physics_only(
        raw, plate_t, t_soak_ref, t_res_ref, cal, heating_rate_min_per_mm, cooling_mode, ac1, ac3, ce
    )
    return _predict_from_precomputed(
        raw,
        plate_t,
        ac1,
        ac3,
        ce,
        float(t_soak_c),
        float(residence_min),
        phys_ref,
        ml_preds,
        cal,
        heating_rate_min_per_mm,
        cooling_mode,
    )


# ----------------------------------------------------------------------------- 7. 2D process window
def normalizing_process_window(
    raw,
    ml_preds=None,
    calib=None,
    t_min=780.0,
    t_max=1020.0,
    r_min=10.0,
    r_max=180.0,
    nt=21,
    nr=21,
    heating_rate_min_per_mm=1.2,
    cooling_mode="still_air",
):
    """Sweep 2D (T_soak x residence_min) process window returning (nr, nt) contour grids."""
    cal = calib if calib is not None else load_calibration()
    plate_t = _extract_thickness_mm(raw)
    ac1 = ac1_temp(raw)
    ac3 = ac3_temp(raw)
    ce = _ce_iiw_from_chem(raw)

    rate = max(float(heating_rate_min_per_mm), 0.1)
    t_heat_min = rate * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
    t_soak_ref = float(cal.get("t_soak_ref_c", 910.0))
    t_res_ref = t_heat_min + float(cal.get("soak_min_per_mm_ref", 0.6)) * plate_t

    phys_ref = _evaluate_physics_only(
        raw, plate_t, t_soak_ref, t_res_ref, cal, rate, cooling_mode, ac1, ac3, ce
    )

    nt_i = max(int(nt), 2)
    nr_i = max(int(nr), 2)
    t_grid = np.linspace(float(t_min), float(t_max), nt_i)
    r_grid = np.linspace(float(r_min), float(r_max), nr_i)

    shape = (nr_i, nt_i)
    ys_grid = np.empty(shape, dtype=float)
    ts_grid = np.empty(shape, dtype=float)
    el_grid = np.empty(shape, dtype=float)
    d_alpha_grid = np.empty(shape, dtype=float)
    d_gamma_grid = np.empty(shape, dtype=float)
    pinning_ratio_grid = np.empty(shape, dtype=float)
    regime_grid = np.empty(shape, dtype="U20")

    for i, res_val in enumerate(r_grid):
        for j, t_val in enumerate(t_grid):
            pred = _predict_from_precomputed(
                raw,
                plate_t,
                ac1,
                ac3,
                ce,
                float(t_val),
                float(res_val),
                phys_ref,
                ml_preds,
                cal,
                rate,
                cooling_mode,
            )
            ys_grid[i, j] = pred["hybrid"]["ys_hybrid_mpa"]
            ts_grid[i, j] = pred["hybrid"]["ts_hybrid_mpa"]
            el_grid[i, j] = pred["hybrid"]["el_hybrid_pct"]
            d_alpha_grid[i, j] = pred["grain"]["d_alpha_um"]
            d_gamma_grid[i, j] = pred["grain"]["d_gamma_final_um"]
            pinning_ratio_grid[i, j] = pred["grain"]["pinning_ratio"]
            regime_grid[i, j] = pred["regime"]

    return {
        "t_grid": t_grid,
        "r_grid": r_grid,
        "ys_grid": ys_grid,
        "ts_grid": ts_grid,
        "el_grid": el_grid,
        "d_alpha_grid": d_alpha_grid,
        "d_gamma_grid": d_gamma_grid,
        "pinning_ratio_grid": pinning_ratio_grid,
        "regime_grid": regime_grid,
        "ac1_c": float(ac1),
        "ac3_c": float(ac3),
        "t_heat_min": float(t_heat_min),
    }
