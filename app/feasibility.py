"""Ranged Process Feasibility (Monte Carlo) engine and interactive Streamlit tab.

Evaluates joint multi-property feasibility (YS_BN, TS_BN, EL_BN) across user-defined
process and geometry ranges using stratified Latin Hypercube Sampling and a single
vectorized call to ``streamlit_app.score_frame``.

Supports both Heavy Plate (``Plate``) and Steckel Coil (``HRC``) routes cleanly:
- Plate: ``TB_Thickness_mm = TB_Ratio * Plate_T_mm`` per sample, integer-uniform
  ``ACC_Laminar_Header`` counts, and physically coupled ``ACC_Temp_Post_Laminar``
  via ``physics.post_laminar_temp(FRT, headers, plate_t)``.
- HRC: ``TB_Thick_mm = TB_Ratio * Coil_T_mm`` per sample and symmetric coiling
  coupling ``Temp_CT_Bot_C = Temp_CT_Top_C`` when bottom CT is not separately ranged.

Import invariant: NEVER import ``modal`` or ``streamlit_app`` at module level; resolve the
active app module lazily through ``app_ref.get_app_module()``.
"""

import html as html_lib
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import physics  # noqa: E402
import spec_checker as sc  # noqa: E402

# ----------------------------------------------------------------------------- palette & metadata
INK = "#0b0d10"
ACCENT = "#1746c8"
HAIR = "#e5e7eb"
GRID = "#f1f2f4"
MUTED = "#6b7280"
SEMANTIC_OK = "#065f46"
SEMANTIC_WARN = "#92400e"
SEMANTIC_BAD = "#991b1b"
WARN_LINE = "#b45309"

FAILURE_MODES = ("PASS", "YS fail", "TS fail", "EL fail", "Multi fail")
FAIL_ONLY_MODES = ("YS fail", "TS fail", "EL fail", "Multi fail")
TARGET_KEYS = ("YS_BN", "TS_BN", "EL_BN")

TARGET_META = {
    "YS_BN": ("Yield Strength (YS_BN)", "MPa"),
    "TS_BN": ("Tensile Strength (TS_BN)", "MPa"),
    "EL_BN": ("Elongation (EL_BN)", "%"),
}

PROCESS_PARAMS = {
    "Plate": [
        ("FM_Temp_Before_Rolling", "Entry temperature", "°C", 850.0, 1150.0, 5.0, "continuous"),
        ("FM_Finish_Temp", "Finishing temperature", "°C", 700.0, 1050.0, 5.0, "continuous"),
        ("ACC_Laminar_Header", "Laminar headers", "count", 0.0, 8.0, 1.0, "int"),
    ],
    "HRC": [
        ("Temp_Transfer_Bar_C", "Transfer bar temperature", "°C", 850.0, 1150.0, 5.0, "continuous"),
        ("Temp_FT_Avg_C", "Finishing temperature", "°C", 750.0, 1000.0, 5.0, "continuous"),
        ("Temp_CT_Top_C", "Coiling temperature", "°C", 400.0, 750.0, 5.0, "continuous"),
    ],
}

GEOMETRY_PARAMS = {
    "Plate": [
        ("Plate_T_mm", "Plate thickness", "mm", 6.0, 120.0, 0.5, "continuous"),
        ("Plate_W_mm", "Plate width", "mm", 1000.0, 4000.0, 25.0, "continuous"),
        ("TB_Ratio", "Transfer bar ratio (TB:t)", "×", 1.1, 8.0, 0.05, "continuous"),
    ],
    "HRC": [
        ("Coil_T_mm", "Coil thickness", "mm", 1.8, 16.0, 0.2, "continuous"),
        ("Coil_W_mm", "Coil width", "mm", 700.0, 1600.0, 25.0, "continuous"),
        ("TB_Ratio", "Transfer bar ratio (TB:t)", "×", 1.2, 15.0, 0.1, "continuous"),
    ],
}

DEFAULT_TARGETS = {
    "Plate": {"YS_BN_min": 345.0, "TS_BN_min": 450.0, "TS_BN_max": 0.0, "EL_BN_min": 18.0},
    "HRC": {"YS_BN_min": 245.0, "TS_BN_min": 400.0, "TS_BN_max": 0.0, "EL_BN_min": 21.0},
}


def _app():
    """Resolve the active streamlit_app module lazily without circular imports."""
    import app_ref  # noqa: PLC0415

    return app_ref.get_app_module()


def _rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def _prune_widget_key(key, options):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        out = float(val)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _infer_tb_ratio(mill, raw):
    """Resolve transfer bar ratio from raw dict with legacy thickness fallback."""
    ratio = _safe_float((raw or {}).get("TB_Ratio"))
    if ratio is not None and ratio > 0.0:
        return ratio
    if mill == "Plate":
        t_val = max(_safe_float((raw or {}).get("Plate_T_mm"), 16.0), 1.0)
        tb_val = _safe_float((raw or {}).get("TB_Thickness_mm"), 41.0)
        return tb_val / t_val
    t_val = max(_safe_float((raw or {}).get("Coil_T_mm"), 6.0), 1.0)
    tb_val = _safe_float((raw or {}).get("TB_Thick_mm"), 24.8)
    return tb_val / t_val


def _normalize_range(val, fallback, is_int=False):
    """Normalize a scalar or (lo, hi) range into ordered numeric bounds."""
    if isinstance(val, (tuple, list, np.ndarray)):
        lo = _safe_float(val[0], fallback) if len(val) > 0 else float(fallback)
        hi = _safe_float(val[1], lo) if len(val) > 1 else lo
    else:
        lo = _safe_float(val, fallback)
        hi = lo
    if lo > hi:
        lo, hi = hi, lo
    if is_int:
        ilo = max(0, int(round(lo)))
        ihi = max(ilo, int(round(hi)))
        return (ilo, ihi)
    return (float(lo), float(hi))


def _normalize_targets(targets):
    """Extract canonical BN target thresholds; TS_BN_max <= 0 or None is unconstrained."""
    t = dict(targets or {})
    ys_min = _safe_float(t.get("YS_BN_min"), None)
    ts_min = _safe_float(t.get("TS_BN_min"), None)
    ts_max = _safe_float(t.get("TS_BN_max"), None)
    el_min = _safe_float(t.get("EL_BN_min"), None)
    if ts_max is not None and ts_max <= 0.0:
        ts_max = None
    return {
        "YS_BN_min": ys_min,
        "TS_BN_min": ts_min,
        "TS_BN_max": ts_max,
        "EL_BN_min": el_min,
    }


def _lhs_unit(n, d, rng):
    """Stratified Latin Hypercube samples in [0, 1) with shape (n, d)."""
    out = np.empty((n, d), dtype=float)
    for j in range(d):
        perm = rng.permutation(n)
        jitter = rng.random(n)
        out[:, j] = (perm + jitter) / float(n)
    return out


def _param_label_map(mill):
    labels = {}
    for key, lbl, _unit, _lo, _hi, _step, _kind in PROCESS_PARAMS[mill] + GEOMETRY_PARAMS[mill]:
        labels[key] = lbl
    labels["Temp_CT_Bot_C"] = "Coiling temperature, bottom"
    labels["ACC_Temp_Post_Laminar"] = "Post-laminar temperature"
    return labels


def _empty_result(mill, engine, n_samples, include_conformal, targets, process_ranges, geometry_ranges, error):
    return {
        "error": error,
        "mill": mill,
        "engine": engine,
        "n_samples": int(n_samples),
        "include_conformal": bool(include_conformal),
        "targets": targets,
        "process_ranges": process_ranges,
        "geometry_ranges": geometry_ranges,
        "summary": {
            "joint_pass_rate_pct": 0.0,
            "conformal_pass_rate_pct": 0.0,
            "point_pass_rate_pct": 0.0,
            "ys_pass_rate_pct": 0.0,
            "ts_pass_rate_pct": 0.0,
            "el_pass_rate_pct": 0.0,
            "dominant_failure": "None",
            "failure_counts": {m: 0 for m in FAILURE_MODES},
            "target_quantiles": {},
            "conformal_half_widths": {},
        },
        "sensitivity": [],
        "samples_df": pd.DataFrame(),
    }


def run_ranged_feasibility(
    mill,
    raw,
    process_ranges,
    geometry_ranges,
    targets,
    samples=1000,
    engine="Champion GBDT (Recommended)",
    include_conformal=False,
    couple_post=True,
    seed=42,
):
    """Monte Carlo feasibility study across ranged process and geometry windows.

    Draws ``samples`` stratified Latin Hypercube samples using ``np.random.default_rng(seed)``
    across the 6 ranged variables (plus any extra explicitly ranged variables), propagates
    coupled geometry (``TB_Thickness_mm = TB_Ratio * Plate_T_mm`` on Plate;
    ``TB_Thick_mm = TB_Ratio * Coil_T_mm`` on HRC) and cooling physics, scores all rows in
    ONE vectorized call to ``app.score_frame(mill, frame, engine=engine)``, and computes
    point & conservative 95% conformal pass rates, failure breakdown, target quantiles, and
    parameter sensitivity.
    """
    norm_targets = _normalize_targets(targets)
    if mill not in ("Plate", "HRC"):
        return _empty_result(
            mill, engine, 0, include_conformal, norm_targets,
            dict(process_ranges or {}), dict(geometry_ranges or {}),
            f"Unsupported route: {mill!r}",
        )

    app_mod = _app()
    base_raw = dict(app_mod.defaults(mill))
    if raw:
        for k, v in dict(raw).items():
            if v is not None:
                base_raw[k] = v
    base_raw["TB_Ratio"] = _infer_tb_ratio(mill, base_raw)

    n = max(1, int(samples))
    proc_in = dict(process_ranges or {})
    geom_in = dict(geometry_ranges or {})

    proc_norm = {}
    proc_kinds = {}
    for key, _lbl, _unit, f_lo, _f_hi, _step, kind in PROCESS_PARAMS[mill]:
        fb = float(base_raw.get(key, f_lo))
        proc_norm[key] = _normalize_range(proc_in.get(key, (fb, fb)), fb, is_int=(kind == "int"))
        proc_kinds[key] = kind
    for extra_key, extra_val in proc_in.items():
        if extra_key not in proc_norm:
            fb = _safe_float(base_raw.get(extra_key), 0.0)
            proc_norm[extra_key] = _normalize_range(extra_val, fb, is_int=False)
            proc_kinds[extra_key] = "continuous"

    geom_norm = {}
    for key, _lbl, _unit, f_lo, _f_hi, _step, _kind in GEOMETRY_PARAMS[mill]:
        fb = float(base_raw.get(key, f_lo))
        geom_norm[key] = _normalize_range(geom_in.get(key, (fb, fb)), fb, is_int=False)
    for extra_key, extra_val in geom_in.items():
        if extra_key not in geom_norm:
            fb = _safe_float(base_raw.get(extra_key), 0.0)
            geom_norm[extra_key] = _normalize_range(extra_val, fb, is_int=False)

    # Canonical sampling list: 3 process + any extra process + 3 geometry + any extra geometry
    sample_specs = []
    for k, bounds in proc_norm.items():
        sample_specs.append((k, bounds, proc_kinds.get(k, "continuous")))
    for k, bounds in geom_norm.items():
        sample_specs.append((k, bounds, "continuous"))

    rng = np.random.default_rng(seed)
    u_mat = _lhs_unit(n, len(sample_specs), rng)

    frame = pd.DataFrame([dict(base_raw) for _ in range(n)])
    for j, (key, (lo, hi), kind) in enumerate(sample_specs):
        u = u_mat[:, j]
        if kind == "int":
            ilo, ihi = int(lo), int(hi)
            if ihi <= ilo:
                arr = np.full(n, ilo, dtype=int)
            else:
                arr = np.minimum(ihi, ilo + np.floor(u * (ihi - ilo + 1)).astype(int))
            frame[key] = arr
        else:
            if hi <= lo:
                frame[key] = np.full(n, float(lo), dtype=float)
            else:
                frame[key] = float(lo) + u * (float(hi) - float(lo))

    if mill == "Plate":
        plate_t = np.asarray(frame["Plate_T_mm"], dtype=float)
        tb_ratio = np.asarray(frame["TB_Ratio"], dtype=float)
        frame["TB_Thickness_mm"] = tb_ratio * plate_t
        headers = np.asarray(frame["ACC_Laminar_Header"], dtype=float)
        frame["ACC_Headers_n"] = headers
        frame["ACC_Active"] = (headers > 0).astype(float)
        if couple_post:
            frt = np.asarray(frame["FM_Finish_Temp"], dtype=float)
            frame["ACC_Temp_Post_Laminar"] = physics.post_laminar_temp(frt, headers, plate_t)
    else:
        coil_t = np.asarray(frame["Coil_T_mm"], dtype=float)
        tb_ratio = np.asarray(frame["TB_Ratio"], dtype=float)
        frame["TB_Thick_mm"] = tb_ratio * coil_t
        if "Temp_CT_Bot_C" not in proc_in:
            frame["Temp_CT_Bot_C"] = np.asarray(frame["Temp_CT_Top_C"], dtype=float)

    try:
        scored = app_mod.score_frame(mill, frame, engine=engine)
    except Exception as exc:  # noqa: BLE001
        return _empty_result(
            mill, engine, n, include_conformal, norm_targets, proc_norm, geom_norm,
            f"Model scoring failed with {engine}: {type(exc).__name__}: {exc}",
        )

    pred_arrays = {}
    for tgt in TARGET_KEYS:
        if tgt in scored.columns:
            pred_arrays[tgt] = pd.to_numeric(scored[tgt], errors="coerce").to_numpy(dtype=float)
        else:
            pred_arrays[tgt] = np.full(n, np.nan, dtype=float)

    # Validate that every constrained target actually returned finite predictions
    constrained_targets = []
    if norm_targets["YS_BN_min"] is not None:
        constrained_targets.append("YS_BN")
    if norm_targets["TS_BN_min"] is not None or norm_targets["TS_BN_max"] is not None:
        constrained_targets.append("TS_BN")
    if norm_targets["EL_BN_min"] is not None:
        constrained_targets.append("EL_BN")

    missing_targets = [t for t in constrained_targets if not np.isfinite(pred_arrays[t]).any()]
    if missing_targets:
        return _empty_result(
            mill, engine, n, include_conformal, norm_targets, proc_norm, geom_norm,
            f"Target(s) {', '.join(missing_targets)} could not be scored with {engine}.",
        )

    halves = {t: max(float(app_mod.get_target_interval_half(mill, t)), 1e-6) for t in TARGET_KEYS}
    ys = pred_arrays["YS_BN"]
    ts = pred_arrays["TS_BN"]
    el = pred_arrays["EL_BN"]

    ys_min = norm_targets["YS_BN_min"]
    ts_min = norm_targets["TS_BN_min"]
    ts_max = norm_targets["TS_BN_max"]
    el_min = norm_targets["EL_BN_min"]

    pt_ys = (np.isfinite(ys) & (ys >= ys_min)) if ys_min is not None else np.ones(n, dtype=bool)
    cf_ys = (np.isfinite(ys) & ((ys - halves["YS_BN"]) >= ys_min)) if ys_min is not None else np.ones(n, dtype=bool)

    pt_ts = np.ones(n, dtype=bool)
    cf_ts = np.ones(n, dtype=bool)
    if ts_min is not None:
        pt_ts &= np.isfinite(ts) & (ts >= ts_min)
        cf_ts &= np.isfinite(ts) & ((ts - halves["TS_BN"]) >= ts_min)
    if ts_max is not None:
        pt_ts &= np.isfinite(ts) & (ts <= ts_max)
        cf_ts &= np.isfinite(ts) & ((ts + halves["TS_BN"]) <= ts_max)

    pt_el = (np.isfinite(el) & (el >= el_min)) if el_min is not None else np.ones(n, dtype=bool)
    cf_el = (np.isfinite(el) & ((el - halves["EL_BN"]) >= el_min)) if el_min is not None else np.ones(n, dtype=bool)

    pt_joint = pt_ys & pt_ts & pt_el
    cf_joint = cf_ys & cf_ts & cf_el

    if include_conformal:
        pass_ys, pass_ts, pass_el, joint_pass = cf_ys, cf_ts, cf_el, cf_joint
    else:
        pass_ys, pass_ts, pass_el, joint_pass = pt_ys, pt_ts, pt_el, pt_joint

    pt_ratios = []
    cf_ratios = []
    if ys_min is not None:
        r_ys = np.where(np.isfinite(ys), (ys - ys_min) / halves["YS_BN"], -np.inf)
        pt_ratios.append(r_ys)
        cf_ratios.append(r_ys - 1.0)
    if ts_min is not None:
        r_ts_lo = np.where(np.isfinite(ts), (ts - ts_min) / halves["TS_BN"], -np.inf)
        pt_ratios.append(r_ts_lo)
        cf_ratios.append(r_ts_lo - 1.0)
    if ts_max is not None:
        r_ts_hi = np.where(np.isfinite(ts), (ts_max - ts) / halves["TS_BN"], -np.inf)
        pt_ratios.append(r_ts_hi)
        cf_ratios.append(r_ts_hi - 1.0)
    if el_min is not None:
        r_el = np.where(np.isfinite(el), (el - el_min) / halves["EL_BN"], -np.inf)
        pt_ratios.append(r_el)
        cf_ratios.append(r_el - 1.0)

    if pt_ratios:
        active_ratios = cf_ratios if include_conformal else pt_ratios
        worst_margin_ratio = np.min(np.vstack(active_ratios), axis=0)
    else:
        worst_margin_ratio = np.zeros(n, dtype=float)

    fail_ys = ~pass_ys
    fail_ts = ~pass_ts
    fail_el = ~pass_el
    n_fail = fail_ys.astype(int) + fail_ts.astype(int) + fail_el.astype(int)
    failure_mode = np.where(
        n_fail == 0,
        "PASS",
        np.where(
            n_fail > 1,
            "Multi fail",
            np.where(fail_ys, "YS fail", np.where(fail_ts, "TS fail", "EL fail")),
        ),
    )

    failure_counts = {mode: int(np.sum(failure_mode == mode)) for mode in FAILURE_MODES}
    max_fail_cnt = max(failure_counts[m] for m in FAIL_ONLY_MODES)
    dominant_failure = (
        "None"
        if max_fail_cnt == 0
        else max(FAIL_ONLY_MODES, key=lambda m: failure_counts[m])
    )

    target_quantiles = {}
    for tgt in TARGET_KEYS:
        arr = pred_arrays[tgt]
        fin = arr[np.isfinite(arr)]
        if fin.size > 0:
            target_quantiles[tgt] = {
                "p05": float(np.percentile(fin, 5)),
                "p50": float(np.percentile(fin, 50)),
                "p95": float(np.percentile(fin, 95)),
                "mean": float(np.mean(fin)),
                "std": float(np.std(fin, ddof=1) if fin.size > 1 else 0.0),
            }
        else:
            target_quantiles[tgt] = {
                "p05": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
            }

    joint_rate_pct = float(np.mean(joint_pass) * 100.0)
    summary = {
        "joint_pass_rate_pct": joint_rate_pct,
        "conformal_pass_rate_pct": float(np.mean(cf_joint) * 100.0),
        "point_pass_rate_pct": float(np.mean(pt_joint) * 100.0),
        "ys_pass_rate_pct": float(np.mean(pass_ys) * 100.0),
        "ts_pass_rate_pct": float(np.mean(pass_ts) * 100.0),
        "el_pass_rate_pct": float(np.mean(pass_el) * 100.0),
        "dominant_failure": dominant_failure,
        "failure_counts": failure_counts,
        "target_quantiles": target_quantiles,
        "conformal_half_widths": halves,
    }

    # Build clean samples_df keeping ranged inputs, coupled geometry/temp, predictions & flags
    ordered_cols = list(proc_norm.keys()) + list(geom_norm.keys())
    if mill == "Plate":
        for c in ("TB_Thickness_mm", "ACC_Temp_Post_Laminar"):
            if c not in ordered_cols:
                ordered_cols.append(c)
    else:
        for c in ("TB_Thick_mm", "Temp_CT_Bot_C"):
            if c not in ordered_cols:
                ordered_cols.append(c)

    samples_df = frame[ordered_cols].copy()
    samples_df["YS_BN"] = ys
    samples_df["TS_BN"] = ts
    samples_df["EL_BN"] = el
    samples_df["pass_ys"] = pass_ys
    samples_df["pass_ts"] = pass_ts
    samples_df["pass_el"] = pass_el
    samples_df["joint_pass"] = joint_pass
    samples_df["conformal_joint_pass"] = cf_joint
    samples_df["failure_mode"] = failure_mode
    samples_df["worst_margin_ratio"] = worst_margin_ratio

    # Sensitivity analysis across ranged variables
    label_map = _param_label_map(mill)
    sensitivity = []
    fin_margin = np.isfinite(worst_margin_ratio)
    for key, _bounds, _kind in sample_specs:
        x = np.asarray(samples_df[key], dtype=float)
        if np.ptp(x) <= 1e-12:
            corr = 0.0
            lo_pct = joint_rate_pct
            hi_pct = joint_rate_pct
        else:
            valid = fin_margin & np.isfinite(x)
            if np.sum(valid) >= 2:
                xv = x[valid]
                yv = worst_margin_ratio[valid]
                if np.std(xv) > 1e-12 and np.std(yv) > 1e-12:
                    xc = xv - np.mean(xv)
                    yc = yv - np.mean(yv)
                    corr = float(np.clip(np.dot(xc, yc) / np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)), -1.0, 1.0))
                else:
                    corr = 0.0
            else:
                corr = 0.0

            med = float(np.median(x))
            lo_mask = x <= med
            hi_mask = x > med
            if not hi_mask.any() or not lo_mask.any():
                lo_mask = x < med
                hi_mask = x >= med
            if lo_mask.any() and hi_mask.any():
                lo_pct = float(np.mean(joint_pass[lo_mask]) * 100.0)
                hi_pct = float(np.mean(joint_pass[hi_mask]) * 100.0)
            else:
                lo_pct = joint_rate_pct
                hi_pct = joint_rate_pct

        sensitivity.append({
            "param": key,
            "label": label_map.get(key, key),
            "corr_worst_margin": corr,
            "low_half_pass_pct": lo_pct,
            "high_half_pass_pct": hi_pct,
            "pass_lift_pct": hi_pct - lo_pct,
        })

    sensitivity.sort(
        key=lambda d: (abs(d["pass_lift_pct"]), abs(d["corr_worst_margin"])),
        reverse=True,
    )

    return {
        "error": None,
        "mill": mill,
        "engine": engine,
        "n_samples": n,
        "include_conformal": bool(include_conformal),
        "targets": norm_targets,
        "process_ranges": proc_norm,
        "geometry_ranges": geom_norm,
        "summary": summary,
        "sensitivity": sensitivity,
        "samples_df": samples_df,
    }


def extract_spec_targets(spec, grade, thickness, mill="Plate"):
    """Read YS_BN, TS_BN, and EL_BN mechanical limits for a specification tier at ``thickness``."""
    row, info = sc.find_matching_spec_row(spec, grade, float(thickness))
    if row is None:
        return None, info
    mech = sc.evaluate_mechanical(row, {}, mill, "AR")
    targets = {"YS_BN_min": None, "TS_BN_min": None, "TS_BN_max": None, "EL_BN_min": None}
    for chk in mech.get("checks", []):
        k = chk.get("key")
        if k == "YS_BN" and chk.get("limit_min") is not None:
            targets["YS_BN_min"] = float(chk["limit_min"])
        elif k == "TS_BN":
            if chk.get("limit_min") is not None:
                targets["TS_BN_min"] = float(chk["limit_min"])
            if chk.get("limit_max") is not None:
                targets["TS_BN_max"] = float(chk["limit_max"])
        elif k == "EL_BN" and chk.get("limit_min") is not None:
            targets["EL_BN_min"] = float(chk["limit_min"])
    return targets, info


# ----------------------------------------------------------------------------- UI helpers & charts
def _spec_bounds_for_param(app_mod, mill, key, fallback_lo, fallback_hi):
    for row in app_mod.SPECS.get(mill, ()):
        if len(row) >= 6 and row[1] == key:
            lo = _safe_float(row[4], fallback_lo)
            hi = _safe_float(row[5], fallback_hi)
            if hi > lo:
                return lo, hi
    return float(fallback_lo), float(fallback_hi)


def _default_window(mill, key, cur_val, lo, hi, kind):
    if kind == "int":
        cur_i = int(round(cur_val))
        ilo = max(int(round(lo)), cur_i - 4)
        ihi = min(int(round(hi)), cur_i + 2)
        if ihi <= ilo and int(round(hi)) > int(round(lo)):
            ilo, ihi = int(round(lo)), int(round(hi))
        return float(ilo), float(ihi)
    if key in ("FM_Temp_Before_Rolling", "FM_Finish_Temp", "Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C"):
        return max(lo, round(cur_val - 25.0, 1)), min(hi, round(cur_val + 25.0, 1))
    if key in ("Plate_T_mm", "Coil_T_mm"):
        return max(lo, round(cur_val * 0.85, 1)), min(hi, round(cur_val * 1.15, 1))
    if key in ("Plate_W_mm", "Coil_W_mm"):
        return max(lo, round(cur_val - 200.0, -1)), min(hi, round(cur_val + 200.0, -1))
    return max(lo, round(cur_val - 0.5, 2)), min(hi, round(cur_val + 0.5, 2))


def _apply_spec_prefill(mill, chosen_label, lookup, median_t):
    spec, grade = lookup[chosen_label]
    targets, info = extract_spec_targets(spec, grade, median_t, mill)
    if targets is None:
        return None, info
    ys_val = float(targets["YS_BN_min"]) if targets["YS_BN_min"] is not None else 0.0
    ts_min_val = float(targets["TS_BN_min"]) if targets["TS_BN_min"] is not None else 0.0
    ts_max_val = float(targets["TS_BN_max"]) if targets["TS_BN_max"] is not None else 0.0
    el_val = float(targets["EL_BN_min"]) if targets["EL_BN_min"] is not None else 0.0
    st.session_state[f"feas_ys_min_{mill}"] = ys_val
    st.session_state[f"feas_ts_min_{mill}"] = ts_min_val
    st.session_state[f"feas_ts_max_{mill}"] = ts_max_val
    st.session_state[f"feas_el_min_{mill}"] = el_val
    return targets, info


def _feasibility_pill(rate_pct):
    if rate_pct >= 85.0:
        return "FEASIBLE", "ok", SEMANTIC_OK, "#ecfdf5", "#a7f3d0"
    if rate_pct >= 50.0:
        return "MARGINAL", "warn", SEMANTIC_WARN, "#fffbeb", "#fde68a"
    return "HIGH RISK", "bad", SEMANTIC_BAD, "#fef2f2", "#fecaca"


def _render_kpi_strip(res):
    s = res["summary"]
    rate = float(s["joint_pass_rate_pct"])
    pill_text, pill_cls, fg, bg, bdr = _feasibility_pill(rate)
    mode_txt = "95% conformal verdict" if res["include_conformal"] else "point prediction verdict"
    pill_html = (
        f'<span class="pill {pill_cls}" '
        f'style="background:{bg};color:{fg};border:1px solid {bdr};">{pill_text}</span>'
    )
    dom = s["dominant_failure"]
    fail_n = res["n_samples"] - s["failure_counts"].get("PASS", 0)
    dom_sub = "all samples pass" if dom == "None" else f"{fail_n:,} of {res['n_samples']:,} runs fail"

    st.markdown(
        '<div class="facts">'
        f'<div class="lead"><span>Joint Pass Rate</span><b>{rate:.1f}%</b> '
        f'{pill_html}<br><em>{mode_txt} · {res["n_samples"]:,} LHS runs</em></div>'
        f'<div><span>YS Pass Rate</span><b>{s["ys_pass_rate_pct"]:.1f}%</b><em>yield</em></div>'
        f'<div><span>TS Pass Rate</span><b>{s["ts_pass_rate_pct"]:.1f}%</b><em>tensile</em></div>'
        f'<div><span>EL Pass Rate</span><b>{s["el_pass_rate_pct"]:.1f}%</b><em>elongation</em></div>'
        f'<div><span>Conformal-Safe Pass Rate</span><b>{s["conformal_pass_rate_pct"]:.1f}%</b>'
        f'<em>±95% interval buffered</em></div>'
        f'<div><span>Dominant Failure Mode</span><b>{html_lib.escape(dom)}</b>'
        f'<em>{dom_sub}</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _histograms_figure(res):
    df = res["samples_df"]
    targets = res["targets"]
    quantiles = res["summary"]["target_quantiles"]
    pass_mask = df["joint_pass"].to_numpy(dtype=bool)

    titles = []
    for key in TARGET_KEYS:
        name, unit = TARGET_META[key]
        q = quantiles.get(key, {})
        titles.append(
            f"{name} [{unit}]<br>"
            f"<span style='font-size:10px;color:{MUTED};'>"
            f"P05 {q.get('p05', np.nan):.1f} · P50 {q.get('p50', np.nan):.1f} · P95 {q.get('p95', np.nan):.1f}"
            "</span>"
        )

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.07)
    limit_specs = {
        "YS_BN": [(targets.get("YS_BN_min"), "YS Min")],
        "TS_BN": [(targets.get("TS_BN_min"), "TS Min"), (targets.get("TS_BN_max"), "TS Max")],
        "EL_BN": [(targets.get("EL_BN_min"), "EL Min")],
    }

    for col_idx, key in enumerate(TARGET_KEYS, start=1):
        vals = pd.to_numeric(df[key], errors="coerce").to_numpy(dtype=float)
        pass_vals = vals[pass_mask & np.isfinite(vals)]
        fail_vals = vals[(~pass_mask) & np.isfinite(vals)]
        show_leg = col_idx == 1

        if pass_vals.size > 0:
            fig.add_trace(
                go.Histogram(
                    x=pass_vals,
                    nbinsx=26,
                    name="Joint PASS",
                    legendgroup="pass",
                    showlegend=show_leg,
                    marker=dict(color=_rgba(SEMANTIC_OK, 0.52), line=dict(color=SEMANTIC_OK, width=0.6)),
                ),
                row=1,
                col=col_idx,
            )
        if fail_vals.size > 0:
            fig.add_trace(
                go.Histogram(
                    x=fail_vals,
                    nbinsx=26,
                    name="FAIL",
                    legendgroup="fail",
                    showlegend=show_leg,
                    marker=dict(color=_rgba(SEMANTIC_BAD, 0.48), line=dict(color=SEMANTIC_BAD, width=0.6)),
                ),
                row=1,
                col=col_idx,
            )

        q = quantiles.get(key, {})
        for q_key, q_col, q_dash in (("p05", MUTED, "dot"), ("p50", INK, "solid"), ("p95", MUTED, "dot")):
            qv = q.get(q_key)
            if qv is not None and np.isfinite(qv):
                fig.add_vline(
                    x=float(qv),
                    line_color=q_col,
                    line_width=1.4 if q_key == "p50" else 1.1,
                    line_dash=q_dash,
                    row=1,
                    col=col_idx,
                )

        for lim_val, lim_lbl in limit_specs[key]:
            if lim_val is not None and np.isfinite(lim_val):
                fig.add_vline(
                    x=float(lim_val),
                    line_color=WARN_LINE,
                    line_width=2.0,
                    line_dash="dash",
                    annotation_text=f"{lim_lbl} {lim_val:g}",
                    annotation_position="top",
                    annotation_font=dict(size=10, color=WARN_LINE),
                    row=1,
                    col=col_idx,
                )

    fig.update_layout(
        barmode="overlay",
        height=350,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=42, r=18, t=64, b=42),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text="Runs", row=1, col=1)
    return fig


def _scatter_figure(res, x_key, y_key, label_map, unit_map):
    df = res["samples_df"]
    pass_mask = df["joint_pass"].to_numpy(dtype=bool)

    fig = go.Figure()
    hover_tpl = (
        f"{label_map.get(x_key, x_key)}: %{{x:.2f}}<br>"
        f"{label_map.get(y_key, y_key)}: %{{y:.2f}}<br>"
        "YS %{customdata[0]:.1f} MPa · TS %{customdata[1]:.1f} MPa · EL %{customdata[2]:.1f}%<br>"
        "Status: %{customdata[3]} (margin %{customdata[4]:+.2f}×)<extra></extra>"
    )

    for mask, name, color, symbol in (
        (pass_mask, "Joint PASS", SEMANTIC_OK, "circle"),
        (~pass_mask, "FAIL", SEMANTIC_BAD, "x"),
    ):
        sub = df.loc[mask]
        if sub.empty:
            continue
        custom = list(zip(
            sub["YS_BN"].to_numpy(dtype=float),
            sub["TS_BN"].to_numpy(dtype=float),
            sub["EL_BN"].to_numpy(dtype=float),
            sub["failure_mode"].astype(str).tolist(),
            sub["worst_margin_ratio"].to_numpy(dtype=float),
        ))
        fig.add_trace(
            go.Scatter(
                x=sub[x_key],
                y=sub[y_key],
                mode="markers",
                name=f"{name} ({len(sub):,})",
                marker=dict(
                    size=7,
                    color=color,
                    symbol=symbol,
                    opacity=0.72 if name == "Joint PASS" else 0.78,
                    line=dict(color="#ffffff", width=0.5) if symbol == "circle" else dict(width=0),
                ),
                customdata=custom,
                hovertemplate=hover_tpl,
            )
        )

    x_u = unit_map.get(x_key, "")
    y_u = unit_map.get(y_key, "")
    fig.update_layout(
        title=dict(
            text=f"2D Process Window: {label_map.get(x_key, x_key)} vs {label_map.get(y_key, y_key)}",
            font=dict(size=12.5, color=INK),
            x=0,
            xanchor="left",
        ),
        height=410,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=54, r=20, t=48, b=46),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        legend=dict(orientation="h", y=1.08, x=1.0, xanchor="right", font=dict(size=11)),
    )
    fig.update_xaxes(
        title_text=f"{label_map.get(x_key, x_key)} [{x_u}]" if x_u else label_map.get(x_key, x_key),
        showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False,
    )
    fig.update_yaxes(
        title_text=f"{label_map.get(y_key, y_key)} [{y_u}]" if y_u else label_map.get(y_key, y_key),
        showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False,
    )
    return fig


def _failure_breakdown_figure(res):
    counts = res["summary"]["failure_counts"]
    n = max(1, res["n_samples"])
    modes = list(FAILURE_MODES)
    vals = [counts.get(m, 0) for m in modes]
    pcts = [100.0 * v / n for v in vals]
    colors = [
        SEMANTIC_OK if m == "PASS" else (SEMANTIC_WARN if m != "Multi fail" else SEMANTIC_BAD)
        for m in modes
    ]

    fig = go.Figure(
        go.Bar(
            x=modes,
            y=vals,
            marker_color=colors,
            text=[f"{v:,} ({p:.1f}%)" for v, p in zip(vals, pcts)],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,} runs<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Failure Mode Breakdown", font=dict(size=12.5, color=INK), x=0, xanchor="left"),
        height=350,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=48, r=18, t=48, b=44),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(
        title_text="Monte Carlo runs",
        range=[0, max(vals + [1]) * 1.18],
        showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False,
    )
    return fig


def _sensitivity_figure(res):
    sens = list(res["sensitivity"])
    ordered = list(reversed(sens))  # largest driver at top of horizontal bar chart
    labels = [
        f"{d['label']} ({d['pass_lift_pct']:+.1f}%)"
        for d in ordered
    ]
    lo_vals = [d["low_half_pass_pct"] for d in ordered]
    hi_vals = [d["high_half_pass_pct"] for d in ordered]
    corrs = [d["corr_worst_margin"] for d in ordered]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=lo_vals,
            name="Low-Half Window Pass %",
            orientation="h",
            marker_color="#9ca3af",
            customdata=corrs,
            hovertemplate="%{y}<br>Low-half pass: %{x:.1f}%<br>Margin r: %{customdata:+.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=hi_vals,
            name="High-Half Window Pass %",
            orientation="h",
            marker_color=ACCENT,
            customdata=corrs,
            hovertemplate="%{y}<br>High-half pass: %{x:.1f}%<br>Margin r: %{customdata:+.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Parameter Sensitivity (Low-Half vs High-Half Pass Rate)",
            font=dict(size=12.5, color=INK),
            x=0,
            xanchor="left",
        ),
        height=350,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=18, r=20, t=54, b=42),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(title_text="Pass Rate [%]", range=[0, 105], showline=True, linecolor=HAIR, gridcolor=GRID)
    fig.update_yaxes(showline=True, linecolor=HAIR, gridcolor=GRID, automargin=True)
    return fig


def _render_feasibility_result(mill, res):
    if res.get("error"):
        st.error(res["error"])
        return

    _render_kpi_strip(res)
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)

    _section("mechanical property pass/fail distributions")
    st.plotly_chart(_histograms_figure(res), use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Green bars: runs clearing all active targets simultaneously. Red bars: failing runs. "
        "Dashed amber lines mark Spec Min/Max limits; black/grey vertical lines mark P05, P50, and P95."
    )

    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    _section("2d pass/fail process window")
    param_keys = [k for k, *_ in PROCESS_PARAMS[mill] + GEOMETRY_PARAMS[mill]]
    label_map = {k: lbl for k, lbl, *_ in PROCESS_PARAMS[mill] + GEOMETRY_PARAMS[mill]}
    unit_map = {k: unit for k, _lbl, unit, *_ in PROCESS_PARAMS[mill] + GEOMETRY_PARAMS[mill]}

    def_x = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    def_x_idx = param_keys.index(def_x) if def_x in param_keys else 0
    _prune_widget_key(f"feas_x_{mill}", param_keys)
    c1, c2 = st.columns(2)
    x_key = c1.selectbox(
        "Horizontal axis (X)",
        param_keys,
        index=def_x_idx,
        format_func=lambda k: label_map.get(k, k),
        key=f"feas_x_{mill}",
    )
    y_opts = [k for k in param_keys if k != x_key]
    def_y = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    def_y_idx = y_opts.index(def_y) if def_y in y_opts else 0
    _prune_widget_key(f"feas_y_{mill}", y_opts)
    y_key = c2.selectbox(
        "Vertical axis (Y)",
        y_opts,
        index=def_y_idx,
        format_func=lambda k: label_map.get(k, k),
        key=f"feas_y_{mill}",
    )
    st.plotly_chart(
        _scatter_figure(res, x_key, y_key, label_map, unit_map),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    _section("failure drivers & parameter sensitivity")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(_failure_breakdown_figure(res), use_container_width=True, config={"displayModeBar": False})
    with right:
        st.plotly_chart(_sensitivity_figure(res), use_container_width=True, config={"displayModeBar": False})

    if res["sensitivity"]:
        top = res["sensitivity"][0]
        st.caption(
            f"Top feasibility lever: **{top['label']}** "
            f"(high-half pass {top['high_half_pass_pct']:.1f}% vs low-half {top['low_half_pass_pct']:.1f}%, "
            f"lift {top['pass_lift_pct']:+.1f}%, margin correlation r = {top['corr_worst_margin']:+.2f})."
        )

    csv_bytes = res["samples_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Monte Carlo Samples (CSV)",
        data=csv_bytes,
        file_name=f"ranged_feasibility_{mill}_{res['n_samples']}.csv",
        mime="text/csv",
        key=f"feas_csv_{mill}",
        use_container_width=True,
    )


# ----------------------------------------------------------------------------- interactive view
def view_ranged_feasibility(mill, raw, engine="Champion GBDT (Recommended)"):
    """Interactive Ranged Process Feasibility (Monte Carlo) view for Plate and HRC."""
    if mill not in ("Plate", "HRC"):
        st.error(f"Unsupported route: {mill!r}")
        return

    app_mod = _app()
    raw = dict(raw or app_mod.defaults(mill))
    t_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    cur_t = float(raw.get(t_key, 16.0 if mill == "Plate" else 6.0))
    tb_ratio_cur = _infer_tb_ratio(mill, raw)

    _section("ranged process feasibility (monte carlo)")
    st.markdown(
        '<div class="note">Define operating min–max windows across finishing/cooling temperatures, '
        'gauge, width, and transfer bar ratio. One vectorized pass scores a stratified Latin '
        'Hypercube ensemble to quantify joint multi-property pass rate, dominant failure modes, '
        'and high-leverage control levers.</div>',
        unsafe_allow_html=True,
    )

    # Optional spec pre-fill picker outside the form
    labels, lookup = sc.spec_grade_options()
    spec_prefill_note = None
    if labels:
        picker_opts = ["Manual / custom targets"] + labels
        c_spec, c_btn = st.columns([3.4, 1.4], vertical_alignment="bottom")
        chosen_spec_label = c_spec.selectbox(
            "Pre-fill mechanical targets from specification (optional)",
            picker_opts,
            index=0,
            key=f"feas_spec_pick_{mill}",
        )
        apply_clicked = c_btn.button(
            "Apply limits at median gauge",
            key=f"feas_spec_apply_{mill}",
            use_container_width=True,
        )
        last_key = f"feas_spec_last_{mill}"
        t_lo_ss = _safe_float(st.session_state.get(f"feas_{t_key}_lo_{mill}"), cur_t * 0.85)
        t_hi_ss = _safe_float(st.session_state.get(f"feas_{t_key}_hi_{mill}"), cur_t * 1.15)
        median_t = 0.5 * (t_lo_ss + t_hi_ss)

        if chosen_spec_label != "Manual / custom targets" and (
            chosen_spec_label != st.session_state.get(last_key) or apply_clicked
        ):
            prefilled, info = _apply_spec_prefill(mill, chosen_spec_label, lookup, median_t)
            st.session_state[last_key] = chosen_spec_label
            if prefilled is not None:
                parts = []
                if prefilled["YS_BN_min"] is not None:
                    parts.append(f"YS ≥ {prefilled['YS_BN_min']:.0f} MPa")
                if prefilled["TS_BN_min"] is not None:
                    parts.append(f"TS ≥ {prefilled['TS_BN_min']:.0f} MPa")
                if prefilled["TS_BN_max"] is not None:
                    parts.append(f"TS ≤ {prefilled['TS_BN_max']:.0f} MPa")
                if prefilled["EL_BN_min"] is not None:
                    parts.append(f"EL ≥ {prefilled['EL_BN_min']:.1f}%")
                spec_prefill_note = (
                    f"Loaded **{chosen_spec_label}** at {median_t:.1f} mm: "
                    + (", ".join(parts) if parts else "no YS/TS/EL limits listed.")
                )
                if info.get("status") == "out_of_gauge":
                    spec_prefill_note += f" ({info.get('message', '')})"
        elif chosen_spec_label == "Manual / custom targets":
            st.session_state[last_key] = chosen_spec_label

    if spec_prefill_note:
        st.caption(spec_prefill_note)

    # Form with process constraints, geometry variables, and target mechanical properties
    with st.form(key=f"feas_form_{mill}"):
        st.markdown("**1. Process Parameter Constraints (Min – Max)**")
        p_cols = st.columns(3)
        proc_ranges_form = {}
        swapped_notes = []
        for col, (key, label, unit, f_lo, f_hi, step, kind) in zip(p_cols, PROCESS_PARAMS[mill]):
            lo_b, hi_b = _spec_bounds_for_param(app_mod, mill, key, f_lo, f_hi)
            cur_v = float(raw.get(key, 0.5 * (lo_b + hi_b)))
            if key == "ACC_Laminar_Header":
                hi_b = max(hi_b, cur_v, 8.0)
            def_lo, def_hi = _default_window(mill, key, cur_v, lo_b, hi_b, kind)

            col.markdown(f"<div style='font-size:12px;font-weight:600;color:{INK};margin-bottom:.2rem;'>"
                         f"{label} [{unit}]</div>", unsafe_allow_html=True)
            k_lo = f"feas_{key}_lo_{mill}"
            k_hi = f"feas_{key}_hi_{mill}"
            if kind == "int":
                v_lo_seed = int(min(max(round(float(st.session_state.get(k_lo, def_lo))), int(lo_b)), int(hi_b)))
                v_hi_seed = int(min(max(round(float(st.session_state.get(k_hi, def_hi))), int(lo_b)), int(hi_b)))
                v_lo = col.number_input("Min", min_value=int(lo_b), max_value=int(hi_b),
                                        value=v_lo_seed, step=1, key=k_lo)
                v_hi = col.number_input("Max", min_value=int(lo_b), max_value=int(hi_b),
                                        value=v_hi_seed, step=1, key=k_hi)
            else:
                v_lo_seed = float(min(max(float(st.session_state.get(k_lo, def_lo)), lo_b), hi_b))
                v_hi_seed = float(min(max(float(st.session_state.get(k_hi, def_hi)), lo_b), hi_b))
                v_lo = col.number_input("Min", min_value=float(lo_b), max_value=float(hi_b),
                                        value=v_lo_seed, step=float(step), key=k_lo)
                v_hi = col.number_input("Max", min_value=float(lo_b), max_value=float(hi_b),
                                        value=v_hi_seed, step=float(step), key=k_hi)
            if v_hi < v_lo:
                swapped_notes.append(label)
                v_lo, v_hi = v_hi, v_lo
            proc_ranges_form[key] = (int(v_lo), int(v_hi)) if kind == "int" else (float(v_lo), float(v_hi))

        if mill == "Plate":
            st.caption("Post-laminar temperature is physically derived per sample from finishing temperature, "
                       "laminar headers, and plate thickness (`couple_post=True`).")
        else:
            st.caption("Coiling temperature applies symmetrically to `Temp_CT_Top_C` and `Temp_CT_Bot_C`.")

        st.markdown("**2. Product Geometry Variables (Min – Max)**")
        g_cols = st.columns(3)
        geom_ranges_form = {}
        for col, (key, label, unit, f_lo, f_hi, step, kind) in zip(g_cols, GEOMETRY_PARAMS[mill]):
            lo_b, hi_b = _spec_bounds_for_param(app_mod, mill, key, f_lo, f_hi)
            cur_v = tb_ratio_cur if key == "TB_Ratio" else float(raw.get(key, 0.5 * (lo_b + hi_b)))
            cur_v = min(max(cur_v, lo_b), hi_b)
            def_lo, def_hi = _default_window(mill, key, cur_v, lo_b, hi_b, kind)

            col.markdown(f"<div style='font-size:12px;font-weight:600;color:{INK};margin-bottom:.2rem;'>"
                         f"{label} [{unit}]</div>", unsafe_allow_html=True)
            k_lo = f"feas_{key}_lo_{mill}"
            k_hi = f"feas_{key}_hi_{mill}"
            v_lo_seed = float(min(max(float(st.session_state.get(k_lo, def_lo)), lo_b), hi_b))
            v_hi_seed = float(min(max(float(st.session_state.get(k_hi, def_hi)), lo_b), hi_b))
            v_lo = col.number_input("Min", min_value=float(lo_b), max_value=float(hi_b),
                                    value=v_lo_seed, step=float(step), key=k_lo)
            v_hi = col.number_input("Max", min_value=float(lo_b), max_value=float(hi_b),
                                    value=v_hi_seed, step=float(step), key=k_hi)
            if v_hi < v_lo:
                swapped_notes.append(label)
                v_lo, v_hi = v_hi, v_lo
            geom_ranges_form[key] = (float(v_lo), float(v_hi))

        t_mid = 0.5 * sum(geom_ranges_form[t_key])
        tb_lo_mm = geom_ranges_form["TB_Ratio"][0] * t_mid
        tb_hi_mm = geom_ranges_form["TB_Ratio"][1] * t_mid
        tb_col_name = "TB_Thickness_mm" if mill == "Plate" else "TB_Thick_mm"
        st.caption(
            f"Each Monte Carlo run computes `{tb_col_name} = TB_Ratio × {t_key}` "
            f"(~{tb_lo_mm:.1f}–{tb_hi_mm:.1f} mm at median {t_mid:.1f} mm gauge)."
        )

        st.markdown("**3. Target Mechanical Properties (Before Normalizing)**")
        t_defaults = DEFAULT_TARGETS[mill]
        t_cols = st.columns(4)
        ys_seed = float(min(max(float(st.session_state.get(f"feas_ys_min_{mill}", t_defaults["YS_BN_min"])), 0.0), 1500.0))
        ts_min_seed = float(min(max(float(st.session_state.get(f"feas_ts_min_{mill}", t_defaults["TS_BN_min"])), 0.0), 1500.0))
        ts_max_seed = float(min(max(float(st.session_state.get(f"feas_ts_max_{mill}", t_defaults["TS_BN_max"])), 0.0), 1500.0))
        el_seed = float(min(max(float(st.session_state.get(f"feas_el_min_{mill}", t_defaults["EL_BN_min"])), 0.0), 60.0))

        ys_min_in = t_cols[0].number_input("YS Min [MPa]", min_value=0.0, max_value=1500.0,
                                           value=ys_seed, step=5.0, key=f"feas_ys_min_{mill}")
        ts_min_in = t_cols[1].number_input("TS Min [MPa]", min_value=0.0, max_value=1500.0,
                                           value=ts_min_seed, step=5.0, key=f"feas_ts_min_{mill}")
        ts_max_in = t_cols[2].number_input("TS Max [MPa] (0 = unconstrained)", min_value=0.0, max_value=1500.0,
                                           value=ts_max_seed, step=5.0, key=f"feas_ts_max_{mill}")
        el_min_in = t_cols[3].number_input("EL Min [%]", min_value=0.0, max_value=60.0,
                                           value=el_seed, step=0.5, key=f"feas_el_min_{mill}")

        c_mc, c_cf = st.columns([2.0, 2.0], vertical_alignment="bottom")
        samples_seed = int(min(max(int(st.session_state.get(f"feas_samples_{mill}", 1000)), 300), 3000))
        n_samples = c_mc.slider(
            "Monte Carlo samples",
            min_value=300,
            max_value=3000,
            value=samples_seed,
            step=100,
            key=f"feas_samples_{mill}",
        )
        conf_seed = bool(st.session_state.get(f"feas_conf_{mill}", False))
        include_conformal = c_cf.checkbox(
            "Use 95% conformal buffer as primary pass/fail verdict",
            value=conf_seed,
            key=f"feas_conf_{mill}",
        )

        submitted = st.form_submit_button("Run Feasibility Study", type="primary", use_container_width=True)

    result_key = f"feas_result_{mill}"
    if submitted:
        if swapped_notes:
            st.info("Swapped inverted min/max bounds for: " + ", ".join(swapped_notes))
        targets_in = {
            "YS_BN_min": float(ys_min_in) if ys_min_in > 0.0 else None,
            "TS_BN_min": float(ts_min_in) if ts_min_in > 0.0 else None,
            "TS_BN_max": float(ts_max_in) if ts_max_in > 0.0 else None,
            "EL_BN_min": float(el_min_in) if el_min_in > 0.0 else None,
        }
        with st.spinner(f"Scoring {int(n_samples):,} stratified Monte Carlo schedules ({engine})..."):
            res = run_ranged_feasibility(
                mill,
                raw,
                proc_ranges_form,
                geom_ranges_form,
                targets_in,
                samples=int(n_samples),
                engine=engine,
                include_conformal=bool(include_conformal),
                couple_post=True,
                seed=42,
            )
        st.session_state[result_key] = res
    else:
        res = st.session_state.get(result_key)
        if not res:
            st.caption("Adjust the process/geometry ranges and mechanical targets above, then press "
                       "**Run Feasibility Study**.")
            return
        st.caption("Shown from your last feasibility study; update windows and press **Run Feasibility Study** "
                   "to recompute.")

    _render_feasibility_result(mill, res)
