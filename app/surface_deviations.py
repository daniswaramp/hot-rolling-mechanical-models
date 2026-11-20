"""Response-surface actual chemistry & process deviation propagation.

Propagates realistic mill aiming/analytical tolerances (chemistry ± wt%, thermal
scatter, gauge/width tolerances, transfer-bar ratio, and laminar header jitter)
through the real Plate or HRC scoring pipeline in a single vectorized batch,
producing P05 / P50 / P95 / nominal / spread envelopes for 1D slices, 2D contours,
3D surfaces, and single-schedule point evaluations.

Import invariant: NEVER import ``modal`` or ``streamlit_app`` at module level;
resolve the active app module lazily via ``app_ref.get_app_module()``.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import physics  # noqa: E402 - read-only post-laminar thermal coupling

# ----------------------------------------------------------------------------- defaults
DEFAULT_CHEM_DEVIATIONS = {
    "Chem_C": 0.01,
    "Chem_Mn": 0.05,
    "Chem_Si": 0.03,
    "Chem_P": 0.002,
    "Chem_S": 0.001,
    "Chem_Al": 0.005,
    "Chem_Nb": 0.003,
    "Chem_V": 0.003,
    "Chem_Ti": 0.002,
    "Chem_Cu": 0.01,
    "Chem_Ni": 0.01,
    "Chem_Cr": 0.01,
    "Chem_Mo": 0.003,
    "Chem_N": 0.001,
    "Chem_B": 0.0002,
}

DEFAULT_PROCESS_DEVIATIONS = {
    "Plate": {
        "FM_Temp_Before_Rolling": 15.0,
        "FM_Finish_Temp": 15.0,
        "ACC_Temp_Post_Laminar": 20.0,
        "ACC_Laminar_Header": 0.0,
        "Plate_T_mm": 0.5,
        "Plate_W_mm": 20.0,
        "TB_Ratio": 0.15,
        "Slab_T_mm": 2.0,
        "FM_Last_Pass_Drafting": 0.5,
    },
    "HRC": {
        "Temp_Transfer_Bar_C": 15.0,
        "Temp_FT_Avg_C": 15.0,
        "Temp_CT_Top_C": 15.0,
        "Temp_CT_Bot_C": 15.0,
        "Coil_T_mm": 0.2,
        "Coil_W_mm": 15.0,
        "TB_Ratio": 0.15,
    },
}

PRESET_SCALES = {
    "Typical mill tolerances": 1.0,
    "Tight (±50%)": 0.5,
    "Wide (±150%)": 1.5,
    "Zero (manual)": 0.0,
}

DISPLAY_MODES = [
    "Conservative lower bound (P05)",
    "Nominal (center)",
    "Upper bound (P95)",
    "Uncertainty spread (P95 − P05)",
]

_THICK_OR_RATIO_KEYS = {
    "Plate_T_mm",
    "Coil_T_mm",
    "Slab_T_mm",
    "TB_Thickness_mm",
    "TB_Thick_mm",
    "TB_Ratio",
}
_WIDTH_KEYS = {"Plate_W_mm", "Coil_W_mm"}
_TEMP_KEYS = {
    "FM_Temp_Before_Rolling",
    "FM_Finish_Temp",
    "ACC_Temp_Post_Laminar",
    "Temp_Transfer_Bar_C",
    "Temp_FT_Avg_C",
    "Temp_CT_Top_C",
    "Temp_CT_Bot_C",
}

_FALLBACK_UI_META = {
    "TB_Ratio": ("Transfer bar ratio", "×", 0.05, 3.0, "%.2f"),
    "ACC_Laminar_Header": ("Laminar headers", "count", 1.0, 8.0, "%.0f"),
    "FM_Last_Pass_Drafting": ("Last pass drafting", "mm", 0.1, 10.0, "%.2f"),
    "Slab_T_mm": ("Slab thickness", "mm", 0.5, 50.0, "%.1f"),
    "Plate_T_mm": ("Plate thickness", "mm", 0.1, 30.0, "%.2f"),
    "Plate_W_mm": ("Plate width", "mm", 5.0, 500.0, "%.1f"),
    "Coil_T_mm": ("Coil thickness", "mm", 0.05, 10.0, "%.2f"),
    "Coil_W_mm": ("Coil width", "mm", 5.0, 400.0, "%.1f"),
    "FM_Temp_Before_Rolling": ("Entry temperature", "°C", 1.0, 150.0, "%.1f"),
    "FM_Finish_Temp": ("Finishing temperature", "°C", 1.0, 150.0, "%.1f"),
    "ACC_Temp_Post_Laminar": ("Post-laminar temperature", "°C", 1.0, 150.0, "%.1f"),
    "Temp_Transfer_Bar_C": ("Transfer bar temperature", "°C", 1.0, 150.0, "%.1f"),
    "Temp_FT_Avg_C": ("Finishing temperature", "°C", 1.0, 150.0, "%.1f"),
    "Temp_CT_Top_C": ("Coiling temperature, top", "°C", 1.0, 150.0, "%.1f"),
    "Temp_CT_Bot_C": ("Coiling temperature, bottom", "°C", 1.0, 150.0, "%.1f"),
}


# ----------------------------------------------------------------------------- internal helpers
def _app():
    """Resolve the active streamlit_app module lazily without circular imports."""
    import app_ref  # noqa: PLC0415

    return app_ref.get_app_module()


def _validate_mill(mill):
    if mill not in ("Plate", "HRC"):
        raise ValueError(f"Unsupported mill route: {mill!r} (expected 'Plate' or 'HRC')")


def _is_finite(val):
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def active_deviations(deviations):
    """Return {key: float(val)} for all finite deviation entries with val > 0."""
    out = {}
    for key, val in (deviations or {}).items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fval) and fval > 0.0:
            out[str(key)] = fval
    return out


def _route_cols(mill):
    if mill == "Plate":
        return "Plate_T_mm", "TB_Thickness_mm", "FM_Finish_Temp", 16.0, 832.0
    return "Coil_T_mm", "TB_Thick_mm", "Temp_FT_Avg_C", 6.0, 835.0


def _base_value_for(mill, raw, key, spec_defaults=None):
    gauge_col, tb_col, _frt_col, default_gauge, _default_frt = _route_cols(mill)
    if key in raw and _is_finite(raw[key]):
        return float(raw[key])
    if key == "TB_Ratio":
        gauge = float(raw[gauge_col]) if (gauge_col in raw and _is_finite(raw[gauge_col])) else default_gauge
        gauge = max(gauge, 1.0)
        tb_mm = float(raw[tb_col]) if (tb_col in raw and _is_finite(raw[tb_col])) else (gauge * 2.0)
        return tb_mm / gauge
    if spec_defaults and key in spec_defaults:
        return float(spec_defaults[key])
    if str(key).startswith("Chem_"):
        return 0.0
    return 0.0


def _clip_perturbed_values(key, vals):
    arr = np.asarray(vals, dtype=float)
    if str(key).startswith("Chem_"):
        return np.maximum(0.0, arr)
    if key == "ACC_Laminar_Header":
        return np.maximum(0.0, np.rint(arr))
    if key in _THICK_OR_RATIO_KEYS:
        return np.maximum(1.0, arr)
    if key in _WIDTH_KEYS:
        return np.maximum(1.0, arr)
    if key == "FM_Last_Pass_Drafting":
        return np.maximum(0.1, arr)
    if key in _TEMP_KEYS:
        return np.maximum(200.0, arr)
    return arr


def _symmetric_unit_draws(n_draws, n_vars, seed):
    """Build (n_draws, n_vars) stratified, antithetic unit offsets in [-1, 1].

    Pairs +u with -u per variable so the perturbed ensemble keeps exact zero-mean
    symmetry around nominal even for small draw counts, while permuting columns
    independently across variables.
    """
    n_draws = max(1, int(n_draws))
    n_vars = max(0, int(n_vars))
    if n_vars == 0:
        return np.zeros((n_draws, 0), dtype=float)
    rng = np.random.default_rng(int(seed))
    half = n_draws // 2
    if half == 0:
        return np.zeros((1, n_vars), dtype=float)
    u_half = np.empty((half, n_vars), dtype=float)
    bins = np.arange(half, dtype=float)
    for col in range(n_vars):
        vals = 2.0 * ((bins + rng.uniform(0.0, 1.0, size=half)) / float(half)) - 1.0
        rng.shuffle(vals)
        u_half[:, col] = vals
    if n_draws % 2 == 0:
        return np.vstack([u_half, -u_half])
    return np.vstack([np.zeros((1, n_vars), dtype=float), u_half, -u_half])


def _apply_derived_couplings(mill, raw, frame, swept_keys, active_devs, couple_post, tpost_deltas=None):
    gauge_col, tb_col, frt_col, default_gauge, default_frt = _route_cols(mill)

    has_tb_ratio = "TB_Ratio" in raw and _is_finite(raw["TB_Ratio"])
    tb_ratio_active = ("TB_Ratio" in swept_keys) or ("TB_Ratio" in active_devs)
    gauge_active = (gauge_col in swept_keys) or (gauge_col in active_devs)
    tb_mm_active = (tb_col in swept_keys) or (tb_col in active_devs)

    if tb_ratio_active or (gauge_active and has_tb_ratio) or (has_tb_ratio and tb_col not in raw):
        if "TB_Ratio" not in frame.columns:
            base_g = max(float(raw.get(gauge_col, default_gauge) or default_gauge), 1.0)
            base_tb = float(raw.get(tb_col, base_g * 2.0) or (base_g * 2.0))
            frame["TB_Ratio"] = base_tb / base_g
        g_vals = np.maximum(
            1.0,
            pd.to_numeric(frame.get(gauge_col, default_gauge), errors="coerce")
            .fillna(default_gauge)
            .to_numpy(dtype=float),
        )
        r_vals = np.maximum(
            1.0,
            pd.to_numeric(frame["TB_Ratio"], errors="coerce")
            .fillna(2.0)
            .to_numpy(dtype=float),
        )
        frame[gauge_col] = g_vals
        frame["TB_Ratio"] = r_vals
        frame[tb_col] = r_vals * g_vals
    elif tb_mm_active and (has_tb_ratio or (mill == "Plate" and "TB_Ratio" in frame.columns)):
        g_vals = np.maximum(
            1.0,
            pd.to_numeric(frame.get(gauge_col, default_gauge), errors="coerce")
            .fillna(default_gauge)
            .to_numpy(dtype=float),
        )
        tb_vals = np.maximum(
            1.0,
            pd.to_numeric(frame.get(tb_col, g_vals * 2.0), errors="coerce")
            .fillna(2.0 * default_gauge)
            .to_numpy(dtype=float),
        )
        frame[tb_col] = tb_vals
        frame["TB_Ratio"] = np.maximum(1.0, tb_vals / g_vals)

    if mill == "Plate":
        hdr_swept = "ACC_Laminar_Header" in swept_keys
        hdr_perturbed = "ACC_Laminar_Header" in active_devs
        if hdr_swept or hdr_perturbed:
            hdrs = np.maximum(
                0.0,
                pd.to_numeric(frame.get("ACC_Laminar_Header", 0.0), errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float),
            )
            frame["ACC_Laminar_Header"] = hdrs
            frame["ACC_Headers_n"] = hdrs
            frame["ACC_Active"] = (hdrs > 0.0).astype(float)
            if couple_post and ("ACC_Temp_Post_Laminar" not in swept_keys):
                frt = (
                    pd.to_numeric(frame.get(frt_col, default_frt), errors="coerce")
                    .fillna(default_frt)
                    .to_numpy(dtype=float)
                )
                pt = np.maximum(
                    1.0,
                    pd.to_numeric(frame.get(gauge_col, default_gauge), errors="coerce")
                    .fillna(default_gauge)
                    .to_numpy(dtype=float),
                )
                t_post = physics.post_laminar_temp(frt, hdrs, pt)
                if tpost_deltas is not None:
                    t_post = np.maximum(200.0, t_post + tpost_deltas)
                frame["ACC_Temp_Post_Laminar"] = t_post


def _build_nominal_frame(mill, raw, swept_arrays, couple_post):
    n_nodes = len(next(iter(swept_arrays.values()))) if swept_arrays else 1
    frame = pd.DataFrame([dict(raw) for _ in range(n_nodes)])
    for key, arr in swept_arrays.items():
        frame[key] = np.asarray(arr, dtype=float)
    _apply_derived_couplings(mill, raw, frame, set(swept_arrays), {}, couple_post)
    return frame


def _build_perturbed_frame(mill, raw, swept_arrays, active_devs, n_draws, couple_post, seed, spec_defaults):
    n_nodes = len(next(iter(swept_arrays.values()))) if swept_arrays else 1
    n_draws = max(1, int(n_draws))
    total_rows = n_draws * n_nodes

    active_keys = list(active_devs.keys())
    u = _symmetric_unit_draws(n_draws, len(active_keys), seed)

    frame = pd.DataFrame([dict(raw) for _ in range(total_rows)])
    for key, arr in swept_arrays.items():
        frame[key] = np.tile(np.asarray(arr, dtype=float), n_draws)

    tpost_deltas = None
    for j, key in enumerate(active_keys):
        if key in swept_arrays:
            center = np.tile(np.asarray(swept_arrays[key], dtype=float), n_draws)
        else:
            base_val = _base_value_for(mill, raw, key, spec_defaults=spec_defaults)
            center = np.full(total_rows, base_val, dtype=float)
        delta = np.repeat(u[:, j], n_nodes) * float(active_devs[key])
        if key == "ACC_Temp_Post_Laminar":
            tpost_deltas = delta
        frame[key] = _clip_perturbed_values(key, center + delta)

    _apply_derived_couplings(
        mill,
        raw,
        frame,
        set(swept_arrays),
        active_devs,
        couple_post,
        tpost_deltas=tpost_deltas,
    )
    return frame


def _spec_default_map(app, mill):
    try:
        return {row[1]: float(row[6]) for row in app.SPECS[mill]}
    except Exception:
        return {}


def _apply_slice_guardrail(app, mill, x_key, xs_arr, df_scores):
    if mill != "Plate" or x_key != "ACC_Laminar_Header" or len(xs_arr) < 2:
        return df_scores
    if np.all(np.diff(xs_arr) >= 0.0):
        return app.monotone_guardrail(df_scores, x_key, mill)
    order = np.argsort(xs_arr, kind="mergesort")
    inv = np.argsort(order)
    guarded = app.monotone_guardrail(df_scores.iloc[order].reset_index(drop=True), x_key, mill)
    return guarded.iloc[inv].reset_index(drop=True)


def _apply_grid_guardrail(app, mill, x_key, y_key, xs_arr, ys_arr, target, arr, axis_offset=0):
    if mill != "Plate":
        return arr
    sign = getattr(app, "HEADER_SWEEP_SIGN", {}).get(target)
    if sign is None:
        return arr
    if x_key == "ACC_Laminar_Header":
        hdr_axis, hdr_vals = 0, xs_arr
    elif y_key == "ACC_Laminar_Header":
        hdr_axis, hdr_vals = 1, ys_arr
    else:
        return arr
    if len(hdr_vals) < 2:
        return arr
    ax = axis_offset + hdr_axis
    ascending = bool(np.all(np.diff(hdr_vals) >= 0.0))
    if ascending:
        return np.maximum.accumulate(arr, axis=ax) if sign > 0 else np.minimum.accumulate(arr, axis=ax)
    order = np.argsort(hdr_vals, kind="mergesort")
    inv = np.argsort(order)
    sorted_arr = np.take(arr, order, axis=ax)
    acc = np.maximum.accumulate(sorted_arr, axis=ax) if sign > 0 else np.minimum.accumulate(sorted_arr, axis=ax)
    return np.take(acc, inv, axis=ax)


# ----------------------------------------------------------------------------- vectorized scoring API
def score_slice_with_deviations(
    mill,
    raw,
    x_key,
    xs,
    y_keys,
    deviations,
    engine="Champion GBDT (Recommended)",
    n_draws=24,
    couple_post=True,
    seed=42,
):
    """Evaluate a 1D slice at nominal plus symmetric MC deviations around each node.

    Returns ``{key: {"nominal": arr, "p05": arr, "p50": arr, "p95": arr, "std": arr}}``
    for each scoreable target in ``y_keys``.
    """
    _validate_mill(mill)
    xs_arr = np.asarray(list(xs), dtype=float)
    y_keys = list(y_keys)
    nx = int(xs_arr.size)
    if not y_keys:
        return {}
    if nx == 0:
        empty = np.array([], dtype=float)
        return {
            key: {"nominal": empty.copy(), "p05": empty.copy(), "p50": empty.copy(),
                  "p95": empty.copy(), "std": empty.copy()}
            for key in y_keys
        }

    app = _app()
    raw = dict(raw or {})
    active_devs = active_deviations(deviations)

    nom_frame = _build_nominal_frame(mill, raw, {x_key: xs_arr}, couple_post)
    nom_scores = app.score_frame(mill, nom_frame, engine=engine)
    nom_scores = _apply_slice_guardrail(app, mill, x_key, xs_arr, nom_scores)

    present_keys = [k for k in y_keys if k in nom_scores.columns]
    if not present_keys:
        return {}

    if not active_devs:
        out = {}
        for key in present_keys:
            nom = pd.to_numeric(nom_scores[key], errors="coerce").to_numpy(dtype=float)
            out[key] = {
                "nominal": nom.copy(),
                "p05": nom.copy(),
                "p50": nom.copy(),
                "p95": nom.copy(),
                "std": np.zeros_like(nom),
            }
        return out

    n_draws = max(1, int(n_draws))
    spec_defaults = _spec_default_map(app, mill)
    pert_frame = _build_perturbed_frame(
        mill, raw, {x_key: xs_arr}, active_devs, n_draws, couple_post, seed, spec_defaults
    )
    scored = app.score_frame(mill, pert_frame, engine=engine)

    mat_by_key = {
        key: pd.to_numeric(scored[key], errors="coerce").to_numpy(dtype=float).reshape(n_draws, nx)
        for key in present_keys
        if key in scored.columns
    }

    if mill == "Plate" and x_key == "ACC_Laminar_Header" and nx >= 2 and mat_by_key:
        order = None if np.all(np.diff(xs_arr) >= 0.0) else np.argsort(xs_arr, kind="mergesort")
        inv_order = np.argsort(order) if order is not None else None
        for d in range(n_draws):
            sub_dict = {
                k: (mat[d][order] if order is not None else mat[d])
                for k, mat in mat_by_key.items()
            }
            guarded = app.monotone_guardrail(pd.DataFrame(sub_dict), x_key, mill)
            for k, mat in mat_by_key.items():
                vals = guarded[k].to_numpy(dtype=float)
                mat[d] = vals[inv_order] if inv_order is not None else vals

    out = {}
    for key in present_keys:
        nom = pd.to_numeric(nom_scores[key], errors="coerce").to_numpy(dtype=float)
        mat = mat_by_key.get(key)
        if mat is None:
            out[key] = {
                "nominal": nom.copy(),
                "p05": nom.copy(),
                "p50": nom.copy(),
                "p95": nom.copy(),
                "std": np.zeros_like(nom),
            }
            continue
        out[key] = {
            "nominal": nom,
            "p05": np.nanpercentile(mat, 5.0, axis=0),
            "p50": np.nanpercentile(mat, 50.0, axis=0),
            "p95": np.nanpercentile(mat, 95.0, axis=0),
            "std": np.nanstd(mat, axis=0, ddof=0),
        }
    return out


def score_grid_with_deviations(
    mill,
    raw,
    x_key,
    y_key,
    xs,
    ys,
    target,
    deviations,
    engine="Champion GBDT (Recommended)",
    n_draws=12,
    couple_post=True,
    seed=42,
):
    """Evaluate a 2D (len(xs), len(ys)) grid with vectorized perturbations per node.

    Returns ``{"nominal", "p05", "p50", "p95", "spread"}`` with each array shaped
    ``(len(xs), len(ys))``, matching ``view_surface``'s ``reshape(grid, grid)`` convention.
    """
    _validate_mill(mill)
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    nx, ny = int(xs_arr.size), int(ys_arr.size)
    if nx == 0 or ny == 0:
        empty = np.empty((nx, ny), dtype=float)
        return {k: empty.copy() for k in ("nominal", "p05", "p50", "p95", "spread")}

    app = _app()
    raw = dict(raw or {})
    active_devs = active_deviations(deviations)

    swept_arrays = {
        x_key: np.repeat(xs_arr, ny),
        y_key: np.tile(ys_arr, nx),
    }
    nom_frame = _build_nominal_frame(mill, raw, swept_arrays, couple_post)
    nom_scores = app.score_frame(mill, nom_frame, engine=engine)
    if target not in nom_scores.columns:
        nans = np.full((nx, ny), np.nan, dtype=float)
        return {k: nans.copy() for k in ("nominal", "p05", "p50", "p95", "spread")}

    z_nom = pd.to_numeric(nom_scores[target], errors="coerce").to_numpy(dtype=float).reshape(nx, ny)
    z_nom = _apply_grid_guardrail(app, mill, x_key, y_key, xs_arr, ys_arr, target, z_nom, axis_offset=0)

    if not active_devs:
        return {
            "nominal": z_nom.copy(),
            "p05": z_nom.copy(),
            "p50": z_nom.copy(),
            "p95": z_nom.copy(),
            "spread": np.zeros_like(z_nom),
        }

    n_draws = max(1, int(n_draws))
    spec_defaults = _spec_default_map(app, mill)
    pert_frame = _build_perturbed_frame(
        mill, raw, swept_arrays, active_devs, n_draws, couple_post, seed, spec_defaults
    )
    scored = app.score_frame(mill, pert_frame, engine=engine)
    if target not in scored.columns:
        return {
            "nominal": z_nom.copy(),
            "p05": z_nom.copy(),
            "p50": z_nom.copy(),
            "p95": z_nom.copy(),
            "spread": np.zeros_like(z_nom),
        }

    mat = pd.to_numeric(scored[target], errors="coerce").to_numpy(dtype=float).reshape(n_draws, nx, ny)
    mat = _apply_grid_guardrail(app, mill, x_key, y_key, xs_arr, ys_arr, target, mat, axis_offset=1)

    z_p05 = np.nanpercentile(mat, 5.0, axis=0)
    z_p50 = np.nanpercentile(mat, 50.0, axis=0)
    z_p95 = np.nanpercentile(mat, 95.0, axis=0)
    return {
        "nominal": z_nom,
        "p05": z_p05,
        "p50": z_p50,
        "p95": z_p95,
        "spread": z_p95 - z_p05,
    }


def evaluate_point_with_deviations(
    mill,
    raw,
    target,
    deviations,
    engine="Champion GBDT (Recommended)",
    n_draws=48,
    couple_post=True,
    seed=42,
):
    """Evaluate nominal and P05/P50/P95/spread at a single schedule point."""
    _validate_mill(mill)
    app = _app()
    raw = dict(raw or {})
    active_devs = active_deviations(deviations)

    nom_frame = _build_nominal_frame(mill, raw, {}, couple_post)
    nom_scores = app.score_frame(mill, nom_frame, engine=engine)
    if target not in nom_scores.columns:
        return {k: float("nan") for k in ("nominal", "p05", "p50", "p95", "spread")}

    nom = float(pd.to_numeric(nom_scores[target], errors="coerce").iloc[0])
    if not active_devs or not np.isfinite(nom):
        return {
            "nominal": nom,
            "p05": nom,
            "p50": nom,
            "p95": nom,
            "spread": 0.0 if np.isfinite(nom) else float("nan"),
        }

    n_draws = max(1, int(n_draws))
    spec_defaults = _spec_default_map(app, mill)
    pert_frame = _build_perturbed_frame(
        mill, raw, {}, active_devs, n_draws, couple_post, seed, spec_defaults
    )
    scored = app.score_frame(mill, pert_frame, engine=engine)
    if target not in scored.columns:
        return {"nominal": nom, "p05": nom, "p50": nom, "p95": nom, "spread": 0.0}

    vals = pd.to_numeric(scored[target], errors="coerce").to_numpy(dtype=float)
    p05 = float(np.nanpercentile(vals, 5.0))
    p50 = float(np.nanpercentile(vals, 50.0))
    p95 = float(np.nanpercentile(vals, 95.0))
    return {
        "nominal": nom,
        "p05": p05,
        "p50": p50,
        "p95": p95,
        "spread": float(p95 - p05),
    }


# ----------------------------------------------------------------------------- Streamlit UI controls
def _field_widget_meta(key, base_default, spec_by_key):
    if key in _FALLBACK_UI_META:
        label, unit, step, max_val, fmt = _FALLBACK_UI_META[key]
        if key in spec_by_key:
            label = str(spec_by_key[key][2])
            raw_unit = str(spec_by_key[key][3])
            unit = "°C" if raw_unit == "C" else raw_unit
        return label, unit, step, max_val, fmt

    if key in spec_by_key:
        row = spec_by_key[key]
        label = str(row[2])
        raw_unit = str(row[3])
        unit = "°C" if raw_unit == "C" else raw_unit
    else:
        label = key.replace("Chem_", "") if str(key).startswith("Chem_") else key
        unit = "wt%" if str(key).startswith("Chem_") else ""

    if str(key).startswith("Chem_"):
        if base_default <= 0.0005:
            step, fmt, max_val = 0.0001, "%.4f", 0.01
        elif base_default <= 0.003:
            step, fmt, max_val = 0.0005 if base_default <= 0.001 else 0.001, "%.4f" if base_default <= 0.001 else "%.3f", 0.05
        else:
            step, fmt, max_val = 0.005 if base_default >= 0.01 else 0.001, "%.3f", max(0.5, round(base_default * 20.0, 2))
        return label, unit, step, max_val, fmt

    return label, unit, 1.0, max(100.0, base_default * 10.0), "%.2f"


def _render_deviation_grid(mill, defaults_map, scale, spec_by_key):
    cols = st.columns(3)
    out = {}
    for idx, (key, base_val) in enumerate(defaults_map.items()):
        label, unit, step, max_val, fmt = _field_widget_meta(key, base_val, spec_by_key)
        wkey = f"sdev_val_{mill}_{key}"
        default_scaled = min(max(round(base_val * scale, 6), 0.0), float(max_val))
        seeded = st.session_state.get(wkey)
        val = float(default_scaled if seeded is None else seeded)
        val = min(max(val, 0.0), float(max_val))
        out[key] = float(
            cols[idx % 3].number_input(
                f"± {label} [{unit}]",
                min_value=0.0,
                max_value=float(max_val),
                value=val,
                step=float(step),
                format=fmt,
                key=wkey,
            )
        )
    return out


def render_deviation_controls(mill, spec_rows):
    """Render collapsible mill tolerance controls and return ``(enabled, deviations_dict, display_mode, n_draws)``."""
    _validate_mill(mill)
    spec_rows = list(spec_rows or [])
    spec_by_key = {row[1]: row for row in spec_rows if len(row) >= 7}

    chem_keys = [row[1] for row in spec_rows if len(row) >= 2 and str(row[1]).startswith("Chem_")]
    if not chem_keys:
        chem_keys = [k for k in DEFAULT_CHEM_DEVIATIONS if mill == "HRC" or k != "Chem_B"]
    chem_defaults = {k: float(DEFAULT_CHEM_DEVIATIONS.get(k, 0.005)) for k in chem_keys}

    proc_defaults = dict(DEFAULT_PROCESS_DEVIATIONS.get(mill, DEFAULT_PROCESS_DEVIATIONS["Plate"]))
    all_defaults = dict(chem_defaults)
    all_defaults.update(proc_defaults)

    deviations_dict = dict(all_defaults)
    with st.expander("Actual chemistry & process deviations (± uncertainty)", expanded=False):
        c_on, c_pre, c_mode, c_draws = st.columns([1.25, 1.25, 1.45, 1.05], vertical_alignment="bottom")
        enabled = c_on.checkbox(
            "Enable ± chemistry & process deviations",
            value=False,
            key=f"sdev_on_{mill}",
            help="Overlay Monte-Carlo P05/P50/P95 response bounds across actual mill chemistry and process tolerances.",
        )
        preset = c_pre.selectbox(
            "Tolerance preset",
            list(PRESET_SCALES.keys()),
            index=0,
            key=f"sdev_preset_{mill}",
        )
        display_mode = c_mode.selectbox(
            "Display mode",
            DISPLAY_MODES,
            index=0,
            key=f"sdev_mode_{mill}",
            help="Controls which quantile surface is plotted on 2D contours and 3D surfaces.",
        )
        n_draws = c_draws.slider(
            "MC draws per node",
            min_value=8,
            max_value=32,
            value=12,
            step=1,
            key=f"sdev_draws_{mill}",
        )

        scale = PRESET_SCALES[preset]
        applied_key = f"sdev_preset_applied_{mill}"
        if st.session_state.get(applied_key) != preset:
            for key, base_val in all_defaults.items():
                _lbl, _unit, _step, max_val, _fmt = _field_widget_meta(key, base_val, spec_by_key)
                scaled = min(max(round(base_val * scale, 6), 0.0), float(max_val))
                st.session_state[f"sdev_val_{mill}_{key}"] = float(scaled)
            st.session_state[applied_key] = preset

        tab_chem, tab_proc = st.tabs(["Chemistry deviations (± wt%)", "Process & geometry deviations (±)"])
        with tab_chem:
            st.caption("Symmetric ± tolerance applied around the sidebar heat chemistry (clipped ≥ 0 wt%).")
            deviations_dict.update(_render_deviation_grid(mill, chem_defaults, scale, spec_by_key))

        with tab_proc:
            st.caption(
                "Symmetric ± tolerance on temperatures, gauge, width, and TB_Ratio "
                "(recomputes transfer-bar thickness and post-laminar cooling when coupled)."
            )
            deviations_dict.update(_render_deviation_grid(mill, proc_defaults, scale, spec_by_key))

    return bool(enabled), deviations_dict, str(display_mode), int(n_draws)
