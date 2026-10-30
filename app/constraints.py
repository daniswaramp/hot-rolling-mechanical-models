"""House limits for process temperatures.

One source of truth shared by the Inverse Optimizer and the Response Surface so that a
setpoint the plant cannot hold is never recommended or plotted.

The limits are seeded from the mill envelope already declared in PLATE_SPEC / HRC_SPEC
(`(group, key, label, unit, lo, hi, default, step)`) and can be narrowed by the user in the
sidebar. They can never be widened beyond the hardware envelope.
"""

import streamlit as st

LIMIT_KEYS = {
    "Plate": ["FM_Temp_Before_Rolling", "FM_Finish_Temp", "ACC_Temp_Post_Laminar"],
    "HRC": ["Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_CT_Bot_C"],
}

# water-stop must sit at least this far below the finishing temperature (plate rule)
POST_LAMINAR_GAP_C = 20.0


def _index(spec_rows):
    return {row[1]: row for row in spec_rows}


def defaults_from_spec(mill, spec_rows):
    """key -> (lo, hi) taken straight from the mill envelope."""
    idx = _index(spec_rows)
    out = {}
    for key in LIMIT_KEYS.get(mill, []):
        row = idx.get(key)
        if row is not None:
            out[key] = (float(row[4]), float(row[5]))
    return out


def get_limits(mill, spec_rows):
    """User limits, clamped to the mill envelope, falling back to the envelope."""
    defaults = defaults_from_spec(mill, spec_rows)
    saved = st.session_state.get(f"limits_{mill}") or {}
    merged = {}
    for key, (hw_lo, hw_hi) in defaults.items():
        lo, hi = saved.get(key, (hw_lo, hw_hi))
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            lo, hi = hw_lo, hw_hi
        lo = max(hw_lo, min(lo, hw_hi))
        hi = max(hw_lo, min(hi, hw_hi))
        if hi <= lo:
            lo, hi = hw_lo, hw_hi
        merged[key] = (lo, hi)
    return merged


def set_limits(mill, limits):
    st.session_state[f"limits_{mill}"] = {k: (float(v[0]), float(v[1])) for k, v in limits.items()}


def within(key, value, limits):
    if key not in limits:
        return True
    lo, hi = limits[key]
    return lo - 1e-9 <= float(value) <= hi + 1e-9


def clamp_raw(mill, raw, limits):
    """Clamp a schedule to the limits. Returns (clamped_copy, violations)."""
    out = dict(raw)
    violations = []
    for key, (lo, hi) in (limits or {}).items():
        if key not in out:
            continue
        try:
            value = float(out[key])
        except (TypeError, ValueError):
            continue
        clamped = min(max(value, lo), hi)
        if abs(clamped - value) > 1e-9:
            violations.append(f"{key}: {value:.0f} -> {clamped:.0f} C (limit {lo:.0f}-{hi:.0f})")
        out[key] = clamped

    if mill == "Plate" and "ACC_Temp_Post_Laminar" in out and "FM_Finish_Temp" in out:
        cap = float(out["FM_Finish_Temp"]) - POST_LAMINAR_GAP_C
        if float(out["ACC_Temp_Post_Laminar"]) > cap:
            violations.append(
                f"ACC_Temp_Post_Laminar: {float(out['ACC_Temp_Post_Laminar']):.0f} -> {cap:.0f} C "
                f"(must be at least {POST_LAMINAR_GAP_C:.0f} C below finish)")
            out["ACC_Temp_Post_Laminar"] = cap
    return out, violations


def render_editor(mill, spec_rows):
    """Sidebar editor. Writes to session state and returns the active limits."""
    idx = _index(spec_rows)
    defaults = defaults_from_spec(mill, spec_rows)
    current = get_limits(mill, spec_rows)
    edited = {}

    with st.sidebar.expander("House limits (°C)", expanded=False):
        st.caption("The solver and the response surface cannot leave these bands. "
                   "They may be narrowed, never widened past the mill envelope.")
        for key in LIMIT_KEYS.get(mill, []):
            row = idx.get(key)
            if row is None:
                continue
            label = row[2]
            hw_lo, hw_hi = float(row[4]), float(row[5])
            lo_now, hi_now = current[key]
            cols = st.columns(2)
            lo = cols[0].number_input(f"{label} min", min_value=hw_lo, max_value=hw_hi,
                                      value=float(lo_now), step=1.0, format="%.0f",
                                      key=f"lim_lo_{mill}_{key}")
            hi = cols[1].number_input(f"{label} max", min_value=hw_lo, max_value=hw_hi,
                                      value=float(hi_now), step=1.0, format="%.0f",
                                      key=f"lim_hi_{mill}_{key}")
            if hi <= lo:
                st.warning(f"{label}: max must exceed min — using the mill envelope for this row.")
                lo, hi = defaults[key]
            edited[key] = (lo, hi)

        if st.button("Reset to mill envelope", key=f"lim_reset_{mill}", use_container_width=True):
            for key in LIMIT_KEYS.get(mill, []):
                st.session_state.pop(f"lim_lo_{mill}_{key}", None)
                st.session_state.pop(f"lim_hi_{mill}_{key}", None)
            set_limits(mill, defaults)
            edited = dict(defaults)
            rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
            if rerun is not None:
                rerun()

    if not edited:
        edited = dict(defaults)
    set_limits(mill, edited)
    return edited


def describe(limits):
    """Short human-readable summary for captions."""
    parts = []
    for key, (lo, hi) in (limits or {}).items():
        parts.append(f"{key} {lo:.0f}-{hi:.0f}")
    return " · ".join(parts) if parts else "no temperature limits"
