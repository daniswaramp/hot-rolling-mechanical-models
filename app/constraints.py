"""House limits for the inverse optimizer.

Two kinds of constraint live here:

  * process window  — an allowed min/max for entry, exit and cooling temperature
  * chemistry caps  — a ceiling on CE_IIW and on P_cm

The window opens on the trained envelope (P01-P99 from models/applicability_domain.json),
not on the mill's hardware range. A band that reaches past the trained range is allowed,
but it raises a warning rather than being silently clipped.
"""

import json
import math
import os

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DOMAIN_PATH = os.path.join(os.path.dirname(APP_DIR), "models", "applicability_domain.json")

# role -> (input key, label)
WINDOW_KEYS = {
    "Plate": [
        ("entry", "FM_Temp_Before_Rolling", "Entry temperature"),
        ("exit", "FM_Finish_Temp", "Exit temperature"),
        ("cool", "ACC_Temp_Post_Laminar", "Post-laminar temperature"),
    ],
    "HRC": [
        ("entry", "Temp_Transfer_Bar_C", "Transfer bar temperature"),
        ("exit", "Temp_FT_Avg_C", "Finishing temperature"),
        ("cool", "Temp_CT_Top_C", "Coiling temperature, top"),
        ("cool_bottom", "Temp_CT_Bot_C", "Coiling temperature, bottom"),
    ],
}

POST_LAMINAR_GAP_C = 20.0

CHEM_CAPS = [
    ("ceq", "CE_IIW max", "Carbon equivalent, IIW"),
    ("pcm", "P_cm max", "Ito-Bessyo cracking parameter"),
]

CHEM_DEFAULTS = {"ceq": 0.42, "pcm": 0.24}


def _index(spec_rows):
    return {row[1]: row for row in spec_rows}


@st.cache_data(show_spinner=False)
def load_domain():
    try:
        with open(DOMAIN_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def training_bounds(mill):
    """Trained P01-P99 window per temperature key. Entries that are missing or not
    physically usable (a coiling temperature of 0 C comes from blank cells, not data)
    are left out so the caller can fall back."""
    domain = load_domain().get(mill, {})
    out = {}
    for _role, key, _label in WINDOW_KEYS.get(mill, []):
        block = domain.get(key)
        if not isinstance(block, dict):
            continue
        try:
            lo = float(block.get("p01"))
            hi = float(block.get("p99"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        if lo <= 0.0 and "Temp" in key:
            continue
        out[key] = (lo, hi)
    return out


def defaults_from_spec(mill, spec_rows):
    """Hardware envelope from the mill spec table. Used for clamping, never as the default."""
    idx = _index(spec_rows)
    out = {}
    for _role, key, _label in WINDOW_KEYS.get(mill, []):
        row = idx.get(key)
        if row is not None:
            out[key] = (float(row[4]), float(row[5]))
    return out


def defaults_from_training(mill, spec_rows):
    """Opening window: trained P01-P99 where available, mill envelope where not."""
    hardware = defaults_from_spec(mill, spec_rows)
    trained = training_bounds(mill)
    out = {}
    for key, (hw_lo, hw_hi) in hardware.items():
        if key in trained:
            lo = max(hw_lo, trained[key][0])
            hi = min(hw_hi, trained[key][1])
            out[key] = (lo, hi) if hi > lo else (hw_lo, hw_hi)
        else:
            out[key] = (hw_lo, hw_hi)
    return out


def bound_source(mill, spec_rows):
    """key -> 'trained' or 'mill envelope', for the caption."""
    trained = training_bounds(mill)
    return {key: ("trained" if key in trained else "mill envelope")
            for key in defaults_from_spec(mill, spec_rows)}


def get_limits(mill, spec_rows):
    """Current window, kept inside the hardware envelope."""
    hardware = defaults_from_spec(mill, spec_rows)
    opening = defaults_from_training(mill, spec_rows)
    saved = st.session_state.get(f"limits_{mill}") or {}
    merged = {}
    for key, (hw_lo, hw_hi) in hardware.items():
        lo, hi = saved.get(key, opening[key])
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            lo, hi = opening[key]
        lo = max(hw_lo, min(lo, hw_hi))
        hi = max(hw_lo, min(hi, hw_hi))
        if hi <= lo:
            lo, hi = opening[key]
        merged[key] = (lo, hi)
    return merged


def set_limits(mill, limits):
    st.session_state[f"limits_{mill}"] = {k: (float(v[0]), float(v[1])) for k, v in limits.items()}


def outside_training(mill, limits):
    """Bands that reach past the trained P01-P99 range."""
    trained = training_bounds(mill)
    issues = []
    for key, (lo, hi) in (limits or {}).items():
        if key not in trained:
            continue
        t_lo, t_hi = trained[key]
        if lo < t_lo - 1e-9 or hi > t_hi + 1e-9:
            issues.append({"key": key, "lo": lo, "hi": hi, "t_lo": t_lo, "t_hi": t_hi})
    return issues


def get_chem_caps(mill):
    saved = st.session_state.get(f"chemcaps_{mill}") or {}
    return {k: (float(saved[k]) if saved.get(k) is not None else None) for k, _l, _h in CHEM_CAPS}


def set_chem_caps(mill, caps):
    st.session_state[f"chemcaps_{mill}"] = dict(caps)


def within(key, value, limits):
    if key not in limits:
        return True
    lo, hi = limits[key]
    return lo - 1e-9 <= float(value) <= hi + 1e-9


def ce_iiw(c, mn, cr, mo, v, ni, cu):
    return c + mn / 6.0 + (cr + mo + v) / 5.0 + (ni + cu) / 15.0


def pcm(c, si, mn, cu, cr, ni, mo, v, b=0.0):
    return c + si / 30.0 + (mn + cu + cr) / 20.0 + ni / 60.0 + mo / 15.0 + v / 10.0 + 5.0 * b


def clamp_raw(mill, raw, limits):
    """Clamp a schedule into the hardware window and report anything that moved."""
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
            violations.append(f"{key}: {value:.0f} -> {clamped:.0f} C (window {lo:.0f}-{hi:.0f})")
        out[key] = clamped

    if mill == "Plate" and "ACC_Temp_Post_Laminar" in out and "FM_Finish_Temp" in out:
        cap = float(out["FM_Finish_Temp"]) - POST_LAMINAR_GAP_C
        if float(out["ACC_Temp_Post_Laminar"]) > cap:
            violations.append(
                f"ACC_Temp_Post_Laminar: {float(out['ACC_Temp_Post_Laminar']):.0f} -> {cap:.0f} C "
                f"(must sit at least {POST_LAMINAR_GAP_C:.0f} C below the exit)")
            out["ACC_Temp_Post_Laminar"] = cap
    return out, violations


def describe(limits):
    parts = [f"{k} {v[0]:.0f}-{v[1]:.0f}" for k, v in (limits or {}).items()]
    return " · ".join(parts) if parts else "no temperature window"


def describe_caps(caps):
    if not caps:
        return "no chemistry caps"
    bits = []
    if caps.get("ceq") is not None:
        bits.append(f"CE_IIW <= {caps['ceq']:.3f}")
    if caps.get("pcm") is not None:
        bits.append(f"P_cm <= {caps['pcm']:.3f}")
    return " · ".join(bits) if bits else "no chemistry caps"


def render_editor(mill, spec_rows, container=None):
    """Window and chemistry caps, rendered inside the Inverse Optimization tab."""
    box = container if container is not None else st
    idx = _index(spec_rows)
    opening = defaults_from_training(mill, spec_rows)
    hardware = defaults_from_spec(mill, spec_rows)
    sources = bound_source(mill, spec_rows)
    current = get_limits(mill, spec_rows)
    caps = get_chem_caps(mill)
    rev = int(st.session_state.get(f"lim_rev_{mill}", 0))

    with box.expander("Permitted window and chemistry caps", expanded=True):
        st.caption("Bands open on the trained envelope (P01-P99). A band that reaches past it "
                   "is still accepted, but the solver warns rather than pretending the model has seen it.")

        edited = {}
        for _role, key, label in WINDOW_KEYS.get(mill, []):
            row = idx.get(key)
            if row is None:
                continue
            hw_lo, hw_hi = float(row[4]), float(row[5])
            lo_now, hi_now = current[key]
            tail = "trained" if sources.get(key) == "trained" else "mill envelope"
            cols = st.columns(2)
            lo = cols[0].number_input(f"{label} min (°C)", min_value=hw_lo, max_value=hw_hi,
                                      value=float(lo_now), step=1.0, format="%.0f",
                                      key=f"lim_lo_{mill}_{key}_{rev}")
            hi = cols[1].number_input(f"{label} max (°C)", min_value=hw_lo, max_value=hw_hi,
                                      value=float(hi_now), step=1.0, format="%.0f",
                                      key=f"lim_hi_{mill}_{key}_{rev}")
            st.caption(f"{label} band opens from the {tail}.")
            if hi <= lo:
                st.error(f"{label}: max must beat min, keeping the opening band for that row.")
                lo, hi = opening[key]
            edited[key] = (lo, hi)

        issues = outside_training(mill, edited)
        if issues:
            lines = [
                f"{i['key']}: {i['lo']:.0f}-{i['hi']:.0f} extends past the trained "
                f"{i['t_lo']:.0f}-{i['t_hi']:.0f} °C"
                for i in issues
            ]
            st.warning("Outside the training envelope. Predictions there are extrapolation, "
                       "not interpolation: " + "; ".join(lines))
        else:
            st.success("Window sits inside the trained envelope for every band the model has seen.")

        st.markdown("---")
        new_caps = {}
        for cap_key, cap_label, cap_help in CHEM_CAPS:
            c1, c2 = st.columns([1.0, 1.6], vertical_alignment="bottom")
            on = c1.checkbox(f"Cap {cap_label}", value=caps.get(cap_key) is not None,
                             key=f"cap_on_{mill}_{cap_key}_{rev}", help=cap_help)
            if on:
                new_caps[cap_key] = c2.number_input(
                    cap_label, min_value=0.05, max_value=1.20,
                    value=float(caps.get(cap_key) if caps.get(cap_key) is not None else CHEM_DEFAULTS[cap_key]),
                    step=0.005, format="%.3f", key=f"cap_val_{mill}_{cap_key}_{rev}")
            else:
                new_caps[cap_key] = None

        reset = st.button("Reset to trained envelope", key=f"lim_reset_{mill}_{rev}",
                          use_container_width=True)

    if reset:
        st.session_state[f"lim_rev_{mill}"] = rev + 1
        set_limits(mill, opening)
        set_chem_caps(mill, {"ceq": None, "pcm": None})
        rerun = getattr(st, "rerun", None)
        if rerun is not None:
            rerun()

    set_limits(mill, edited)
    set_chem_caps(mill, new_caps)
    return edited, new_caps
