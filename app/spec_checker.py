"""Specification checking engine for dual-mill hot rolling platform.

Loads GRP product specification list (GRP.ST.QA.09 / SPECIFICATION LIST.xlsx)
and correlates user-entered thickness and chemical composition with standard
limits and ML-predicted mechanical properties (Yield, Tensile, Elongation, Impact).
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(APP_DIR)
SPEC_FILE_PATH = os.path.join(PROJECT_DIR, "specification-list", "SPECIFICATION LIST.xlsx")

# Chemical mapping: (app_key, col_min, col_max, label)
CHEM_ELEMENTS = [
    ("Chem_C", "C Min [%]", "C Max [%]", "Carbon (C)"),
    ("Chem_Si", "Si Min [%]", "Si Max [%]", "Silicon (Si)"),
    ("Chem_Mn", "Mn Min [%]", "Mn Max [%]", "Manganese (Mn)"),
    ("Chem_P", None, "P Max [%]", "Phosphorus (P)"),
    ("Chem_S", None, "S Max [%]", "Sulfur (S)"),
    ("Chem_Al", "Al Min [%]", "Al Max [%]", "Aluminum (Al)"),
    ("Chem_Nb", "Nb Min [%]", "Nb Max [%]", "Niobium (Nb)"),
    ("Chem_V", "V Min [%]", "V Max [%]", "Vanadium (V)"),
    ("Chem_Ti", "Ti Min [%]", "Ti Max [%]", "Titanium (Ti)"),
    ("Chem_Cu", "Cu Min [%]", "Cu Max [%]", "Copper (Cu)"),
    ("Chem_Ni", "Ni Min [%]", "Ni Max [%]", "Nickel (Ni)"),
    ("Chem_Cr", "Cr Min [%]", "Cr Max [%]", "Chromium (Cr)"),
    ("Chem_Mo", None, "Mo Max [%]", "Molybdenum (Mo)"),
    ("Chem_N", None, "N Max [%]", "Nitrogen (N)"),
    ("Chem_B", None, "B Max [%]", "Boron (B)"),
]

# Carbon equivalents mapping
CE_ELEMENTS = [
    ("ce_iiw", None, "CE Max [%]", "Carbon Equivalent (CE IIW)"),
    ("pcm", None, "PCM Max [%]", "Cracking Parameter (Pcm)"),
]


def parse_limit(val):
    """Extract numeric limit and any footnote note from strings like '250A', '1.35K', '-'."""
    if pd.isna(val):
        return None, ""
    s = str(val).strip()
    if s in ("", "-", "None", "nan", "NaN"):
        return None, ""
    # Extract leading float or integer
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", s)
    if m:
        num = float(m.group(1))
        # Note letters if any (e.g. 'A', 'K', 'O')
        letters = re.findall(r"[a-zA-Z]+", s)
        note = "".join(letters)
        return num, note
    return None, s


@st.cache_data(show_spinner=False)
def load_specifications():
    """Load and clean the Specification sheet from SPECIFICATION LIST.xlsx."""
    if not os.path.exists(SPEC_FILE_PATH):
        return pd.DataFrame()

    df = pd.read_excel(SPEC_FILE_PATH, sheet_name="Specification", header=1)
    df["SPEC"] = df["SPEC"].astype(str).str.strip()
    df["GRADE"] = df["GRADE"].fillna("-").astype(str).str.strip()
    df["THICK MIN (mm)"] = pd.to_numeric(df["THICK MIN (mm)"], errors="coerce").fillna(0.0)
    df["THICK MAX (mm)"] = pd.to_numeric(df["THICK MAX (mm)"], errors="coerce").fillna(99999.0)
    return df


def get_available_specs():
    """Return sorted unique specification names."""
    df = load_specifications()
    if df.empty or "SPEC" not in df.columns:
        return []
    specs = sorted([s for s in df["SPEC"].unique() if s and s != "nan"])
    return specs


def get_available_grades(spec):
    """Return sorted unique grades for a given specification."""
    df = load_specifications()
    if df.empty:
        return []
    sub = df[df["SPEC"] == spec]
    if sub.empty:
        return []
    grades = sorted([g for g in sub["GRADE"].unique() if g and g != "nan"])
    return grades


def find_matching_spec_row(spec, grade, thickness):
    """Find the exact tier row matching SPEC, GRADE, and THICK MIN <= thickness <= THICK MAX.

    Returns:
        (matched_row_series_or_None, status_dict)
    """
    df = load_specifications()
    if df.empty:
        return None, {"status": "error", "message": "Specification dataset not found."}

    sub = df[(df["SPEC"] == spec) & (df["GRADE"] == grade)]
    if sub.empty:
        return None, {"status": "no_grade", "message": f"No entries for {spec} grade {grade}."}

    # Match thickness tier
    t = float(thickness)
    matched = sub[(sub["THICK MIN (mm)"] <= t) & (sub["THICK MAX (mm)"] >= t)]

    if not matched.empty:
        return matched.iloc[0], {"status": "ok", "message": "Exact tier matched."}

    # If not in range, return closest row for reference
    sub_sorted = sub.sort_values("THICK MIN (mm)")
    min_t = sub_sorted["THICK MIN (mm)"].min()
    max_t = sub_sorted["THICK MAX (mm)"].max()
    closest = sub_sorted.iloc[-1] if t > max_t else sub_sorted.iloc[0]

    return closest, {
        "status": "out_of_gauge",
        "message": f"Thickness {t:.1f} mm outside standard range ({min_t:.1f} – {max_t:.1f} mm). Showing closest tier.",
        "min_t": min_t,
        "max_t": max_t,
    }


def evaluate_thickness(row, thickness, mill):
    """Evaluate thickness against standard tier and GRP plant capability."""
    t = float(thickness)
    t_min = float(row.get("THICK MIN (mm)", 0.0))
    t_max = float(row.get("THICK MAX (mm)", 99999.0))
    prod_thick = str(row.get("PROD THICK (mm)", "")).strip()

    # Gauge tier check
    in_standard = (t >= t_min - 1e-4) and (t <= t_max + 1e-4)

    # Mill physical bounds
    mill_min = 6.0 if mill == "Plate" else 1.8
    mill_max = 120.0 if mill == "Plate" else 16.0
    in_mill_envelope = (t >= mill_min) and (t <= mill_max)

    # Check GRP production capability text
    cannot_supply = "can't" in prod_thick.lower() or "cannot" in prod_thick.lower()

    if cannot_supply:
        verdict = "CAPABILITY_BLOCKED"
        msg = f"GRP Note: {prod_thick}"
    elif not in_standard:
        verdict = "OUT_OF_STANDARD"
        msg = f"Standard tier requires {t_min:.1f}–{t_max:.1f} mm (actual: {t:.1f} mm)."
    elif not in_mill_envelope:
        verdict = "MILL_ENVELOPE_WARN"
        msg = f"Outside normal {mill} capability ({mill_min:.1f}–{mill_max:.1f} mm)."
    else:
        verdict = "COMPLIANT"
        msg = f"Within standard tier ({t_min:.1f}–{t_max:.1f} mm) and GRP capability ({prod_thick})."

    return {
        "verdict": verdict,
        "in_standard": in_standard,
        "in_mill_envelope": in_mill_envelope,
        "standard_min": t_min,
        "standard_max": t_max,
        "prod_thick": prod_thick,
        "message": msg,
    }


def evaluate_chemistry(row, raw_inputs, physical_indicators=None):
    """Evaluate chemical composition and carbon equivalents against standard limits.

    Splits elements into:
    - active_checks: elements that have a specified Min and/or Max
    - unconstrained: elements with no specified limit (-)
    """
    active_checks = []
    unconstrained = []
    pass_count = 0
    fail_count = 0

    all_specs = list(CHEM_ELEMENTS)
    if physical_indicators:
        all_specs += CE_ELEMENTS

    for app_key, col_min, col_max, label in all_specs:
        actual_val = float(
            raw_inputs.get(app_key, physical_indicators.get(app_key, 0.0) if physical_indicators else 0.0)
        )

        min_val, min_note = parse_limit(row.get(col_min)) if col_min else (None, "")
        max_val, max_note = parse_limit(row.get(col_max)) if col_max else (None, "")

        has_limit = (min_val is not None) or (max_val is not None)

        if not has_limit:
            unconstrained.append({
                "key": app_key,
                "label": label,
                "actual": actual_val,
            })
            continue

        # Evaluate compliance
        status = "PASS"
        margin = None
        notes = []
        if min_note:
            notes.append(f"Min footnote [{min_note}]")
        if max_note:
            notes.append(f"Max footnote [{max_note}]")

        if min_val is not None and actual_val < (min_val - 1e-4):
            status = "LOW"
            margin = actual_val - min_val
            fail_count += 1
        elif max_val is not None and actual_val > (max_val + 1e-4):
            status = "HIGH"
            margin = actual_val - max_val
            fail_count += 1
        else:
            status = "PASS"
            pass_count += 1
            if max_val is not None:
                margin = max_val - actual_val  # Headroom below max
            elif min_val is not None:
                margin = actual_val - min_val  # Margin above min

        active_checks.append({
            "key": app_key,
            "label": label,
            "actual": actual_val,
            "min": min_val,
            "max": max_val,
            "margin": margin,
            "status": status,
            "notes": ", ".join(notes),
        })

    overall_status = "ALL_PASS" if fail_count == 0 else f"{fail_count}_OUT_OF_SPEC"

    return {
        "status": overall_status,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "active_checks": active_checks,
        "unconstrained": unconstrained,
    }


def evaluate_mechanical(row, predictions, mill, rolling_type="AR"):
    """Evaluate predicted mechanical properties (with 95% bootstrap CI) against standard limits.

    Determines active route (As-rolled vs Normalised) and evaluates decency:
    - COMPLIANT: predicted mean and conservative 95% CI bound both satisfy limit
    - MARGINAL_RISK: predicted mean satisfies limit, but 95% CI dips below/above limit
    - NON_COMPLIANT: predicted mean violates limit
    """
    checks = []
    pass_count = 0
    risk_count = 0
    fail_count = 0

    supply_cond = str(row.get("SUPPLY CONDITION", "AR")).strip().upper()
    is_normalised = (rolling_type == "N") or ("N" in supply_cond and "AR" not in supply_cond and "TMCP" not in supply_cond)

    # 1. Yield Strength
    ys_target = "YS_AN" if (mill == "Plate" and is_normalised) else "YS_BN"
    ys_pred_obj = predictions.get(ys_target, {})
    ys_min_tr, ys_note_tr = parse_limit(row.get("YS min (TR) [N/mm2]"))
    ys_min_lg, ys_note_lg = parse_limit(row.get("YS min (LG) [N/mm2]"))
    ys_min = ys_min_tr if ys_min_tr is not None else ys_min_lg
    ys_dir = "TR" if ys_min_tr is not None else ("LG" if ys_min_lg is not None else "")
    ys_note = ys_note_tr if ys_min_tr is not None else ys_note_lg

    if ys_min is not None and ys_pred_obj and np.isfinite(ys_pred_obj.get("pred", np.nan)):
        pred = ys_pred_obj["pred"]
        ci_low = ys_pred_obj.get("ci_low", pred)
        ci_high = ys_pred_obj.get("ci_high", pred)
        margin = pred - ys_min

        if pred < ys_min:
            verdict = "NON_COMPLIANT"
            fail_count += 1
        elif ci_low < ys_min:
            verdict = "MARGINAL_RISK"
            risk_count += 1
        else:
            verdict = "COMPLIANT"
            pass_count += 1

        checks.append({
            "property": f"Yield strength ({'normalised' if is_normalised else 'as rolled'})",
            "key": ys_target,
            "unit": "MPa",
            "req_label": f"≥ {ys_min:.0f} MPa ({ys_dir})",
            "limit_min": ys_min,
            "limit_max": None,
            "pred": pred,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "margin": margin,
            "verdict": verdict,
            "note": f"Footnote [{ys_note}]" if ys_note else "",
        })

    # 2. Tensile Strength
    ts_target = "TS_AN" if (mill == "Plate" and is_normalised) else "TS_BN"
    ts_pred_obj = predictions.get(ts_target, {})
    ts_min_tr, ts_note_min = parse_limit(row.get("TS min (TR) [N/mm2]"))
    ts_min_lg, _ = parse_limit(row.get("TS min (LG) [N/mm2]"))
    ts_min = ts_min_tr if ts_min_tr is not None else ts_min_lg

    ts_max_tr, ts_note_max = parse_limit(row.get("TS max (TR) [N/mm2]"))
    ts_max_lg, _ = parse_limit(row.get("TS max (LG) [N/mm2]"))
    ts_max = ts_max_tr if ts_max_tr is not None else ts_max_lg
    ts_note = ts_note_min or ts_note_max

    if (ts_min is not None or ts_max is not None) and ts_pred_obj and np.isfinite(ts_pred_obj.get("pred", np.nan)):
        pred = ts_pred_obj["pred"]
        ci_low = ts_pred_obj.get("ci_low", pred)
        ci_high = ts_pred_obj.get("ci_high", pred)

        req_parts = []
        if ts_min is not None:
            req_parts.append(f"≥ {ts_min:.0f}")
        if ts_max is not None:
            req_parts.append(f"≤ {ts_max:.0f}")
        req_label = " & ".join(req_parts) + " MPa"

        # Evaluate bounds
        violates_min = (ts_min is not None) and (pred < ts_min)
        violates_max = (ts_max is not None) and (pred > ts_max)
        risk_min = (ts_min is not None) and (ci_low < ts_min)
        risk_max = (ts_max is not None) and (ci_high > ts_max)

        if violates_min or violates_max:
            verdict = "NON_COMPLIANT"
            fail_count += 1
            margin = (pred - ts_min) if violates_min else (ts_max - pred)
        elif risk_min or risk_max:
            verdict = "MARGINAL_RISK"
            risk_count += 1
            margin = (pred - ts_min) if ts_min is not None else (ts_max - pred)
        else:
            verdict = "COMPLIANT"
            pass_count += 1
            margin = (pred - ts_min) if ts_min is not None else (ts_max - pred)

        checks.append({
            "property": f"Tensile strength ({'normalised' if is_normalised else 'as rolled'})",
            "key": ts_target,
            "unit": "MPa",
            "req_label": req_label,
            "limit_min": ts_min,
            "limit_max": ts_max,
            "pred": pred,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "margin": margin,
            "verdict": verdict,
            "note": f"Footnote [{ts_note}]" if ts_note else "",
        })

    # 3. Elongation
    el_target = "EL_AN" if (mill == "Plate" and is_normalised) else "EL_BN"
    el_pred_obj = predictions.get(el_target, {})

    # Check available elongation types in order: 5.65√So, 200, 50, 80
    el_col_matches = [
        ("EL 5.65√So (TR) [%]", "5.65√S₀ (TR)"),
        ("EL 200 (TR) [%]", "200 mm (TR)"),
        ("EL 50 (TR) [%]", "50 mm (TR)"),
        ("EL 80 (TR) [%]", "80 mm (TR)"),
        ("EL 5.65√So (LG) [%]", "5.65√S₀ (LG)"),
        ("EL 200 (LG) [%]", "200 mm (LG)"),
        ("EL 50 (LG) [%]", "50 mm (LG)"),
    ]
    el_min = None
    el_gauge_type = ""
    el_note = ""
    for col, g_type in el_col_matches:
        v, note = parse_limit(row.get(col))
        if v is not None:
            el_min = v
            el_gauge_type = g_type
            el_note = note
            break

    if el_min is not None and el_pred_obj and np.isfinite(el_pred_obj.get("pred", np.nan)):
        pred = el_pred_obj["pred"]
        ci_low = el_pred_obj.get("ci_low", pred)
        ci_high = el_pred_obj.get("ci_high", pred)
        margin = pred - el_min

        if pred < el_min:
            verdict = "NON_COMPLIANT"
            fail_count += 1
        elif ci_low < el_min:
            verdict = "MARGINAL_RISK"
            risk_count += 1
        else:
            verdict = "COMPLIANT"
            pass_count += 1

        checks.append({
            "property": f"Elongation ({el_gauge_type})",
            "key": el_target,
            "unit": "%",
            "req_label": f"≥ {el_min:.1f} %",
            "limit_min": el_min,
            "limit_max": None,
            "pred": pred,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "margin": margin,
            "verdict": verdict,
            "note": f"Footnote [{el_note}]" if el_note else "",
        })

    # 4. Impact Toughness (Plate route only)
    if mill == "Plate":
        impact_map = [
            ("Impact_0C", "Impact 0 (TR) [J]", "Impact 0 (LG) [J]", "Charpy 0 °C"),
            ("Impact_m20C", "Impact -20 (TR) [J]", "Impact -20 (LG) [J]", "Charpy -20 °C"),
            ("Impact_m40C", "Impact -40 (TR) [J]", "Impact -40 (LG) [J]", "Charpy -40 °C"),
        ]
        for imp_key, col_tr, col_lg, imp_label in impact_map:
            imp_tr, imp_note_tr = parse_limit(row.get(col_tr))
            imp_lg, imp_note_lg = parse_limit(row.get(col_lg))
            imp_min = imp_tr if imp_tr is not None else imp_lg
            imp_dir = "TR" if imp_tr is not None else ("LG" if imp_lg is not None else "")
            imp_note = imp_note_tr if imp_tr is not None else imp_note_lg

            imp_pred_obj = predictions.get(imp_key, {})
            if imp_min is not None and imp_pred_obj and np.isfinite(imp_pred_obj.get("pred", np.nan)):
                pred = imp_pred_obj["pred"]
                ci_low = imp_pred_obj.get("ci_low", pred)
                ci_high = imp_pred_obj.get("ci_high", pred)
                margin = pred - imp_min

                if pred < imp_min:
                    verdict = "NON_COMPLIANT"
                    fail_count += 1
                elif ci_low < imp_min:
                    verdict = "MARGINAL_RISK"
                    risk_count += 1
                else:
                    verdict = "COMPLIANT"
                    pass_count += 1

                checks.append({
                    "property": f"{imp_label} ({imp_dir})",
                    "key": imp_key,
                    "unit": "J",
                    "req_label": f"≥ {imp_min:.0f} J",
                    "limit_min": imp_min,
                    "limit_max": None,
                    "pred": pred,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "margin": margin,
                    "verdict": verdict,
                    "note": f"Footnote [{imp_note}]" if imp_note else "",
                })

    if fail_count > 0:
        overall_status = "NON_COMPLIANT"
    elif risk_count > 0:
        overall_status = "MARGINAL_RISK"
    elif pass_count > 0:
        overall_status = "COMPLIANT"
    else:
        overall_status = "NO_REQUIREMENTS_SPECIFIED"

    return {
        "status": overall_status,
        "is_normalised": is_normalised,
        "pass_count": pass_count,
        "risk_count": risk_count,
        "fail_count": fail_count,
        "checks": checks,
    }


def evaluate_all(spec=None, grade=None, thickness=None, raw_inputs=None, predictions=None,
                 physical_indicators=None, mill="Plate"):
    """Perform complete end-to-end specification validation."""
    matched_row, match_info = find_matching_spec_row(spec, grade, thickness)
    if matched_row is None:
        return {
            "matched": False,
            "message": match_info.get("message", "Specification could not be found."),
        }

    gauge_eval = evaluate_thickness(matched_row, thickness, mill)
    chem_eval = evaluate_chemistry(matched_row, raw_inputs, physical_indicators)
    rolling_type = str(raw_inputs.get("Rolling_Type", "AR"))
    mech_eval = evaluate_mechanical(matched_row, predictions, mill, rolling_type)

    notes = str(matched_row.get("NOTES", "")).strip()
    if notes in ("", "-", "nan", "None"):
        notes = ""

    return {
        "matched": True,
        "match_info": match_info,
        "row": matched_row,
        "spec_type": str(matched_row.get("TYPE", "-")),
        "supply_condition": str(matched_row.get("SUPPLY CONDITION", "-")),
        "general_purpose": str(matched_row.get("GENERAL PURPOSE", "-")),
        "notes": notes,
        "gauge": gauge_eval,
        "chemistry": chem_eval,
        "mechanical": mech_eval,
    }


def spec_available():
    """The GRP standard workbook is not published with this repository."""
    return os.path.exists(SPEC_FILE_PATH)
