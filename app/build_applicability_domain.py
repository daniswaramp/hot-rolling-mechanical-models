"""Rebuild models/applicability_domain.json from the cleaned training tables.

The shipped envelope was computed from the raw logged columns, so a few entries carry the very
contamination the sanitising pass removes: Plate ACC_Laminar_Header holds header counts swapped
with temperatures (p01 671.67 on a 0-20 column), HRC coiling temperatures hold blank cells logged
as 0 C, and several maxima are decimal-shift typos. The result was a false extrapolation alert on
every ordinary Plate schedule and no low-side monitoring on Steckel coiling temperature.

Run this where the campaign sheets live:

    python app/build_applicability_domain.py

It applies each mill's PHYSICAL_BOUNDS first, then takes P01/P05/P50/P95/P99 plus min/max of the
cleaned columns the app actually accepts as inputs. Columns that are still unusable (a percentile
of exactly 0 on a temperature, a count above the mill's hardware) are dropped with a reason
written into the _caveats block rather than shipped as a wrong envelope.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
DEST = os.path.join(ROOT, "models", "applicability_domain.json")

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Plate"))
sys.path.insert(0, os.path.join(ROOT, "HRC", "script"))

from HRC.script.ml.config import PHYSICAL_BOUNDS as HRC_BOUNDS  # noqa: E402

TABLES = {
    "Plate": os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
    "HRC": os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"),
}

# the raw input columns the app exposes, per mill (mirrors streamlit_app.SPECS minus Rolling_Type)
COLUMNS = {
    "Plate": ["Plate_T_mm", "Plate_W_mm", "Slab_T_mm", "TB_Thickness_mm",
              "FM_Temp_Before_Rolling", "FM_Finish_Temp", "FM_Last_Pass_Drafting",
              "ACC_Laminar_Header", "ACC_Temp_Post_Laminar",
              "Chem_C", "Chem_Mn", "Chem_Si", "Chem_P", "Chem_S", "Chem_Al", "Chem_Nb",
              "Chem_V", "Chem_Ti", "Chem_Cu", "Chem_Ni", "Chem_Cr", "Chem_Mo", "Chem_N"],
    "HRC": ["Coil_T_mm", "Coil_W_mm", "TB_Thick_mm",
            "Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_CT_Bot_C",
            "Chem_C", "Chem_Mn", "Chem_Si", "Chem_P", "Chem_S", "Chem_Al", "Chem_Nb",
            "Chem_V", "Chem_Ti", "Chem_Cu", "Chem_Ni", "Chem_Cr", "Chem_Mo", "Chem_N", "Chem_B"],
}

# a count, not a temperature: the mill never runs more headers than this
HARDWARE_CAPS = {"ACC_Laminar_Header": 20.0}


def percentiles(values):
    q = np.percentile(values, [1, 5, 50, 95, 99])
    return {"min": float(values.min()), "p01": float(q[0]), "p05": float(q[1]),
            "p50": float(q[2]), "p95": float(q[3]), "p99": float(q[4]),
            "max": float(values.max())}


def usable(column, values):
    """A percentile envelope is only worth shipping if it can actually bound an input."""
    if column in HARDWARE_CAPS and values.max() > HARDWARE_CAPS[column]:
        return (f"logged max {values.max():.1f} exceeds the hardware cap "
                f"{HARDWARE_CAPS[column]:.0f}; the column is contaminated")
    if "Temp" in column and values.min() <= 0.0:
        return "blank cells were logged as 0 C, so the low side cannot bound an input"
    return None


def build_mill(mill):
    path = TABLES[mill]
    if not os.path.exists(path):
        raise SystemExit(f"{mill}: cleaned table not found at {path} — run the sanitising pass first")
    frame = pd.read_csv(path, low_memory=False)
    out, caveats = {}, {}
    for column in COLUMNS[mill]:
        if column not in frame.columns:
            caveats[f"{mill}.{column}"] = "column absent from the cleaned table"
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        lo, hi = HRC_BOUNDS.get(column, (None, None)) if mill == "HRC" else (None, None)
        if lo is not None:
            values = values[(values >= lo) & (values <= hi)]
        if values.size < 40:
            caveats[f"{mill}.{column}"] = f"only {values.size} usable rows"
            continue
        reason = usable(column, values)
        if reason:
            caveats[f"{mill}.{column}"] = reason
            continue
        out[column] = percentiles(values.to_numpy())
    return out, caveats


def main():
    domain, caveats = {}, {}
    for mill in ("Plate", "HRC"):
        envelope, why = build_mill(mill)
        domain[mill] = envelope
        for key, reason in why.items():
            caveats[key] = reason

    payload = {
        "registry_version": "3.0",
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        **domain,
        "_caveats": {
            "note": "Rebuilt from the cleaned training tables with PHYSICAL_BOUNDS applied, "
                    "by app/build_applicability_domain.py.",
            "dropped": {k: v for k, v in caveats.items()},
        },
    }
    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", DEST)
    for key, reason in caveats.items():
        print(f"  dropped {key}: {reason}")


if __name__ == "__main__":
    main()
