"""Build models/trend_stats.json: observed property trends along each process driver.

The 1D slice draws the model through a swept input. That line is piecewise constant and
wiggles by a few MPa, which reads as a wrong physical law unless the observed scatter sits
beside it.

Raw means are unusable for that: in the 15-18 mm band the plant runs controlled-rolled
steel (Mn ~1.3, CE ~0.40) below 830 C and as-rolled steel (Mn ~0.6, CE ~0.28) above it, so
the unadjusted average drops 65 MPa across a route change, not a temperature change.

Each bin therefore carries a partial-residual mean: the target is regressed on every other
numeric driver, the swept driver's fitted contribution is added back, and the result is
shifted to the target's overall level. That isolates the driver's own shape. The UI labels
it as adjusted.

    python app/build_trend_bundle.py
"""

import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

import training_stats as ts  # noqa: E402

OUT = os.path.join(ROOT, "models", "trend_stats.json")

THICKNESS = {"Plate": "Plate_T_mm", "HRC": "Coil_T_mm"}

VARIABLES = {
    "Plate": ["FM_Finish_Temp", "FM_Temp_Before_Rolling", "ACC_Temp_Post_Laminar",
              "ACC_Laminar_Header", "Plate_T_mm", "Chem_C", "Chem_Mn", "Chem_Nb"],
    "HRC": ["Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_Transfer_Bar_C",
            "Coil_T_mm", "Chem_C", "Chem_Mn", "Chem_Nb"],
}

FEATURES = {
    "Plate": ["Slab_T_mm", "Plate_T_mm", "Plate_W_mm", "TB_Ratio",
              "FM_Temp_Before_Rolling", "FM_Finish_Temp", "FM_Last_Pass_Drafting",
              "ACC_Laminar_Header", "ACC_Temp_Post_Laminar",
              "Chem_C", "Chem_Mn", "Chem_Si", "Chem_Al", "Chem_S", "Chem_P",
              "Chem_Nb", "Chem_V", "Chem_Mo", "Chem_Ti", "Chem_Cu", "Chem_Ni",
              "Chem_Cr", "Chem_N", "Chem_CE"],
    "HRC": ["Coil_T_mm", "Coil_W_mm", "TB_Thick_mm", "Temp_Transfer_Bar_C",
            "Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_CT_Bot_C",
            "Chem_C", "Chem_Mn", "Chem_Si", "Chem_Al", "Chem_S", "Chem_P",
            "Chem_Nb", "Chem_V", "Chem_Ti", "Chem_N", "Chem_CE"],
}

TARGETS = {
    "Plate": {
        "YS_BN": "YS_Before_N_MPa", "TS_BN": "TS_Before_N_MPa", "EL_BN": "EL_Before_N_pct",
        "YS_AN": "YS_After_N_MPa", "TS_AN": "TS_After_N_MPa", "EL_AN": "EL_After_N_pct",
        "Impact_0C": "Impact_0C_E_Avg_J", "Impact_m20C": "Impact_m20C_E_Avg_J",
        "Impact_m40C": "Impact_m40C_E_Avg_J", "Z_Test": "Z_Test_Avg_pct",
    },
    "HRC": {"YS_BN": "YS_BN_MPa", "TS_BN": "TS_BN_MPa", "EL_BN": "EL_BN_pct"},
}

BINS_X = 12
BINS_THICKNESS = 9
MIN_ROWS = 20
METHOD = "partial residual: target on all other numeric drivers, swept driver added back"


def design(table, mill):
    frames = [ts._numeric(table[name]).rename(name) for name in FEATURES[mill]
              if name in table.columns]
    X = pd.concat(frames, axis=1)
    X = X.fillna(X.median(numeric_only=True))
    if mill == "Plate" and "Rolling_Type" in table.columns:
        X = pd.concat([X, pd.get_dummies(table["Rolling_Type"].fillna("UNKNOWN"),
                                         prefix="route", dtype=float)], axis=1)
    return X


def driver_series(table, mill, column):
    """{driver: residual + driver contribution} plus the target's overall level."""
    y = ts._numeric(table[column])
    mask = y.notna() & np.isfinite(y)
    if mask.sum() < 200:
        return None, None

    X = design(table, mill)
    names = list(X.columns)
    Xm = X[mask].to_numpy(float)
    ym = y[mask].to_numpy(float)
    A = np.column_stack([np.ones(len(Xm)), Xm])
    beta, *_ = np.linalg.lstsq(A, ym, rcond=None)
    fitted = A @ beta

    per_driver = {}
    for j, name in enumerate(names):
        centred = Xm[:, j] - Xm[:, j].mean()
        series = pd.Series(np.nan, index=table.index)
        series[mask] = ym - fitted + centred * beta[j + 1]
        per_driver[name] = series
    return per_driver, float(ym.mean())


def bin_edges(values):
    clean = values.dropna()
    if len(clean) < BINS_X * MIN_ROWS:
        return []
    edges = np.unique(np.quantile(clean, np.linspace(0, 1, BINS_X + 1)))
    return [float(e) for e in edges] if len(edges) >= 3 else []


def bins_for(x, rows_mask, adjusted, level, driver):
    edges = bin_edges(x[rows_mask])
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        mask = rows_mask & (x >= lo) & ((x <= hi) if last else (x < hi))
        if mask.sum() < MIN_ROWS:
            continue
        targets = {}
        for key, per_driver in adjusted.items():
            series = per_driver.get(driver)
            if series is None:
                continue
            values = (series[mask] + level[key]).dropna()
            if values.empty:
                continue
            targets[key] = {
                "mean": round(float(values.mean()), 2),
                "se": round(float(values.std(ddof=1) / np.sqrt(values.size)), 2),
                "n": int(values.size),
            }
        if targets:
            out.append({"x": round((lo + hi) / 2.0, 2), "lo": round(lo, 2),
                        "hi": round(hi, 2), "targets": targets})
    return out


def build(mill):
    table = ts.load_table(mill)
    if table.empty:
        print(f"  {mill}: no cleaned table, skipped")
        return None

    thickness = ts._numeric(table[THICKNESS[mill]])
    edges = []
    if thickness.notna().sum() > BINS_THICKNESS * MIN_ROWS:
        edges = sorted({round(float(v), 2) for v in
                        thickness.quantile(np.linspace(0, 1, BINS_THICKNESS + 1)).dropna()})

    adjusted, level = {}, {}
    for key, column in TARGETS[mill].items():
        if column not in table.columns:
            continue
        per_driver, mean = driver_series(table, mill, column)
        if per_driver is None:
            continue
        adjusted[key], level[key] = per_driver, mean

    rows = pd.Series(True, index=table.index)
    variables = {}
    for variable in VARIABLES[mill]:
        if variable not in table.columns:
            continue
        x = ts._numeric(table[variable])
        entry = {"all": bins_for(x, rows, adjusted, level, variable), "by_thickness": []}
        if variable != THICKNESS[mill] and len(edges) > 2:
            for i in range(len(edges) - 1):
                lo, hi = edges[i], edges[i + 1]
                last = i == len(edges) - 2
                band = (thickness >= lo) & ((thickness <= hi) if last else (thickness < hi))
                bins = bins_for(x, band, adjusted, level, variable)
                if bins:
                    entry["by_thickness"].append({"lo": lo, "hi": hi, "bins": bins})
        if entry["all"]:
            variables[variable] = entry
            print(f"  {mill} {variable}: {len(entry['all'])} bins, "
                  f"{len(entry['by_thickness'])} thickness bands")

    return {"rows": int(len(table)), "thickness_edges": edges, "variables": variables}


def main():
    bundle = {"generated": date.today().isoformat(), "bins": BINS_X,
              "min_rows": MIN_ROWS, "method": METHOD}
    for mill in ("Plate", "HRC"):
        payload = build(mill)
        if payload:
            bundle[mill] = payload

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"written: {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
