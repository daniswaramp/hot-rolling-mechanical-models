"""Build models/training_stats.json from the cleaned training tables.

The Streamlit deployment does not ship the raw CSVs (they are production heats), so the
sidebar statistics fall back to this bundled summary. It holds the same figures as the
local path, bucketed by finished thickness so the numbers still respond to the gauge
entered in the form.

Run from the repo root or the app folder:

    python app/build_stats_bundle.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

import training_stats as ts  # noqa: E402
from streamlit_app import SPECS  # noqa: E402  (only the input table is used)

OUT = os.path.join(os.path.dirname(APP_DIR), "models", "training_stats.json")

THICKNESS_KEY = {"Plate": "Plate_T_mm", "HRC": "Coil_T_mm"}
BINS = 9
MIN_ROWS = ts.MIN_ROWS


def summarise(values, step):
    values = values.dropna()
    if values.empty:
        return None
    rounded = values.round(ts._decimals(step))
    modes = rounded.mode()
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mode": float(modes.iloc[0]) if not modes.empty else None,
        "mean": float(values.mean()),
    }


def build(mill):
    table = ts.load_table(mill)
    if table.empty:
        print(f"  {mill}: no cleaned table, skipped")
        return None

    thickness_col = THICKNESS_KEY[mill]
    thickness = ts._numeric(table[thickness_col]) if thickness_col in table.columns else None
    edges = []
    if thickness is not None and thickness.notna().sum() > BINS * MIN_ROWS:
        quantiles = np.linspace(0, 1, BINS + 1)
        edges = sorted({float(v) for v in thickness.quantile(quantiles).dropna().values})
        edges = [round(e, 2) for e in edges]

    variables = {}
    spec_rows = [r for r in SPECS[mill]]
    for _group, key, _label, _unit, lo, hi, _default, step in spec_rows:
        if key not in table.columns:
            continue
        values = ts._numeric(table[key])
        keep = (values >= float(lo)) & (values <= float(hi))
        values = values[keep]

        entry = {"all": summarise(values, step), "bins": []}
        if edges and len(edges) > 2:
            for i in range(len(edges) - 1):
                low, high = edges[i], edges[i + 1]
                last = i == len(edges) - 2
                mask = (thickness >= low) & ((thickness <= high) if last else (thickness < high))
                bucket = summarise(values[mask.reindex(values.index, fill_value=False)], step)
                entry["bins"].append({"lo": low, "hi": high, "stats": bucket})

        if entry["all"] is not None:
            variables[key] = entry

    return {"rows": int(len(table)), "thickness_edges": edges, "variables": variables}


def main():
    bundle = {"generated_from": "cleaned training tables", "bins": BINS}
    for mill in ("Plate", "HRC"):
        payload = build(mill)
        if payload:
            bundle[mill] = payload
            print(f"  {mill}: {payload['rows']:,} rows, {len(payload['variables'])} variables, "
                  f"{max(0, len(payload['thickness_edges']) - 1)} thickness bins")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"written: {OUT} ({os.path.getsize(OUT)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
