"""Observed property means along a swept driver, for the 1D slice overlay.

Reads models/trend_stats.json, built by app/build_trend_bundle.py from the cleaned
training tables. Returns an empty list when the bundle is absent so the deployment
degrades to the model line only.
"""

import json
import os

import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
BUNDLE_PATH = os.path.join(ROOT, "models", "trend_stats.json")

THICKNESS = {"Plate": "Plate_T_mm", "HRC": "Coil_T_mm"}


@st.cache_data(show_spinner=False)
def load():
    try:
        with open(BUNDLE_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def bins_for(mill, x_key, thickness=None):
    """Observed bins for one driver, on the thickness band nearest the entered gauge."""
    payload = load().get(mill)
    if not payload:
        return []
    entry = payload.get("variables", {}).get(x_key)
    if not entry:
        return []

    if x_key == THICKNESS.get(mill) or thickness is None:
        return entry.get("all", [])

    try:
        gauge = float(thickness)
    except (TypeError, ValueError):
        return entry.get("all", [])

    bands = entry.get("by_thickness") or []
    for i, band in enumerate(bands):
        last = i == len(bands) - 1
        hit = (band["lo"] <= gauge <= band["hi"]) if last else (band["lo"] <= gauge < band["hi"])
        if hit:
            return band.get("bins", [])
    return entry.get("all", [])
