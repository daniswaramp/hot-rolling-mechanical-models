"""Training-set statistics for the sidebar.

Each input in the schedule form gets a button that shows the distribution of that
variable in the cleaned training table, narrowed to heats rolled at a similar gauge
and width. The band widens until enough rows are left, and the caption says which
band was used so a thin sample is never presented as the whole population.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)

TABLES = {
    "Plate": os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
    "HRC": os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"),
}

GEOMETRY = {
    "Plate": ["Plate_T_mm", "Plate_W_mm"],
    "HRC": ["Coil_T_mm", "Coil_W_mm"],
}

MIN_ROWS = 40


@st.cache_data(show_spinner=False)
def load_table(mill):
    path = TABLES.get(mill)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError):
        return pd.DataFrame()


def _numeric(series):
    """Numbers, including cells stored as text such as '2H' for two laminar headers."""
    direct = pd.to_numeric(series, errors="coerce")
    if direct.notna().mean() > 0.5:
        return direct
    text = series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)")[0]
    return pd.to_numeric(text, errors="coerce")


def _within(df, mill, raw, tolerance):
    out = df
    for key in GEOMETRY.get(mill, []):
        value = raw.get(key)
        if value is None or key not in out.columns:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        column = _numeric(out[key])
        band = abs(value) * tolerance
        if band <= 0:
            continue
        out = out[(column >= value - band) & (column <= value + band)]
    return out


def _decimals(step):
    try:
        step = float(step)
    except (TypeError, ValueError):
        return 3
    if step <= 0 or step >= 1:
        return 0
    return int(max(0, min(6, round(-np.log10(step)))))


def variable_stats(mill, key, raw, step=None, bounds=None):
    """Min, quartiles, max, mode and mean for one input, filtered by the entered geometry.

    Values outside the mill's declared range are dropped first: the cleaned table still
    carries a few decimal-shift typos (an 9111 C finishing temperature, a 0 % carbon heat)
    and they would wreck the summary.
    """
    table = load_table(mill)
    if table.empty or key not in table.columns:
        return None

    subset, band = None, "all rows"
    for tolerance in (0.10, 0.25):
        candidate = _within(table, mill, raw, tolerance)
        if len(candidate) >= MIN_ROWS:
            subset = candidate
            band = f"within {int(tolerance * 100)}% of the entered thickness and width"
            break
    if subset is None:
        subset = table

    values = _numeric(subset[key]).dropna()
    dropped = 0
    if bounds is not None:
        low, high = float(bounds[0]), float(bounds[1])
        keep = (values >= low) & (values <= high)
        dropped = int((~keep).sum())
        values = values[keep]
    if values.empty:
        return None

    rounded = values.round(_decimals(step))
    modes = rounded.mode()
    return {
        "rows": int(values.size),
        "dropped": dropped,
        "band": band,
        "min": float(values.min()),
        "q1": float(values.quantile(0.25)),
        "median": float(values.median()),
        "q3": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mode": float(modes.iloc[0]) if not modes.empty else float("nan"),
        "mean": float(values.mean()),
    }


@st.cache_data(show_spinner=False)
def cached_stats(mill, key, geometry, step, bounds=None):
    """Cache on primitives so a rerun does not refilter the table twenty times."""
    return variable_stats(mill, key, dict(geometry), step, bounds)


def summary_frame(stats, unit=""):
    labels = ["Min", "Q1", "Median (Q2)", "Q3", "Max", "Mode", "Mean"]
    values = [stats["min"], stats["q1"], stats["median"], stats["q3"],
              stats["max"], stats["mode"], stats["mean"]]
    pretty = []
    for value in values:
        if value is None or not np.isfinite(value):
            pretty.append("—")
        elif abs(value) >= 100:
            pretty.append(f"{value:,.0f}")
        elif abs(value) >= 1:
            pretty.append(f"{value:,.2f}")
        else:
            pretty.append(f"{value:.4f}")
    return pd.DataFrame([pretty], columns=labels)
