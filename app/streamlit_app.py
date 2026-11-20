"""Plate Mill & Steckel Mill Prediction Platform: fresh technical white UI.

Heavy plate and Steckel mill coil, one model layer. Sections: simulator, response surface, batch,
retrain, diagnostics.
"""

import base64
import html as html_lib
import io
import json
import math
import os
import sys
from datetime import datetime

# Cap the BLAS/OpenMP worker pools BEFORE numpy is imported: each OpenBLAS
# worker thread commits ~28 MB of buffer memory on a many-core host, which by
# itself blew past the Streamlit Community Cloud free-tier budget (measured:
# numpy+pandas 271 MB uncapped vs 79 MB at 2 threads on 8 cores). This app's
# linear algebra is trivial (tree-model prediction, isotonic PAV), so 2 worker
# threads cost nothing in latency and leave the in-app refit some parallelism.
# setdefault keeps an operator-chosen setting in the deployment env authoritative.
for _thread_cap in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_cap, "2")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# NOTE: scipy is intentionally NOT imported at module level — its committed
# private memory is substantial and only monotone_guardrail (header-sweep
# isotonic correction) needs it. Imported on first use; Python caches it.

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import importlib


def _maybe_reload(mod):
    """Reload a sibling app module only when HRCM_DEV_RELOAD=1, avoiding cache resets."""
    if os.environ.get("HRCM_DEV_RELOAD") == "1":
        try:
            importlib.reload(mod)
        except Exception:
            pass
    return mod


import model_loader as ml  # noqa: E402
_maybe_reload(ml)

import spec_checker as sc  # noqa: E402
_maybe_reload(sc)

import explainer  # noqa: E402
_maybe_reload(explainer)

import optimizer  # noqa: E402
_maybe_reload(optimizer)

import constraints as cons  # noqa: E402
_maybe_reload(cons)

import training_stats as ts  # noqa: E402
_maybe_reload(ts)

import grade_templates as gt  # noqa: E402
_maybe_reload(gt)

import trend_stats as tst  # noqa: E402
_maybe_reload(tst)

import physics  # noqa: E402
_maybe_reload(physics)

import precipitation  # noqa: E402
_maybe_reload(precipitation)

# Heavy calculation dispatch (Modal when configured, this process otherwise). Importing it is
# free: the modal package inside it is only imported when a calculation is actually asked for.
import remote_compute  # noqa: E402
_maybe_reload(remote_compute)

# What-if atlas (stored four-axis response grid) and digital shadow (bias against measured
# results). Importing either is free: modal is only imported inside remote_jobs when a build
# or a calibration is actually asked for, and both modules import this one lazily.
import atlas  # noqa: E402
_maybe_reload(atlas)

import shadow  # noqa: E402
_maybe_reload(shadow)

# Pass schedule recrystallization & penetration screening. Importing it is free: the tab is
# closed-form numpy behind its own submit button and imports precipitation read-only.
import pass_schedule  # noqa: E402
_maybe_reload(pass_schedule)

st.set_page_config(
    page_title="Plate Mill & Steckel Mill Prediction Platform",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUILD = "2026-11-20 · 4-pillar upgrade: DBTT 18-col impact · correlated CVaR & capability · ladle triage · shadow EWMA drift · schedule A/B & setup card"
LOG_PATH = os.path.join(APP_DIR, "training_log.json")
LOGO_PATH = os.path.join(os.path.dirname(APP_DIR), "assets", "grp.png")
INK = "#0b0d10"
MUTED = "#6b7280"
HAIR = "#e5e7eb"
ACCENT = "#1746c8"   # GRP brand blue, taken from the mill mark (assets/grp.png)
PANEL = "#f7f8fa"


def _shade(hex_color, factor):
    """Lighter (factor > 0, toward white) or darker (factor < 0, toward black) step of a hex."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + factor)) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _rgba(hex_color, alpha):
    """Plotly-friendly rgba() string of a token hex, so chart fills stay on the palette."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# Ordered chart ramp derived from the accent: darker -> base -> lighter.
ACCENT_DARK = _shade(ACCENT, -0.45)
ACCENT_RAMP = [ACCENT, _shade(ACCENT, 0.30), _shade(ACCENT, 0.55),
               _shade(ACCENT, 0.75), _shade(ACCENT, 0.88)]

# Semantic state tokens, matching the pill colours (#065f46 / #92400e / #991b1b family).
SEMANTIC_OK = "#065f46"
SEMANTIC_WARN = "#92400e"
SEMANTIC_BAD = "#991b1b"
WARN_LINE = "#b45309"    # limit / reference lines on charts
WARN_STAR = "#f59e0b"    # optimal-setpoint marker
BAD_LINE = "#f87171"     # failing-coverage bars on charts

# Multi-series chart palette: neutral + accent ramp + the two semantic counter-hues.
CHART_SERIES = [INK, ACCENT, ACCENT_RAMP[2], ACCENT_RAMP[3], WARN_LINE, SEMANTIC_OK]


# ----------------------------------------------------------------------------- input specs
PLATE_SPEC = [
    ("Geometry", "Slab_T_mm", "Slab thickness", "mm", 120.0, 320.0, 218.0, 1.0),
    ("Geometry", "Plate_T_mm", "Plate thickness", "mm", 6.0, 120.0, 16.0, 0.5),
    ("Geometry", "Plate_W_mm", "Plate width", "mm", 1000.0, 4000.0, 1889.0, 10.0),
    ("Geometry", "TB_Thickness_mm", "Transfer bar thickness", "mm", 15.0, 120.0, 41.0, 0.5),
    ("Finishing mill", "FM_Temp_Before_Rolling", "Entry temperature", "C", 850.0, 1150.0, 985.0, 1.0),
    ("Finishing mill", "FM_Finish_Temp", "Finishing temperature", "C", 700.0, 1050.0, 832.0, 1.0),
    ("Finishing mill", "FM_Last_Pass_Drafting", "Last pass drafting", "mm", 0.5, 25.0, 4.0, 0.5),
    ("Cooling", "ACC_Laminar_Header", "Laminar headers", "count", 0.0, 8.0, 8.0, 1.0),
    ("Cooling", "ACC_Temp_Post_Laminar", "Post-laminar temperature", "C", 200.0, 850.0, 744.0, 1.0),
    ("Chemistry", "Chem_C", "C", "wt%", 0.02, 0.35, 0.177, 0.005),
    ("Chemistry", "Chem_Mn", "Mn", "wt%", 0.2, 2.0, 0.613, 0.01),
    ("Chemistry", "Chem_Si", "Si", "wt%", 0.02, 0.9, 0.151, 0.005),
    ("Chemistry", "Chem_P", "P", "wt%", 0.001, 0.04, 0.0121, 0.001),
    ("Chemistry", "Chem_S", "S", "wt%", 0.001, 0.03, 0.005, 0.001),
    ("Chemistry", "Chem_Al", "Al", "wt%", 0.005, 0.12, 0.03, 0.001),
    ("Chemistry", "Chem_Nb", "Nb", "wt%", 0.0, 0.09, 0.002, 0.001),
    ("Chemistry", "Chem_V", "V", "wt%", 0.0, 0.12, 0.0039, 0.001),
    ("Chemistry", "Chem_Ti", "Ti", "wt%", 0.0, 0.06, 0.002, 0.001),
    ("Chemistry", "Chem_Cu", "Cu", "wt%", 0.0, 0.6, 0.0175, 0.005),
    ("Chemistry", "Chem_Ni", "Ni", "wt%", 0.0, 1.2, 0.02, 0.005),
    ("Chemistry", "Chem_Cr", "Cr", "wt%", 0.0, 1.2, 0.043, 0.005),
    ("Chemistry", "Chem_Mo", "Mo", "wt%", 0.0, 0.5, 0.01, 0.001),
    ("Chemistry", "Chem_N", "N", "wt%", 0.001, 0.02, 0.0042, 0.0005),
]

HRC_SPEC = [
    ("Geometry", "Coil_T_mm", "Coil thickness", "mm", 1.8, 16.0, 6.0, 0.1),
    ("Geometry", "Coil_W_mm", "Coil width", "mm", 700.0, 1600.0, 1200.0, 10.0),
    ("Geometry", "TB_Thick_mm", "Transfer bar thickness", "mm", 15.0, 105.0, 24.8, 0.5),
    ("Steckel mill", "Temp_Transfer_Bar_C", "Transfer bar temperature", "C", 850.0, 1150.0, 937.0, 1.0),
    ("Steckel mill", "Temp_FT_Avg_C", "Finishing temperature", "C", 750.0, 1000.0, 835.0, 1.0),
    ("Cooling", "Temp_CT_Top_C", "Coiling temperature, top", "C", 400.0, 750.0, 595.0, 1.0),
    ("Cooling", "Temp_CT_Bot_C", "Coiling temperature, bottom", "C", 400.0, 750.0, 558.0, 1.0),
    ("Chemistry", "Chem_C", "C", "wt%", 0.02, 0.35, 0.176, 0.005),
    ("Chemistry", "Chem_Mn", "Mn", "wt%", 0.2, 2.0, 0.581, 0.01),
    ("Chemistry", "Chem_Si", "Si", "wt%", 0.02, 0.9, 0.127, 0.005),
    ("Chemistry", "Chem_P", "P", "wt%", 0.001, 0.04, 0.015, 0.001),
    ("Chemistry", "Chem_S", "S", "wt%", 0.001, 0.03, 0.005, 0.001),
    ("Chemistry", "Chem_Al", "Al", "wt%", 0.005, 0.12, 0.036, 0.001),
    ("Chemistry", "Chem_Nb", "Nb", "wt%", 0.0, 0.09, 0.001, 0.001),
    ("Chemistry", "Chem_V", "V", "wt%", 0.0, 0.12, 0.004, 0.001),
    ("Chemistry", "Chem_Ti", "Ti", "wt%", 0.0, 0.06, 0.002, 0.001),
    ("Chemistry", "Chem_Cu", "Cu", "wt%", 0.0, 0.6, 0.02, 0.005),
    ("Chemistry", "Chem_Ni", "Ni", "wt%", 0.0, 1.2, 0.02, 0.005),
    ("Chemistry", "Chem_Cr", "Cr", "wt%", 0.0, 1.2, 0.03, 0.005),
    ("Chemistry", "Chem_Mo", "Mo", "wt%", 0.0, 0.5, 0.005, 0.001),
    ("Chemistry", "Chem_N", "N", "wt%", 0.001, 0.02, 0.004, 0.0005),
    ("Chemistry", "Chem_B", "B", "wt%", 0.0, 0.005, 0.0, 0.0002),
]


def _validated_specs(rows, mill):
    """Coerce every spec row once at import: 8 fields, numeric bounds, sane defaults.

    A malformed row would otherwise raise inside schedule_inputs' tuple unpacking and take
    both routes down with it, so repair here and say what moved.
    """
    out, repaired = [], []
    for row in rows:
        key = row[1] if len(row) > 1 else repr(row)
        if len(row) != 8:
            repaired.append(f"{key}: {len(row)} fields")
            row = tuple(row[:8]) + (0.0,) * max(0, 8 - len(row))
        group, key, label, unit, lo, hi, default, step = row
        try:
            lo, hi, default, step = float(lo), float(hi), float(default), float(step)
        except (TypeError, ValueError):
            repaired.append(f"{key}: non-numeric bounds")
            lo, hi, default, step = 0.0, 1.0, 0.0, 0.1
        if hi <= lo:
            repaired.append(f"{key}: max <= min")
            hi = lo + 1.0
        if not lo <= default <= hi:
            repaired.append(f"{key}: default outside range")
            default = min(max(default, lo), hi)
        if step <= 0:
            repaired.append(f"{key}: step <= 0")
            step = (hi - lo) / 100.0
        out.append((str(group), str(key), str(label), str(unit), lo, hi, default, step))
    if repaired:
        print(f"[{mill}] repaired specification rows: " + "; ".join(repaired), file=sys.stderr)
    return out


SPECS = {"Plate": _validated_specs(PLATE_SPEC, "Plate"), "HRC": _validated_specs(HRC_SPEC, "HRC")}

SURFACE = {
    "Plate": ["FM_Finish_Temp", "ACC_Laminar_Header", "ACC_Temp_Post_Laminar", "Plate_T_mm",
              "Chem_C", "Chem_Mn", "Chem_Nb", "FM_Temp_Before_Rolling"],
    "HRC": ["Temp_FT_Avg_C", "Temp_CT_Top_C", "Coil_T_mm", "Temp_Transfer_Bar_C",
            "Chem_C", "Chem_Mn", "Chem_Nb"],
}

TARGET_GROUPS = {
    "Plate": [("as rolled", ["YS_BN", "TS_BN", "EL_BN"]),
              ("after normalising", ["YS_AN", "TS_AN", "EL_AN"]),
              ("toughness, Charpy V-notch", ["Impact_0C", "Impact_m20C", "Impact_m40C"]),
              ("through thickness", ["Z_Test"])],
    "HRC": [("as rolled", ["YS_BN", "TS_BN", "EL_BN"])],
}

TARGET_LABEL = {
    "Plate": {"YS_BN": ("yield, as rolled", "MPa"), "TS_BN": ("tensile, as rolled", "MPa"),
              "EL_BN": ("elongation, as rolled", "%"), "YS_AN": ("yield, normalised", "MPa"),
              "TS_AN": ("tensile, normalised", "MPa"), "EL_AN": ("elongation, normalised", "%"),
              "Impact_0C": ("Charpy 0 C", "J"), "Impact_m20C": ("Charpy -20 C", "J"),
              "Impact_m40C": ("Charpy -40 C", "J"), "Z_Test": ("Z-test", "%")},
    "HRC": {"YS_BN": ("yield", "MPa"), "TS_BN": ("tensile", "MPa"), "EL_BN": ("elongation", "%")},
}

# Readable names for the feature columns the boosted artifacts carry, used by the
# Diagnostics importance chart; unmapped names render as the raw column name.
FEATURE_LABELS = {
    "sigma_ss": "Pickering solid solution (sigma_ss)",
    "inv_sqrt_t": "Hall-Petch gauge proxy (t^-0.5)",
    "CE_IIW": "Carbon equivalent (CE_IIW)",
    "Chem_CE": "Carbon equivalent (Chem_CE)",
    "P_cm": "Cracking parameter (P_cm)",
    "Microalloy_Total": "Microalloy total (Nb+V+Ti)",
    "Nb_eff": "Effective niobium (Nb_eff)",
    "Ti_to_N_ratio": "Ti/N ratio",
    "W_over_T": "Width over gauge (W/t)",
    "eps_total": "True strain, total",
    "eps_STM": "True strain, Steckel mill",
    "FM_Temp_Drop": "Entry-to-finish temperature drop",
    "ACC_Temp_Drop": "ACC temperature drop (cooling severity)",
    "Cooling_Rate_Proxy": "Cooling rate proxy (ACC laminar)",
    "Ar3": "Ar3 transformation temperature",
    "Tnr": "Recrystallisation stop temperature",
    "FRT_minus_Ar3": "Finishing margin above Ar3",
    "FRT_minus_Tnr": "FRT below recrystallisation stop",
    "Tpost_minus_Ar3": "Post-laminar temperature above Ar3",
    "FM_Temp_Before_Rolling": "Entry temperature",
    "Temp_Transfer_Bar_C": "Transfer bar (entry) temperature",
    "FM_Finish_Temp": "Finishing temperature",
    "Temp_FT_Avg_C": "Finishing temperature",
    "Temp_CT_Top_C": "Coiling temperature, top",
    "Temp_CT_Bot_C": "Coiling temperature, bottom",
    "ACC_Laminar_Header": "Laminar header count",
    "ACC_Headers_n": "Laminar header count",
    "ACC_Temp_Post_Laminar": "Post-laminar temperature",
    "ACC_Active": "ACC active flag",
    "Slab_T_mm": "Slab thickness",
    "TB_Thickness_mm": "Transfer bar thickness",
    "TB_Ratio": "Transfer bar / gauge ratio",
    "FM_Last_Pass_Drafting": "Last pass drafting",
    "Plate_T_mm": "Plate gauge",
    "Coil_T_mm": "Coil gauge",
    "Plate_W_mm": "Plate width",
    "Coil_W_mm": "Coil width",
    "Chem_C": "Carbon", "Chem_Mn": "Manganese", "Chem_Si": "Silicon", "Chem_P": "Phosphorus",
    "Chem_S": "Sulphur", "Chem_Al": "Aluminium", "Chem_Nb": "Niobium", "Chem_V": "Vanadium",
    "Chem_Ti": "Titanium", "Chem_Cu": "Copper", "Chem_Ni": "Nickel", "Chem_Cr": "Chromium",
    "Chem_Mo": "Molybdenum", "Chem_N": "Nitrogen", "Chem_B": "Boron",
}


# ----------------------------------------------------------------------------- styling
@st.cache_resource(show_spinner=False)
def boot_check(mill):
    """Score the active route once at start-up using Champion GBDT (instant bootstrap).

    Scoped to one mill on purpose: a both-routes loop eagerly pulls every route's joblib
    bundles (predict_champion's fallback included) into RAM at cold boot, which is what
    pushed the free-tier Streamlit Cloud container over its memory cliff. The other route
    loads lazily the first time an operator actually switches to it.
    """
    try:
        raw = defaults(mill)
        if mill == "Plate":
            raw["Rolling_Type"] = "AR"
        out = predict_champion(mill, raw)
        scored = {k: v["pred"] for k, v in out.items()}
        good = sum(1 for v in scored.values() if np.isfinite(v))
        first_error = next((v["error"] for v in out.values() if v.get("error")), "")
        return {mill: (good, len(scored), first_error)}
    except Exception as exc:
        return {mill: (0, 0, f"{type(exc).__name__}: {exc}")}


def logo_uri():
    try:
        with open(LOGO_PATH, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    except OSError:
        return ""


def inject_css():
    st.markdown("""
<style>
/* type: Inter for labels and JetBrains Mono for numbers, because the instrument's whole
   value is tabular alignment of MPa/°C readouts; the scale is 11/13/15/22/30 px, 11 px floor */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --fs-11: 11px;
    --fs-13: 13px;
    --fs-15: 15px;
    --fs-22: 22px;
    --fs-30: 30px;
    --accent: __ACCENT__;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.stApp, .stMarkdown, p, label, div[data-baseweb="tab"], div[data-testid="stSidebar"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
code, pre, .mono, [data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace;
}
.stApp { background: #ffffff; }
.block-container { max-width: 1320px; padding-top: 4.4rem; padding-bottom: 3rem; }

h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: #0b0d10; letter-spacing: -.02em; }

.kicker { font-size: var(--fs-11); letter-spacing: .18em; text-transform: uppercase; color: var(--accent); font-weight: 600; line-height: 1.6; }
.title { font-size: var(--fs-30); line-height: 1.15; font-weight: 600; margin: .35rem 0 .5rem; letter-spacing: -.03em; color: #0b0d10; }
.lede { font-size: var(--fs-15); color: #4b5563; max-width: 72ch; line-height: 1.62; text-wrap: pretty; }
.hair { height: 1px; background: #e5e7eb; margin: 1.4rem 0 1.2rem; border: 0; }

/* named spacer tokens: .breath = intra-section gap, .break = gap between major sections */
.breath { height: 1rem; }
.break { height: 2rem; }

.section { font-size: var(--fs-11); letter-spacing: .12em; text-transform: uppercase; color: #4b5563;
           font-weight: 600; display: flex; align-items: center; gap: .7rem; margin-bottom: .7rem; }
.section i { flex: 1; height: 1px; background: #e5e7eb; display: block; }

.card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 0 1rem; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
table.ut { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
table.ut th { font-size: var(--fs-11); letter-spacing: .08em; text-transform: uppercase; color: #6b7280;
              font-weight: 600; text-align: right; padding: .75rem .5rem .55rem;
              border-bottom: 1px solid #e5e7eb; background: #fafbfc; }
table.ut th:first-child { text-align: left; padding-left: .25rem; }
table.ut td { padding: .65rem .5rem; border-bottom: 1px solid #f1f2f4; text-align: right;
              vertical-align: middle; font-size: var(--fs-13); color: #0b0d10; }
table.ut td:first-child { text-align: left; padding-left: .25rem; font-weight: 500; }
table.ut tbody tr:hover td { background: #f8fafc; }
table.ut tbody tr:last-child td { border-bottom: 0; }
table.ut tr.grp td { text-align: left; font-size: var(--fs-11); letter-spacing: .12em; text-transform: uppercase;
                     color: var(--accent); font-weight: 600; padding: .9rem .25rem .35rem;
                     border-bottom: 1px solid #e5e7eb; background: #ffffff; }
table.ut tbody tr:first-child.grp td { padding-top: .55rem; }
table.ut td.name { text-align: left; font-size: var(--fs-13); font-weight: 500; color: #0b0d10; }
table.ut td.val { font-family: 'JetBrains Mono', monospace; font-size: var(--fs-15); font-weight: 600; color: #0b0d10; }
table.ut td.alt.muted { color: #6b7280; }
table.ut td.alt { font-family: 'JetBrains Mono', monospace; font-size: var(--fs-13); color: #4b5563; }
table.ut td.val .unit { font-size: var(--fs-11); color: #6b7280; margin-left: .25rem; font-weight: 400; }
table.ut td.rng { font-family: 'JetBrains Mono', monospace; font-size: var(--fs-13); color: #4b5563; }
table.ut td.sub { font-size: var(--fs-13); color: #6b7280; }
table.ut td.flag { width: 110px; text-align: center; }
.card .foot { border-top: 1px solid #e5e7eb; padding: .6rem 0 .7rem; font-size: var(--fs-11); color: #6b7280; }

.pill { display: inline-flex; align-items: center; justify-content: center; font-size: var(--fs-11);
        letter-spacing: .05em; font-weight: 600; text-transform: uppercase; padding: .2rem .55rem;
        border-radius: 4px; white-space: nowrap; line-height: 1.2; }
.pill.ok { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.pill.warn { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.pill.bad { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.pill.neutral { background: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }

.tag { display: inline-block; font-size: var(--fs-11); letter-spacing: .06em; text-transform: uppercase;
       padding: .15rem .4rem; border: 1px solid #e5e7eb; color: #4b5563; border-radius: 3px; }

/* facts cell anatomy used by every tab: span label / b mono value / em unit / optional pill */
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0;
         border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #ffffff; }
.facts div { padding: .75rem 1rem; border-right: 1px solid #e5e7eb; }
.facts div:last-child { border-right: 0; }
.facts span { display: block; font-size: var(--fs-11); letter-spacing: .08em; text-transform: uppercase;
              color: #6b7280; margin-bottom: .25rem; font-weight: 500; }
.facts b { font-family: 'JetBrains Mono', monospace; font-size: var(--fs-15); font-weight: 600; color: #0b0d10; }
.facts em { font-style: normal; font-size: var(--fs-11); color: #6b7280; margin-left: .25rem; }
.facts div.lead b { font-size: var(--fs-22); }
.facts.facts-text b { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                      font-size: var(--fs-13); font-weight: 600; letter-spacing: 0; }

.note { font-size: var(--fs-13); color: #4b5563; line-height: 1.5; border-left: 2px solid #e5e7eb;
        padding-left: .7rem; margin-top: .6rem; }

/* flat provenance list: plain rows, no card shells */
.prov { margin: .4rem 0 0; padding: 0; list-style: none; border-top: 1px solid #e5e7eb; }
.prov li { padding: .55rem .25rem; border-bottom: 1px solid #e5e7eb; font-size: var(--fs-13);
           color: #4b5563; line-height: 1.55; }
.prov li b { color: #0b0d10; }
.prov li.prov-lead { color: #0b0d10; }

div[data-testid="stSidebar"] { background: #fbfbfc; border-right: 1px solid #e5e7eb; }
div[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
div[data-testid="stSidebar"] h2 { font-size: var(--fs-13); letter-spacing: .1em; text-transform: uppercase; }

.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid #e5e7eb; }
.stTabs [data-baseweb="tab"] { font-size: var(--fs-11); letter-spacing: .06em; text-transform: uppercase;
                               color: #6b7280; padding: .5rem 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #0b0d10 !important; font-weight: 600; }

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: var(--fs-22); }
#MainMenu, footer, [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stExpandSidebarButton"] { display: flex !important; visibility: visible !important; }

/* narrow screens: keep input text at 16px so iOS Safari does not auto-zoom on focus */
@media (max-width: 640px) {
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input,
  [data-testid="stTextArea"] textarea { font-size: 16px; }
}
</style>
""".replace("__ACCENT__", ACCENT), unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def hair():
    st.markdown('<hr class="hair">', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data helpers
def spec_float(row, index, fallback=0.0):
    """A spec cell may arrive as text, None or '-'; never let that take the page down."""
    return cons.spec_float(row, index, fallback)


def prune_stale_widget_key(key, options):
    """A selectbox whose options shrink would otherwise raise on its persisted value."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def spec_map(mill):
    return {row[1]: row for row in SPECS[mill]}


def surface_keys(mill):
    """Every numeric input is sweepable; the historical thermal keys stay first."""
    numeric = [row[1] for row in SPECS[mill]]
    preferred = [k for k in SURFACE[mill] if k in numeric]
    return preferred + [k for k in numeric if k not in preferred]


def axis_range(mill, key, limits):
    """Sweep range for an axis: mill envelope intersected with the house limits."""
    row = spec_map(mill)[key]
    lo, hi = spec_float(row, 4), spec_float(row, 5)
    if limits and key in limits:
        lo = max(lo, float(limits[key][0]))
        hi = min(hi, float(limits[key][1]))
    if hi <= lo:
        lo, hi = spec_float(row, 4), spec_float(row, 5)
    return lo, hi


def defaults(mill):
    return {row[1]: spec_float(row, 6) for row in SPECS[mill]}


@st.cache_resource(show_spinner=False)
def bundles(mill):
    return {k: ml.load_model_bundle(mill, k) for k in TARGET_LABEL[mill]}


def prepared(raw, mill):
    """The cleaned sheet already carries TB_Ratio, drafting and ACC header count; derive them here."""
    out = dict(raw)
    if mill == "Plate":
        plate_t = max(float(out.get("Plate_T_mm", 16.0)), 1.0)
        tb = float(out.get("TB_Thickness_mm", plate_t * 2))
        out["TB_Ratio"] = tb / plate_t
        if "FM_Last_Pass_Drafting" in out:
            out["FM_Last_Pass_Drafting"] = float(out["FM_Last_Pass_Drafting"])
        else:
            out["FM_Last_Pass_Drafting"] = float(out.get("FM_Last_Pass_Thick", plate_t + 4.0)) - plate_t
        headers = float(out.get("ACC_Laminar_Header", 0.0))
        out["ACC_Headers_n"] = headers
        out["ACC_Active"] = 1.0 if headers > 0 else 0.0
        out["Rolling_Type"] = str(out.get("Rolling_Type", "AR"))
    return out


def prepare_frame(mill, frame):
    frame = frame.copy()
    if mill == "Plate":
        if "Plate_T_mm" not in frame.columns:
            frame["Plate_T_mm"] = 16.0
        plate_t = pd.to_numeric(frame["Plate_T_mm"], errors="coerce").fillna(16.0).clip(lower=1.0)
        if "TB_Ratio" not in frame:
            if "TB_Thickness_mm" in frame.columns:
                tb = pd.to_numeric(frame["TB_Thickness_mm"], errors="coerce").fillna(plate_t * 2)
            else:
                tb = plate_t * 2
            frame["TB_Ratio"] = tb / plate_t
        if "FM_Last_Pass_Drafting" not in frame:
            if "FM_Last_Pass_Thick" in frame.columns:
                last = pd.to_numeric(frame["FM_Last_Pass_Thick"], errors="coerce").fillna(plate_t * 1.15)
            else:
                last = plate_t * 1.15
            frame["FM_Last_Pass_Drafting"] = last - plate_t
        if "ACC_Headers_n" not in frame:
            if "ACC_Laminar_Header" in frame.columns:
                frame["ACC_Headers_n"] = pd.to_numeric(frame["ACC_Laminar_Header"],
                                                       errors="coerce").fillna(0.0)
            else:
                frame["ACC_Headers_n"] = 0.0
        if "ACC_Active" not in frame:
            frame["ACC_Active"] = (frame["ACC_Headers_n"] > 0).astype(float)
        if "Rolling_Type" not in frame:
            frame["Rolling_Type"] = "AR"
    else:
        # the Steckel feature module reads Chem_<el> for all sixteen elements including CE;
        # a missing one raises rather than defaulting, so fill the blanks before it runs
        for el in ("C", "Si", "Mn", "P", "S", "Nb", "Cu", "Cr", "Ni", "Mo", "V", "Al", "Ti",
                   "N", "B", "CE"):
            col = f"Chem_{el}"
            if col not in frame:
                frame[col] = 0.0
        if float(pd.to_numeric(frame["Chem_CE"], errors="coerce").fillna(0).abs().sum()) == 0.0:
            frame["Chem_CE"] = (frame["Chem_C"] + frame["Chem_Mn"] / 6.0
                                + (frame["Chem_Cr"] + frame["Chem_Mo"] + frame["Chem_V"]) / 5.0
                                + (frame["Chem_Ni"] + frame["Chem_Cu"]) / 15.0)
    return frame


def feature_matrix(mill, frame):
    """One pass through the mill's own feature module; the models expect exactly this."""
    if mill == "Plate":
        return ml.engineer_plate_features(frame)
    X, _ = ml.engineer_hrc_features(frame, is_train=False)
    return X


def candidate_columns(bundle, X):
    """Ordered guesses at the feature list a shipped estimator expects.

    The normalised targets were fitted on per-target subsets, and the bundled name list does not
    always match the fitted one, so try the estimator's own record first and fall back.
    """
    seen, out = set(), []
    for cols in (getattr(bundle["model"], "feature_names_in_", None),
                 bundle.get("feature_names"), bundle.get("features"), list(X.columns)):
        if cols is None:
            continue
        key = tuple(cols)
        if key not in seen:
            seen.add(key)
            out.append(list(cols))
    return out


def align(X, cols):
    return pd.DataFrame({c: (X[c] if c in X.columns else 0.0) for c in cols}, index=X.index)


def _nlr_scores(mill, frame):
    """Vectorised NLR: the same five terms predict_nlr builds, for a whole frame."""
    table = nlr_models(mill)
    out = pd.DataFrame(index=frame.index)
    if not table:
        return out
    thick = pd.to_numeric(frame.get("Plate_T_mm" if mill == "Plate" else "Coil_T_mm"),
                          errors="coerce").fillna(16.0 if mill == "Plate" else 6.0).clip(lower=1.0)
    slab = pd.to_numeric(frame.get("Slab_T_mm" if mill == "Plate" else "TB_Thick_mm"),
                         errors="coerce").fillna(218.0 if mill == "Plate" else 32.0).clip(lower=1.0)
    headers = (pd.to_numeric(frame.get("ACC_Laminar_Header"), errors="coerce").fillna(0.0)
               if mill == "Plate" else pd.Series(0.0, index=frame.index))

    def num(col, default):
        return pd.to_numeric(frame.get(col), errors="coerce").fillna(default)

    # NLR term definitions: FROZEN. These are NOT the Pickering sigma_ss or the ACC cooling rate
    # the models and the derived-metrics card use: the NLR coefficients in models/nlr_models.json
    # were fitted on exactly these narrower definitions (scripts/parse_nlr.py::term_values), so
    # changing them here would invalidate the fit. The keys must keep matching the "var" names in
    # that JSON. HRC has no NLR fit, so this path only runs for Plate.
    terms = {
        "t": thick,
        "sigma_ss": (32 * num("Chem_Mn", .9) + 84 * num("Chem_Si", .2)
                     + 38 * num("Chem_Cu", .02) + 32 * num("Chem_Ni", .02)),
        "CE": (num("Chem_C", .16) + num("Chem_Mn", .9) / 6
               + (num("Chem_Cr", .02) + num("Chem_Mo", .01) + num("Chem_V", .005)) / 5
               + (num("Chem_Ni", .02) + num("Chem_Cu", .02)) / 15),
        "CoolingRate": (headers + 0.1) / thick ** 1.5,
        "eps_total": np.log(slab / thick),
    }
    for key, model in table.items():
        value = pd.Series(float(model["intercept"]), index=frame.index)
        for term in model["terms"]:
            values = terms.get(term["var"])
            if values is None:
                continue
            value = value + term["coef"] * values ** term["power"]
        out[key] = value
    return out


def score_frame(mill, frame, engine="Champion GBDT (Recommended)"):
    """Vectorised scoring: features once, then every target.

    The engine argument must be honoured identically on every tab; an unsupported
    engine raises rather than silently scoring with something else.
    """
    frame = prepare_frame(mill, frame)
    X = feature_matrix(mill, frame)
    out = pd.DataFrame(index=frame.index)
    if engine == "OLS / Linear":
        for key, spec in linear_families(mill).items():
            if spec["ols"]["r2"] < 0.15:
                continue          # a linear fit this bad is not an answer, it is noise in MPa
            missing = [c for c in spec["features"] if c not in X.columns]
            if missing:
                continue
            values = X[spec["features"]].to_numpy(dtype=float)
            values = np.nan_to_num(values, nan=0.0)
            out[key] = spec["ols"]["intercept"] + values @ np.asarray(spec["ols"]["coef"], dtype=float)
    elif engine == "NLR":
        out = _nlr_scores(mill, frame)
    elif engine in ("Champion GBDT (Recommended)", "CatBoost", "XGBoost", "LightGBM"):
        plan = champion_plan(mill)
        idx = boosted_index(mill)
        for key in TARGET_LABEL[mill]:
            if engine == "Champion GBDT (Recommended)":
                cfg = plan.get(key) or {}
                fam = cfg.get("family")
            else:
                fam = {"CatBoost": "cat", "XGBoost": "xgb", "LightGBM": "lgbm"}[engine]
            model = boosted_model(mill, fam, key) if fam in ("cat", "xgb", "lgbm") else None
            if model is not None and fam in idx and key in idx[fam]:
                cols = idx[fam][key].get("features") or list(X.columns)
                aligned = pd.DataFrame({c: (X[c] if c in X.columns else 0.0) for c in cols},
                                       index=X.index).to_numpy(dtype=float)
                try:
                    out[key] = np.asarray(model.predict(aligned)).ravel()
                    continue
                except Exception:
                    pass
            # Fallback to a joblib bundle. Some HRC tuned ensembles fail to unpickle into the
            # shared StackingEnsemble class, so try the plain best_model as well.
            for use_tuned in (True, False):
                bundle = ml.load_model_bundle(mill, key, use_tuned=use_tuned)
                if bundle is None:
                    continue
                filled = False
                for cols in candidate_columns(bundle, X):
                    try:
                        out[key] = bundle["model"].predict(align(X, cols))
                        filled = True
                        break
                    except Exception:
                        continue
                if filled:
                    break
    elif engine == "Ensemble (.joblib)":
        for key, bundle in bundles(mill).items():
            if bundle is None:
                continue
            out[key] = np.nan
            for cols in candidate_columns(bundle, X):
                try:
                    out[key] = bundle["model"].predict(align(X, cols))
                    break
                except Exception:
                    continue
    else:
        raise ValueError(f"Unsupported scoring engine: {engine!r}")
    if mill == "Plate" and "TS_BN" in out.columns and "TS_AN" in out.columns:
        # frame-path twin of ml.apply_plate_normalizing_coupling, which handles the one-row dict.
        # .mask rather than .loc so a float32 model column can take a float64 replacement.
        delta_ts, delta_ys, delta_el = ml.plate_softening_deltas_frame(frame)

        ts_cap = out["TS_BN"] - delta_ts
        out["TS_AN"] = out["TS_AN"].mask(out["TS_AN"].notna() & (out["TS_AN"] > ts_cap), ts_cap)
        if "YS_BN" in out.columns and "YS_AN" in out.columns:
            ys_cap = out["YS_BN"] - delta_ys
            out["YS_AN"] = out["YS_AN"].mask(out["YS_AN"].notna() & (out["YS_AN"] > ys_cap), ys_cap)
        if "EL_BN" in out.columns and "EL_AN" in out.columns:
            el_floor = out["EL_BN"] + delta_el
            out["EL_AN"] = out["EL_AN"].mask(out["EL_AN"].notna() & (out["EL_AN"] < el_floor), el_floor)
    return out



def get_target_interval_half(mill, key, default_rmse=None):
    """Returns the 95% split-conformal prediction interval half-width guaranteeing >= 95% empirical coverage."""
    conf = pd.DataFrame()
    try:
        if hasattr(ml, "get_conformal_intervals"):
            conf = ml.get_conformal_intervals(mill)
        else:
            conf_path = os.path.join(os.path.dirname(APP_DIR), mill, "reports", "conformal_intervals.csv")
            if os.path.exists(conf_path):
                conf = pd.read_csv(conf_path)
    except Exception:
        conf = pd.DataFrame()

    if not conf.empty and "Target" in conf.columns:
        row = conf[conf["Target"] == key]
        if not row.empty and "Conformal_Half_Width_95" in row.columns and pd.notna(row["Conformal_Half_Width_95"].values[0]):
            return float(row["Conformal_Half_Width_95"].values[0])
    if default_rmse and default_rmse > 0:
        return float(1.96 * default_rmse)
    try:
        ci = ml.get_ci_data(mill)
        if not ci.empty and "Target" in ci.columns:
            row = ci[ci["Target"] == key]
            if not row.empty and "Val_RMSE" in row.columns and pd.notna(row["Val_RMSE"].values[0]):
                return float(1.96 * float(row["Val_RMSE"].values[0]))
    except Exception:
        pass
    try:
        return float(ml.TARGET_CONFIG[mill][key]["tol"] * 0.5)
    except Exception:
        return 15.0

def predict_one(mill, raw):
    """One schedule, every target, with the split-conformal interval the shipped report implies."""
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    out, last_error = {}, {}
    for key, bundle in bundles(mill).items():
        if bundle is None:
            continue
        value = np.nan
        for cols in candidate_columns(bundle, X):
            try:
                value = float(bundle["model"].predict(align(X, cols))[0])
                last_error.pop(key, None)
                break
            except Exception as exc:
                last_error[key] = f"{type(exc).__name__}: {exc}"
        half = get_target_interval_half(mill, key)
        r2_val = _holdout_r2(mill, key)
        label, unit = TARGET_LABEL[mill].get(key, (key, ""))
        out[key] = {"name": label, "pred": round(value, 2) if np.isfinite(value) else float("nan"),
                    "unit": unit, "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
                    "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan"),
                    "r2": r2_val,
                    "error": last_error.get(key, "")}
    # coupling is applied once, in predict_selected_engine
    return out


def physical(raw, mill):
    merged = dict(raw)
    merged.setdefault("ACC_Headers_n", raw.get("ACC_Laminar_Header", 8.0))
    merged.setdefault("FM_Entry_Temp", raw.get("FM_Temp_Before_Rolling", 985.0))
    return ml.compute_physical_indicators(merged, mill)


@st.cache_resource(show_spinner=False)
def linear_families(mill):
    path = os.path.join(os.path.dirname(APP_DIR), "models", "linear_families.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get(mill, {})
    except (OSError, ValueError):
        return {}


@st.cache_resource(show_spinner=False)
def nlr_models(mill):
    path = os.path.join(os.path.dirname(APP_DIR), "models", "nlr_models.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get(mill, {})
    except (OSError, ValueError):
        return {}


def predict_nlr(mill, raw):
    table = nlr_models(mill)
    if not table:
        return {}
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    scores = _nlr_scores(mill, frame)
    out = {}
    for key, model in table.items():
        if key in scores and np.isfinite(scores[key].iloc[0]):
            out[key] = {"pred": float(scores[key].iloc[0]), "r2": float(model.get("r2", 0.0))}
    return out


NLR_TERM_NAMES = {"t": "t", "sigma_ss": "σ<sub>ss</sub>", "CE": "CE",
                  "CoolingRate": "CR", "eps_total": "ε<sub>total</sub>"}


def _nlr_card(label, model):
    """One constitutive-equation card, built only from what the shipped JSON states."""
    parts = [f"{model['intercept']:.4g}"]
    for term in model["terms"]:
        coef = float(term["coef"])
        if coef == 0.0:
            continue          # a zero coefficient is not a mechanism, drop it from the card
        name = NLR_TERM_NAMES.get(term["var"], term["var"])
        piece = f"{coef:+.4g} · {name}"
        if term["power"] not in (1, 1.0):
            piece += f"<sup>{term['power']:g}</sup>"
        parts.append(piece)
    r2 = model.get("r2")
    metrics = f"holdout R² = {r2:.3f}" if r2 is not None else "holdout R² not recorded"
    return ('<div class="card" style="padding:1rem;">'
            f'<span class="pill ok">{label}</span>'
            '<div style="font-family:monospace; font-size:13px; color:#0b0d10; '
            'margin-top:.6rem; line-height:1.6;">' + " ".join(parts) +
            f'</div><div class="note" style="margin-top:.5rem;">{metrics} · '
            'no finishing-temperature term by construction</div></div>')


def _render_nlr_cards(mill):
    nlrs = nlr_models(mill)
    if not nlrs:
        st.info("No constitutive fit is shipped for this route: the NLR engine scores no "
                "target here, and no equations are shown rather than invented.")
        return
    items = list(nlrs.items())
    cols = st.columns(min(3, len(items)))
    for col, (key, model) in zip(cols, items):
        label = TARGET_LABEL[mill].get(key, (key, ""))[0]
        with col:
            st.markdown(_nlr_card(label, model), unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def boosted_index(mill):
    """Which boosted artifacts exist for this route, and their holdout scores."""
    base = os.path.join(os.path.dirname(APP_DIR), "models", "boosted_out")
    try:
        with open(os.path.join(base, "boosted_metrics.json"), "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
    except (OSError, ValueError):
        return {}
    index = {}
    for key, meta in metrics.items():
        m, target, family = key.split("|")
        if m == mill:
            index.setdefault(family, {})[target] = meta
    return index


@st.cache_resource(show_spinner=False)
def boosted_model(mill, family, target):
    base = os.path.join(os.path.dirname(APP_DIR), "models", "boosted_out")
    stem = os.path.join(base, f"{mill}_{target}_{family}")
    try:
        if family == "xgb":
            from xgboost import XGBRegressor
            model = XGBRegressor()
            model.load_model(stem + ".ubj")
        elif family == "lgbm":
            import lightgbm as lgb
            model = lgb.Booster(model_file=stem + ".txt")
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(stem + ".cbm")
        return model
    except Exception:
        return None


def predict_boosted(mill, raw, family):
    index = boosted_index(mill)
    if family not in index:
        return {}
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    out = {}
    for target, meta in index[family].items():
        model = boosted_model(mill, family, target)
        if model is None:
            continue
        cols = meta.get("features") or list(X.columns)
        aligned = pd.DataFrame({c: (X[c] if c in X.columns else 0.0) for c in cols},
                               index=X.index).to_numpy(dtype=float)
        out[target] = {"pred": float(np.asarray(model.predict(aligned)).ravel()[0]),
                       "r2": float(meta.get("r2", 0.0))}
    return out


def predict_linear(mill, raw, family):
    """OLS or Ridge from the committed coefficients, on the same frame the ensemble sees."""
    table = linear_families(mill)
    if not table:
        return {}
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    row = X.iloc[0]
    out = {}
    for key, spec in table.items():
        # NaN is truthy, so `or` will not save you here; a missing feature must be 0.0
        values = np.array([float(row[c]) if c in row.index and np.isfinite(row[c]) else 0.0
                           for c in spec["features"]], dtype=float)
        if family == "OLS":
            value = spec["ols"]["intercept"] + float(np.dot(spec["ols"]["coef"], values))
            r2 = spec["ols"]["r2"]
        else:
            z = (values - np.array(spec["ridge"]["scaler_mean"])) / np.array(spec["ridge"]["scaler_scale"])
            value = spec["ridge"]["intercept"] + float(np.dot(spec["ridge"]["coef"], z))
            r2 = spec["ridge"]["r2"]
        if r2 < 0.15:
            continue          # a linear fit this bad is not an answer, it is noise in MPa
        out[key] = {"pred": float(value), "r2": float(r2)}
    return out


@st.cache_resource(show_spinner=False)
def champion_plan(mill):
    """Determines the champion model family and holdout stats per target.

    Targets without a boosted artifact get None here; predict_champion then falls back
    to the shipped joblib bundle and reports the leaderboard's own numbers. Never
    invent a score for a target the artifacts cannot back.
    """
    index = boosted_index(mill)
    plan = {}
    for target in TARGET_LABEL[mill]:
        best_fam, best_r2, best_rmse = None, -999.0, None
        for fam in ("cat", "xgb", "lgbm"):
            if fam in index and target in index[fam]:
                r2 = float(index[fam][target].get("r2", -999.0))
                if r2 > best_r2:
                    best_r2 = r2
                    best_fam = fam
                    best_rmse = float(index[fam][target].get("rmse", 0.0))
        if best_fam is not None:
            plan[target] = {"family": best_fam, "r2": best_r2, "rmse": best_rmse}
        else:
            plan[target] = None
    return plan


def _importance_array(value):
    """Accept only a 1-D numeric, finite, non-empty array; anything else is unusable."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def _read_importances(model):
    """First usable importance array a native model exposes, or None.

    CatBoost's sklearn shim exposes feature_importances_ but raises on models loaded
    from .cbm: numpy swallows that raise and hands back a 0-d object array, so each
    accessor's output is validated and an unusable one falls through to the next.
    """
    for accessor in ("feature_importances_", "get_feature_importance", "feature_importance"):
        try:
            value = getattr(model, accessor)
            arr = _importance_array(value() if callable(value) else value)
        except Exception:
            continue
        if arr is not None:
            return arr
    return None


@st.cache_data(show_spinner=False)
def champion_importances(mill):
    """Split importances computed from the shipped champion artifacts, per target.

    Reads whatever importance API the native model exposes (sklearn-style
    feature_importances_, CatBoost's get_feature_importance, LightGBM's Booster
    feature_importance) and normalises to a share of total. A target whose artifact
    exposes nothing usable is left out of the dict entirely, so the UI never has to
    invent a number.
    """
    plan = champion_plan(mill)
    out = {}
    for target, cfg in plan.items():
        fam = (cfg or {}).get("family")
        if fam not in ("cat", "xgb", "lgbm"):
            continue
        model = boosted_model(mill, fam, target)
        if model is None:
            continue
        raw = _read_importances(model)
        if raw is None:
            continue
        cols = boosted_index(mill).get(fam, {}).get(target, {}).get("features")
        if cols is None or len(cols) != raw.size or raw.size == 0:
            continue
        if not np.isfinite(raw).all() or raw.sum() <= 0:
            continue
        shares = raw / raw.sum()
        out[target] = sorted(zip(cols, shares.tolist()), key=lambda pair: pair[1])
    return out


def _holdout_r2(mill, key):
    """The shipped holdout R2 for one target, from the bootstrap report or leaderboard."""
    try:
        ci = ml.get_ci_data(mill)
        if not ci.empty and "Target" in ci.columns:
            row = ci[ci["Target"] == key]
            if not row.empty and "Val_R2" in row.columns and pd.notna(row["Val_R2"].values[0]):
                return float(row["Val_R2"].values[0])
        lb = ml.get_leaderboard_data(mill)
        if not lb.empty and "Target" in lb.columns:
            row = lb[lb["Target"] == key]
            if not row.empty and "Final_Val_R2" in row.columns and pd.notna(row["Final_Val_R2"].values[0]):
                return float(row["Final_Val_R2"].values[0])
    except Exception:
        pass
    return None


def predict_champion(mill, raw):
    """Fast, high-R2 scoring using champion native GBDT models with fallback for Z-Test."""
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    plan = champion_plan(mill)
    out = {}
    for key in TARGET_LABEL[mill]:
        label, unit = TARGET_LABEL[mill].get(key, (key, ""))
        cfg = plan.get(key) or {}
        fam = cfg.get("family")
        value = np.nan
        err = ""
        if fam in ("xgb", "lgbm", "cat"):
            model = boosted_model(mill, fam, key)
            if model is not None:
                meta = boosted_index(mill).get(fam, {}).get(key, {})
                cols = meta.get("features") or list(X.columns)
                aligned = pd.DataFrame({c: (X[c] if c in X.columns else 0.0) for c in cols},
                                       index=X.index).to_numpy(dtype=float)
                try:
                    value = float(np.asarray(model.predict(aligned)).ravel()[0])
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
        if not np.isfinite(value):
            bundle = ml.load_model_bundle(mill, key)
            if bundle is not None:
                for cols in candidate_columns(bundle, X):
                    try:
                        value = float(bundle["model"].predict(align(X, cols))[0])
                        err = ""
                        fam = "ensemble"
                        break
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
        if not np.isfinite(value) and not err:
            err = "no scoring artifact for this target"
        half = get_target_interval_half(mill, key, default_rmse=cfg.get("rmse"))
        out[key] = {
            "name": label,
            "pred": round(value, 2) if np.isfinite(value) else float("nan"),
            "unit": unit,
            "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
            "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan"),
            "family": fam,
            "r2": cfg.get("r2", _holdout_r2(mill, key)),
            "error": err
        }
    # coupling is applied once, in predict_selected_engine
    return out


def predict_selected_engine(mill, raw, engine):
    """Dispatches prediction to the chosen engine."""
    if engine == "Champion GBDT (Recommended)":
        out = predict_champion(mill, raw)
    elif engine == "Ensemble (.joblib)":
        out = predict_one(mill, raw)
    elif engine in ("CatBoost", "XGBoost", "LightGBM"):
        fam_key = {"CatBoost": "cat", "XGBoost": "xgb", "LightGBM": "lgbm"}[engine]
        boosted = predict_boosted(mill, raw, fam_key)
        frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
        X = feature_matrix(mill, frame)
        ci = ml.get_ci_data(mill)
        index = boosted_index(mill).get(fam_key, {})
        out = {}
        for key in TARGET_LABEL[mill]:
            label, unit = TARGET_LABEL[mill].get(key, (key, ""))
            if key in boosted and np.isfinite(boosted[key].get("pred", np.nan)):
                value = boosted[key]["pred"]
                meta = index.get(key, {})
                half = get_target_interval_half(mill, key, default_rmse=meta.get("rmse"))
                out[key] = {
                    "name": label,
                    "pred": round(value, 2),
                    "unit": unit,
                    "ci_low": round(value - half, 2),
                    "ci_high": round(value + half, 2),
                    "family": fam_key,
                    "r2": meta.get("r2", 0.0),
                    "error": ""
                }
            else:
                bundle = ml.load_model_bundle(mill, key)
                value = np.nan
                if bundle is not None:
                    for cols in candidate_columns(bundle, X):
                        try:
                            value = float(bundle["model"].predict(align(X, cols))[0])
                            break
                        except Exception:
                            continue
                half = ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
                out[key] = {
                    "name": label,
                    "pred": round(value, 2) if np.isfinite(value) else float("nan"),
                    "unit": unit,
                    "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
                    "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan"),
                    "family": "ensemble",
                    "r2": 0.0,
                    "error": "" if np.isfinite(value) else f"Not available in {engine}"
                }
    elif engine == "OLS / Linear":
        lin_table = linear_families(mill)
        lin = predict_linear(mill, raw, "OLS")
        champ = predict_champion(mill, raw)
        out = {}
        for key in TARGET_LABEL[mill]:
            label, unit = TARGET_LABEL[mill].get(key, (key, ""))
            if key in lin and np.isfinite(lin[key].get("pred", np.nan)):
                val = lin[key]["pred"]
                r2 = lin[key]["r2"]
                half = ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
                out[key] = {
                    "name": label,
                    "pred": round(val, 2),
                    "unit": unit,
                    "ci_low": round(val - half, 2),
                    "ci_high": round(val + half, 2),
                    "family": "OLS",
                    "r2": r2,
                    "error": ""
                }
            else:
                item = dict(champ[key])
                item["error"] = ("no linear family is shipped for this route"
                                 if not lin_table else "Linear R2 < 0.15 (showing Champion GBDT)")
                out[key] = item
    else:
        out = predict_champion(mill, raw)

    if mill == "Plate":
        out = ml.apply_plate_normalizing_coupling(raw, out)
    return out


def single_target(mill, raw, engine, key):
    """One target from the chosen engine, without letting a missing key kill the page."""
    try:
        preds = predict_selected_engine(mill, raw, engine)
    except Exception as exc:  # noqa: BLE001
        return np.nan, f"{type(exc).__name__}: {exc}"
    res = preds.get(key)
    if not res:
        return np.nan, f"{engine} returned no result for {key}"
    value = res.get("pred", np.nan)
    if not np.isfinite(value):
        return np.nan, res.get("error") or f"{engine} could not score {key}"
    return float(value), ""


def _scaled_interval(res, key, mult=1.0):
    """Scale a prediction's 95% conformal interval around `pred` by `mult`."""
    lo = float(res["ci_low"])
    hi = float(res["ci_high"])
    pred = float(res["pred"])
    mult = float(mult)
    if abs(mult - 1.0) < 1e-9 or not (np.isfinite(pred) and np.isfinite(lo) and np.isfinite(hi)):
        return lo, hi
    adj_lo = pred - max(0.0, pred - lo) * mult
    adj_hi = pred + max(0.0, hi - pred) * mult
    if key.startswith(("EL_", "Impact_", "Z_")):
        adj_lo = max(0.0, adj_lo)
    return round(adj_lo, 2), round(adj_hi, 2)


def targets_table(mill, predictions, extras=None, engine_label="Champion GBDT",
                  conformal_mult=1.0, conformal_reason=None):
    """Grouped property card: clean readable name, tabular numbers, model holdout uncertainty."""
    extras = extras or {}
    rows, total = [], 0
    span = 4 + len(extras)
    for group, keys in TARGET_GROUPS[mill]:
        rows.append(f'<tr class="grp"><td colspan="{span}">{group}</td></tr>')
        for key in keys:
            res = predictions.get(key)
            if not res or not np.isfinite(res.get("pred", np.nan)):
                continue
            total += 1
            label, unit = TARGET_LABEL[mill].get(key, (res["name"], res["unit"]))
            rmse_tol = ml.TARGET_CONFIG[mill][key]["tol"]
            ci_lo, ci_hi = _scaled_interval(res, key, conformal_mult)
            fam = res.get("family", "")
            fam_badge = f'<span class="tag" style="margin-left:.4rem;">{fam}</span>' if fam else ""
            if mill == "Plate" and key in ("YS_AN", "TS_AN", "EL_AN"):
                coupled_tip = res.get("note", "Coupled to blackplate via dynamic transformation kinetics")
                fam_badge += (f'<span class="tag" style="margin-left:.3rem; background:{ACCENT_RAMP[4]}; '
                              f'color:{ACCENT_DARK}; border:1px solid {ACCENT_RAMP[3]};" '
                              f'title="{coupled_tip}">coupled</span>')
            extra_cells = "".join(
                (f'<td class="alt">{extras[name][key]["pred"]:.1f}</td>'
                 if key in extras[name] and np.isfinite(extras[name][key].get("pred", np.nan)) else '<td class="alt muted">—</td>')
                for name in extras)
            rows.append(f"""<tr>
              <td class="name">{label}{fam_badge}</td>
              <td class="val">{res['pred']:.1f}<span class="unit">{unit}</span></td>
              <td class="rng">{ci_lo:.1f} – {ci_hi:.1f}</td>
              <td class="sub" style="font-family:monospace;">±{rmse_tol:g}&nbsp;{unit}</td>{extra_cells}</tr>""")
    cqr_note = ""
    if conformal_reason:
        cqr_note = (f" · Adaptive CQR: <b>{float(conformal_mult):.2f}&times;</b> "
                    f"({html_lib.escape(str(conformal_reason))})")
    html = (f"<div class=\"card\"><table class=\"ut\"><thead><tr><th>property</th><th>{engine_label.lower()}</th>"
            "<th>95% Split-Conformal Interval</th><th>Certification Tolerance</th>"
            + "".join(f"<th>{n}</th>" for n in extras)
            + "</tr></thead><tbody>" + "".join(rows)
            + f"</tbody></table><div class=\"foot\">{total} certified properties scored through "
              f"{mill} model layer ({engine_label}){cqr_note}.</div></div>")
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- header
def header(mill, n_targets):
    route = ("the roughing and finishing mills with laminar accelerated cooling, plus the optional "
             "furnace-normalising route" if mill == "Plate"
             else "the reversing Steckel mill, the runout table and the downcoiler")
    lede = (
        f"The engine takes a rolling schedule and a heat chemistry off {route} and returns the "
        f"{n_targets} mechanical properties that schedule should certify. The features are the mill's "
        f"own physics: true strain accumulated through the passes, the recrystallisation stop, the "
        f"finishing margin above Ar3, and the cooling rate the steel actually sees. Plate and Steckel "
        f"coil each run a separate feature module, so plate numbers never borrow from coil data."
        "<br><br>"
        "Every property is fitted three ways. Ordinary least squares and a metallurgical non-linear "
        "fit give the whitebox reading, gradient-boosted trees give the accuracy. All of it is scored "
        "on melts held out as whole heats, so sister plates from one heat never straddle the split, and "
        "the interval beside each number comes from split-conformal calibration rather than a bare RMSE. "
        "The diagnostics tab shows where the residuals actually sit."
        "<br><br>"
        "The same page turns those predictions into Cp and Cpk against limits you set, draws a 1D slice, "
        "a 2D contour or a 3D surface across any two inputs, and gives the inverse optimizer the window "
        "it needs to propose setpoints."
    )
    st.markdown(f"""
<div class="kicker">Plate Mill & Steckel Mill Prediction Platform</div>
<div class="title">Specification Limit & Mechanical Models</div>
<div class="lede">{lede}</div>
""", unsafe_allow_html=True)
    hair()


# ----------------------------------------------------------------------------- inputs
def chemistry_template(mill):
    """Seed the chemistry inputs from an internal grade's band midpoint."""
    choices = gt.names()
    if not choices:
        return
    choice = st.sidebar.selectbox(
        "Chemistry template", [gt.MANUAL] + choices, key=f"{mill}_grade_template",
        help="Midpoint of an internal grade's permitted MIN/MAX band from the grade database. "
             "Choosing one fills the chemistry inputs; the rolling schedule is left untouched.")
    marker = f"{mill}_grade_template_applied"
    previous = st.session_state.get(marker)
    if choice == gt.MANUAL:
        st.session_state[marker] = choice
        return

    record = gt.get(choice)
    if not record:
        return
    median = record.get("median", {})
    caption = " · ".join(f"{el} {median[el]:.3f}" for el in ("C", "Mn", "Si", "Nb")
                         if el in median)
    if record.get("ce_median") is not None:
        caption += f" · CE {record['ce_median']:.3f}"
    st.sidebar.caption(caption)
    st.sidebar.caption("band midpoint, not a mill aim: edit any element below before calculating")

    ranges = {row[1]: (spec_float(row, 4), spec_float(row, 5)) for row in SPECS[mill]}
    applied, held = {}, []
    for key, value in gt.chemistry(choice).items():
        if key not in ranges:
            continue
        lo, hi = ranges[key]
        bounded = float(min(max(value, lo), hi))
        if bounded != value:
            held.append(f"{key.replace('Chem_', '')} {value:g} → {bounded:g}")
        applied[key] = bounded
    if held:
        st.sidebar.caption("Outside the app's input range, held at the limit: " + " · ".join(held))

    if choice == previous:
        return
    for key, value in applied.items():
        st.session_state[f"{mill}_{key}"] = value
    st.session_state[marker] = choice


def schedule_inputs(mill):
    st.sidebar.markdown(f"### {mill} schedule")
    st.sidebar.caption("Every value here feeds the mill's own feature module. Edits apply as you "
                       "make them; there is no separate calculate step.")
    chemistry_template(mill)
    values = {}
    groups = []
    for row in SPECS[mill]:
        if row[0] not in groups:
            groups.append(row[0])
    for group in groups:
        with st.expander(group, expanded=group in ("Finishing mill", "Steckel mill")):
            for g, key, label, unit, lo, hi, default, step in SPECS[mill]:
                if g != group:
                    continue
                fmt = "%.4f" if step < 0.001 else ("%.3f" if step < 0.01 else
                                                   ("%.2f" if step < 1 else "%.0f"))
                shown_unit = "°C" if unit == "C" else unit
                widget_key = f"{mill}_{key}"
                seeded = st.session_state.get(widget_key)
                try:
                    value = float(default if seeded is None else seeded)
                except (TypeError, ValueError):
                    value = float(default)
                value = min(max(value, float(lo)), float(hi))
                auto_post = False
                if mill == "Plate" and key == "ACC_Temp_Post_Laminar":
                    auto_post = st.checkbox(
                        "Derive post-laminar T from headers (physical coupling)",
                        key=f"{mill}_post_laminar_auto", value=True)
                    st.caption("Sweeping laminar headers now moves the end temperature, like the "
                               "mill actually runs.")
                    if auto_post:
                        derived = physics.post_laminar_temp(values["FM_Finish_Temp"],
                                                            values["ACC_Laminar_Header"],
                                                            values["Plate_T_mm"])
                        value = min(max(derived, float(lo)), float(hi))
                        st.session_state[widget_key] = value
                values[key] = st.number_input(f"{label} [{shown_unit}]", min_value=float(lo),
                                              max_value=float(hi), step=float(step), format=fmt,
                                              key=widget_key, value=value,
                                              disabled=auto_post)
    if mill == "Plate":
        st.markdown('<div class="note">Delivery route</div>', unsafe_allow_html=True)
        values["Rolling_Type"] = st.selectbox(
            "Rolling type", ["AR", "TMCP", "N"], key=f"{mill}_rolling_type",
            help="AR = as rolled, TMCP = thermomechanically controlled, N = normalised delivery.")
    st.session_state["schedule"] = dict(values)
    st.session_state["schedule_mill"] = mill

    with st.sidebar.expander("Training statistics", expanded=False):
        st.caption("How one input actually ran in the cleaned training table, narrowed to heats "
                   "near the gauge and width above.")
        picks = [row for row in SPECS[mill] if row[0] != "Geometry"]
        unit_of = lambda row: "°C" if row[3] == "C" else row[3]
        # a spec-table edit can shrink this list while the widget key persists an old index
        prune_stale_widget_key(f"stat_var_{mill}", list(range(len(picks))))
        choice = st.selectbox("Input", range(len(picks)),
                              format_func=lambda i: f"{picks[i][2]}  [{unit_of(picks[i])}]",
                              key=f"stat_var_{mill}")
        group, key, label, unit, _lo, _hi, _default, step = picks[choice]
        geometry = tuple((g, float(values.get(g, 0.0))) for g in ts.GEOMETRY[mill])
        stats = ts.cached_stats(mill, key, geometry, float(step), (float(_lo), float(_hi)))
        if stats is None:
            have_table = not ts.load_table(mill).empty
            have_bundle = bool(ts.load_bundle())
            st.warning("No training rows for this input in this build.")
            st.caption(f"cleaned CSV: {'found' if have_table else 'not shipped'} · "
                       f"bundled summary: {'found' if have_bundle else 'not shipped'}")
            if not have_table and not have_bundle:
                st.caption("Running an older build. models/training_stats.json is part of the current "
                           "commit, so reboot the app to pick it up.")
        else:
            st.dataframe(ts.summary_frame(stats), hide_index=True, use_container_width=True)
            note = f"{stats['rows']:,} training rows · {stats['band']}"
            if stats.get("dropped"):
                note += f" · {stats['dropped']} out-of-range values dropped"
            st.caption(note)
            st.caption("source: " + ("bundled summary" if stats.get("source") == "bundle"
                                     else "cleaned training table"))

    return st.session_state["schedule"]


def _fmt_spec_value(val, step):
    """Format a schedule value cleanly using its spec step resolution."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return html_lib.escape(str(val))
    if not np.isfinite(v):
        return "—"
    if step < 0.001:
        return f"{v:.4f}"
    if step < 0.01:
        return f"{v:.3f}"
    if step < 1.0:
        return f"{v:.2f}"
    return f"{v:.0f}"


def build_mill_setup_card_html(mill, raw, preds, ind, engine="Champion GBDT (Recommended)"):
    """Build a standalone, print-ready HTML Mill Setup Card for the active schedule."""
    raw = raw or {}
    preds = preds or {}
    ind = ind or physical(raw, mill)

    ar3_val = ind.get("ar3") if isinstance(ind, dict) else None
    cqr_mult, cqr_reason = explainer.adaptive_conformal_multiplier(mill, raw, ar3=ar3_val)
    warnings, notices = explainer.check_applicability_domain(mill, raw)

    frt_key = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    finish = float(raw.get(frt_key, 0.0) or 0.0)
    ar3_f = float(ar3_val) if ar3_val is not None and np.isfinite(ar3_val) else 0.0
    frt_margin = finish - ar3_f

    route_title = "Heavy Plate Mill" if mill == "Plate" else "Steckel Coil Mill (HRC)"
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    status_cls = "warn" if warnings else "ok"
    status_txt = f"EXTRAPOLATION ({len(warnings)} OOD)" if warnings else "IN-DOMAIN SCHEDULE"

    # Rolling & cooling setpoints
    setpoint_rows = []
    for group, key, label, unit, _lo, _hi, _default, step in SPECS[mill]:
        if group == "Chemistry":
            continue
        shown_unit = "°C" if unit == "C" else unit
        val_str = _fmt_spec_value(raw[key], step) if key in raw else "—"
        setpoint_rows.append(
            f"<tr><td>{html_lib.escape(group)}</td>"
            f"<td><b>{html_lib.escape(label)}</b></td>"
            f"<td class=\"num\">{val_str} <span class=\"u\">{html_lib.escape(shown_unit)}</span></td></tr>"
        )
    if mill == "Plate" and "Rolling_Type" in raw:
        setpoint_rows.append(
            f"<tr><td>Route</td><td><b>Delivery condition</b></td>"
            f"<td class=\"num\">{html_lib.escape(str(raw.get('Rolling_Type', 'AR')))}</td></tr>"
        )

    # Heat chemistry
    chem_cells = []
    for group, key, label, unit, _lo, _hi, _default, step in SPECS[mill]:
        if group != "Chemistry":
            continue
        val_str = _fmt_spec_value(raw[key], step) if key in raw else "—"
        chem_cells.append(
            f"<div class=\"chem-item\"><span>{html_lib.escape(label)}</span>"
            f"<b>{val_str}%</b></div>"
        )

    # Derived physics metrics
    physics_items = [
        ("Finishing margin (FRT − Ar3)", f"{frt_margin:+.1f} K", f"Ar3 {ar3_f:.1f} °C"),
        ("Andrews Ar1", f"{float(ind.get('ar1', 0.0)):.1f} °C", "equilibrium lower"),
        ("Pickering σ<sub>ss</sub>", f"{float(ind.get('sigma_ss', 0.0)):.1f} MPa", "solid solution"),
        ("CE<sub>IIW</sub>", f"{float(ind.get('ce_iiw', 0.0)):.3f}", "weldability index"),
        ("P<sub>cm</sub>", f"{float(ind.get('pcm', 0.0)):.3f}", "cold-cracking index"),
        ("True strain (ε)", f"{float(ind.get('true_strain', 0.0)):.2f}", "log thickness ratio"),
        ("Cooling rate proxy", f"{float(ind.get('cooling_rate', 0.0)):.2f}", "route proxy"),
    ]
    physics_html = "".join(
        f"<div class=\"phys-item\"><span>{lbl}</span><b>{val}</b><em>{sub}</em></div>"
        for lbl, val, sub in physics_items
    )

    # Certified property predictions
    prop_rows = []
    for group, keys in TARGET_GROUPS[mill]:
        group_rows = []
        for key in keys:
            res = preds.get(key)
            if not res or not np.isfinite(res.get("pred", np.nan)):
                continue
            label, unit = TARGET_LABEL[mill].get(key, (res.get("name", key), res.get("unit", "")))
            tol = ml.TARGET_CONFIG.get(mill, {}).get(key, {}).get("tol", 0.0)
            ci_lo, ci_hi = _scaled_interval(res, key, cqr_mult)
            fam = html_lib.escape(str(res.get("family", "") or ""))
            fam_tag = f"<span class=\"tag\">{fam}</span>" if fam else ""
            note = res.get("note", "")
            note_html = f"<div class=\"subnote\">{html_lib.escape(str(note))}</div>" if note else ""
            group_rows.append(
                f"<tr><td>{html_lib.escape(label)} {fam_tag}{note_html}</td>"
                f"<td class=\"num\"><b>{float(res['pred']):.1f}</b> <span class=\"u\">{html_lib.escape(unit)}</span></td>"
                f"<td class=\"num\">{ci_lo:.1f} – {ci_hi:.1f} <span class=\"u\">{html_lib.escape(unit)}</span></td>"
                f"<td class=\"num\">±{tol:g} {html_lib.escape(unit)}</td></tr>"
            )
        if group_rows:
            prop_rows.append(f"<tr class=\"grp\"><td colspan=\"4\">{html_lib.escape(group)}</td></tr>")
            prop_rows.extend(group_rows)

    warn_block = ""
    if warnings:
        items = "".join(
            f"<li><code>{html_lib.escape(str(w['param']))}</code> = {float(w['val']):.3g} "
            f"({html_lib.escape(str(w['direction']))}, bound {float(w['bound']):.3g})</li>"
            for w in warnings
        )
        warn_block = f"<div class=\"alert warn\"><b>Extrapolation warnings (P01–P99):</b><ul>{items}</ul></div>"

    notice_block = ""
    if notices:
        items = "".join(f"<li>{html_lib.escape(str(n).replace('**', ''))}</li>" for n in notices)
        notice_block = f"<div class=\"alert info\"><b>Domain notices:</b><ul>{items}</ul></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mill Setup Card — {html_lib.escape(route_title)}</title>
<style>
  @page {{ size: A4; margin: 12mm; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #0b0d10; background: #fff; margin: 0; padding: 24px; line-height: 1.45; }}
  .wrap {{ max-width: 920px; margin: 0 auto; }}
  .hdr {{ display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0b0d10; padding-bottom: 12px; margin-bottom: 16px; }}
  .kicker {{ font-size: 11px; letter-spacing: .14em; text-transform: uppercase; color: {ACCENT}; font-weight: 700; }}
  h1 {{ margin: 4px 0; font-size: 22px; letter-spacing: -.02em; }}
  .meta {{ font-size: 12px; color: #4b5563; }}
  .pill {{ display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; padding: 4px 10px; border-radius: 4px; }}
  .pill.ok {{ background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }}
  .pill.warn {{ background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }}
  .print-btn {{ margin-top: 8px; font-size: 12px; padding: 5px 12px; border: 1px solid #d1d5db; border-radius: 4px; background: #f9fafb; cursor: pointer; }}
  .sec {{ font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #4b5563; font-weight: 700; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; margin: 18px 0 10px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  th {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .08em; color: #6b7280; text-align: left; padding: 6px; border-bottom: 1px solid #e5e7eb; background: #fafbfc; }}
  th.num, td.num {{ text-align: right; font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }}
  td {{ padding: 5px 6px; border-bottom: 1px solid #f1f2f4; vertical-align: middle; }}
  tr.grp td {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: .1em; color: {ACCENT}; font-weight: 700; padding-top: 10px; background: #fff; border-bottom: 1px solid #e5e7eb; }}
  .u {{ color: #6b7280; font-size: 11px; font-weight: 400; }}
  .tag {{ display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: .05em; padding: 1px 5px; border: 1px solid #e5e7eb; border-radius: 3px; color: #4b5563; margin-left: 4px; }}
  .subnote {{ font-size: 11px; color: #6b7280; }}
  .chem-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }}
  .chem-item, .phys-item {{ border: 1px solid #e5e7eb; border-radius: 4px; padding: 6px 8px; background: #fafbfc; }}
  .chem-item span, .phys-item span {{ display: block; font-size: 10.5px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }}
  .chem-item b, .phys-item b {{ font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 13px; color: #0b0d10; }}
  .phys-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }}
  .phys-item em {{ display: block; font-style: normal; font-size: 10.5px; color: #6b7280; }}
  .cqr-note {{ margin-top: 10px; padding: 8px 12px; border-left: 3px solid {ACCENT}; background: #f8fafc; font-size: 12.5px; }}
  .alert {{ margin-top: 10px; padding: 8px 12px; border-radius: 4px; font-size: 12px; }}
  .alert.warn {{ background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }}
  .alert.info {{ background: #eff6ff; border: 1px solid #bfdbfe; color: #1e40af; }}
  .alert ul {{ margin: 4px 0 0 16px; padding: 0; }}
  .foot {{ margin-top: 16px; padding-top: 8px; border-top: 1px solid #e5e7eb; font-size: 11px; color: #6b7280; }}
  @media print {{
    body {{ padding: 0; }}
    .no-print {{ display: none !important; }}
    .card-block {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <div>
      <div class="kicker">Mill Setup &amp; Certification Card</div>
      <h1>{html_lib.escape(route_title)} ({html_lib.escape(mill)})</h1>
      <div class="meta">Scoring engine: <b>{html_lib.escape(str(engine))}</b> &middot; Generated {html_lib.escape(stamp)}</div>
    </div>
    <div style="text-align:right;">
      <span class="pill {status_cls}">{html_lib.escape(status_txt)}</span><br>
      <button class="print-btn no-print" onclick="window.print()">Print / Save PDF</button>
    </div>
  </div>

  <div class="grid-2 card-block">
    <div>
      <div class="sec">Rolling &amp; Cooling Setpoints</div>
      <table>
        <thead><tr><th>Stage</th><th>Parameter</th><th class="num">Setpoint</th></tr></thead>
        <tbody>{''.join(setpoint_rows)}</tbody>
      </table>
    </div>
    <div>
      <div class="sec">Heat Chemistry (wt%)</div>
      <div class="chem-grid">{''.join(chem_cells)}</div>
    </div>
  </div>

  <div class="card-block">
    <div class="sec">Derived Physics Metrics</div>
    <div class="phys-grid">{physics_html}</div>
  </div>

  <div class="card-block">
    <div class="sec">Certified Property Predictions &amp; Adaptive 95% Conformal Intervals</div>
    <table>
      <thead><tr><th>Property</th><th class="num">Predicted</th><th class="num">95% Conformal Interval</th><th class="num">Cert. Tolerance</th></tr></thead>
      <tbody>{''.join(prop_rows)}</tbody>
    </table>
    <div class="cqr-note">
      <b>Adaptive Conformal Multiplier (CQR): {cqr_mult:.2f}&times;</b> &mdash; {html_lib.escape(cqr_reason)}.
      Nominal split-conformal half-widths are scaled by this factor to reflect local domain density and Ar<sub>3</sub> proximity.
    </div>
    {warn_block}
    {notice_block}
  </div>

  <div class="foot">Build: {html_lib.escape(BUILD)}</div>
</div>
</body>
</html>"""


def _schedule_compare_rows(mill, raw, preds, ind, baseline):
    """Build comparison rows (Schedule B current vs Schedule A pinned baseline)."""
    b_raw = baseline.get("raw") or {}
    b_preds = baseline.get("preds") or {}
    b_ind = baseline.get("ind") or {}

    frt_key = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    cur_margin = float(raw.get(frt_key, 0.0)) - float(ind.get("ar3", 0.0))
    base_margin = float(b_raw.get(frt_key, 0.0)) - float(b_ind.get("ar3", 0.0))

    specs = [
        ("YS_BN", "Yield strength (YS_BN)", "MPa", ".1f",
         preds.get("YS_BN", {}).get("pred", np.nan), b_preds.get("YS_BN", {}).get("pred", np.nan)),
        ("TS_BN", "Tensile strength (TS_BN)", "MPa", ".1f",
         preds.get("TS_BN", {}).get("pred", np.nan), b_preds.get("TS_BN", {}).get("pred", np.nan)),
        ("EL_BN", "Elongation (EL_BN)", "%", ".1f",
         preds.get("EL_BN", {}).get("pred", np.nan), b_preds.get("EL_BN", {}).get("pred", np.nan)),
        ("CE_IIW", "Carbon equivalent (CE_IIW)", "", ".3f",
         ind.get("ce_iiw", np.nan), b_ind.get("ce_iiw", np.nan)),
        ("FRT_Ar3", "Finishing margin (FRT − Ar3)", "K", ".1f",
         cur_margin, base_margin),
    ]
    rows = []
    for key, label, unit, fmt, cur_val, base_val in specs:
        cur_f = float(cur_val) if cur_val is not None else np.nan
        base_f = float(base_val) if base_val is not None else np.nan
        delta = cur_f - base_f if (np.isfinite(cur_f) and np.isfinite(base_f)) else np.nan
        rows.append({
            "key": key, "label": label, "unit": unit, "fmt": fmt,
            "baseline": base_f, "current": cur_f, "delta": delta,
        })
    return rows


def _render_schedule_compare_card(mill, raw, preds, ind, baseline):
    """Render the Schedule Compare (Current vs. Pinned Baseline) comparison card."""
    rows = _schedule_compare_rows(mill, raw, preds, ind, baseline)
    tr_html = []
    for r in rows:
        fmt = r["fmt"]
        unit = f'<span class="unit">{r["unit"]}</span>' if r["unit"] else ""
        base_s = f"{r['baseline']:{fmt}}{unit}" if np.isfinite(r["baseline"]) else "—"
        cur_s = f"{r['current']:{fmt}}{unit}" if np.isfinite(r["current"]) else "—"
        if np.isfinite(r["delta"]):
            sign = "+" if r["delta"] > 0 else ""
            d_color = ACCENT if abs(r["delta"]) > 1e-6 else MUTED
            delta_s = f'<span style="color:{d_color}; font-weight:600;">{sign}{r["delta"]:{fmt}}{unit}</span>'
        else:
            delta_s = "—"
        tr_html.append(
            f'<tr><td class="name">{r["label"]}</td>'
            f'<td class="rng">{base_s}</td>'
            f'<td class="val">{cur_s}</td>'
            f'<td class="val">{delta_s}</td></tr>'
        )
    b_raw = baseline.get("raw") or {}
    diffs = []
    for _grp, key, lbl, _u, _lo, _hi, _d, step in SPECS[mill]:
        if key in raw and key in b_raw:
            try:
                if abs(float(raw[key]) - float(b_raw[key])) > max(1e-9, step * 0.05):
                    diffs.append(f"{lbl}: {_fmt_spec_value(b_raw[key], step)} → {_fmt_spec_value(raw[key], step)}")
            except (TypeError, ValueError):
                continue
    diff_note = " · Changed inputs: " + ", ".join(diffs[:5]) + ("…" if len(diffs) > 5 else "") if diffs else " · Identical inputs to Schedule A."
    pinned_at = html_lib.escape(str(baseline.get("pinned_at", "")))
    pinned_eng = html_lib.escape(str(baseline.get("engine", "")))

    section("Schedule Compare (Current vs. Pinned Baseline)")
    card_html = (
        '<div class="card"><table class="ut">'
        '<thead><tr><th>Metric</th><th>Schedule A (Baseline)</th>'
        '<th>Schedule B (Current)</th><th>Delta (Current − Baseline)</th></tr></thead>'
        f'<tbody>{"".join(tr_html)}</tbody></table>'
        f'<div class="foot">Pinned at {pinned_at} ({pinned_eng}){html_lib.escape(diff_note)}</div></div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sections
def view_simulator(mill, raw, engine="Champion GBDT (Recommended)"):
    # Applicability Domain Monitoring
    warnings, notices = explainer.check_applicability_domain(mill, raw)
    if warnings:
        warn_msg = "**Historical Extrapolation Alert**: The current schedule parameters fall outside the historical 98% mill envelope (P01–P99): " + "; ".join([f"`{w['param']}` = {w['val']:.1f} ({w['direction']})" for w in warnings[:3]])
        st.warning(warn_msg)
    for ntc in notices:
        st.info(ntc)

    section("predicted properties")
    preds = predict_selected_engine(mill, raw, engine)
    ind = physical(raw, mill)
    cqr_mult, cqr_reason = explainer.adaptive_conformal_multiplier(mill, raw, ar3=ind["ar3"])

    options = []
    if linear_families(mill):
        options += ["OLS", "Ridge"]
    if nlr_models(mill):
        options.append("NLR")
    if "Ensemble" not in engine:
        options.append("Ensemble (.joblib)")
    for family in ("xgb", "lgbm", "cat"):
        disp = {"xgb": "XGBoost", "lgbm": "LightGBM", "cat": "CatBoost"}[family]
        if disp != engine and family in boosted_index(mill):
            options.append(disp)
    chosen = st.multiselect("Also fit", options, default=[],
                            help="Alternative models to compare side-by-side with the active scoring engine.",
                            key=f"fams_{mill}_{engine}")
    extras = {}
    for family in chosen:
        if family in ("OLS", "Ridge"):
            extras[family] = predict_linear(mill, raw, family)
        elif family == "NLR":
            extras[family] = predict_nlr(mill, raw)
        elif family == "Ensemble (.joblib)":
            extras[family] = predict_one(mill, raw)
        else:
            key = {"XGBoost": "xgb", "LightGBM": "lgbm", "CatBoost": "cat"}[family]
            extras[family] = predict_boosted(mill, raw, key)
    targets_table(mill, preds, extras, engine_label=engine,
                  conformal_mult=cqr_mult, conformal_reason=cqr_reason)
    if extras:
        parts = []
        for f in extras:
            r2s = [float(v["r2"]) for v in extras[f].values() if isinstance(v, dict) and "r2" in v and pd.notna(v.get("r2"))]
            if r2s:
                parts.append(f"{f} {np.nanmean(r2s):.2f}")
        if parts:
            st.caption("Holdout R² of the added families, averaged over targets: "
                       + ", ".join(parts)
                       + ". Where a linear fit is hopeless the number says so.")

    broken = {k: v["error"] for k, v in preds.items() if v.get("error")}
    if broken:
        with st.expander(f"{len(broken)} target(s) could not be scored", expanded=True):
            st.caption("The shipped artifact refused the feature frame. Paste this when reporting it.")
            for key, why in broken.items():
                st.code(f"{key}: {why[:400]}", language="text")

    # Schedule A vs. B Pinning & Printable Mill Setup Card toolbar
    st.markdown('<div class="breath"></div>', unsafe_allow_html=True)
    pin_key = f"pinned_baseline_{mill}"
    has_pin = pin_key in st.session_state
    tb_cols = st.columns([1.45, 1.05, 1.75, 1.75] if has_pin else [1.45, 1.75, 2.80])
    if tb_cols[0].button("Pin as Baseline (Schedule A)", key=f"pin_base_{mill}", use_container_width=True):
        st.session_state[pin_key] = {
            "raw": dict(raw),
            "preds": {k: dict(v) for k, v in preds.items()},
            "ind": dict(ind),
            "engine": engine,
            "pinned_at": datetime.now().strftime("%H:%M:%S"),
        }
    if has_pin:
        if tb_cols[1].button("Clear baseline", key=f"clear_base_{mill}", use_container_width=True):
            st.session_state.pop(pin_key, None)
        dl_col = tb_cols[2]
    else:
        dl_col = tb_cols[1]
    setup_html = build_mill_setup_card_html(mill, raw, preds, ind, engine)
    dl_col.download_button(
        "Download Mill Setup Card (.html)",
        data=setup_html.encode("utf-8"),
        file_name=f"{mill.lower()}_mill_setup_card.html",
        mime="text/html",
        key=f"dl_setup_card_{mill}",
        use_container_width=True,
    )

    pinned = st.session_state.get(pin_key)
    if pinned:
        st.markdown('<div class="breath"></div>', unsafe_allow_html=True)
        _render_schedule_compare_card(mill, raw, preds, ind, pinned)

    finish = raw.get("FM_Finish_Temp", raw.get("Temp_FT_Avg_C", 0.0))
    margin = finish - ind["ar3"]
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    section("derived process metrics")
    st.markdown(f"""
<div class="facts">
  <div class="lead"><span>finishing margin above Ar3</span><b>{margin:+.0f}</b><em>K · Ar3 {ind['ar3']:.0f} °C</em></div>
  <div><span>Pickering σ<sub>ss</sub></span><b>{ind['sigma_ss']:.0f}</b><em>MPa</em></div>
  <div><span>CE IIW</span><b>{ind['ce_iiw']:.3f}</b></div>
  <div><span>Pcm</span><b>{ind['pcm']:.3f}</b></div>
  <div><span>true strain</span><b>{ind['true_strain']:.2f}</b></div>
</div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="note">Finishing at {finish:.0f} °C sits {margin:+.0f} K above Ar3 '
                f'({ind["ar3"]:.0f} °C). Below zero the last passes are intercritical, which is where '
                f'properties start to scatter.</div>', unsafe_allow_html=True)

    # In-App Explainability
    st.markdown('<div class="breath"></div>', unsafe_allow_html=True)
    with st.expander("Feature contribution breakdown (TreeSHAP attribution)", expanded=False):
        tgt_choices = list(TARGET_LABEL[mill])
        def_idx = 1 if len(tgt_choices) > 1 else 0  # Default to TS_BN
        chosen_tgt = st.selectbox("Inspect target attribution", tgt_choices, index=def_idx,
                                  format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"shap_tgt_{mill}")
        frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
        X_explain = feature_matrix(mill, frame)
        unit = TARGET_LABEL[mill][chosen_tgt][1]
        explainer.render_feature_contributions_ui(mill, chosen_tgt, X_explain, unit=unit,
                                                  family=(champion_plan(mill).get(chosen_tgt) or {}).get("family"))


def view_spec_check(mill, raw, engine="Champion GBDT (Recommended)"):
    labels, lookup = sc.spec_grade_options()
    if not labels:
        st.error("No specifications could be loaded from specification-list/SPECIFICATION LIST.xlsx. "
                 "The workbook ships with the repository and is looked up relative to the app "
                 "package, so confirm the file is still at that path in this checkout; if you are "
                 "running a copied or deployed app, make sure the specification-list/ folder was "
                 "uploaded beside app/. Then rerun this tab.")
        return

    # Smart defaults: A 572 Gr 50 for plate, JIS G 3101 SS400 for coil.
    preferred_key = ("ASTMA57250" if mill == "Plate" else "JISG3101SS400")
    if sc.norm_key(preferred_key) not in map(sc.norm_key, labels):
        preferred = next((label for label in labels
                          if ("A 572" in label if mill == "Plate" else "JIS G 3101" in label)),
                         labels[0])
    else:
        preferred = next(label for label in labels if sc.norm_key(label) == preferred_key)

    c1, c3 = st.columns([4.3, 1.7], vertical_alignment="bottom")
    chosen_label = c1.selectbox("Specification + grade", labels,
                                index=labels.index(preferred), key=f"sc_sg_{mill}")
    chosen_spec, chosen_grade = lookup[chosen_label]

    t_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    thickness = float(raw.get(t_key, 16.0 if mill == "Plate" else 6.0))
    c3.markdown(f'<div style="padding-bottom:.4rem; font-size:13px; color:#4b5563;">Active Gauge: '
                f'<b style="font-family:monospace; color:#0b0d10;">{thickness:.1f} mm</b><br>'
                f'<span style="font-size:11px; color:{ACCENT}; font-weight:600;">Engine: {engine}</span></div>',
                unsafe_allow_html=True)

    preds = predict_selected_engine(mill, raw, engine)
    ind = physical(raw, mill)
    supplied = {row[1] for row in SPECS[mill]} | {"ce_iiw", "pcm"}
    eval_res = sc.evaluate_all(chosen_spec, chosen_grade, thickness, raw, preds, ind, mill,
                               available_keys=supplied)

    if not eval_res["matched"]:
        st.warning(eval_res.get("message", "No matching tier found."))
        return

    # Executive Summary Strip
    gauge_info = eval_res["gauge"]
    chem_info = eval_res["chemistry"]
    mech_info = eval_res["mechanical"]

    # Gauge verdict pill
    g_verdict = gauge_info["verdict"]
    g_pill_cls = "ok" if g_verdict == "COMPLIANT" else ("warn" if "WARN" in g_verdict else "bad")
    g_pill_text = "IN SPEC" if g_verdict == "COMPLIANT" else g_verdict.replace("_", " ")

    # Chem verdict pill
    c_status = chem_info["status"]
    c_pill_cls = "ok" if c_status == "ALL_PASS" else ("warn" if c_status == "PARTIAL_VERIFICATION" else "bad")
    if c_status == "ALL_PASS":
        c_pill_text = "ALL PASS"
    elif c_status == "PARTIAL_VERIFICATION":
        c_pill_text = f"PARTIAL: {len(chem_info.get('unverifiable', []))} NOT VERIFIABLE"
    else:
        c_pill_text = f"{chem_info['fail_count']} OUT OF SPEC"

    # Mech verdict pill
    m_status = mech_info["status"]
    if m_status == "COMPLIANT":
        m_pill_cls, m_pill_text = "ok", "COMPLIANT"
    elif m_status in ("MARGINAL_RISK", "PARTIAL_VERIFICATION"):
        m_pill_cls, m_pill_text = "warn", "MARGINAL RISK" if m_status == "MARGINAL_RISK" else "PARTIAL: IMPACT NOT VERIFIABLE"
    elif m_status == "NON_COMPLIANT":
        m_pill_cls, m_pill_text = "bad", "NON-COMPLIANT"
    elif m_status == "NOT_VERIFIABLE":
        m_pill_cls, m_pill_text = "warn", "NOT VERIFIABLE"
    else:
        m_pill_cls, m_pill_text = "neutral", "NO REQ SPECIFIED"

    st.markdown('<div class="breath"></div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="facts facts-text">
  <div>
    <span>Specification &amp; Type</span>
    <b>{chosen_spec} · Gr {chosen_grade} · {eval_res['spec_type']}</b>
  </div>
  <div>
    <span>Standard Gauge &amp; GRP Limit</span>
    <b>{thickness:.1f} mm</b><em><span class="pill {g_pill_cls}">{g_pill_text}</span></em>
  </div>
  <div>
    <span>Chemical Composition</span>
    <b>{chem_info['pass_count']} / {len(chem_info['active_checks'])} elements in spec</b><em><span class="pill {c_pill_cls}">{c_pill_text}</span></em>
  </div>
  <div>
    <span>Mechanical compliance</span>
    <b>{mech_info['pass_count']} requirements passed</b><em><span class="pill {m_pill_cls}">{m_pill_text}</span></em>
  </div>
</div>""", unsafe_allow_html=True)

    if gauge_info.get("message"):
        st.markdown(f'<div class="note">{gauge_info["message"]}</div>', unsafe_allow_html=True)

    # 1. Mechanical compliance table
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    section("mechanical properties: compliance check")
    st.caption(f"Correlates {mill} ML predictions and 95% split-conformal prediction intervals against "
               f"{chosen_spec} Grade {chosen_grade} requirements. "
               f"A marginal risk indicates the point prediction passes but the 95% interval bound crosses the limit under process variance.")

    mech_checks = mech_info.get("checks", [])
    if mech_checks:
        m_rows = []
        for chk in mech_checks:
            verdict = chk["verdict"]
            if verdict == "NOT_VERIFIABLE":
                m_rows.append(f"""<tr>
                  <td class="name">{chk['property']} <span class="sub" style="margin-left:.4rem;">{chk.get('note', 'cannot be verified')}</span></td>
                  <td class="sub" style="font-weight:600; color:#374151;">{chk['req_label']}</td>
                  <td class="val" style="color:#6b7280;">—</td>
                  <td class="rng">—</td>
                  <td class="sub">—</td>
                  <td class="flag"><span class="pill warn">NOT VERIFIABLE</span></td>
                </tr>""")
                continue
            v_pill = "ok" if verdict == "COMPLIANT" else ("warn" if verdict == "MARGINAL_RISK" else "bad")
            v_label = "COMPLIANT" if verdict == "COMPLIANT" else ("MARGINAL RISK" if verdict == "MARGINAL_RISK" else "NON-COMPLIANT")
            sign = "+" if chk["margin"] >= 0 else ""
            m_rows.append(f"""<tr>
              <td class="name">{chk['property']} <span class="sub" style="margin-left:.4rem;">{chk['note']}</span></td>
              <td class="sub" style="font-weight:600; color:#374151;">{chk['req_label']}</td>
              <td class="val">{chk['pred']:.1f}<span class="unit">{chk['unit']}</span></td>
              <td class="rng">{chk['ci_low']:.1f} – {chk['ci_high']:.1f}</td>
              <td class="val" style="color:{'#065f46' if chk['margin']>=0 else '#991b1b'};">{sign}{chk['margin']:.1f}<span class="unit">{chk['unit']}</span></td>
              <td class="flag"><span class="pill {v_pill}">{v_label}</span></td>
            </tr>""")
        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>property</th><th>standard requirement</th><th>predicted</th>'
            '<th>95% prediction interval</th><th>safety margin</th><th>verdict</th></tr></thead><tbody>'
            + "".join(m_rows) +
            '</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No specific mechanical property limits specified for {chosen_spec} Grade {chosen_grade} in standard table.")

    # 2. Chemistry Verification Table
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    section("chemical composition verification")
    st.caption("Active element limits specified for this standard. Trace unconstrained elements are summarized below.")

    active_chem = chem_info.get("active_checks", [])
    if active_chem:
        c_rows = []
        for ac in active_chem:
            c_status = ac["status"]
            c_pill = "ok" if c_status == "PASS" else "bad"
            min_str = f"{ac['min']:.3g} %" if ac['min'] is not None else "—"
            max_str = f"{ac['max']:.3g} %" if ac['max'] is not None else "—"
            margin_str = f"{ac['margin']:+.3g} %" if ac['margin'] is not None else "—"
            c_rows.append(f"""<tr>
              <td class="name">{ac['label']} <span class="sub" style="margin-left:.3rem;">{ac['notes']}</span></td>
              <td class="val">{ac['actual']:.3f}<span class="unit">%</span></td>
              <td class="rng">{min_str}</td>
              <td class="rng">{max_str}</td>
              <td class="sub" style="font-family:monospace; color:{'#065f46' if c_status=='PASS' else '#991b1b'};">{margin_str}</td>
              <td class="flag"><span class="pill {c_pill}">{c_status}</span></td>
            </tr>""")
        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>element</th><th>schedule input</th><th>spec min</th><th>spec max</th>'
            '<th>headroom / margin</th><th>status</th></tr></thead><tbody>'
            + "".join(c_rows) +
            '</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No chemical composition constraints listed for this standard tier.")

    # Unconstrained elements compact summary
    unconstrained = chem_info.get("unconstrained", [])
    if unconstrained:
        uncon_pills = " ".join([f'<span class="tag" style="margin:.15rem .2rem; background:#f9fafb;">{u["label"].split()[0]}: {u["actual"]:.3g}%</span>'
                                for u in unconstrained])
        st.markdown(f'<div style="margin-top:.6rem; font-size:11px; color:#6b7280;">Unconstrained trace inputs (no standard limit):<br>{uncon_pills}</div>',
                    unsafe_allow_html=True)

    unverifiable_chem = chem_info.get("unverifiable", [])
    if unverifiable_chem:
        names = ", ".join(u["label"] for u in unverifiable_chem)
        st.markdown(f'<div class="note">The standard states limits for {names}, but the '
                    f'{mill} schedule form does not carry that element, so the limit cannot be '
                    f'verified here. Enter it in an uploaded sheet or judge it at the ladle.',
                    unsafe_allow_html=True)

    # 3. Standard Notes & Footnotes Callout
    if eval_res.get("notes"):
        st.markdown('<div class="break"></div>', unsafe_allow_html=True)
        section("standard footnotes & technical notes")
        st.markdown(f'<div class="note" style="border-left-color:{ACCENT}; background:#fbfcfe; padding:.75rem .9rem; border-radius:0 4px 4px 0;">'
                    f'<b>Standard Notes (GRP.ST.QA.09):</b><br>{eval_res["notes"]}</div>',
                    unsafe_allow_html=True)


def iso_line_value(mill, label, target, thickness, rolling_type="AR"):
    """Spec-required minimum for `target` at a merged spec+grade tier (pure helper).

    Resolves the picker label ("ASTM A 572 50") through spec_checker.spec_grade_options and
    reads the tier's requirements via optimizer.spec_requirements (chemistry=None: the limit
    itself does not depend on the candidate chemistry). Returns float(required) for the
    FIRST min-type requirement of the drawn target, or None when the tier is unmatched or
    carries no min-type requirement for it. Max-type ceilings (CE, S, P, and range specs'
    upper TS bounds) are deliberately skipped: an iso-line of a ceiling is drawn as a
    "keep-out" region, not a spec floor, and the overlay targets floor compliance only.
    """
    _labels, lookup = sc.spec_grade_options()
    pair = lookup.get(label)
    if pair is None:
        return None
    spec, grade = pair
    try:
        reqs, _info = optimizer.spec_requirements(spec, grade, float(thickness), mill,
                                                  rolling_type=rolling_type, chemistry=None)
    except Exception:  # noqa: BLE001 - a malformed workbook row must never break the surface
        return None
    for req in reqs:
        if req.get("key") == target and req.get("direction") == "min":
            try:
                val = float(req.get("required"))
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                return val
    return None


def surface_ood_mask(mill, xs, ys, x_key, y_key, base):
    """Out-of-envelope boolean mask over the 2-D contour grid (pure helper).

    True where a grid point sweeps either axis outside the historical P01-P99 envelope of
    models/applicability_domain.json — the same source view_simulator checks through
    explainer.check_applicability_domain / optimizer.ood_mask. All-False when the domain
    carries no usable entry for the swept axes, so callers can skip the shading silently.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    nx, ny = len(xs), len(ys)
    frame = pd.DataFrame([dict(base) for _ in range(nx * ny)])
    frame[x_key] = np.repeat(xs, ny)
    frame[y_key] = np.tile(ys, nx)
    mask = optimizer.ood_mask(mill, frame, [x_key, y_key])
    return mask.to_numpy(dtype=bool).reshape(nx, ny)


def view_surface(mill, raw, engine="Champion GBDT (Recommended)"):
    limits = cons.get_limits(mill, SPECS[mill])
    keys = surface_keys(mill)
    labels = {row[1]: row[2] for row in SPECS[mill]}
    axes = dict(showline=True, linecolor=HAIR, gridcolor="#f1f2f4", zeroline=False,
                tickfont=dict(size=10, color=MUTED), title_font=dict(size=11, color=INK))

    head = st.columns([1.6, 1.2, 1.4], vertical_alignment="bottom")
    view = head[0].radio("View", ["2D contour", "3D surface", "1D slice"],
                         horizontal=True, key=f"v_{mill}")
    head[2].caption(("Bounded by house limits: " + cons.describe(limits)) if limits
                    else f"Engine: {engine}. Rest of schedule held at sidebar values.")

    if view == "1D slice":
        _surface_slice(mill, raw, engine, keys, labels, limits, axes)
        return

    if not keys:
        st.error("This route ships no sweepable inputs, so there is nothing to plot.")
        return
    left, mid, right = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")
    x_key = left.selectbox("Horizontal axis", keys, index=0, key=f"x_{mill}",
                           format_func=lambda k: labels.get(k, k))
    y_opts = [k for k in keys if k != x_key]
    prune_stale_widget_key(f"y_{mill}", y_opts)
    y_key = mid.selectbox("Vertical axis", y_opts, index=min(1, len(y_opts) - 1), key=f"y_{mill}",
                          format_func=lambda k: labels.get(k, k))
    target = right.selectbox("Property", list(TARGET_LABEL[mill]),
                             format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"t_{mill}")
    grid = st.slider("Grid", 15, 41, 25, step=2, key=f"g_{mill}")

    sx, sy = spec_map(mill)[x_key], spec_map(mill)[y_key]
    xs = np.linspace(*axis_range(mill, x_key, limits), grid)
    ys = np.linspace(*axis_range(mill, y_key, limits), grid)
    frame = pd.DataFrame([dict(raw) for _ in range(grid * grid)])
    frame[x_key] = np.repeat(xs, grid)
    frame[y_key] = np.tile(ys, grid)
    scored = score_frame(mill, frame, engine=engine)
    if target not in scored.columns:
        st.error(f"{TARGET_LABEL[mill][target][0]} could not be scored on this route, so the "
                 f"surface has nothing to draw. Switch engine or check models/ for {mill}.")
        return
    z = scored[target].values.reshape(grid, grid)

    # grid rows run along x (axis 0) and columns along y (axis 1), both ascending, so the
    # header guardrail is a running max/min along whichever axis carries the headers.
    if mill == "Plate" and target in HEADER_SWEEP_SIGN:
        hdr_axis = 0 if x_key == "ACC_Laminar_Header" else \
            (1 if y_key == "ACC_Laminar_Header" else None)
        if hdr_axis is not None:
            sign = HEADER_SWEEP_SIGN[target]
            z = (np.maximum.accumulate(z, axis=hdr_axis) if sign > 0
                 else np.minimum.accumulate(z, axis=hdr_axis))

    label, unit = TARGET_LABEL[mill][target]
    point = cons.clamp_raw(mill, raw, limits)[0]
    here, here_err = single_target(mill, point, engine, target)
    if not np.isfinite(here):
        st.error(f"The current schedule could not be scored for {label} with {engine}: {here_err}")
        return

    if view == "2D contour":
        # --- spec iso-line overlay (2D branch only): min-type mechanical requirements ------
        ovr_labels, _ovr_lookup = sc.spec_grade_options()
        overlay = st.selectbox("Overlay spec limit", ["None"] + ovr_labels, key=f"ovr_{mill}",
                               help="Draws the chosen tier's minimum requirement for the "
                                    "plotted property as a labelled contour line at the "
                                    "current gauge.")
        thickness = float(raw.get("Plate_T_mm" if mill == "Plate" else "Coil_T_mm",
                                  16.0 if mill == "Plate" else 6.0))
        iso_val = (None if overlay == "None"
                   else iso_line_value(mill, overlay, target, thickness,
                                       rolling_type=str(raw.get("Rolling_Type", "AR"))))
        iso_caption, ood_caption = None, None

        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z, colorscale="Blues", contours=dict(showlabels=True,
            labelfont=dict(size=9, color="white")), line=dict(width=.4, color="rgba(255,255,255,.6)"),
            hovertemplate=f"{sx[2]}: %{{x:.1f}}<br>{sy[2]}: %{{y:.1f}}<br>{label}: %{{z:.1f}} {unit}<extra></extra>",
        ))
        if iso_val is not None:
            zf = z[np.isfinite(z)]
            z_lo, z_hi = (float(zf.min()), float(zf.max())) if zf.size else (None, None)
            if z_lo is not None and z_lo <= iso_val <= z_hi:
                # Constant-z contours never render (marching squares has no crossings in a
                # flat field), so draw the line from a +-eps band field: the contour level
                # iso_val then falls exactly where z crosses the requirement, and the
                # contour label still prints the required value.
                eps = max(1e-6, 1e-3 * abs(iso_val))
                band = np.where(z >= iso_val, iso_val + eps, iso_val - eps)
                fig.add_trace(go.Contour(
                    x=xs, y=ys, z=band,
                    contours=dict(start=iso_val, end=iso_val, size=eps, coloring="lines",
                                  showlabels=True, labelfont=dict(size=10, color=INK)),
                    line=dict(color="#b91c1c", width=2.2, dash="dash"),
                    showscale=False, hoverinfo="skip"))
                iso_caption = (f"Dashed red line: {overlay} minimum {label} ≥ {iso_val:.0f} "
                               f"{unit} at {thickness:.1f} mm gauge.")
            else:
                rng = (f"the plotted range ({z_lo:.0f}–{z_hi:.0f} {unit})"
                       if z_lo is not None else "the plotted range")
                iso_caption = (f"{overlay} requires {label} ≥ {iso_val:.0f} {unit}, which lies "
                               f"outside {rng}; no iso-line drawn.")
        elif overlay != "None":
            iso_caption = f"No requirement for this property in {overlay}."

        # --- out-of-envelope shading (2D branch only) ---------------------------------------
        ood = surface_ood_mask(mill, xs, ys, x_key, y_key, raw)
        if ood.any():
            xx = np.repeat(xs, grid).reshape(grid, grid)
            yy = np.tile(ys, grid).reshape(grid, grid)
            fig.add_trace(go.Scatter(x=xx[ood], y=yy[ood], mode="markers",
                                     name="outside historical envelope",
                                     marker=dict(size=4, color="#9ca3af", opacity=0.35),
                                     hoverinfo="skip"))
            ood_caption = ("grey points lie outside the mill's historical P01-P99 envelope; "
                           "treat them as extrapolation")

        fig.add_trace(go.Scatter(x=[point[x_key]], y=[point[y_key]], mode="markers",
                                 marker=dict(size=13, color=INK, line=dict(color="white", width=2.5)),
                                 hovertemplate=f"this schedule<br>{label}: {here:.1f} {unit}<extra></extra>",
                                 showlegend=False))
        fig.add_annotation(x=point[x_key], y=point[y_key], text=f"<b>{here:.1f} {unit}</b>",
                           showarrow=True, arrowhead=0, ax=0, ay=-30, bgcolor="white",
                           bordercolor=INK, borderwidth=1, font=dict(size=11, family="JetBrains Mono"))
        fig.update_xaxes(title=sx[2], **axes); fig.update_yaxes(title=sy[2], **axes)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    else:
        fig = go.Figure(go.Surface(x=xs, y=ys, z=z, colorscale="Blues", showscale=False,
                                   contours=dict(z=dict(show=True, usecolormap=True, project_z=True))))
        fig.add_trace(go.Scatter3d(x=[point[x_key]], y=[point[y_key]], z=[here], mode="markers",
                                   marker=dict(size=5, color=INK), name="this schedule"))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                          scene=dict(xaxis=dict(title=sx[2], **axes), yaxis=dict(title=sy[2], **axes),
                                     zaxis=dict(title=f"{label} [{unit}]", **axes),
                                     camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))),
                          paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
    if view == "2D contour":
        for cap in (iso_caption, ood_caption):
            if cap:
                st.caption(cap)
    st.markdown(f'<div class="note">Surface held on the mill\'s own feature engineering, not a proxy: '
                f'every node is a full pass through the {mill} pipeline. The marker is the current '
                f'schedule at {here:.1f} {unit}.</div>', unsafe_allow_html=True)


HEADER_SWEEP_SIGN = {"TS_BN": 1, "YS_BN": 1, "EL_BN": -1, "TS_AN": -1, "YS_AN": -1,
                     "EL_AN": 1, "Impact_0C": 1, "Impact_m20C": 1, "Impact_m40C": 1}


def monotone_guardrail(scores, x_key, mill):
    """Isotonic correction of a header sweep: residual GBDT tree noise removed per
    physically-defensible sign. Targets with an ambiguous sign are left untouched."""
    if mill != "Plate" or x_key != "ACC_Laminar_Header":
        return scores
    from scipy.optimize import isotonic_regression
    out = scores.copy()
    for key, sign in HEADER_SWEEP_SIGN.items():
        if key not in out.columns:
            continue
        y = pd.to_numeric(out[key], errors="coerce").to_numpy(dtype=float)
        fit = isotonic_regression(y, increasing=(sign > 0))
        out[key] = fit.x if hasattr(fit, "x") else fit
    return out


@st.cache_data(show_spinner=False)
def _slice_scores(mill, x_key, xs, engine, raw_tuple, couple_post=False):
    """One vectorised pass: sweep X, hold everything else, score every target."""
    frame = pd.DataFrame([dict(raw_tuple) for _ in range(len(xs))])
    frame[x_key] = np.asarray(xs, dtype=float)
    if mill == "Plate" and couple_post and x_key == "ACC_Laminar_Header":
        frame["ACC_Temp_Post_Laminar"] = physics.post_laminar_temp(
            frame["FM_Finish_Temp"], frame[x_key], frame["Plate_T_mm"])
    scores = score_frame(mill, frame, engine=engine)
    return monotone_guardrail(scores, x_key, mill)


def _surface_slice(mill, raw, engine, keys, labels, limits, axes):
    """1D X-Y slice: X = any numeric input, Y = one or more properties."""
    c = st.columns([1.5, 2.6, 1.0], vertical_alignment="bottom")
    x_key = c[0].selectbox("X axis (independent)", keys, index=0, key=f"xs_{mill}",
                           format_func=lambda k: labels.get(k, k))
    y_keys = c[1].multiselect("Y axis (dependent)", list(TARGET_LABEL[mill]),
                              default=["YS_BN", "TS_BN"],
                              format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"ys_{mill}")
    n = c[2].slider("Points", 11, 121, 41, step=5, key=f"np_{mill}")

    if not y_keys:
        st.info("Select at least one property for the Y axis.")
        return

    lo, hi = axis_range(mill, x_key, limits)
    xs = np.linspace(lo, hi, n)
    point = cons.clamp_raw(mill, raw, limits)[0]
    try:
        scores = _slice_scores(mill, x_key, tuple(xs), engine,
                               tuple(sorted(dict(raw).items())),
                               couple_post=bool(st.session_state.get(
                                   f"{mill}_post_laminar_auto", True)))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Sweep failed: {exc}")
        return

    conf = ml.get_conformal_intervals(mill)
    gauge = raw.get(ts.GEOMETRY[mill][0])
    observed = tst.bins_for(mill, x_key, gauge)

    missing = [k for k in y_keys if k not in scores.columns]
    if missing:
        st.warning("Not scored on this route: "
                   + ", ".join(TARGET_LABEL[mill][k][0] for k in missing))
        y_keys = [k for k in y_keys if k in scores.columns]
    if not y_keys:
        return

    def half_width(key):
        if conf is not None and not conf.empty and "Target" in conf.columns:
            row = conf[conf["Target"] == key]
            if not row.empty and "Conformal_Half_Width_95" in row.columns:
                return float(row["Conformal_Half_Width_95"].values[0])
        return None

    by_unit = {}
    for key in y_keys:
        by_unit.setdefault(TARGET_LABEL[mill][key][1], []).append(key)
    units = list(by_unit)

    fig = make_subplots(rows=len(units), cols=1, shared_xaxes=True, vertical_spacing=0.12,
                        subplot_titles=[f"[{u}]" for u in units])
    palette = CHART_SERIES
    first_key, hw = y_keys[0], half_width(y_keys[0])

    for r, unit in enumerate(units, start=1):
        for i, key in enumerate(by_unit[unit]):
            label, _ = TARGET_LABEL[mill][key]
            y = scores[key].to_numpy(dtype=float)
            color = palette[i % len(palette)]
            if key == first_key and hw:
                fig.add_trace(go.Scatter(x=xs, y=y + hw, line=dict(width=0), hoverinfo="skip",
                                         showlegend=False), row=r, col=1)
                fig.add_trace(go.Scatter(x=xs, y=y - hw, line=dict(width=0), fill="tonexty",
                                         fillcolor=_rgba(ACCENT, .12), hoverinfo="skip",
                                         showlegend=False, name=f"{label} 95% band"), row=r, col=1)
            fig.add_trace(go.Scatter(
                x=xs, y=y, mode="lines", name=label, line=dict(color=color, width=2.2),
                hovertemplate=f"{labels.get(x_key, x_key)} %{{x:.2f}}<br>{label} %{{y:.1f}} {unit}<extra></extra>",
            ), row=r, col=1)

            points = [(b["x"], b["targets"][key]) for b in observed if key in b.get("targets", {})]
            if points:
                fig.add_trace(go.Scatter(
                    x=[p[0] for p in points], y=[p[1]["mean"] for p in points],
                    error_y=dict(type="data", array=[p[1]["se"] for p in points],
                                 thickness=1, width=3, color=MUTED),
                    mode="markers", name=f"{label} · observed, adjusted",
                    marker=dict(size=7, symbol="diamond-open", color=color),
                    customdata=[p[1]["n"] for p in points],
                    hovertemplate=(f"{labels.get(x_key, x_key)} %{{x:.1f}}<br>"
                                   f"observed {label} %{{y:.1f}} {unit}<br>n=%{{customdata}}"
                                   "<extra></extra>"),
                ), row=r, col=1)
        fig.update_yaxes(title=unit, row=r, col=1, **axes)
        if x_key in limits:
            for xv in limits[x_key]:
                fig.add_vline(x=xv, line=dict(color=MUTED, width=1, dash="dash"), row=r, col=1)

    op = predict_selected_engine(mill, point, engine)
    op_x = float(point.get(x_key, xs[len(xs) // 2]))
    for r, unit in enumerate(units, start=1):
        for key in by_unit[unit]:
            val = op.get(key, {}).get("pred")
            if val is not None and np.isfinite(val):
                fig.add_trace(go.Scatter(
                    x=[op_x], y=[float(val)], mode="markers", showlegend=False,
                    marker=dict(size=9, color=INK, line=dict(color="white", width=2)),
                    hovertemplate=f"this schedule<br>{TARGET_LABEL[mill][key][0]} {float(val):.1f} {unit}<extra></extra>",
                ), row=r, col=1)

    fig.update_xaxes(title=labels.get(x_key, x_key), row=len(units), col=1, **axes)
    fig.update_layout(height=280 * len(units) + 90, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="white", legend=dict(orientation="h", y=1.12, x=0))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
    if mill == "Plate" and x_key == "ACC_Laminar_Header" \
            and any(k in HEADER_SWEEP_SIGN for k in y_keys):
        st.caption("monotone physics guardrail applied to this header sweep (residual tree noise "
                   "removed); single-schedule predictions remain the raw model.")

    show_row = {labels.get(x_key, x_key): round(op_x, 3)}
    for key in y_keys:
        val = op.get(key, {}).get("pred")
        show_row[TARGET_LABEL[mill][key][0]] = (round(float(val), 2) if val is not None and np.isfinite(val) else None)
    st.dataframe(pd.DataFrame([show_row]), hide_index=True, use_container_width=True)

    note = (f"Slice through the {mill} pipeline at the sidebar schedule; "
            f"{labels.get(x_key, x_key)} swept over {lo:.0f}-{hi:.0f}"
            + (", bounded by the house limits" if x_key in limits else "")
            + (f". Shaded band is the 95% split-conformal interval for {TARGET_LABEL[mill][first_key][0]}." if hw else "."))
    if observed:
        note += (" Open diamonds are population means from the cleaned training table, adjusted for "
                 "chemistry, gauge and cooling (partial residual) and narrowed to the entered "
                 "gauge band, not this exact schedule; bars are ±1 standard error. Where that "
                 "scatter is wider than the step you are reading, the step is not distinguishable.")
    st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)


def template_bytes(mill):
    cols = [row[1] for row in SPECS[mill]]
    blank = pd.DataFrame([{c: row[6] for c, row in zip(cols, SPECS[mill])}])
    buf = io.BytesIO()
    blank.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def capability_resample(mill, raw, target, lsl, usl, samples, bands,
                        engine="Champion GBDT (Recommended)"):
    """Monte-Carlo capability study over the drifting band, as one plain result dict.

    Resamples each named input uniformly inside ± band, scores every sampled schedule
    through the mill's feature module, and returns the capability summary. It is pure
    apart from score_frame (no Streamlit calls), so the Capability tab and the Modal
    function in modal_app.py run this same body: the reported Cp/Cpk cannot drift between
    the local and the remote path. The RNG is seeded, so the sampled schedule is identical
    wherever the study runs.

    Keys: error (str or None), points (the sampled inputs that scored, one column per
    band), scored (the scored values), mean, sd, cp, cpk, oos (out-of-spec percent),
    sigma_model, cpk_total, target, unit, lsl, usl, bands, n_points.
    """
    label, unit = TARGET_LABEL[mill].get(target, (target, ""))
    bands = {str(key): float(band) for key, band in (bands or {}).items()}
    samples = max(int(samples), 1)
    lsl, usl = float(lsl), float(usl)
    blank = {"error": None, "points": pd.DataFrame(), "scored": np.array([], dtype=float),
             "mean": np.nan, "sd": np.nan, "cp": np.nan, "cpk": np.nan, "oos": np.nan,
             "sigma_model": np.nan, "cpk_total": np.nan, "target": target, "unit": unit,
             "lsl": lsl, "usl": usl, "bands": dict(bands), "n_points": 0}

    rng = np.random.default_rng(42)
    frame = pd.DataFrame([dict(raw) for _ in range(samples)])
    band_items = list(bands.items())
    if len(band_items) >= 2:
        # Gaussian copula so co-varying mill temperatures (rho=0.55) and process/chemistry
        # variables (rho=0.30) drift together while preserving exact uniform [-band, +band] marginals.
        thermal_keys = {
            "FM_Temp_Before_Rolling", "FM_Finish_Temp", "ACC_Temp_Post_Laminar",
            "Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_CT_Bot_C",
        } | {row[1] for row in SPECS.get(mill, ()) if row[3] == "C"}
        keys = [k for k, _ in band_items]
        d = len(keys)
        corr = np.eye(d, dtype=float)
        for i in range(d):
            for j in range(i + 1, d):
                rho = 0.55 if (keys[i] in thermal_keys and keys[j] in thermal_keys) else 0.30
                corr[i, j] = rho
                corr[j, i] = rho
        chol = np.linalg.cholesky(corr)
        z = rng.standard_normal((samples, d)) @ chol.T
        erf_vec = np.frompyfunc(math.erf, 1, 1)
        u = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)).astype(float))
        u = np.clip(u, 0.0, 1.0)
        for idx, (key, band) in enumerate(band_items):
            centre = float(raw[key])
            frame[key] = centre + (2.0 * u[:, idx] - 1.0) * band
    else:
        for key, band in band_items:
            centre = float(raw[key])
            frame[key] = rng.uniform(centre - band, centre + band, samples)
    scored_frame = score_frame(mill, frame, engine=engine)
    if target not in scored_frame.columns:
        return dict(blank, error=f"{label} cannot be scored with the {engine} engine "
                                 "(no linear fit clears the R² floor for it). Switch engine "
                                 "or property.")
    scored_all = np.asarray(pd.to_numeric(scored_frame[target], errors="coerce"), dtype=float)
    keep = np.isfinite(scored_all)
    scored = scored_all[keep]
    if scored.size < 30:
        return dict(blank, error="Too few scoring points came back: widen the band or "
                                 "change property.")
    points = frame.loc[keep, [key for key in bands if key in frame.columns]].reset_index(drop=True)

    mean, sd = float(np.mean(scored)), float(np.std(scored, ddof=1))
    cp = (usl - lsl) / (6 * sd) if sd > 0 else float("inf")
    cpk = min(usl - mean, mean - lsl) / (3 * sd) if sd > 0 else float("inf")
    oos = float(np.mean((scored < lsl) | (scored > usl)) * 100)
    # the sampled band only carries process drift; the model's own residual is a separate
    # source of spread and belongs in a certification decision
    sigma_model = get_target_interval_half(mill, target) / 1.96
    sigma_total = float(np.sqrt(sd ** 2 + sigma_model ** 2))
    cpk_total = min(usl - mean, mean - lsl) / (3 * sigma_total) if sigma_total > 0 else float("inf")

    return {"error": None, "points": points, "scored": scored, "mean": mean, "sd": sd,
            "cp": float(cp), "cpk": float(cpk), "oos": oos, "sigma_model": float(sigma_model),
            "cpk_total": float(cpk_total), "target": target, "unit": unit,
            "lsl": lsl, "usl": usl, "bands": dict(bands), "n_points": int(scored.size)}


def _render_capability_result(mill, res, source):
    """Render one capability outcome: the source caption, metrics, histogram and band fits.

    Renders from the stored result alone and never recomputes, so the button path and a
    plain rerun that replays st.session_state[f"cp_result_{mill}"] produce identical
    output. Limits and bands come from the result rather than the live widgets, because
    those are the values the numbers were computed for.
    """
    if res.get("error"):
        st.error(res["error"])
        return
    target = res["target"]
    unit = res.get("unit") or TARGET_LABEL[mill][target][1]
    lsl, usl, bands = res["lsl"], res["usl"], res["bands"]
    scored = np.asarray(res["scored"], dtype=float)
    points = res["points"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Cp", f"{res['cp']:.2f}")
    m2.metric("Cpk", f"{res['cpk']:.2f}")
    m3.metric("Cpk incl. model σ", f"{res['cpk_total']:.2f}")
    m4.metric("mean", f"{res['mean']:.1f} {unit}")
    m5.metric("out of spec", f"{res['oos']:.1f} %")
    st.caption(remote_compute.source_label(source))
    st.caption(f"process σ {res['sd']:.2f} {unit} · model σ {res['sigma_model']:.2f} {unit} · "
               f"{res['n_points']} scored points · band "
               + ", ".join(f"{spec_map(mill)[k][2]} ±{bands[k]:g}" for k in bands))
    st.caption("Cp and Cpk describe the sampled process band only. Cpk incl. model σ adds the "
               "model's residual spread: use that one before promising a certification limit.")

    fig = go.Figure(go.Histogram(x=scored, nbinsx=45, marker_color=ACCENT_RAMP[4],
                                 marker_line_color=ACCENT, marker_line_width=.6))
    for value, label in ((lsl, "LSL"), (usl, "USL")):
        fig.add_vline(x=value, line_color=WARN_LINE, line_width=1.4, line_dash="dash",
                      annotation_text=label, annotation_position="top", annotation_font_size=10)
    fig.add_vline(x=res["mean"], line_color="#111418", line_width=1)
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=20, b=10), plot_bgcolor="white",
                      paper_bgcolor="white", bargap=.04,
                      xaxis=dict(title=f"{TARGET_LABEL[mill][target][0]} [{unit}]", showline=True,
                                 linecolor=HAIR, gridcolor="#f4f5f7"),
                      yaxis=dict(title="count", showline=False, gridcolor="#f4f5f7"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("linear band per input")
    st.caption("Property against each drifting input, with a fitted line and a ±3σ band on the "
               "scatter. A steep slope here is the band width that costs you capability.")
    for key in bands:
        if key not in points.columns:
            continue
        x = pd.to_numeric(points[key], errors="coerce").values
        ok = np.isfinite(x)
        if ok.sum() < 2:
            continue
        slope, intercept = np.polyfit(x[ok], scored[ok], 1)
        resid_sd = float(np.std(scored[ok] - (slope * x[ok] + intercept)))
        grid = np.linspace(x[ok].min(), x[ok].max(), 60)
        band_fig = go.Figure()
        band_fig.add_trace(go.Scatter(x=x[ok], y=scored[ok], mode="markers",
                                      marker=dict(size=4, color=ACCENT_RAMP[3], opacity=.45),
                                      name="sampled"))
        band_fig.add_trace(go.Scatter(x=grid, y=slope * grid + intercept, mode="lines",
                                      line=dict(color=ACCENT, width=1.6), name="fit"))
        band_fig.add_trace(go.Scatter(x=np.concatenate([grid, grid[::-1]]),
                                      y=np.concatenate([slope * grid + intercept + 3 * resid_sd,
                                                        (slope * grid + intercept - 3 * resid_sd)[::-1]]),
                                      fill="toself", fillcolor=_rgba(ACCENT, .10),
                                      line=dict(width=0), name="±3σ"))
        for value, label in ((lsl, "LSL"), (usl, "USL")):
            band_fig.add_hline(y=value, line_color=WARN_LINE, line_width=1, line_dash="dash")
        band_fig.update_layout(height=250, margin=dict(l=10, r=10, t=26, b=10), plot_bgcolor="white",
                               paper_bgcolor="white", showlegend=False,
                               title=dict(text=f"{spec_map(mill)[key][2]}: {slope:+.2f} "
                                               f"{unit} per {spec_map(mill)[key][3]}",
                                          font=dict(size=11), x=0),
                               xaxis=dict(showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                               yaxis=dict(title=unit, showline=False, gridcolor="#f4f5f7"))
        st.plotly_chart(band_fig, use_container_width=True, config={"displayModeBar": False})


def view_capability(mill, raw, engine="Champion GBDT (Recommended)"):
    """Cp/Cpk over a band the user defines, resampled through the mill's feature module.

    Nothing is resampled while the operator is setting the limits: the "Calculate
    capability" button is the only gate, and its outcome is kept in
    st.session_state[f"cp_result_{mill}"] so later reruns replay the study instead of
    spending Modal credits (or local CPU) on it again. The compute body lives in
    capability_resample, which is what the Modal function calls too.
    """
    section("process capability")
    st.markdown('<div class="note">Set the specification limits, then say how far two or three '
                'inputs may drift. The app resamples that band, scores every point and reports '
                'capability: Cp and Cpk on the predicted distribution, not on a single schedule.'
                '</div>', unsafe_allow_html=True)

    keys = list(TARGET_LABEL[mill])
    c1, c2 = st.columns([1.4, 1.6], vertical_alignment="bottom")
    target = c1.selectbox("Property", keys, format_func=lambda k: TARGET_LABEL[mill][k][0],
                          key=f"cp_t_{mill}")
    unit = TARGET_LABEL[mill][target][1]
    tolerance = ml.TARGET_CONFIG[mill][target]["tol"]
    base, base_err = single_target(mill, raw, engine, target)
    if not np.isfinite(base):
        st.error(f"Could not score {TARGET_LABEL[mill][target][0]} with {engine}: {base_err}")
        return

    c3, c4, c5 = st.columns([1, 1, 1], vertical_alignment="bottom")
    lsl = c3.number_input(f"LSL [{unit}]", value=float(round(base - tolerance, 1)), step=1.0,
                          key=f"cp_l_{mill}")
    usl = c4.number_input(f"USL [{unit}]", value=float(round(base + tolerance, 1)), step=1.0,
                          key=f"cp_u_{mill}")
    samples = c5.slider("Samples", 200, 3000, 800, step=100, key=f"cp_n_{mill}")

    section("inputs allowed to drift")
    order = [k for k in SURFACE[mill] if k in spec_map(mill)]
    c6, c7 = st.columns([2, 3], vertical_alignment="bottom")
    chosen = c6.multiselect("Variables", order, default=order[:2], max_selections=3,
                            format_func=lambda k: spec_map(mill)[k][2], key=f"cp_v_{mill}")
    if not chosen:
        st.caption("Pick at least one variable.")
        return
    bands = {}
    cols = st.columns(len(chosen) or 1)
    for col, key in zip(cols, chosen):
        row = spec_map(mill)[key]
        span = (row[5] - row[4]) / 12.0
        band_unit = "°C" if row[3] == "C" else row[3]
        bands[key] = col.number_input(f"± band, {row[2]} [{band_unit}]", min_value=0.0,
                                      value=float(round(span, 3)), step=float(row[7]),
                                      key=f"cp_b_{mill}_{key}")

    result_key = f"cp_result_{mill}"
    clicked = st.button("Calculate capability", type="primary", key=f"cp_go_{mill}")
    if clicked:
        where = " on Modal" if remote_compute.remote_ready() else ""
        with st.spinner(f"Running capability study{where}..."):
            try:
                res, source = remote_compute.capability(
                    mill, raw, target, lsl, usl, samples, bands, engine=engine)
            except Exception as exc:  # noqa: BLE001 - a failed study must not kill the tab
                st.error(f"Capability study failed: {type(exc).__name__}: {exc}")
                return
        st.session_state[result_key] = {"res": res, "source": source}
    else:
        stored = st.session_state.get(result_key)
        if not stored:
            st.caption(f"Reference schedule predicts {base:.1f} {unit}; limits default to "
                       f"±{tolerance:g}. Press Calculate capability to run the study.")
            return
        st.caption("Shown from your last calculation. Press Calculate capability to run again "
                   "with the current inputs.")
        res, source = stored["res"], stored["source"]

    _render_capability_result(mill, res, source)


def _upload_key(uploaded):
    """Stable identity for one uploaded file: its name, size and (when present) file_id.

    Used to decide whether a stored batch result still belongs to the sheet in the
    uploader: the same file keeps its table, any other file clears it.
    """
    parts = [str(getattr(uploaded, "name", "") or ""), str(getattr(uploaded, "size", "") or "")]
    file_id = getattr(uploaded, "file_id", None)
    if file_id:
        parts.append(str(file_id))
    return "|".join(parts)


def _batch_frame_notes(mill, frame):
    """Input-level notices for an uploaded sheet: rows outside the envelope, rows with no chemistry.

    Returned as (level, text) pairs so a rerun can replay the same warnings from session
    state without re-reading the sheet. Neither notice depends on the scores.
    """
    notes = []
    domain = explainer.get_applicability_domain().get(mill, {})
    if domain:
        outside = pd.Series(0, index=frame.index)
        for param, bounds in domain.items():
            if param not in frame.columns:
                continue
            p01, p99 = bounds.get("p01"), bounds.get("p99")
            if p01 is None or p99 is None:
                continue
            values = pd.to_numeric(frame[param], errors="coerce")
            outside = outside + ((values < p01) | (values > p99)).fillna(False).astype(int)
        ood_rows = int((outside > 0).sum())
        if ood_rows:
            notes.append(("warning", f"{ood_rows:,} of {len(frame):,} rows sit outside the "
                                     f"historical P01–P99 envelope on at least one input: those "
                                     f"predictions are extrapolation and should be read as such."))

    chem_cols = [c for c in frame.columns if c.startswith("Chem_")]
    if chem_cols:
        blank_chem = int(frame[chem_cols].isna().all(axis=1).sum())
        if blank_chem:
            notes.append(("warning", f"{blank_chem:,} row(s) carry no chemistry at all: every "
                                     f"element was imputed, so those predictions describe a "
                                     f"hypothetical heat, not one you measured. Fill the "
                                     f"chemistry columns before certifying anything."))
    return notes


def _render_batch_notes(notes):
    for level, text in notes or []:
        if level == "warning":
            st.warning(text)
        else:
            st.info(text)


def _render_batch_result(mill, stored, replay):
    """Render a stored batch outcome: source caption, warnings, preview and download."""
    st.caption(f"{stored['n_rows']:,} rows · Scoring engine: {stored.get('engine', '')}")
    st.caption(remote_compute.source_label(stored.get("source", "local")))
    if replay:
        st.caption("Shown from this upload's last calculation. Press Calculate scores to "
                   "re-score it.")
    _render_batch_notes(stored.get("notes"))
    table = stored["table"]
    st.dataframe(table.head(200), use_container_width=True, height=320)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="scored")
    st.download_button("Download scored sheet", buf.getvalue(),
                       file_name=f"{mill.lower()}_scored.xlsx", type="primary",
                       key=f"batch_dl_{mill}")


def view_batch(mill, engine="Champion GBDT (Recommended)"):
    """Score an uploaded sheet, and only when the operator asks for it.

    The upload is validated as soon as it arrives, but nothing is scored until the
    "Calculate scores" button is pressed: batch scoring is the heavy calculation on this
    tab and, with Modal configured, a paid one. The scored table and its source are kept
    in st.session_state[f"batch_result_{mill}"] keyed by the upload, so a rerun replays the
    table instead of re-scoring it, and a different upload clears it.
    """
    section("upload a schedule")
    st.markdown('<div class="note">Download the template, fill one row per plate or coil, upload it '
                'back. Every row is scored through the same feature module the simulator uses.</div>',
                unsafe_allow_html=True)
    a, b = st.columns([1, 3], vertical_alignment="bottom")
    a.download_button("Download template (.xlsx)", template_bytes(mill),
                      file_name=f"{mill.lower()}_template.xlsx", use_container_width=True)
    uploaded = b.file_uploader("Filled sheet", type=["xlsx", "csv"], key=f"up_{mill}",
                               label_visibility="collapsed")
    state_key = f"batch_result_{mill}"
    if uploaded is None:
        st.session_state.pop(state_key, None)
        return

    upload_key = _upload_key(uploaded)
    stored = st.session_state.get(state_key)
    if stored is not None and stored.get("upload_key") != upload_key:
        st.session_state.pop(state_key, None)
        stored = None

    clicked = st.button("Calculate scores", type="primary", key=f"batch_go_{mill}")
    if not clicked and stored is not None:
        _render_batch_result(mill, stored, replay=True)
        return

    frame = (pd.read_excel(uploaded) if uploaded.name.lower().endswith("xlsx")
             else pd.read_csv(uploaded))
    st.caption(f"{len(frame):,} rows read · Scoring engine: {engine}")
    missing_inputs = [row[1] for row in SPECS[mill] if row[1] not in frame.columns]
    if missing_inputs:
        st.error("These input columns are missing from the sheet: "
                 + ", ".join(missing_inputs)
                 + ". Download the template for the exact headers: scoring would otherwise "
                   "substitute defaults without telling you.")
        return
    if len(frame) > 20000:
        st.error(f"{len(frame):,} rows is above the 20,000-row limit; split the sheet and "
                 "upload twice so the browser does not run out of memory.")
        return
    notes = _batch_frame_notes(mill, frame)

    if not clicked:
        _render_batch_notes(notes)
        st.caption("Nothing is scored until you press Calculate scores.")
        return

    try:
        where = " on Modal" if remote_compute.remote_ready() else ""
        with st.spinner(f"Scoring {len(frame):,} rows{where}..."):
            scored, source = remote_compute.score_batch(mill, frame, engine=engine)
    except Exception as exc:
        st.error(f"Scoring failed: {type(exc).__name__}: {exc}")
        st.markdown('<div class="note">To recover, check in order: <b>1.</b> every column header '
                    'matches the downloaded template exactly (headers, units and spelling — '
                    'scoring substitutes defaults silently for anything it cannot find); '
                    '<b>2.</b> every numeric cell holds a plain number, with no thousand '
                    'separators, units or text such as "hot" left in; <b>3.</b> the sheet stays '
                    'under the 20,000-row limit and the gauges and temperatures sit inside the '
                    'mill\'s permitted bands. Fix the offending rows and upload again.</div>',
                    unsafe_allow_html=True)
        return
    table = pd.concat([frame[[c for c in frame.columns if c in spec_map(mill)]].reset_index(drop=True),
                       scored.add_prefix("pred_").reset_index(drop=True)], axis=1)
    for key in scored.columns:
        half = get_target_interval_half(mill, key)
        table[f"pred_{key}_lo"] = (table[f"pred_{key}"] - half).round(2)
        table[f"pred_{key}_hi"] = (table[f"pred_{key}"] + half).round(2)

    stored = {"upload_key": upload_key, "table": table, "notes": notes,
              "n_rows": int(len(frame)), "source": source, "engine": engine}
    st.session_state[state_key] = stored
    _render_batch_result(mill, stored, replay=False)


def retrain_template_bytes(mill):
    """Headers the retrain tab expects: an id, the mill's raw inputs, and one measured column."""
    cols = ["Heat_ID"] + [row[1] for row in SPECS[mill]] + ["Measured"]
    row = {"Heat_ID": "HEAT-0001", "Measured": ""}
    row.update({row_[1]: row_[6] for row_ in SPECS[mill]})
    buf = io.BytesIO()
    pd.DataFrame([row], columns=cols).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def read_log():
    try:
        with open(LOG_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return []


def append_log(entry):
    rows = read_log()
    rows.append(entry)
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    except Exception:
        pass          # a read-only container still keeps the session copy
    st.session_state["train_log"] = rows
    return rows


def log_table(rows, key):
    if not rows:
        st.caption("No refits logged yet on this instance.")
        return
    frame = pd.DataFrame(rows[-12:])[["when", "mill", "target", "model", "rows", "melts",
                                      "r2", "rmse", "split"]]
    st.dataframe(frame.iloc[::-1], use_container_width=True, hide_index=True)
    st.download_button("Download training log (CSV)", frame.to_csv(index=False).encode(),
                       file_name="training_log.csv", key=f"logdl_{key}")
    st.caption("The log file lives beside the app, which is read-only on Streamlit Cloud: "
               "there the history is session-only. Download the CSV to keep it.")


def view_retrain(mill):
    section("refit from your own sheet")
    st.markdown('<div class="note">The shipped models are baked artefacts, not incrementally trainable: '
                'a stacking ensemble cannot be fine-tuned by feeding it more rows. What this does '
                'instead is honest: it refits a fresh model for one property from your sheet, using '
                'the same feature module and a melt-grouped holdout.</div>', unsafe_allow_html=True)

    tl, tr = st.columns([1, 3], vertical_alignment="bottom")
    tl.download_button("Download retrain template (.xlsx)", retrain_template_bytes(mill),
                       file_name=f"{mill.lower()}_retrain_template.xlsx", use_container_width=True,
                       key=f"rttpl_{mill}")
    up = tr.file_uploader("Sheet with a measured property column", type=["xlsx", "csv"],
                          key=f"rt_{mill}", label_visibility="collapsed")
    if up is None:
        section("refit log")
        log_table(st.session_state.get("train_log") or read_log(), mill)
        return
    frame = pd.read_excel(up) if up.name.lower().endswith("xlsx") else pd.read_csv(up)
    missing_inputs = [row[1] for row in SPECS[mill] if row[1] not in frame.columns]
    if missing_inputs:
        st.error("These input columns are missing from the sheet: "
                 + ", ".join(missing_inputs)
                 + ". Download the template for the exact headers, or the refit would train "
                   "on substituted defaults.")
        return
    if len(frame) > 50000:
        st.error(f"{len(frame):,} rows is above the 50,000-row limit for an in-browser refit; "
                 "train on Modal instead.")
        return
    candidates = [c for c in frame.columns if c not in spec_map(mill)]
    c1, c2, c3 = st.columns(3, vertical_alignment="bottom")
    target = c1.selectbox("Measured column", candidates, key=f"rt_col_{mill}")
    family = c2.selectbox("Model", ["Ridge", "Random forest", "XGBoost"], key=f"rt_fam_{mill}")
    test_share = c3.slider("Holdout share", .1, .4, .2, step=.05, key=f"rt_ts_{mill}")

    if not st.button("Refit", type="primary", key=f"rt_go_{mill}"):
        st.caption(f"{len(frame):,} rows · target `{target}` · {len(spec_map(mill))} input columns expected")
        return

    y = pd.to_numeric(frame[target], errors="coerce")
    # production logs carry decimal shifts; a single 41,000 MPa entry would wreck the R2
    lowered = target.lower()
    if "pct" in lowered or lowered.startswith("el"):
        lo, hi = 0.0, 70.0
    elif "impact" in lowered or "charpy" in lowered:
        lo, hi = 0.0, 500.0
    else:
        lo, hi = 100.0, 1500.0
    plausible = y.between(lo, hi)
    dropped = int((y.notna() & ~plausible).sum())
    ok = y.notna() & plausible
    if dropped:
        st.caption(f"{dropped} measured values fell outside {lo:g}–{hi:g} and were dropped.")
    if ok.sum() < 120:
        st.error(f"Only {int(ok.sum())} usable measured rows: too few to hold anything out.")
        return
    X = feature_matrix(mill, prepare_frame(mill, frame.loc[ok])).fillna(0.0)
    y = y[ok]
    groups = frame.loc[ok, "Heat_ID"] if "Heat_ID" in frame else pd.Series(np.arange(ok.sum()))

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    if groups.nunique() > 50:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_share, random_state=42)
                      .split(X, y, groups))
        split_note = f"grouped by Heat_ID · {groups.nunique():,} melts"
    else:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_share, random_state=42)
        split_note = "random row split (no usable melt id)"

    model = {"Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))),
             "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                    n_jobs=-1, random_state=42),
             "XGBoost": XGBRegressor(n_estimators=500, max_depth=5, learning_rate=.04,
                                     subsample=.85, colsample_bytree=.85, tree_method="hist",
                                     n_jobs=-1, random_state=42)}[family]
    with st.spinner("Fitting model..."):
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        r2 = 1 - ((y.iloc[te] - pred) ** 2).sum() / ((y.iloc[te] - y.iloc[te].mean()) ** 2).sum()
        rmse = float(np.sqrt(((y.iloc[te] - pred) ** 2).mean()))

    m1, m2, m3 = st.columns(3)
    m1.metric("holdout R²", f"{r2:.3f}")
    m2.metric("holdout RMSE", f"{rmse:.2f}")
    m3.metric("rows", f"{len(y):,}", delta=split_note, delta_color="off")
    lb = ml.get_leaderboard_data(mill)
    first_key = list(TARGET_LABEL[mill])[0]
    ref_r2 = None
    if not lb.empty and "Target" in lb.columns:
        ref_row = lb[lb["Target"] == first_key]
        if not ref_row.empty and "Final_Val_R2" in ref_row.columns and pd.notna(ref_row["Final_Val_R2"].values[0]):
            ref_r2 = float(ref_row["Final_Val_R2"].values[0])
    if ref_r2 is not None:
        st.markdown(f'<div class="note">Compared against the shipped model for this route: '
                    f'R² {ref_r2:.3f} '
                    f'on the campaign holdout. Different data, so read the pair as a sanity check, '
                    f'not a contest.</div>', unsafe_allow_html=True)

    import joblib
    buf = io.BytesIO()
    joblib.dump({"model": model, "feature_names": list(X.columns), "target": target,
                 "mill": mill, "holdout_r2": float(r2)}, buf)
    append_log({"when": datetime.now().strftime("%Y-%m-%d %H:%M"), "mill": mill,
                "target": target, "model": family, "rows": int(len(y)),
                "melts": int(groups.nunique()), "r2": round(float(r2), 3),
                "rmse": round(float(rmse), 2), "split": split_note})

    st.download_button("Download fitted model (.joblib)", buf.getvalue(),
                       file_name=f"{mill.lower()}_{target}_refit.joblib", type="primary")

    section("refit log")
    log_table(read_log(), mill)


def view_about():
    section("what this is")
    st.markdown("""
A property model for both rolling routes: heavy plate off the roughing and finishing mills with
laminar accelerated cooling, and Steckel mill coil off the reversing mill and runout table. It
answers one operational question, what will this heat certify, from chemistry, drafting and
cooling alone.

Predictions come with the interval their split-conformal calibration implies, not a bare number,
and the scores behind them use `GroupShuffleSplit` on the melt so sister plates from one heat never
straddle the train/test boundary. That is stricter than a row split and it is why the numbers here
are lower than a naive fit would show.
""")

    section("how it is built")
    st.markdown("""
- **Features**: Pickering solid-solution strengthening, Hall–Petch `1/sqrt(t)`, true strain
  `ln(t_slab / t_plate)`, carbon equivalents, `Ar3`, and the finishing-temperature margin above it.
  No raw log dump: every input has a metallurgical definition.
- **Models**: nine families fitted per target (OLS, metallurgical NLR, Ridge, XGBoost, LightGBM,
  CatBoost, stacking and bagged ensembles), selected by melt-held-out leaderboard rather than
  assumption. One caveat to keep visible: the NLR constitutive fit carries five terms (gauge,
  solid solution, CE, header cooling, total strain) and **no finishing-temperature term**, so
  read it for chemistry and gauge effects, not for how the exit temperature moves properties.
- **Uncertainty**: 95 % intervals from split-conformal calibration on held-out melts, checked
  against melt-grouped bootstrap resampling; both reports ship beside every score.
- **Stack**: Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, Plotly, Streamlit.
""")

    section("status")
    st.markdown("""
**Personal project.** Developed independently as an exploratory modelling study, this project received no
external funding, client backing, or commercial licensing. It is not an official PT Gunung Raja Paksi Tbk
product and has not been reviewed, approved, or endorsed by the company. The underlying dataset is derived
from production logs across the specified process routes; the ranges shown reflect actual historical
rolling parameters, and any values beyond these boundaries represent extrapolation and should be disregarded.
""")

    c1, c2 = st.columns(2)
    c1.markdown("**Code and notebooks**  \n[github.com/daniswaramp/hot-rolling-mechanical-models]"
                "(https://github.com/daniswaramp/hot-rolling-mechanical-models): models, reports and "
                "both executed notebooks. The campaign sheets themselves are not published.")
    c2.markdown("**Author**  \nSulaeman Daniswara Mulya Putera, process and metallurgical engineer, "
                "steel hot rolling. [LinkedIn](https://www.linkedin.com/in/daniswaramp)")


@st.cache_data
def load_val_predictions(mill):
    """Loads held-out actual vs predicted records for parity chart audits."""
    fname = "validation_predictions.csv" if mill == "Plate" else "hrc_validation_predictions.csv"
    p = os.path.join(os.path.dirname(APP_DIR), mill, "reports", fname)
    if os.path.exists(p):
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def view_diagnostics(mill):
    section("model governance & audit")

    # Governance numbers come from the generated manifest, never from constants in
    # this file: models/data_audit.json separates what the artifacts were trained on
    # from what the cleaned tables contain today.
    audit_path = os.path.join(os.path.dirname(APP_DIR), "models", "data_audit.json")
    try:
        with open(audit_path, "r", encoding="utf-8") as fh:
            audit = json.load(fh).get(mill, {})
    except (OSError, ValueError):
        audit = {}
    trained = audit.get("as_trained") or {}
    current = audit.get("current") or {}
    targets_trained = trained.get("targets") or {}
    targets_current = current.get("targets") or {}

    if not targets_trained:
        st.warning("models/data_audit.json is missing for this route: run "
                   "`python app\\build_data_audit.py` and redeploy. The split audit below "
                   "cannot be shown without it, and no numbers are invented in its place.")

    # 1. High-Level Campaign Audit Cards
    total_records = current.get("ingested_rows")
    usable_rows = current.get("usable_rows")
    total_heats = trained.get("heats_total")
    current_heats = current.get("unique_heats")
    train_heats = trained.get("heats_train")
    holdout_heats = trained.get("heats_holdout")

    def _fmt(value):
        return f"{value:,}" if isinstance(value, (int, float)) else "—"

    st.markdown(f"""
<div class="facts" style="margin: 0.8rem 0 1.2rem;">
  <div><span>Records Ingested (as trained)</span><b>{_fmt(total_records)}</b><em>{mill} products</em></div>
  <div><span>Usable Rows (cleaned today)</span><b>{_fmt(usable_rows)}</b><em>after sanitising</em></div>
  <div><span>Heats: training campaign</span><b>{_fmt(total_heats)}</b><em>{_fmt(train_heats)} train · {_fmt(holdout_heats)} holdout</em></div>
  <div><span>Distinct Melts (cleaned today)</span><b>{_fmt(current_heats)}</b><em>current cleaned tables</em></div>
</div>""", unsafe_allow_html=True)

    # 2. Dataset & Training Split Audit Table
    st.markdown("#### 1. Training split")
    st.caption("Every melt heat (ladle cast) is assigned exclusively to Train or Holdout. "
               "The first table is the split the shipped artifacts were fitted with, recorded "
               "at training time; the second is what the cleaned tables contain today. The two "
               "differ because the sanitising pass repaired cells and removed duplicate rows "
               "after the campaign.")

    if targets_trained:
        audit_rows = []
        for key, split in targets_trained.items():
            label, unit = TARGET_LABEL[mill].get(key, (key, ""))
            score = split.get("val_r2")
            audit_rows.append(f"""<tr>
              <td class="name">{label} <span class="unit">[{unit}]</span></td>
              <td class="val">{split.get('train_rows', 0):,}</td>
              <td class="val" style="color:{ACCENT};">{split.get('holdout_rows', 0):,}</td>
              <td class="sub" style="font-family:monospace; text-align:right;">{split.get('holdout_heats', 0):,}</td>
              <td class="sub" style="font-family:monospace; text-align:right;">{split.get('report_samples', 0):,.0f}</td>
              <td class="sub" style="text-align:left;">{'holdout R² ' + format(score, '.3f') if score is not None else '—'}</td>
            </tr>""")
        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>Target Property</th><th>Trained Rows</th><th>Holdout Rows</th>'
            '<th>Holdout Melts</th><th>Report Samples</th><th>As-Trained Score</th></tr></thead><tbody>'
            + "".join(audit_rows) +
            f'</tbody></table><div class="foot"><b>Melt-grouped holdout check</b>: sister plates or coils from one melt stay exclusively in Train or Holdout. '
            f'<code>{audit.get("split_method", "GroupShuffleSplit on Heat_ID")}</code>, plus 100 heat-level bootstrap resamples. '
            f'Recorded at training time; report samples on this route = {audit.get("samples_semantics", "see campaign report")}.</div></div>',
            unsafe_allow_html=True,
        )

    if current:
        duplicates = current.get("duplicate_rows_removed", 0)
        st.markdown("##### Cleaned tables: current state")
        st.markdown(f"""
<div class="facts">
  <div><span>Rows removed as duplicates</span><b>{duplicates:,}</b><em>exact records</em></div>
  <div><span>Cells rescaled</span><b>{current.get('cells_rescaled', 0):,}</b><em>decimal-shift repair</em></div>
  <div><span>Cells blanked</span><b>{current.get('cells_blanked', 0):,}</b><em>beyond physical bounds</em></div>
  <div><span>Sanitising audit</span><b>{'on file' if current.get('sanitise_log_found') else 'not shipped'}</b><em>reports/sanitise_log.csv</em></div>
</div>""", unsafe_allow_html=True)
        current_rows = []
        for key, stats in targets_current.items():
            label, unit = TARGET_LABEL[mill].get(key, (key, ""))
            current_rows.append(f"""<tr>
              <td class="name">{label} <span class="unit">[{unit}]</span></td>
              <td class="val">{stats.get('usable_rows', 0):,}</td>
              <td class="sub" style="font-family:monospace; text-align:right;">{stats.get('unique_heats', 0):,}</td>
            </tr>""")
        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>Target Property</th><th>Usable Rows (today)</th><th>Distinct Melts</th></tr></thead><tbody>'
            + "".join(current_rows) +
            '</tbody></table><div class="foot">Refreshed by <code>app\\build_data_audit.py</code> from the cleaned '
            'tables and the sanitising audit trail; ship the regenerated JSON with the deployment.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="break"></div>', unsafe_allow_html=True)

    # 3. Holdout accuracy by model family
    st.markdown("#### 2. Holdout accuracy by model family")
    st.caption("Side-by-side holdout scores for every model tier fitted on this route, evaluated "
               "on whole held-out melts. Champion GBDT is the default scoring engine.")

    lb = ml.get_leaderboard_data(mill)
    ci = ml.get_ci_data(mill)
    plan = champion_plan(mill)
    lfam = linear_families(mill)

    if not lb.empty:
        merged = pd.merge(lb, ci[["Target", "R2_95_CI_Low", "R2_95_CI_High"]], on="Target", how="left")
        hit_col = "Val_HitRate_20MPa" if "Val_HitRate_20MPa" in merged.columns else "Val_HitRate_Tol"

        rows = []
        for _, r in merged.iterrows():
            target_key = r["Target"]
            tname = r.get("Target_Name", target_key)
            unit = r.get("Unit", "")

            # Champion GBDT info
            champ_cfg = plan.get(target_key) or {}
            fam_key = champ_cfg.get("family")
            fam_name = {"cat": "CatBoost", "xgb": "XGBoost", "lgbm": "LightGBM",
                        "ensemble": "Stacking"}.get(fam_key, fam_key or "not shipped")
            champ_r2 = champ_cfg.get("r2")
            champ_badge_cls = "ok" if fam_key in ("cat", "xgb", "lgbm") else "warn"
            champ_cell = (f'<span class="pill {champ_badge_cls}">{fam_name}</span> '
                          f'<b style="font-family:monospace; font-size:13px; color:{ACCENT};">{champ_r2:.3f}</b>'
                          if champ_r2 is not None else '<span class="muted">—</span>')

            # Ensemble info
            ens_model = str(r.get("Final_Selected_Model", "Ensemble")).replace("_", " ")
            ens_r2 = float(r["Final_Val_R2"])
            ens_cell = f'<span class="tag">{ens_model}</span> <span style="font-family:monospace; font-weight:600;">{ens_r2:.3f}</span>'

            # Linear baseline info (OLS)
            ols_spec = lfam.get(target_key, {}).get("ols", {})
            ols_r2 = ols_spec.get("r2")
            ols_cell = f"{ols_r2:.3f}" if ols_r2 is not None and ols_r2 >= 0.15 else f'<span style="color:{MUTED};">&lt;0.15</span>'

            # Uncertainty & tolerance
            r2_ci = f"{r['R2_95_CI_Low']:.3f} – {r['R2_95_CI_High']:.3f}" if pd.notna(r.get("R2_95_CI_Low")) else "—"
            rmse = f"{r['Final_Val_RMSE']:.2f} {unit}"
            hit = f"{r[hit_col]:.1f} %" if hit_col in r and pd.notna(r[hit_col]) else "—"
            samples = f"{int(r['Samples']):,}" if "Samples" in r and pd.notna(r["Samples"]) else "—"

            rows.append(f"""<tr>
              <td class="name">{tname}</td>
              <td style="text-align:right;">{champ_cell}</td>
              <td style="text-align:right;">{ens_cell}</td>
              <td class="sub" style="font-family:monospace; text-align:right;">{ols_cell}</td>
              <td class="sub" style="font-family:monospace;">{rmse}</td>
              <td class="rng">{r2_ci}</td>
              <td class="val" style="color:{ACCENT};">{hit}</td>
              <td class="rng">{samples}</td>
            </tr>""")

        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>Target Property</th><th>Champion GBDT (Active Default)</th><th>Stacking Ensemble (.joblib)</th>'
            '<th>Linear Baseline (OLS)</th><th>Holdout RMSE</th><th>95% Bootstrap CI</th>'
            '<th>Within Tol (±20 MPa)</th><th>Test Samples</th></tr></thead><tbody>'
            + "".join(rows) +
            '</tbody></table><div class="foot"><b>Engine selector</b>: scoring engines (Champion GBDT, CatBoost, XGBoost, LightGBM, Stacking Ensemble, OLS/Linear) can be toggled in the sidebar dropdown.</div></div>',
            unsafe_allow_html=True,
        )

        # Artifact registry: shipped weights, hashes and the metrics they were fitted with
        registry_path = os.path.join(os.path.dirname(APP_DIR), "models", "registry.json")
        try:
            with open(registry_path, "r", encoding="utf-8") as fh:
                registry = json.load(fh)
        except (OSError, ValueError):
            registry = {}
        reg_models = [m for m in registry.get("models", []) if m.get("mill") == mill]
        if reg_models:
            reg_frame = pd.DataFrame([{
                "target": m.get("target"), "family": m.get("family"),
                "artifact": m.get("model_id"), "sha256": (m.get("sha256") or "")[:10],
                "val_r2": m.get("val_r2"), "val_rmse": m.get("val_rmse"),
                "conformal ±95": m.get("conformal_half_width_95"), "status": m.get("status"),
            } for m in reg_models]).sort_values(["target", "family"])
            with st.expander(f"Artifact registry: v{registry.get('registry_version')}, "
                             f"{registry.get('total_models')} artifacts shipped, "
                             f"updated {str(registry.get('updated_at'))[:10]}", expanded=False):
                st.caption("Every shipped artifact carries the SHA-256 recorded at build time. "
                           "A hash that no longer matches the weights means the metrics and the "
                           "model no longer belong together.")
                st.dataframe(reg_frame, hide_index=True, use_container_width=True)

    # 4. Interactive inspection subtabs
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. How the models are trained and served")
    st.caption("Model calibration, inverse setpoints, physics features, and what runs where.")

    val_df = load_val_predictions(mill)
    conf_df = ml.get_conformal_intervals(mill)

    diag_tabs = st.tabs([
        "Parity & residuals",
        "Conformal calibration",
        "Inverse optimization",
        "Metallurgy features",
        "System architecture",
        "Process window & drivers"
    ])
    
    with diag_tabs[0]:
        st.caption("Visual audit of predictions versus actual measured test values on held-out melt heats.")
        avail_targets = []
        if mill == "Plate":
            if "YS_BN_Actual" in val_df.columns: avail_targets.append(("YS_BN", "Yield Strength (YS Before Normalizing)", "YS_BN_Actual", "YS_BN_Pred_Best", "MPa", 20.0))
            if "TS_BN_Actual" in val_df.columns: avail_targets.append(("TS_BN", "Tensile Strength (TS Before Normalizing)", "TS_BN_Actual", "TS_BN_Pred_Best", "MPa", 20.0))
        else:
            if "y_true_YS_BN" in val_df.columns: avail_targets.append(("YS_BN", "Yield Strength (YS Before Normalizing)", "y_true_YS_BN", "y_pred_YS_BN", "MPa", 20.0))
            if "y_true_TS_BN" in val_df.columns: avail_targets.append(("TS_BN", "Tensile Strength (TS Before Normalizing)", "y_true_TS_BN", "y_pred_TS_BN", "MPa", 20.0))
            if "y_true_EL_BN" in val_df.columns: avail_targets.append(("EL_BN", "Elongation (EL Before Normalizing)", "y_true_EL_BN", "y_pred_EL_BN", "%", 2.5))
            
        if avail_targets and not val_df.empty:
            sel_target = st.selectbox("Select Target Property for Parity Plot", avail_targets, format_func=lambda x: x[1], key=f"sel_parity_{mill}")
            t_key, t_label, act_col, pred_col, t_unit, t_tol = sel_target
            
            y_act = pd.to_numeric(val_df[act_col], errors="coerce").dropna()
            y_pr = pd.to_numeric(val_df[pred_col], errors="coerce").loc[y_act.index]
            ok_mask = np.isfinite(y_act) & np.isfinite(y_pr)
            y_act, y_pr = y_act[ok_mask], y_pr[ok_mask]
            
            residuals = y_pr - y_act
            r2_val = 1 - np.sum(residuals**2) / np.sum((y_act - y_act.mean())**2)
            rmse_val = float(np.sqrt(np.mean(residuals**2)))
            hit_val = float(np.mean(np.abs(residuals) <= t_tol) * 100.0)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Holdout R²", f"{r2_val:.3f}")
            m2.metric("Holdout RMSE", f"{rmse_val:.2f} {t_unit}")
            m3.metric(f"Within ±{t_tol:g} {t_unit}", f"{hit_val:.1f} %")
            m4.metric("Mean Residual Bias", f"{residuals.mean():+.2f} {t_unit}")
            
            col_chart1, col_chart2 = st.columns([1.3, 1.0])
            with col_chart1:
                p_min = min(float(y_act.min()), float(y_pr.min())) - 10
                p_max = max(float(y_act.max()), float(y_pr.max())) + 10
                fig_par = go.Figure()
                
                fig_par.add_trace(go.Scatter(
                    x=[p_min, p_max, p_max, p_min],
                    y=[p_min + t_tol, p_max + t_tol, p_max - t_tol, p_min - t_tol],
                    fill="toself", fillcolor=_rgba(ACCENT, 0.08), line=dict(width=0),
                    name=f"±{t_tol:g} {t_unit} tolerance", hoverinfo="skip"
                ))
                fig_par.add_trace(go.Scatter(
                    x=[p_min, p_max], y=[p_min, p_max], mode="lines",
                    line=dict(color=WARN_LINE, width=1.5, dash="dash"), name="Ideal 1:1"
                ))
                fig_par.add_trace(go.Scatter(
                    x=y_act, y=y_pr, mode="markers",
                    marker=dict(size=4.5, color=ACCENT, opacity=0.45),
                    text=[f"Actual: {a:.1f}<br>Pred: {p:.1f}<br>Diff: {p-a:+.1f}" for a, p in zip(y_act, y_pr)],
                    hoverinfo="text", name="Held-Out Samples"
                ))
                fig_par.update_layout(
                    title=dict(text=f"Parity chart: {t_label} (N = {len(y_act):,} test samples)", font=dict(size=12)),
                    xaxis=dict(title=f"Actual Measured [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title=f"Predicted [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white", showlegend=False
                )
                st.plotly_chart(fig_par, use_container_width=True)
                
            with col_chart2:
                fig_res = go.Figure(go.Histogram(
                    x=residuals, nbinsx=35, marker_color=ACCENT_RAMP[4],
                    marker_line_color=ACCENT, marker_line_width=0.6
                ))
                fig_res.add_vline(x=0, line_color="#111418", line_width=1.2)
                fig_res.add_vline(x=t_tol, line_color=WARN_LINE, line_width=1, line_dash="dash")
                fig_res.add_vline(x=-t_tol, line_color=WARN_LINE, line_width=1, line_dash="dash")
                fig_res.update_layout(
                    title=dict(text=f"Residual Error Spread (Bias: {residuals.mean():+.2f} {t_unit})", font=dict(size=12)),
                    xaxis=dict(title=f"Prediction Error (Pred - Actual) [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title="Sample Count", showline=False, gridcolor="#f4f5f7"),
                    height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
                )
                st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.caption("Validation prediction logs not available.")

    with diag_tabs[1]:
        st.caption("Inductive split-conformal prediction intervals guarantee distribution-free finite-sample coverage on held-out ladle melts.")
        st.markdown(r"""
        <div class="note" style="margin-bottom:1rem;">
          <b>Why Gaussian Standard Errors Fail in Production</b>: The naive formula <code>±1.96 · RMSE / √n</code> measures the confidence of an aggregate sample mean, <b>not an individual rolled plate or coil</b>. In single-unit plant testing, naive intervals capture only ~25% of certified heats. Split-conformal calibration computes the exact empirical quantile of nonconformity scores $\hat{q}_{0.95}$, guaranteeing $\ge 95\%$ coverage on every future heat without assuming normal distributions.
        </div>
        """, unsafe_allow_html=True)

        if not conf_df.empty and "Target" in conf_df.columns:
            avail_conf_targets = conf_df["Target"].tolist()
            sel_conf_t = st.selectbox("Select Target Property to Inspect Conformal Calibration", avail_conf_targets,
                                      format_func=lambda k: TARGET_LABEL[mill].get(k, (k, ""))[0], key=f"sel_conf_{mill}")
            conf_row = conf_df[conf_df["Target"] == sel_conf_t].iloc[0]

            c_m1, c_m2, c_m3, c_m4, c_m5 = st.columns(5)
            c_m1.metric("Held-Out Test Samples", f"{int(conf_row['Val_N']):,}")
            c_m2.metric("Holdout RMSE", f"{conf_row['Val_RMSE']:.2f} {conf_row.get('Unit', '')}")
            c_m3.metric("Naive Formula (Old)", f"±{conf_row['Old_Naive_Half_Width']:.2f} {conf_row.get('Unit', '')}",
                        delta=f"Coverage: {conf_row['Old_Naive_Coverage_Pct']:.1f}% (FAIL)", delta_color="inverse")
            c_m4.metric("95% Conformal (Active)", f"±{conf_row['Conformal_Half_Width_95']:.2f} {conf_row.get('Unit', '')}",
                        delta=f"Coverage: {conf_row['Coverage_95_Pct']:.1f}% (VALID)", delta_color="normal")
            c_m5.metric("90% Conformal", f"±{conf_row['Conformal_Half_Width_90']:.2f} {conf_row.get('Unit', '')}",
                        delta=f"Coverage: {conf_row['Coverage_90_Pct']:.1f}%", delta_color="normal")

            col_cc1, col_cc2 = st.columns([1.2, 1.0])
            with col_cc1:
                fig_cov = go.Figure()
                fig_cov.add_trace(go.Bar(
                    x=conf_df["Target"], y=conf_df["Old_Naive_Coverage_Pct"],
                    name="Old Naive Formula (~25%)", marker_color=BAD_LINE,
                    text=[f"{v:.1f}%" for v in conf_df["Old_Naive_Coverage_Pct"]], textposition="auto"
                ))
                fig_cov.add_trace(go.Bar(
                    x=conf_df["Target"], y=conf_df["Coverage_95_Pct"],
                    name="95% Split-Conformal (Calibrated)", marker_color=ACCENT,
                    text=[f"{v:.1f}%" for v in conf_df["Coverage_95_Pct"]], textposition="auto"
                ))
                fig_cov.add_hline(y=95.0, line_dash="dash", line_color=SEMANTIC_OK, line_width=1.5,
                                  annotation_text="95% Nominal Guarantee", annotation_position="top left")
                fig_cov.update_layout(
                    title=dict(text="Empirical Coverage Audit: Naive Gaussian SE vs Split-Conformal Calibration", font=dict(size=12)),
                    xaxis=dict(title="Target Mechanical Property", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title="Empirical Coverage on Held-Out Heats (%)", range=[0, 105], showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    barmode="group", height=360, margin=dict(l=10, r=10, t=35, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.12, x=0.0)
                )
                st.plotly_chart(fig_cov, use_container_width=True)

            with col_cc2:
                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(
                    x=conf_df["Target"], y=conf_df["Conformal_Half_Width_95"],
                    name="95% Conformal Width (±)", marker_color=ACCENT_RAMP[2],
                    text=[f"±{v:.1f} {u}" for v, u in zip(conf_df["Conformal_Half_Width_95"], conf_df["Unit"])], textposition="auto"
                ))
                fig_w.add_trace(go.Bar(
                    x=conf_df["Target"], y=conf_df["Old_Naive_Half_Width"],
                    name="Old Naive Width (±)", marker_color=HAIR,
                    text=[f"±{v:.1f}" for v in conf_df["Old_Naive_Half_Width"]], textposition="auto"
                ))
                fig_w.update_layout(
                    title=dict(text=f"Prediction Interval Half-Width Comparison ({mill})", font=dict(size=12)),
                    xaxis=dict(title="Target Property", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title="Interval Half-Width (± Value)", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    barmode="group", height=360, margin=dict(l=10, r=10, t=35, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.12, x=0.0)
                )
                st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.caption("Conformal interval data not available.")

    with diag_tabs[2]:
        st.caption("Solves for chemistry and rolling setpoints that meet the spec at the 95% conformal bound.")
        presets = optimizer.PRESET_GRADES[mill]
        c_s1, c_s2, c_s3 = st.columns([2.0, 1.5, 1.5], vertical_alignment="bottom")
        diag_opt_spec = c_s1.selectbox("Customer Specification Standard", list(presets.keys()), index=0, key=f"diag_opt_spec_{mill}")
        s_cfg = presets[diag_opt_spec]
        d_thick = c_s2.slider("Finish Gauge (mm)", min_value=6.0 if mill=="HRC" else 10.0, max_value=16.0 if mill=="HRC" else 60.0,
                              value=10.0 if mill=="HRC" else 20.0, step=1.0, key=f"diag_opt_thick_{mill}")
        d_obj = c_s3.selectbox("Optimization Objective", ["Minimum CE_IIW (Cost)", "Maximum Safety Buffer (Cpk)"], index=0, key=f"diag_opt_obj_{mill}")
        obj_key = "min_ce" if "Cost" in d_obj else "max_buffer"

        default_base = {spec[1]: spec[6] for spec in SPECS[mill] if len(spec) > 6}
        opt_run = optimizer.run_inverse_optimization(mill, s_cfg, d_thick, default_base, objective_mode=obj_key)

        if opt_run and "candidates_df" in opt_run and not opt_run["candidates_df"].empty:
            cands = opt_run["candidates_df"]
            best = opt_run["best_schedule"]
            q_ys = opt_run["q_ys"]
            q_ts = opt_run["q_ts"]

            o_m1, o_m2, o_m3, o_m4 = st.columns(4)
            o_m1.metric("Evaluated Schedules", f"{opt_run['n_evaluated']:,}")
            o_m2.metric("Feasible Schedules (95% Compliant)", f"{opt_run['n_compliant']:,}",
                        delta=f"{opt_run['n_compliant']/opt_run['n_evaluated']*100:.1f}% Yield")
            o_m3.metric("Champion CE_IIW", f"{best['ce_iiw']:.3f}")
            o_m4.metric("Optimal Chemistry", f"C: {best['Chem_C']:.2f}% · Mn: {best['Chem_Mn']:.2f}% · Nb: {best.get('Chem_Nb', 0):.3f}%")

            fig_opt = go.Figure()

            non_comp = cands[(cands["lower_ys"] < s_cfg.get("ys_min", 0)) | (cands["lower_ts"] < s_cfg.get("ts_min", 0))]
            if not non_comp.empty:
                fig_opt.add_trace(go.Scatter(
                    x=non_comp["ce_iiw"], y=non_comp["pred_ys"],
                    mode="markers", marker=dict(size=6, color=HAIR, opacity=0.6),
                    name="Infeasible (Below 95% Spec)",
                    text=[f"C: {c:.2f}%, Mn: {mn:.2f}%<br>Pred YS: {y} MPa (Lower 95%: {ly} MPa)<br>Req YS: {s_cfg.get('ys_min', 0)} MPa"
                          for c, mn, y, ly in zip(non_comp["Chem_C"], non_comp["Chem_Mn"], non_comp["pred_ys"], non_comp["lower_ys"])],
                    hoverinfo="text"
                ))

            comp = cands[(cands["lower_ys"] >= s_cfg.get("ys_min", 0)) & (cands["lower_ts"] >= s_cfg.get("ts_min", 0))]
            if not comp.empty:
                fig_opt.add_trace(go.Scatter(
                    x=comp["ce_iiw"], y=comp["pred_ys"],
                    mode="markers", marker=dict(size=7, color=ACCENT, opacity=0.7),
                    name="Feasible (Cpk ≥ 1.33 Conformal)",
                    text=[f"C: {c:.2f}%, Mn: {mn:.2f}%<br>Pred YS: {y} MPa (Lower 95%: {ly} MPa)<br>Req YS: {s_cfg.get('ys_min', 0)} MPa"
                          for c, mn, y, ly in zip(comp["Chem_C"], comp["Chem_Mn"], comp["pred_ys"], comp["lower_ys"])],
                    hoverinfo="text"
                ))

            fig_opt.add_trace(go.Scatter(
                x=[best["ce_iiw"]], y=[best["pred_ys"]],
                mode="markers+text",
                marker=dict(size=16, color=WARN_STAR, symbol="star", line=dict(color=WARN_LINE, width=1.5)),
                name="Optimal Setpoint", text=["Optimal Champion"], textposition="top center"
            ))

            if "ys_min" in s_cfg:
                fig_opt.add_hline(
                    y=s_cfg["ys_min"] + q_ys, line_dash="dot", line_color=SEMANTIC_OK, line_width=1.5,
                    annotation_text=f"Compliance Threshold (Spec Min + 95% Conformal Buffer: {s_cfg['ys_min'] + q_ys:.1f} MPa)",
                    annotation_position="bottom right"
                )

            fig_opt.update_layout(
                title=dict(text=f"Inverse setpoint search landscape: {diag_opt_spec} ({d_thick} mm Plate)", font=dict(size=12)),
                xaxis=dict(title="Carbon Equivalent (CE_IIW = C + Mn/6 + 5Nb)", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                yaxis=dict(title="Predicted Yield Strength [MPa]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                height=380, margin=dict(l=10, r=10, t=35, b=10),
                plot_bgcolor="white", paper_bgcolor="white", legend=dict(orientation="h", y=1.12, x=0.0)
            )
            st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.caption("Inverse optimization evaluation could not be completed.")

    with diag_tabs[3]:
        st.caption("Interactive physics sandbox: adjust chemical composition and thermomechanical rolling parameters to see the real-time contribution of each physical dislocation mechanism.")
        p_col1, p_col2, p_col3 = st.columns(3)
        sb_c = p_col1.slider("Carbon [%C]", 0.08, 0.25, 0.16, 0.01, format="%.2f", key=f"sb_c_{mill}")
        sb_mn = p_col2.slider("Manganese [%Mn]", 0.60, 1.80, 1.20, 0.05, format="%.2f", key=f"sb_mn_{mill}")
        sb_si = p_col3.slider("Silicon [%Si]", 0.10, 0.50, 0.25, 0.02, format="%.2f", key=f"sb_si_{mill}")

        p_col4, p_col5, p_col6 = st.columns(3)
        sb_nb = p_col4.slider("Niobium [%Nb]", 0.000, 0.040, 0.015, 0.002, format="%.3f", key=f"sb_nb_{mill}")
        sb_t = p_col5.slider("Finish Thickness [mm]", 6.0 if mill=="HRC" else 8.0, 16.0 if mill=="HRC" else 80.0, 10.0 if mill=="HRC" else 20.0, 1.0, key=f"sb_t_{mill}")
        sb_frt = p_col6.slider("Finish Rolling Temp [°C]", 780.0, 920.0, 830.0, 5.0, key=f"sb_frt_{mill}")

        sigma_ss = 32.0 * sb_mn + 84.0 * sb_si + 43.0 * 0.02 + 38.0 * 0.02 + 80.0 * 0.002 + 354.0 * 0.003
        inv_sqrt_t = 1.0 / np.sqrt(max(1.0, sb_t))
        ce_iiw = sb_c + (sb_mn / 6.0) + (0.02 + 0.005 + 0.005) / 5.0 + 0.04 / 15.0
        ar3 = 910.0 - 310.0 * sb_c - 80.0 * sb_mn - 20.0 * 0.02 - 15.0 * 0.02 - 55.0 * 0.02
        supercool = max(0.0, ar3 - sb_frt)

        sb_m1, sb_m2, sb_m3, sb_m4, sb_m5 = st.columns(5)
        sb_m1.metric("Pickering Solid Solution (σ_ss)", f"{sigma_ss:.1f} MPa")
        sb_m2.metric("Hall-Petch Proxy (t^-0.5)", f"{inv_sqrt_t:.3f} mm^-0.5")
        sb_m3.metric("Carbon Equivalent (CE_IIW)", f"{ce_iiw:.3f}")
        sb_m4.metric("Ar3 Phase Transformation", f"{ar3:.1f} °C")
        sb_m5.metric("Transformation Supercooling", f"{supercool:.1f} °C")

        d_sigma0 = 96.5
        d_hp = 300.0 * inv_sqrt_t
        d_ss = 2.73 * (sb_mn * 32.0 + sb_si * 84.0) / 100.0
        d_nb = 2500.0 * sb_nb
        d_cr = 15.0 if mill == "Plate" else 25.0

        phys_df = pd.DataFrame([
            {"Mechanism": "1. Base Lattice Friction (σ₀)", "Strength_MPa": d_sigma0},
            {"Mechanism": "2. Hall-Petch Grain Refinement (k_hp · t^-0.5)", "Strength_MPa": d_hp},
            {"Mechanism": "3. Solid Solution Hardening (Pickering σ_ss)", "Strength_MPa": d_ss},
            {"Mechanism": "4. Microalloy Precipitation (Nb Carbonitrides)", "Strength_MPa": d_nb},
            {"Mechanism": "5. TMCP Cooling Severity / True Strain", "Strength_MPa": d_cr},
        ])
        total_calc_ys = phys_df["Strength_MPa"].sum()

        fig_phys = go.Figure(go.Bar(
            x=phys_df["Mechanism"], y=phys_df["Strength_MPa"],
            marker_color=[MUTED, ACCENT_DARK, ACCENT, ACCENT_RAMP[1], ACCENT_RAMP[2]],
            text=[f"+{v:.1f} MPa" for v in phys_df["Strength_MPa"]], textposition="auto"
        ))
        fig_phys.update_layout(
            title=dict(text=f"Decomposed Strengthening Mechanism Breakdown (Calculated Physical YS: ~{total_calc_ys:.1f} MPa)", font=dict(size=12)),
            xaxis=dict(showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
            yaxis=dict(title="Strength Contribution [MPa]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
            height=340, margin=dict(l=10, r=10, t=35, b=10), plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig_phys, use_container_width=True)

    with diag_tabs[4]:
        st.caption("What runs where, from the campaign sheets to the served artifacts. The first "
                   "two rows are the claims a number on this platform depends on.")
        n_holdout = _fmt(holdout_heats)
        st.markdown(f"""
        <ul class="prov">
          <li class="prov-lead"><b>Melt-grouped holdout:</b> sister plates or coils cut from one melt share
          chemistry, so the split never cuts a heat in half. <code>GroupShuffleSplit</code> on <code>Heat_ID</code>
          holds out {n_holdout} whole melts (80% train / 20% holdout).</li>
          <li class="prov-lead"><b>Split-conformal intervals:</b> the 95% interval beside every prediction
          comes from split-conformal calibration on held-out melts, not a bare RMSE. Empirical coverage is
          audited per target on the Conformal calibration subtab.</li>
          <li>Campaign workbooks are read with <code>python-calamine</code> (no Excel COM); decimal-shift
          typos such as 41,205 &rarr; 412.05 MPa are rescaled before training, and every repair is logged
          in <code>reports/sanitise_log.csv</code>.</li>
          <li>One row per plate is scored through the same feature module the simulator uses: Pickering
          &sigma;<sub>ss</sub>, Hall-Petch <i>t</i><sup>-0.5</sup>, true strain, <i>FRT &minus; Ar<sub>3</sub></i>.</li>
          <li>Every target is benchmarked across OLS, a metallurgical non-linear fit, Ridge, XGBoost,
          LightGBM, CatBoost and a stacking ensemble; the melt-held-out leaderboard picks the champion.</li>
          <li>Heavy training, hyperparameter search and the 100-iteration heat-level bootstrap run on
          Modal Cloud containers; this app only serves fitted artifacts.</li>
          <li>Champion GBDT models ship as native binaries in <code>models/boosted_out/</code>, each with
          the SHA-256 recorded at build time (see the artifact registry above).</li>
        </ul>
        """, unsafe_allow_html=True)

    with diag_tabs[5]:
        st.caption("How each schedule input actually ran in the cleaned campaign. When the full "
                   "cleaned table is not shipped with the deployment, the bundled quantile summary "
                   "is used instead; nothing synthetic is plotted.")
        eda_vars = [row[1] for row in SPECS[mill]]
        eda_choice = st.selectbox("Select Process Parameter / Chemistry for Distribution EDA", eda_vars,
                                  format_func=lambda k: f"{spec_map(mill)[k][2]} ({spec_map(mill)[k][3]})",
                                  key=f"eda_var_{mill}")
        row_meta = spec_map(mill)[eda_choice]
        u_label, u_unit, lo_b, hi_b = row_meta[2], row_meta[3], row_meta[4], row_meta[5]

        table = ts.load_table(mill)
        values = None
        if not table.empty and eda_choice in table.columns:
            column = ts._numeric(table[eda_choice]).dropna()
            column = column[(column >= lo_b) & (column <= hi_b)]  # decimal-shift typos out
            if len(column) >= 40:
                values = column

        if values is not None:
            q05, q50, q95 = np.percentile(values, [5, 50, 95])
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Min allowed", f"{lo_b:g} {u_unit}")
            e2.metric("P05, measured", f"{q05:.2f} {u_unit}")
            e3.metric("Median, measured", f"{q50:.2f} {u_unit}")
            e4.metric("P95, measured", f"{q95:.2f} {u_unit}")
            e5.metric("Max allowed", f"{hi_b:g} {u_unit}")
            fig_dist = go.Figure(go.Histogram(
                x=values, nbinsx=40, marker_color=ACCENT_RAMP[4],
                marker_line_color=ACCENT, marker_line_width=0.6
            ))
            fig_dist.add_vline(x=q50, line_color="#111418", line_width=1.5, annotation_text="Median", annotation_position="top")
            fig_dist.add_vline(x=q05, line_color=MUTED, line_width=1, line_dash="dash", annotation_text="P05", annotation_position="top")
            fig_dist.add_vline(x=q95, line_color=MUTED, line_width=1, line_dash="dash", annotation_text="P95", annotation_position="top")
            fig_dist.update_layout(
                title=dict(text=f"Campaign distribution: {u_label} (N = {len(values):,} cleaned {mill} records)", font=dict(size=12)),
                xaxis=dict(title=f"{u_label} [{u_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                yaxis=dict(title="Frequency", showline=False, gridcolor="#f4f5f7"),
                height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
            st.caption("Source: the cleaned training table. Values outside the schedule form's "
                       "allowed band are dropped before plotting.")
        else:
            stats = ts.load_bundle().get(mill, {}).get("variables", {}).get(eda_choice, {}).get("all")
            if stats:
                fig_dist = go.Figure(go.Box(
                    x=[0],
                    q1=[stats["q1"]], median=[stats["median"]], q3=[stats["q3"]],
                    lowerfence=[stats["min"]], upperfence=[stats["max"]],
                    marker_color=ACCENT, fillcolor=ACCENT_RAMP[4], name=u_label,
                    hoverinfo="x",
                ))
                fig_dist.update_layout(
                    title=dict(text=f"Campaign distribution: {u_label} "
                                    f"(bundled summary, N = {stats['n']:,} records)", font=dict(size=12)),
                    xaxis=dict(title=f"{u_label} [{u_unit}]", showline=True, linecolor=HAIR,
                               gridcolor="#f4f5f7", range=[min(lo_b, stats["min"]),
                                                           max(hi_b, stats["max"])]),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    height=220, margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption(f"Measured quantiles of this input: min {stats['min']:g}, Q1 {stats['q1']:g}, "
                           f"median {stats['median']:g}, Q3 {stats['q3']:g}, max {stats['max']:g} {u_unit} "
                           f"over {stats['n']:,} records. Source: the bundled training summary "
                           "(models/training_stats.json); the full cleaned table is not shipped with "
                           "this deployment, so no per-record histogram is drawn.")
            else:
                fig_dist = go.Figure(go.Scatter(
                    x=[lo_b, hi_b], y=[0, 0], mode="lines",
                    line=dict(color=ACCENT_RAMP[3], width=10), name="allowed band",
                    hoverinfo="x",
                ))
                fig_dist.update_layout(
                    title=dict(text=f"Input band only: {u_label} (no measured distribution shipped)", font=dict(size=12)),
                    xaxis=dict(title=f"{u_label} [{u_unit}]", showline=True, linecolor=HAIR,
                               gridcolor="#f4f5f7", range=[lo_b, hi_b]),
                    yaxis=dict(showgrid=False, showticklabels=False),
                    height=180, margin=dict(l=10, r=10, t=30, b=10),
                    plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
                )
                st.plotly_chart(fig_dist, use_container_width=True)
                st.caption("Neither the cleaned table nor the bundled summary carries this input, so "
                           "no distribution is drawn. The bar marks the schedule form's allowed range "
                           "only; nothing on this chart is measured data.")

        st.markdown('<div class="breath"></div>', unsafe_allow_html=True)
        st.caption("Top variables driving the champion model's predictions, ranked by the split "
                   "importance computed from the shipped artifact at render time.")
        importances = champion_importances(mill)
        if not importances:
            st.caption("No shipped boosted artifact on this route exposes feature importances, "
                       "so no ranking is shown rather than a guessed one.")
        else:
            prune_stale_widget_key(f"feat_tgt_{mill}", list(importances))
            feat_target = st.selectbox("Target", list(importances),
                                       format_func=lambda k: TARGET_LABEL[mill].get(k, (k, ""))[0],
                                       key=f"feat_tgt_{mill}")
            feat_fam = (champion_plan(mill).get(feat_target) or {}).get("family")
            feat_fam_name = {"cat": "CatBoost", "xgb": "XGBoost", "lgbm": "LightGBM"}.get(feat_fam, feat_fam)
            ranked = importances[feat_target][-12:]
            feat_df = pd.DataFrame([(FEATURE_LABELS.get(name, name), float(weight))
                                    for name, weight in ranked],
                                   columns=["Feature", "Importance"])
            fig_feat = go.Figure(go.Bar(
                x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
                marker=dict(color=ACCENT, line=dict(color=ACCENT_DARK, width=0.5))
            ))
            fig_feat.update_layout(
                title=dict(text=f"Feature split importance: {TARGET_LABEL[mill].get(feat_target, (feat_target, ''))[0]} "
                                f"({mill}, {feat_fam_name} champion artifact)", font=dict(size=12)),
                xaxis=dict(title="Share of total split importance", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                yaxis=dict(showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.caption(f"Computed from the shipped {feat_fam_name} artifact in models/boosted_out at "
                       f"render time; the ranking changes if the models are retrained.")

    # 5. Physical Basis & Constitutive Metallurgy (Clean HTML & LaTeX without messy KaTeX breaks)
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    section("physical basis & constitutive metallurgy")
    st.markdown("""
<div class="note">
Room-temperature mechanical properties are governed by austenite-to-ferrite phase transformation thermodynamics, 
solid-solution lattice distortion, and dislocation glide impedance. 
Unlike pure empirical black-box models, this platform embeds classical metallurgical constitutive laws 
fitted via Levenberg-Marquardt / Trust Region Reflective non-linear optimization.
</div>
""", unsafe_allow_html=True)

    if mill == "Plate":
        st.markdown("#### Heavy Plate Constitutive Formulation")
        st.latex(r"\sigma\ (\text{MPa}) = \sigma_0 + k_{\mathrm{hp}} \cdot d_{\alpha}^{-1/2} + k_{\mathrm{ss}} \cdot \sigma_{\mathrm{ss}} + k_{\mathrm{ce}} \cdot CE_{\mathrm{IIW}}^2 + k_{\mathrm{cr}} \cdot CR + k_{\varepsilon} \cdot \varepsilon_{\mathrm{total}}")

        st.markdown(f"""
<div class="card" style="padding: 1rem 1.25rem; margin: 1rem 0;">
  <div style="font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:{ACCENT}; margin-bottom:.6rem;">
    Physical Variable &amp; Dislocation Mechanism Breakdown
  </div>
  <ul style="margin:0; padding-left:1.2rem; font-size:13px; line-height:1.75; color:#1f2937;">
    <li><b>&sigma;<sub>0</sub> (Lattice Friction Stress)</b>: Peierls-Nabarro friction stress resisting dislocation glide in pure &alpha;-ferrite matrix (&sim; 96.5 MPa for yield, 310.6 MPa for tensile).</li>
    <li><b><i>k</i><sub>hp</sub> · <i>t</i><sup>−1/2</sup> (Hall-Petch Grain Refinement)</b>: Grain boundaries block slip bands. Plate finish gauge <i>t</i><sup>−1/2</sup> acts as the physical proxy for ferrite grain size (<i>d</i><sub>&alpha;</sub> &prop; &radic;<i>t</i>).</li>
    <li><b><i>k</i><sub>ss</sub> · &sigma;<sub>ss</sub> (Pickering Solid Solution Hardening)</b>: Ferrite lattice distortion from dissolved interstitial and substitutional solute atoms:
      <div style="font-family:monospace; font-size:13px; color:{ACCENT}; padding:.25rem 0; font-weight:500;">
        &sigma;<sub>ss</sub> (MPa) = 32[%Mn] + 84[%Si] + 43[%Ni] − 30[%Cr] + 38[%Cu] + 80[%Ti] + 11[%Mo] + 354[%N]
      </div>
    </li>
    <li><b><i>k</i><sub>ce</sub> · <i>CE</i><sub>IIW</sub><sup>2</sup> (Second-Phase Volume Fraction)</b>: Hard microconstituents (pearlite/bainite volume fraction) driven by Carbon Equivalent (<i>CE</i><sub>IIW</sub>).</li>
    <li><b><i>k</i><sub>cr</sub> · <i>CR</i> (ACC Laminar Cooling Rate)</b>: Accelerated water cooling suppresses transformation start temperature (<i>Ar</i><sub>3</sub>), refining polygonal ferrite packet size.</li>
    <li><b><i>k</i><sub>&epsilon;</sub> · &epsilon;<sub>total</sub> (True Deformation Strain)</b>: Total logarithmic rolling reduction ln(<i>t</i><sub>slab</sub> / <i>t</i><sub>plate</sub>) driving dislocation forest multiplication.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("##### Calibrated Constitutive Equations (Heavy Plate)")
        _render_nlr_cards(mill)

    else:
        st.markdown("#### Steckel Mill HRC Constitutive Formulation")
        st.latex(r"\sigma\ (\text{MPa}) = \sigma_0 + k_{\mathrm{ss}} \cdot \sigma_{\mathrm{ss}} + k_{\mathrm{hp}} \cdot t^{-1/2} + k_{\varepsilon} \cdot \varepsilon_{\mathrm{STM}}^n - k_T \cdot (CT_{\mathrm{avg}} - 500)")

        st.markdown(f"""
<div class="card" style="padding: 1rem 1.25rem; margin: 1rem 0;">
  <div style="font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:{ACCENT}; margin-bottom:.6rem;">
    Physical Variable &amp; Dislocation Mechanism Breakdown
  </div>
  <ul style="margin:0; padding-left:1.2rem; font-size:13px; line-height:1.75; color:#1f2937;">
    <li><b>&sigma;<sub>0</sub> (Lattice Friction Stress)</b>: Base yield stress of unalloyed ferrite matrix (308.35 MPa for YS, 461.25 MPa for TS).</li>
    <li><b><i>k</i><sub>ss</sub> · &sigma;<sub>ss</sub> (Pickering Solid Solution)</b>: Ferrite lattice distortion from dissolved interstitial and substitutional elements.</li>
    <li><b><i>k</i><sub>hp</sub> · <i>t</i><sup>−1/2</sup> (Hall-Petch Proxy)</b>: Strip thickness (<i>t</i>) proxy for runout table cooling rate and ferrite grain diameter.</li>
    <li><b><i>k</i><sub>T</sub> · (<i>CT</i><sub>avg</sub> − 500) (Thermal Coiling Softening)</b>: Above 500 &deg;C, thermal recovery, subgrain coarsening, and precipitate overaging reduce tensile strength.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("##### Calibrated Constitutive Equations (Steckel HRC)")
        _render_nlr_cards(mill)

    # 6. Linear Whitebox Coefficients Inspector
    st.markdown('<div class="break"></div>', unsafe_allow_html=True)
    with st.expander("Linear whitebox model weights (standardized OLS / Ridge regression)", expanded=False):
        st.caption("Standardized z-score weights indicate the relative strength and direction of each physical input on room-temperature mechanical properties.")
        target_choice = st.selectbox("Inspect target coefficients", list(TARGET_LABEL[mill]),
                                     format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"diag_cf_{mill}")
        spec = lfam.get(target_choice)
        if spec and "features" in spec and "ols" in spec:
            ols_coefs = pd.Series(spec["ols"]["coef"], index=spec["features"]).sort_values(ascending=False)
            st.caption(f"OLS Intercept: {spec['ols']['intercept']:.2f} · Model Holdout R²: {spec['ols']['r2']:.3f}")
            c_pos = ols_coefs[ols_coefs > 0].head(8)
            c_neg = ols_coefs[ols_coefs < 0].tail(8)
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Top Strengthening Factors (+)**")
                for feat, weight in c_pos.items():
                    st.markdown(f"- `{feat}`: **+{weight:.3f}**")
            with col_b:
                st.markdown("**Top Softening / Inverting Factors (−)**")
                for feat, weight in c_neg.items():
                    st.markdown(f"- `{feat}`: **{weight:.3f}**")
        else:
            st.caption("No standardized linear weights recorded for this target.")


# ----------------------------------------------------------------------------- main
def main():
    inject_css()
    if os.path.exists(LOGO_PATH):
        try:
            st.logo(LOGO_PATH, size="large")
        except Exception:
            pass
    with st.sidebar:
        st.markdown("### Route")
        mill = st.radio("Route", ["Plate", "HRC"], horizontal=True, label_visibility="collapsed",
                        format_func=lambda m: "Heavy plate" if m == "Plate" else "Steckel coil")
        st.markdown("---")
        raw = schedule_inputs(mill)
        st.markdown("---")
        st.markdown("### Model Engine")
        engine_opts = ["Champion GBDT (Recommended)", "CatBoost", "XGBoost", "LightGBM",
                       "Ensemble (.joblib)"]
        if linear_families(mill):
            engine_opts.append("OLS / Linear")
        prune_stale_widget_key(f"engine_{mill}", engine_opts)
        engine = st.selectbox(
            "Scoring engine",
            engine_opts,
            index=0,
            key=f"engine_{mill}",
            help="Champion GBDT selects the highest R² tree architecture per target. "
                 "OLS / Linear is only offered where a linear family was actually fitted."
        )
        st.markdown("---")
        st.caption("Simulator predictions run locally; recipe search, batch and capability "
                   "calculations run on Modal when configured.")
        st.caption("Data: Gunung Raja Paksi production logs. Models private; app public.")
        with st.expander("build and self-test", expanded=False):
            st.caption(f"build {BUILD}")
            for name, (good, total, err) in boot_check(mill).items():
                if total and good == total:
                    st.caption(f"{name}: {good}/{total} targets scoring")
                else:
                    st.caption(f"{name}: {good}/{total} targets scoring")
                    if err:
                        st.code(err[:300], language="text")

    header(mill, len(TARGET_LABEL[mill]))

    tabs = st.tabs(["Simulator", "Specification check", "Inverse Optimizer", "Response surface",
                    "Precipitation & Solubility", "Capability",
                    "Batch", "Retrain", "Diagnostics", "Atlas", "Shadow", "Pass schedule",
                    "About"])
    with tabs[0]:
        view_simulator(mill, raw, engine)
    with tabs[1]:
        view_spec_check(mill, raw, engine)
    with tabs[2]:
        optimizer.view_optimizer(mill, raw, engine, spec_rows=SPECS[mill])
    with tabs[3]:
        view_surface(mill, raw, engine)
    with tabs[4]:
        precipitation.view_precipitation(mill, raw)
    with tabs[5]:
        view_capability(mill, raw, engine)
    with tabs[6]:
        view_batch(mill, engine)
    with tabs[7]:
        view_retrain(mill)
    with tabs[8]:
        view_diagnostics(mill)
    with tabs[9]:
        atlas.view_atlas(mill, raw)
    with tabs[10]:
        shadow.view_shadow(mill, raw)
    with tabs[11]:
        pass_schedule.view_pass_schedule(mill, raw)
    with tabs[12]:
        view_about()


if __name__ == "__main__":
    main()
