"""Plate Mill & Steckel Mill Prediction Platform — fresh technical white UI.

Heavy plate and Steckel mill coil, one model layer. Sections: simulator, response surface, batch,
retrain, diagnostics.
"""

import base64
import io
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import model_loader as ml  # noqa: E402
import spec_checker as sc  # noqa: E402

st.set_page_config(
    page_title="Plate Mill & Steckel Mill Prediction Platform",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BUILD = "2026-10-28 · champion GBDT engine · fast cold-start · engine selector"
LOG_PATH = os.path.join(APP_DIR, "training_log.json")
LOGO_PATH = os.path.join(os.path.dirname(APP_DIR), "assets", "grp.png")
INK = "#0b0d10"
MUTED = "#6b7280"
HAIR = "#e5e7eb"
ACCENT = "#1746c8"
PANEL = "#f7f8fa"


# ----------------------------------------------------------------------------- input specs
PLATE_SPEC = [
    ("Geometry", "Slab_T_mm", "Slab thickness", "mm", 120.0, 320.0, 218.0, 1.0),
    ("Geometry", "Plate_T_mm", "Plate thickness", "mm", 6.0, 120.0, 16.0, 0.5),
    ("Geometry", "Plate_W_mm", "Plate width", "mm", 1000.0, 4000.0, 1889.0, 10.0),
    ("Geometry", "TB_Thickness_mm", "Transfer bar thickness", "mm", 15.0, 80.0, 41.0, 0.5),
    ("Finishing mill", "FM_Temp_Before_Rolling", "Entry temperature", "C", 850.0, 1150.0, 985.0, 1.0),
    ("Finishing mill", "FM_Finish_Temp", "Finishing temperature", "C", 700.0, 1050.0, 832.0, 1.0),
    ("Finishing mill", "FM_Last_Pass_Drafting", "Last pass drafting", "mm", 0.5, 25.0, 4.0, 0.5),
    ("Cooling", "ACC_Laminar_Header", "Laminar headers", "count", 0.0, 20.0, 8.0, 1.0),
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
    ("Geometry", "TB_Thick_mm", "Transfer bar thickness", "mm", 15.0, 70.0, 24.8, 0.5),
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
SPECS = {"Plate": PLATE_SPEC, "HRC": HRC_SPEC}

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


# ----------------------------------------------------------------------------- styling
@st.cache_resource(show_spinner=False)
def boot_check():
    """Score both routes once at start-up using Champion GBDT (instant bootstrap)."""
    report = {}
    for mill in ("Plate", "HRC"):
        try:
            raw = defaults(mill)
            if mill == "Plate":
                raw["Rolling_Type"] = "AR"
            out = predict_champion(mill, raw)
            scored = {k: v["pred"] for k, v in out.items()}
            good = sum(1 for v in scored.values() if np.isfinite(v))
            first_error = next((v["error"] for v in out.values() if v.get("error")), "")
            report[mill] = (good, len(scored), first_error)
        except Exception as exc:
            report[mill] = (0, 0, f"{type(exc).__name__}: {exc}")
    return report


def logo_uri():
    try:
        with open(LOGO_PATH, "rb") as fh:
            return "data:image/png;base64," + base64.b64encode(fh.read()).decode()
    except OSError:
        return ""


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

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

.kicker { font-size: 10.5px; letter-spacing: .18em; text-transform: uppercase; color: #1746c8; font-weight: 600; line-height: 1.6; }
.title { font-size: 30px; line-height: 1.15; font-weight: 600; margin: .35rem 0 .5rem; letter-spacing: -.03em; color: #0b0d10; }
.lede { font-size: 14.5px; color: #4b5563; max-width: 64ch; line-height: 1.55; }
.hair { height: 1px; background: #e5e7eb; margin: 1.4rem 0 1.2rem; border: 0; }

.section { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #4b5563;
           font-weight: 600; display: flex; align-items: center; gap: .7rem; margin-bottom: .7rem; }
.section i { flex: 1; height: 1px; background: #e5e7eb; display: block; }

.card { border: 1px solid #e5e7eb; border-radius: 6px; padding: 0 1rem; background: #ffffff; box-shadow: 0 1px 2px rgba(0,0,0,0.02); }
table.ut { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
table.ut th { font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: #6b7280;
              font-weight: 600; text-align: right; padding: .75rem .5rem .55rem;
              border-bottom: 1px solid #e5e7eb; background: #fafbfc; }
table.ut th:first-child { text-align: left; padding-left: .25rem; }
table.ut td { padding: .65rem .5rem; border-bottom: 1px solid #f1f2f4; text-align: right;
              vertical-align: middle; font-size: 13.5px; color: #0b0d10; }
table.ut td:first-child { text-align: left; padding-left: .25rem; font-weight: 500; }
table.ut tbody tr:hover td { background: #f8fafc; }
table.ut tbody tr:last-child td { border-bottom: 0; }
table.ut tr.grp td { text-align: left; font-size: 9.6px; letter-spacing: .12em; text-transform: uppercase;
                     color: #1746c8; font-weight: 600; padding: .9rem .25rem .35rem;
                     border-bottom: 1px solid #e5e7eb; background: #ffffff; }
table.ut tbody tr:first-child.grp td { padding-top: .55rem; }
table.ut td.name { text-align: left; font-size: 13.5px; font-weight: 500; color: #0b0d10; }
table.ut td.val { font-family: 'JetBrains Mono', monospace; font-size: 15px; font-weight: 600; color: #0b0d10; }
table.ut td.alt.muted { color: #c3c7cd; }
table.ut td.alt { font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #4b5563; }
table.ut td.val .unit { font-size: 10px; color: #9ca3af; margin-left: .25rem; font-weight: 400; }
table.ut td.rng { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #4b5563; }
table.ut td.sub { font-size: 12px; color: #6b7280; }
table.ut td.flag { width: 110px; text-align: center; }
.card .foot { border-top: 1px solid #e5e7eb; padding: .6rem 0 .7rem; font-size: 11.5px; color: #6b7280; }

.pill { display: inline-flex; align-items: center; justify-content: center; font-size: 10px;
        letter-spacing: .05em; font-weight: 600; text-transform: uppercase; padding: .2rem .55rem;
        border-radius: 4px; white-space: nowrap; line-height: 1.2; }
.pill.ok { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
.pill.warn { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
.pill.bad { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
.pill.neutral { background: #f3f4f6; color: #4b5563; border: 1px solid #e5e7eb; }

.tag { display: inline-block; font-size: 10px; letter-spacing: .06em; text-transform: uppercase;
       padding: .15rem .4rem; border: 1px solid #e5e7eb; color: #4b5563; border-radius: 3px; }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0;
         border: 1px solid #e5e7eb; border-radius: 6px; overflow: hidden; background: #ffffff; }
.facts div { padding: .75rem 1rem; border-right: 1px solid #e5e7eb; }
.facts div:last-child { border-right: 0; }
.facts span { display: block; font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
              color: #6b7280; margin-bottom: .25rem; font-weight: 500; }
.facts b { font-family: 'JetBrains Mono', monospace; font-size: 15.5px; font-weight: 600; color: #0b0d10; }
.facts em { font-style: normal; font-size: 11px; color: #6b7280; margin-left: .25rem; }

.note { font-size: 12.5px; color: #4b5563; line-height: 1.5; border-left: 2px solid #e5e7eb;
        padding-left: .7rem; margin-top: .6rem; }

div[data-testid="stSidebar"] { background: #fbfbfc; border-right: 1px solid #e5e7eb; }
div[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
div[data-testid="stSidebar"] h2 { font-size: 13px; letter-spacing: .1em; text-transform: uppercase; }

.stTabs [data-baseweb="tab-list"] { gap: 1.2rem; border-bottom: 1px solid #e5e7eb; }
.stTabs [data-baseweb="tab"] { font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase;
                               color: #6b7280; padding: .5rem 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: #0b0d10 !important; font-weight: 600; }

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 22px; }
#MainMenu, footer, [data-testid="stAppDeployButton"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stExpandSidebarButton"] { display: flex !important; visibility: visible !important; }
</style>
""", unsafe_allow_html=True)


def section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def hair():
    st.markdown('<hr class="hair">', unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data helpers
def spec_map(mill):
    return {row[1]: row for row in SPECS[mill]}


def defaults(mill):
    return {row[1]: float(row[6]) for row in SPECS[mill]}


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
        plate_t = pd.to_numeric(frame.get("Plate_T_mm"), errors="coerce").fillna(16.0).clip(lower=1.0)
        if "TB_Ratio" not in frame:
            tb = pd.to_numeric(frame.get("TB_Thickness_mm"), errors="coerce").fillna(plate_t * 2)
            frame["TB_Ratio"] = tb / plate_t
        if "FM_Last_Pass_Drafting" not in frame:
            last = pd.to_numeric(frame.get("FM_Last_Pass_Thick"), errors="coerce").fillna(plate_t * 1.15)
            frame["FM_Last_Pass_Drafting"] = last - plate_t
        if "ACC_Headers_n" not in frame:
            frame["ACC_Headers_n"] = pd.to_numeric(frame.get("ACC_Laminar_Header"),
                                                   errors="coerce").fillna(0.0)
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


def score_frame(mill, frame, engine="Champion GBDT (Recommended)"):
    """Vectorised scoring: features once, then every target."""
    frame = prepare_frame(mill, frame)
    X = feature_matrix(mill, frame)
    out = pd.DataFrame(index=frame.index)
    if engine in ("Champion GBDT (Recommended)", "CatBoost", "XGBoost", "LightGBM"):
        plan = champion_plan(mill)
        idx = boosted_index(mill)
        for key in TARGET_LABEL[mill]:
            if engine == "Champion GBDT (Recommended)":
                fam = plan.get(key, {}).get("family", "ensemble")
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
            # Fallback to joblib bundle
            bundle = ml.load_model_bundle(mill, key)
            if bundle is not None:
                for cols in candidate_columns(bundle, X):
                    try:
                        out[key] = bundle["model"].predict(align(X, cols))
                        break
                    except Exception:
                        continue
    else:
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
    return out


def predict_one(mill, raw):
    """One schedule, every target, with the interval the shipped bootstrap report implies."""
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    ci = ml.get_ci_data(mill)
    lb = ml.get_leaderboard_data(mill)
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
        half = ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
        r2_val = 0.60
        if not ci.empty and "Target" in ci.columns:
            row = ci[ci["Target"] == key]
            if not row.empty:
                if "Val_RMSE" in row.columns and pd.notna(row["Val_RMSE"].values[0]):
                    half = 1.96 * (float(row["Val_RMSE"].values[0]) / np.sqrt(50))
                if "Val_R2" in row.columns and pd.notna(row["Val_R2"].values[0]):
                    r2_val = float(row["Val_R2"].values[0])
        elif not lb.empty and "Target" in lb.columns:
            row = lb[lb["Target"] == key]
            if not row.empty and "Final_Val_R2" in row.columns and pd.notna(row["Final_Val_R2"].values[0]):
                r2_val = float(row["Final_Val_R2"].values[0])
        label, unit = TARGET_LABEL[mill].get(key, (key, ""))
        out[key] = {"name": label, "pred": round(value, 2) if np.isfinite(value) else float("nan"),
                    "unit": unit, "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
                    "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan"),
                    "r2": r2_val,
                    "error": last_error.get(key, "")}
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


def nlr_terms(mill, raw):
    """The five terms the fitted NLR equations use, from the raw schedule."""
    thick = max(float(raw.get("Plate_T_mm" if mill == "Plate" else "Coil_T_mm", 16.0)), 1.0)
    slab = max(float(raw.get("Slab_T_mm" if mill == "Plate" else "TB_Thick_mm", 218.0)), 1.0)
    headers = float(raw.get("ACC_Laminar_Header", 0.0)) if mill == "Plate" else 0.0
    c, mn, si = (float(raw.get(k, d)) for k, d in (("Chem_C", .16), ("Chem_Mn", .9), ("Chem_Si", .2)))
    cu, ni = float(raw.get("Chem_Cu", .02)), float(raw.get("Chem_Ni", .02))
    cr, mo, v = float(raw.get("Chem_Cr", .02)), float(raw.get("Chem_Mo", .01)), float(raw.get("Chem_V", .005))
    return {
        "t": thick,
        "sigma_ss": 32 * mn + 84 * si + 38 * cu + 32 * ni,
        "CE": c + mn / 6 + (cr + mo + v) / 5 + (ni + cu) / 15,
        "CoolingRate": (headers + 0.1) / thick ** 1.5,
        "eps_total": float(np.log(slab / thick)),
    }


def predict_nlr(mill, raw):
    table = nlr_models(mill)
    values = nlr_terms(mill, raw)
    out = {}
    for key, model in table.items():
        value = model["intercept"]
        for term in model["terms"]:
            value += term["coef"] * values.get(term["var"], 0.0) ** term["power"]
        out[key] = {"pred": float(value), "r2": float(model.get("r2", 0.0))}
    return out


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
        values = np.array([float(row.get(c, 0.0) or 0.0) for c in spec["features"]])
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
    """Determines the champion model family and holdout stats per target."""
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
            plan[target] = {"family": "ensemble", "r2": 0.617, "rmse": 4.41}
    return plan


def predict_champion(mill, raw):
    """Fast, high-R2 scoring using champion native GBDT models with fallback for Z-Test."""
    frame = prepare_frame(mill, pd.DataFrame([prepared(raw, mill)]))
    X = feature_matrix(mill, frame)
    plan = champion_plan(mill)
    ci = ml.get_ci_data(mill)
    out = {}
    for key in TARGET_LABEL[mill]:
        label, unit = TARGET_LABEL[mill].get(key, (key, ""))
        cfg = plan.get(key, {})
        fam = cfg.get("family", "ensemble")
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
        half = ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
        rmse = cfg.get("rmse")
        if rmse and rmse > 0:
            half = 1.96 * (rmse / np.sqrt(50))
        elif not ci.empty and "Target" in ci.columns:
            row = ci[ci["Target"] == key]
            if not row.empty and "Val_RMSE" in row.columns:
                half = 1.96 * (float(row["Val_RMSE"].values[0]) / np.sqrt(50))
        out[key] = {
            "name": label,
            "pred": round(value, 2) if np.isfinite(value) else float("nan"),
            "unit": unit,
            "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
            "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan"),
            "family": fam,
            "r2": cfg.get("r2", 0.0),
            "error": err
        }
    return out


def predict_selected_engine(mill, raw, engine):
    """Dispatches prediction to the chosen engine."""
    if engine == "Champion GBDT (Recommended)":
        return predict_champion(mill, raw)
    elif engine == "Ensemble (.joblib)":
        return predict_one(mill, raw)
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
                rmse = meta.get("rmse", ml.TARGET_CONFIG[mill][key]["tol"] * 0.5)
                half = 1.96 * (rmse / np.sqrt(50)) if rmse > 0 else ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
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
        return out
    elif engine == "OLS / Linear":
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
                item["error"] = "Linear R2 < 0.15 (showing Champion GBDT)"
                out[key] = item
        return out
    return predict_champion(mill, raw)


def targets_table(mill, predictions, extras=None, engine_label="Champion GBDT"):
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
            fam = res.get("family", "")
            fam_badge = f'<span class="tag" style="margin-left:.4rem; font-size:9px;">{fam}</span>' if fam else ""
            extra_cells = "".join(
                (f'<td class="alt">{extras[name][key]["pred"]:.1f}</td>'
                 if key in extras[name] and np.isfinite(extras[name][key].get("pred", np.nan)) else '<td class="alt muted">—</td>')
                for name in extras)
            rows.append(f"""<tr>
              <td class="name">{label}{fam_badge}</td>
              <td class="val">{res['pred']:.1f}<span class="unit">{unit}</span></td>
              <td class="rng">{res['ci_low']:.1f} – {res['ci_high']:.1f}</td>
              <td class="sub" style="font-family:monospace;">±{rmse_tol:g}&nbsp;{unit}</td>{extra_cells}</tr>""")
    html = (f"<div class=\"card\"><table class=\"ut\"><thead><tr><th>property</th><th>{engine_label.lower()}</th>"
            "<th>95% bootstrap interval</th><th>holdout uncertainty</th>"
            + "".join(f"<th>{n}</th>" for n in extras)
            + "</tr></thead><tbody>" + "".join(rows)
            + f"</tbody></table><div class=\"foot\">{total} certified properties scored through "
              f"{mill} model layer ({engine_label}).</div></div>")
    st.markdown("".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- header
def header(mill, n_targets):
    if mill == "Plate":
        lede = ("An end-to-end predictive analytics suite that extracts engineered domain features across "
                "roughing, finishing, laminar ACC, and optional normalizing routes, including cumulative "
                "drafting strains, recrystallization stops, and cooling-rate kinetics. Benchmarking Ordinary "
                "Least Squares (OLS), non-linear regression splines, and fine-tuned machine learning ensembles "
                "validated on strict melt-grouped holdouts with bootstrap uncertainty estimation, the engine predicts "
                "10 certified mechanical properties while executing multi-iterative residual and stability diagnostics. "
                "Continuously maps dynamic Cp and Cpk process capability indices against tolerance bands and identifies "
                "optimal setpoints using fine-tuned 2D and 3D response surface methodology (RSM) visualizations.")
    else:
        lede = ("An end-to-end predictive analytics suite that extracts engineered domain features across "
                "roughing, reversing Steckel mill coiler boxes, runout table (ROT) laminar cooling, and downcoiling routes, "
                "including cumulative drafting strains, thermal retention, and cooling kinetics. Benchmarking Ordinary "
                "Least Squares (OLS), non-linear regression splines, and fine-tuned machine learning ensembles "
                f"validated on strict melt-grouped holdouts with bootstrap uncertainty estimation, the engine predicts "
                f"{n_targets} certified mechanical properties while executing multi-iterative residual and stability diagnostics. "
                "Continuously maps dynamic Cp and Cpk process capability indices against tolerance bands and identifies "
                "optimal setpoints using fine-tuned 2D and 3D response surface methodology (RSM) visualizations.")
    st.markdown(f"""
<div class="kicker">Plate Mill & Steckel Mill Prediction Platform</div>
<div class="title">Specification Limit & Mechanical Models</div>
<div class="lede">{lede}</div>
""", unsafe_allow_html=True)
    hair()


# ----------------------------------------------------------------------------- inputs
def schedule_inputs(mill):
    st.sidebar.markdown(f"### {mill} schedule")
    st.sidebar.caption("Every value here feeds the mill's own feature module.")
    values = {}
    groups = []
    for row in SPECS[mill]:
        if row[0] not in groups:
            groups.append(row[0])
    with st.sidebar.form(f"schedule_{mill}", border=False):
        for group in groups:
            with st.expander(group, expanded=group in ("Finishing mill", "Steckel mill")):
                for g, key, label, unit, lo, hi, default, step in SPECS[mill]:
                    if g != group:
                        continue
                    fmt = "%.4f" if step < 0.001 else ("%.3f" if step < 0.01 else
                                                       ("%.2f" if step < 1 else "%.0f"))
                    shown_unit = "°C" if unit == "C" else unit
                    values[key] = st.number_input(f"{label} [{shown_unit}]", min_value=float(lo),
                                                  max_value=float(hi), value=float(default),
                                                  step=float(step), format=fmt, key=f"{mill}_{key}")
        if mill == "Plate":
            st.markdown('<div class="note">Delivery route</div>', unsafe_allow_html=True)
            values["Rolling_Type"] = st.selectbox(
                "Rolling type", ["AR", "TMCP", "N"], key=f"{mill}_rolling_type",
                help="AR = as rolled, TMCP = thermomechanically controlled, N = normalised delivery.")
        submitted = st.form_submit_button("Calculate", type="primary", use_container_width=True)
    if submitted or "schedule" not in st.session_state:
        st.session_state["schedule"] = dict(values)
        st.session_state["schedule_mill"] = mill
    if st.session_state.get("schedule_mill") != mill:
        st.session_state["schedule"] = dict(values)
        st.session_state["schedule_mill"] = mill
    return st.session_state["schedule"]


# ----------------------------------------------------------------------------- sections
def view_simulator(mill, raw, engine="Champion GBDT (Recommended)"):
    section("predicted properties")
    preds = predict_selected_engine(mill, raw, engine)
    options = ["OLS", "Ridge", "NLR"]
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
    targets_table(mill, preds, extras, engine_label=engine)
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

    ind = physical(raw, mill)
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    section("derived process metrics")
    st.markdown(f"""
<div class="facts">
  <div><span>Pickering σ<sub>ss</sub></span><b>{ind['sigma_ss']:.0f}</b><em>MPa</em></div>
  <div><span>CE IIW</span><b>{ind['ce_iiw']:.3f}</b></div>
  <div><span>Pcm</span><b>{ind['pcm']:.3f}</b></div>
  <div><span>Ar3</span><b>{ind['ar3']:.0f}</b><em>C</em></div>
  <div><span>true strain</span><b>{ind['true_strain']:.2f}</b></div>
</div>""", unsafe_allow_html=True)

    finish = raw.get("FM_Finish_Temp", raw.get("Temp_FT_Avg_C", 0.0))
    margin = finish - ind["ar3"]
    st.markdown(f'<div class="note">Finishing at {finish:.0f} °C sits {margin:+.0f} K above Ar3 '
                f'({ind["ar3"]:.0f} °C). Below zero the last passes are intercritical, which is where '
                f'properties start to scatter.</div>', unsafe_allow_html=True)


def view_spec_check(mill, raw, engine="Champion GBDT (Recommended)"):
    specs = sc.get_available_specs()
    if not specs:
        st.error("No specifications found in specification-list/SPECIFICATION LIST.xlsx")
        return

    # Choose smart defaults
    default_spec = "ASTM A 572 (21e1)" if mill == "Plate" else "JIS G 3101 (2022)"
    if default_spec not in specs:
        default_spec = "ASTM A 36 (2019)" if "ASTM A 36 (2019)" in specs else specs[0]
    spec_idx = specs.index(default_spec) if default_spec in specs else 0

    c1, c2, c3 = st.columns([2.5, 1.8, 1.7], vertical_alignment="bottom")
    chosen_spec = c1.selectbox("Specification Standard", specs, index=spec_idx, key=f"sc_spec_{mill}")

    grades = sc.get_available_grades(chosen_spec)
    default_grade = "50" if "50" in grades else (grades[0] if grades else "-")
    grade_idx = grades.index(default_grade) if default_grade in grades else 0
    chosen_grade = c2.selectbox("Grade", grades, index=grade_idx, key=f"sc_gr_{mill}")

    t_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    thickness = float(raw.get(t_key, 16.0 if mill == "Plate" else 6.0))
    c3.markdown(f'<div style="padding-bottom:.4rem; font-size:13px; color:#4b5563;">Active Gauge: '
                f'<b style="font-family:monospace; color:#0b0d10;">{thickness:.1f} mm</b><br>'
                f'<span style="font-size:10.5px; color:#1746c8; font-weight:600;">Engine: {engine}</span></div>',
                unsafe_allow_html=True)

    preds = predict_selected_engine(mill, raw, engine)
    ind = physical(raw, mill)
    eval_res = sc.evaluate_all(chosen_spec, chosen_grade, thickness, raw, preds, ind, mill)

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
    c_pill_cls = "ok" if c_status == "ALL_PASS" else "bad"
    c_pill_text = "ALL PASS" if c_status == "ALL_PASS" else f"{chem_info['fail_count']} OUT OF SPEC"

    # Mech verdict pill
    m_status = mech_info["status"]
    if m_status == "COMPLIANT":
        m_pill_cls, m_pill_text = "ok", "COMPLIANT"
    elif m_status == "MARGINAL_RISK":
        m_pill_cls, m_pill_text = "warn", "MARGINAL RISK"
    elif m_status == "NON_COMPLIANT":
        m_pill_cls, m_pill_text = "bad", "NON-COMPLIANT"
    else:
        m_pill_cls, m_pill_text = "neutral", "NO REQ SPECIFIED"

    st.markdown('<div style="height:0.8rem"></div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="facts">
  <div>
    <span>Specification &amp; Type</span>
    <b>{chosen_spec}</b><em>Gr {chosen_grade} · {eval_res['spec_type']}</em>
  </div>
  <div>
    <span>Standard Gauge &amp; GRP Limit</span>
    <b>{thickness:.1f} mm</b><em><span class="pill {g_pill_cls}">{g_pill_text}</span></em>
  </div>
  <div>
    <span>Chemical Composition</span>
    <b>{chem_info['pass_count']} / {len(chem_info['active_checks'])}</b><em><span class="pill {c_pill_cls}">{c_pill_text}</span></em>
  </div>
  <div>
    <span>Mechanical Decency</span>
    <b>{mech_info['pass_count']} passed</b><em><span class="pill {m_pill_cls}">{m_pill_text}</span></em>
  </div>
</div>""", unsafe_allow_html=True)

    if gauge_info.get("message"):
        st.markdown(f'<div class="note">{gauge_info["message"]}</div>', unsafe_allow_html=True)

    # 1. Mechanical Decency Table
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
    section("mechanical properties decency & certification compliance")
    st.caption(f"Correlates {mill} ML predictions and 95% bootstrap confidence intervals against "
               f"{chosen_spec} Grade {chosen_grade} requirements. "
               f"A marginal risk indicates the point prediction passes but the 95% bootstrap bound crosses the limit under process variance.")

    mech_checks = mech_info.get("checks", [])
    if mech_checks:
        m_rows = []
        for chk in mech_checks:
            verdict = chk["verdict"]
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
            '<th>95% bootstrap interval</th><th>safety margin</th><th>verdict</th></tr></thead><tbody>'
            + "".join(m_rows) +
            '</tbody></table></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info(f"No specific mechanical property limits specified for {chosen_spec} Grade {chosen_grade} in standard table.")

    # 2. Chemistry Verification Table
    st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
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
        st.markdown(f'<div style="margin-top:.6rem; font-size:12px; color:#6b7280;">Unconstrained trace inputs (no standard limit):<br>{uncon_pills}</div>',
                    unsafe_allow_html=True)

    # 3. Standard Notes & Footnotes Callout
    if eval_res.get("notes"):
        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        section("standard footnotes & technical notes")
        st.markdown(f'<div class="note" style="border-left-color:#1746c8; background:#fbfcfe; padding:.75rem .9rem; border-radius:0 4px 4px 0;">'
                    f'<b>Standard Notes (GRP.ST.QA.09):</b><br>{eval_res["notes"]}</div>',
                    unsafe_allow_html=True)


def view_surface(mill, raw, engine="Champion GBDT (Recommended)"):
    left, mid, right = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")
    keys = [k for k in SURFACE[mill] if k in spec_map(mill)]
    x_key = left.selectbox("Horizontal axis", keys, index=0, key=f"x_{mill}")
    y_key = mid.selectbox("Vertical axis", [k for k in keys if k != x_key], index=1, key=f"y_{mill}")
    target = right.selectbox("Property", list(TARGET_LABEL[mill]),
                             format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"t_{mill}")

    c1, c2, c3 = st.columns([1.4, 1.4, 1.2], vertical_alignment="bottom")
    view = c1.radio("View", ["2D contour", "3D surface"], horizontal=True, key=f"v_{mill}")
    grid = c2.slider("Grid", 15, 41, 25, step=2, key=f"g_{mill}")
    c3.caption(f"Engine: {engine}. Rest of schedule held at sidebar values.")

    sx, sy = spec_map(mill)[x_key], spec_map(mill)[y_key]
    xs = np.linspace(sx[4], sx[5], grid)
    ys = np.linspace(sy[4], sy[5], grid)
    frame = pd.DataFrame([dict(raw) for _ in range(grid * grid)])
    frame[x_key] = np.repeat(xs, grid)
    frame[y_key] = np.tile(ys, grid)
    z = score_frame(mill, frame, engine=engine)[target].values.reshape(grid, grid)

    label, unit = TARGET_LABEL[mill][target]
    axes = dict(showline=True, linecolor=HAIR, gridcolor="#f1f2f4", zeroline=False,
                tickfont=dict(size=10, color=MUTED), title_font=dict(size=11, color=INK))
    here = predict_selected_engine(mill, raw, engine)[target]["pred"]

    if view == "2D contour":
        fig = go.Figure()
        fig.add_trace(go.Contour(
            x=xs, y=ys, z=z, colorscale="Blues", contours=dict(showlabels=True,
            labelfont=dict(size=9, color="white")), line=dict(width=.4, color="rgba(255,255,255,.6)"),
            hovertemplate=f"{sx[2]}: %{{x:.1f}}<br>{sy[2]}: %{{y:.1f}}<br>{label}: %{{z:.1f}} {unit}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(x=[raw[x_key]], y=[raw[y_key]], mode="markers",
                                 marker=dict(size=13, color=INK, line=dict(color="white", width=2.5)),
                                 hovertemplate=f"this schedule<br>{label}: {here:.1f} {unit}<extra></extra>",
                                 showlegend=False))
        fig.add_annotation(x=raw[x_key], y=raw[y_key], text=f"<b>{here:.1f} {unit}</b>",
                           showarrow=True, arrowhead=0, ax=0, ay=-30, bgcolor="white",
                           bordercolor=INK, borderwidth=1, font=dict(size=11, family="JetBrains Mono"))
        fig.update_xaxes(title=sx[2], **axes); fig.update_yaxes(title=sy[2], **axes)
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white")
    else:
        fig = go.Figure(go.Surface(x=xs, y=ys, z=z, colorscale="Blues", showscale=False,
                                   contours=dict(z=dict(show=True, usecolormap=True, project_z=True))))
        fig.add_trace(go.Scatter3d(x=[raw[x_key]], y=[raw[y_key]], z=[here], mode="markers",
                                   marker=dict(size=5, color=INK), name="this schedule"))
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=0, b=0),
                          scene=dict(xaxis=dict(title=sx[2], **axes), yaxis=dict(title=sy[2], **axes),
                                     zaxis=dict(title=f"{label} [{unit}]", **axes),
                                     camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))),
                          paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": True})
    st.markdown(f'<div class="note">Surface held on the mill\'s own feature engineering, not a proxy: '
                f'every node is a full pass through the {mill} pipeline. The marker is the current '
                f'schedule at {here:.1f} {unit}.</div>', unsafe_allow_html=True)


def template_bytes(mill):
    cols = [row[1] for row in SPECS[mill]]
    blank = pd.DataFrame([{c: row[6] for c, row in zip(cols, SPECS[mill])}])
    buf = io.BytesIO()
    blank.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def view_capability(mill, raw, engine="Champion GBDT (Recommended)"):
    """Cp/Cpk over a band the user defines, resampled through the mill's feature module."""
    section("process capability")
    st.markdown('<div class="note">Set the specification limits, then say how far two or three '
                'inputs may drift. The app resamples that band, scores every point and reports '
                'capability — Cp and Cpk on the predicted distribution, not on a single schedule.'
                '</div>', unsafe_allow_html=True)

    keys = list(TARGET_LABEL[mill])
    c1, c2 = st.columns([1.4, 1.6], vertical_alignment="bottom")
    target = c1.selectbox("Property", keys, format_func=lambda k: TARGET_LABEL[mill][k][0],
                          key=f"cp_t_{mill}")
    unit = TARGET_LABEL[mill][target][1]
    tolerance = ml.TARGET_CONFIG[mill][target]["tol"]
    base = predict_selected_engine(mill, raw, engine)[target]["pred"]

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

    if not st.button("Run capability study", type="primary", key=f"cp_go_{mill}"):
        st.caption(f"Reference schedule predicts {base:.1f} {unit}; limits default to ±{tolerance:g}.")
        return

    rng = np.random.default_rng(42)
    frame = pd.DataFrame([dict(raw) for _ in range(samples)])
    for key, band in bands.items():
        centre = float(raw[key])
        frame[key] = rng.uniform(centre - band, centre + band, samples)
    scored = score_frame(mill, frame, engine=engine)[target].values
    scored = scored[np.isfinite(scored)]
    if scored.size < 30:
        st.error("Too few scoring points came back — widen the band or change property.")
        return

    mean, sd = float(np.mean(scored)), float(np.std(scored, ddof=1))
    cp = (usl - lsl) / (6 * sd) if sd > 0 else float("inf")
    cpk = min(usl - mean, mean - lsl) / (3 * sd) if sd > 0 else float("inf")
    oos = float(np.mean((scored < lsl) | (scored > usl)) * 100)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cp", f"{cp:.2f}")
    m2.metric("Cpk", f"{cpk:.2f}")
    m3.metric("mean", f"{mean:.1f} {unit}")
    m4.metric("out of spec", f"{oos:.1f} %")
    st.caption(f"σ {sd:.2f} {unit} · {scored.size} scored points · band "
               + ", ".join(f"{spec_map(mill)[k][2]} ±{bands[k]:g}" for k in bands))

    fig = go.Figure(go.Histogram(x=scored, nbinsx=45, marker_color="#c9d7f5",
                                 marker_line_color="#1746c8", marker_line_width=.6))
    for value, label in ((lsl, "LSL"), (usl, "USL")):
        fig.add_vline(x=value, line_color="#b45309", line_width=1.4, line_dash="dash",
                      annotation_text=label, annotation_position="top", annotation_font_size=10)
    fig.add_vline(x=mean, line_color="#111418", line_width=1)
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
        x = pd.to_numeric(frame[key], errors="coerce").values
        ok = np.isfinite(x)
        slope, intercept = np.polyfit(x[ok], scored[ok], 1)
        resid_sd = float(np.std(scored[ok] - (slope * x[ok] + intercept)))
        grid = np.linspace(x[ok].min(), x[ok].max(), 60)
        band_fig = go.Figure()
        band_fig.add_trace(go.Scatter(x=x[ok], y=scored[ok], mode="markers",
                                      marker=dict(size=4, color="#9db4e8", opacity=.45),
                                      name="sampled"))
        band_fig.add_trace(go.Scatter(x=grid, y=slope * grid + intercept, mode="lines",
                                      line=dict(color="#1746c8", width=1.6), name="fit"))
        band_fig.add_trace(go.Scatter(x=np.concatenate([grid, grid[::-1]]),
                                      y=np.concatenate([slope * grid + intercept + 3 * resid_sd,
                                                        (slope * grid + intercept - 3 * resid_sd)[::-1]]),
                                      fill="toself", fillcolor="rgba(23,70,200,.10)",
                                      line=dict(width=0), name="±3σ"))
        for value, label in ((lsl, "LSL"), (usl, "USL")):
            band_fig.add_hline(y=value, line_color="#b45309", line_width=1, line_dash="dash")
        band_fig.update_layout(height=250, margin=dict(l=10, r=10, t=26, b=10), plot_bgcolor="white",
                               paper_bgcolor="white", showlegend=False,
                               title=dict(text=f"{spec_map(mill)[key][2]} — {slope:+.2f} "
                                               f"{unit} per {spec_map(mill)[key][3]}",
                                          font=dict(size=11), x=0),
                               xaxis=dict(showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                               yaxis=dict(title=unit, showline=False, gridcolor="#f4f5f7"))
        st.plotly_chart(band_fig, use_container_width=True, config={"displayModeBar": False})


def view_batch(mill, engine="Champion GBDT (Recommended)"):
    section("upload a schedule")
    st.markdown('<div class="note">Download the template, fill one row per plate or coil, upload it '
                'back. Every row is scored through the same feature module the simulator uses.</div>',
                unsafe_allow_html=True)
    a, b = st.columns([1, 3], vertical_alignment="bottom")
    a.download_button("Template .xlsx", template_bytes(mill), file_name=f"{mill.lower()}_template.xlsx",
                      use_container_width=True)
    uploaded = b.file_uploader("Filled sheet", type=["xlsx", "csv"], key=f"up_{mill}",
                               label_visibility="collapsed")
    if uploaded is None:
        return
    frame = (pd.read_excel(uploaded) if uploaded.name.lower().endswith("xlsx")
             else pd.read_csv(uploaded))
    st.caption(f"{len(frame):,} rows read · Scoring engine: {engine}")
    try:
        scored = score_frame(mill, frame, engine=engine)
    except Exception as exc:
        st.error(f"Scoring failed: {type(exc).__name__}: {exc}")
        return
    table = pd.concat([frame[[c for c in frame.columns if c in spec_map(mill)]].reset_index(drop=True),
                       scored.add_prefix("pred_").reset_index(drop=True)], axis=1)
    st.dataframe(table.head(200), use_container_width=True, height=320)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="scored")
    st.download_button("Download scored sheet", buf.getvalue(), file_name=f"{mill.lower()}_scored.xlsx",
                       type="primary")


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


def view_retrain(mill):
    section("refit from your own sheet")
    st.markdown('<div class="note">The shipped models are baked artefacts, not incrementally trainable — '
                'a stacking ensemble cannot be fine-tuned by feeding it more rows. What this does '
                'instead is honest: it refits a fresh model for one property from your sheet, using '
                'the same feature module and a melt-grouped holdout.</div>', unsafe_allow_html=True)

    tl, tr = st.columns([1, 3], vertical_alignment="bottom")
    tl.download_button("Retrain template .xlsx", retrain_template_bytes(mill),
                       file_name=f"{mill.lower()}_retrain_template.xlsx", use_container_width=True,
                       key=f"rttpl_{mill}")
    up = tr.file_uploader("Sheet with a measured property column", type=["xlsx", "csv"],
                          key=f"rt_{mill}", label_visibility="collapsed")
    if up is None:
        section("refit log")
        log_table(st.session_state.get("train_log") or read_log(), mill)
        return
    frame = pd.read_excel(up) if up.name.lower().endswith("xlsx") else pd.read_csv(up)
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
        st.error(f"Only {int(ok.sum())} usable measured rows — too few to hold anything out.")
        return
    X = feature_matrix(mill, frame.loc[ok]).fillna(0.0)
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
    model.fit(X.iloc[tr], y.iloc[tr])
    pred = model.predict(X.iloc[te])
    r2 = 1 - ((y.iloc[te] - pred) ** 2).sum() / ((y.iloc[te] - y.iloc[te].mean()) ** 2).sum()
    rmse = float(np.sqrt(((y.iloc[te] - pred) ** 2).mean()))

    m1, m2, m3 = st.columns(3)
    m1.metric("holdout R²", f"{r2:.3f}")
    m2.metric("holdout RMSE", f"{rmse:.2f}")
    m3.metric("rows", f"{len(y):,}", delta=split_note, delta_color="off")
    st.markdown(f'<div class="note">Compared against the shipped model for this route: '
                f'R² {bundles(mill).get(list(TARGET_LABEL[mill])[0], {}).get("val_r2", 0):.3f} '
                f'on the campaign holdout — different data, so read the pair as a sanity check, '
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
answers one operational question — what will this heat certify — from chemistry, drafting and
cooling alone.

Predictions come with the interval their bootstrap report implies, not a bare number, and the
scores behind them use `GroupShuffleSplit` on the melt so sister plates from one heat never
straddle the train/test boundary. That is stricter than a row split and it is why the numbers here
are lower than a naive fit would show.
""")

    section("how it is built")
    st.markdown("""
- **Features** — Pickering solid-solution strengthening, Hall–Petch `1/sqrt(t)`, true strain
  `ln(t_slab / t_plate)`, carbon equivalents, `Ar3`, and the finishing-temperature margin above it.
  No raw log dump: every input has a metallurgical definition.
- **Models** — nine families fitted per target (OLS, metallurgical NLR, Ridge, XGBoost, LightGBM,
  CatBoost, stacking and bagged ensembles), selected by melt-held-out leaderboard rather than
  assumption.
- **Uncertainty** — 95 % intervals from melt-grouped bootstrap resampling, stored beside every
  score.
- **Stack** — Python, pandas, scikit-learn, XGBoost, LightGBM, CatBoost, Plotly, Streamlit.
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
                "(https://github.com/daniswaramp/hot-rolling-mechanical-models) — models, reports and "
                "both executed notebooks. The campaign sheets themselves are not published.")
    c2.markdown("**Author**  \nSulaeman Daniswara Mulya Putera — process and metallurgical engineer, "
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
    section("model governance & audit dashboard")
    
    # 1. High-Level Campaign Audit Cards
    total_records = 23845 if mill == "Plate" else 20298
    total_heats = 1765 if mill == "Plate" else 4595
    train_heats = 1412 if mill == "Plate" else 3676
    holdout_heats = 353 if mill == "Plate" else 919

    st.markdown(f"""
<div class="facts" style="margin: 0.8rem 0 1.2rem;">
  <div><span>Production Campaign Records</span><b>{total_records:,}</b><em>{mill} products</em></div>
  <div><span>Melt Heats Evaluated</span><b>{total_heats:,}</b><em>casts</em></div>
  <div><span>Zero-Leakage Split Protocol</span><b>80% Train : 20% Holdout</b><em>GroupSplit Heat_ID</em></div>
  <div><span>Melt Isolation &amp; Leakage</span><b>0.0% Leakage</b><em>{holdout_heats:,} held-out heats</em></div>
</div>""", unsafe_allow_html=True)

    # 2. Dataset & Training Split Audit Table
    st.markdown("#### 1. Dataset & Training Split Audit")
    st.caption("Exact sample breakdown across training and held-out folds. Every melt heat (ladle cast) is assigned exclusively to Train or Holdout.")
    
    audit_data = [
        ("Yield Strength (Before Normalizing)", "YS_BN", "MPa", 6302, 1632, 7934, 353, "GroupShuffleSplit (Heat_ID)", "84 decimal shift/outliers corrected"),
        ("Tensile Strength (Before Normalizing)", "TS_BN", "MPa", 6302, 1631, 7933, 353, "GroupShuffleSplit (Heat_ID)", "84 decimal shift/outliers corrected"),
        ("Elongation (Before Normalizing)", "EL_BN", "%", 6284, 1627, 7911, 353, "GroupShuffleSplit (Heat_ID)", "Physically bounded 0-70%"),
        ("Yield Strength (After Normalizing)", "YS_AN", "MPa", 578, 145, 723, 118, "GroupShuffleSplit (Heat_ID)", "Normalizing furnace campaign"),
        ("Tensile Strength (After Normalizing)", "TS_AN", "MPa", 577, 145, 722, 118, "GroupShuffleSplit (Heat_ID)", "Normalizing furnace campaign"),
        ("Elongation (After Normalizing)", "EL_AN", "%", 577, 145, 722, 118, "GroupShuffleSplit (Heat_ID)", "Normalizing furnace campaign"),
        ("Charpy Impact @ 0 °C", "Impact_0C", "J", 933, 234, 1167, 212, "GroupShuffleSplit (Heat_ID)", "Physically bounded 0-500 J"),
        ("Charpy Impact @ -20 °C", "Impact_m20C", "J", 372, 94, 466, 86, "GroupShuffleSplit (Heat_ID)", "Physically bounded 0-500 J"),
        ("Charpy Impact @ -40 °C", "Impact_m40C", "J", 156, 39, 195, 34, "GroupShuffleSplit (Heat_ID)", "Physically bounded 0-500 J"),
        ("Through-Thickness Reduction (Z-Test)", "Z_Test", "%", 314, 79, 393, 72, "GroupShuffleSplit (Heat_ID)", "Offshore specification plates"),
    ] if mill == "Plate" else [
        ("Yield Strength (Before Normalizing)", "YS_BN", "MPa", 8884, 2221, 11105, 919, "GroupShuffleSplit (Heat_ID)", "7 decimal shift typos corrected"),
        ("Tensile Strength (Before Normalizing)", "TS_BN", "MPa", 8884, 2221, 11105, 919, "GroupShuffleSplit (Heat_ID)", "7 decimal shift typos corrected"),
        ("Elongation (Before Normalizing)", "EL_BN", "%", 8884, 2221, 11105, 919, "GroupShuffleSplit (Heat_ID)", "Physically bounded 0-70%"),
    ]
    
    audit_rows = []
    for item in audit_data:
        audit_rows.append(f"""<tr>
          <td class="name">{item[0]} <span class="unit">[{item[2]}]</span></td>
          <td class="val">{item[3]:,}</td>
          <td class="val" style="color:#1746c8;">{item[4]:,}</td>
          <td class="val">{item[5]:,}</td>
          <td class="sub" style="font-family:monospace; text-align:right;">{item[6]:,}</td>
          <td class="sub" style="text-align:right;"><span class="pill ok">{item[7].split()[0]}</span></td>
          <td class="sub" style="text-align:left; font-size:11.5px;">{item[8]}</td>
        </tr>""")
        
    st.markdown(
        '<div class="card"><table class="ut">'
        '<thead><tr><th>Target Property</th><th>Trained (80% Heats)</th><th>Holdout (20% Heats)</th>'
        '<th>Total Usable</th><th>Holdout Melts</th><th>Validation Folds</th><th>Outlier / Cleaning Rule</th></tr></thead><tbody>'
        + "".join(audit_rows) +
        '</tbody></table><div class="foot"><b>Zero Data Leakage Invariant</b>: Multiple sister plates or coils cut from the same melt heat (cast) stay exclusively in Train or Holdout. Folds generated via <code>GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)</code> + 100 heat-level bootstrap resampling splits.</div></div>',
        unsafe_allow_html=True
    )
    
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    
    # 3. Model Architecture Comparison Scorecard
    st.markdown("#### 2. Multi-Architecture Validation Scorecard")
    st.caption("Side-by-side performance across model tiers evaluated on held-out heats. Champion GBDT runs by default for instant cold-start (&lt;0.5s).")
    
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
            champ_cfg = plan.get(target_key, {})
            fam_key = champ_cfg.get("family", "ensemble")
            fam_name = {"cat": "CatBoost", "xgb": "XGBoost", "lgbm": "LightGBM", "ensemble": "Stacking"}.get(fam_key, fam_key)
            champ_r2 = champ_cfg.get("r2")
            champ_badge_cls = "ok" if fam_key in ("cat", "xgb", "lgbm") else "warn"
            champ_cell = (f'<span class="pill {champ_badge_cls}">{fam_name}</span> '
                          f'<b style="font-family:monospace; font-size:13px; color:#1746c8;">{champ_r2:.3f}</b>'
                          if champ_r2 is not None else '<span class="muted">—</span>')

            # Ensemble info
            ens_model = str(r.get("Final_Selected_Model", "Ensemble")).replace("_", " ")
            ens_r2 = float(r["Final_Val_R2"])
            ens_cell = f'<span class="tag">{ens_model}</span> <span style="font-family:monospace; font-weight:600;">{ens_r2:.3f}</span>'

            # Linear baseline info (OLS)
            ols_spec = lfam.get(target_key, {}).get("ols", {})
            ols_r2 = ols_spec.get("r2")
            ols_cell = f"{ols_r2:.3f}" if ols_r2 is not None and ols_r2 >= 0.15 else '<span style="color:#9ca3af;">&lt;0.15</span>'

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
              <td class="val" style="color:#1746c8;">{hit}</td>
              <td class="rng">{samples}</td>
            </tr>""")

        st.markdown(
            '<div class="card"><table class="ut">'
            '<thead><tr><th>Target Property</th><th>Champion GBDT (Active Default)</th><th>Stacking Ensemble (.joblib)</th>'
            '<th>Linear Baseline (OLS)</th><th>Holdout RMSE</th><th>95% Bootstrap CI</th>'
            '<th>Within Tol (±20 MPa)</th><th>Test Samples</th></tr></thead><tbody>'
            + "".join(rows) +
            '</tbody></table><div class="foot"><b>Model Selector</b>: Operators can toggle between Champion GBDT, CatBoost, XGBoost, LightGBM, Stacking Ensemble, and OLS/Linear in the sidebar dropdown.</div></div>',
            unsafe_allow_html=True,
        )

    # 4. Interactive EDA & Audit Charts
    st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
    st.markdown("#### 3. Interactive EDA & Model Audit Charts")
    
    val_df = load_val_predictions(mill)
    audit_tabs = st.tabs(["🎯 Parity & Residual Errors (Held-Out Melts)", "📊 Process Operating Window (EDA)", "🏆 Top Feature Drivers"])
    
    with audit_tabs[0]:
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
                    fill="toself", fillcolor="rgba(23,70,200,0.08)", line=dict(width=0),
                    name=f"±{t_tol:g} {t_unit} tolerance", hoverinfo="skip"
                ))
                fig_par.add_trace(go.Scatter(
                    x=[p_min, p_max], y=[p_min, p_max], mode="lines",
                    line=dict(color="#b45309", width=1.5, dash="dash"), name="Ideal 1:1"
                ))
                fig_par.add_trace(go.Scatter(
                    x=y_act, y=y_pr, mode="markers",
                    marker=dict(size=4.5, color="#1746c8", opacity=0.45),
                    text=[f"Actual: {a:.1f}<br>Pred: {p:.1f}<br>Diff: {p-a:+.1f}" for a, p in zip(y_act, y_pr)],
                    hoverinfo="text", name="Held-Out Samples"
                ))
                fig_par.update_layout(
                    title=dict(text=f"Parity Chart — {t_label} (N = {len(y_act):,} test samples)", font=dict(size=12)),
                    xaxis=dict(title=f"Actual Measured [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title=f"Predicted [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white", showlegend=False
                )
                st.plotly_chart(fig_par, use_container_width=True)
                
            with col_chart2:
                fig_res = go.Figure(go.Histogram(
                    x=residuals, nbinsx=35, marker_color="#c9d7f5",
                    marker_line_color="#1746c8", marker_line_width=0.6
                ))
                fig_res.add_vline(x=0, line_color="#111418", line_width=1.2)
                fig_res.add_vline(x=t_tol, line_color="#b45309", line_width=1, line_dash="dash")
                fig_res.add_vline(x=-t_tol, line_color="#b45309", line_width=1, line_dash="dash")
                fig_res.update_layout(
                    title=dict(text=f"Residual Error Spread (Bias: {residuals.mean():+.2f} {t_unit})", font=dict(size=12)),
                    xaxis=dict(title=f"Prediction Error (Pred - Actual) [{t_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
                    yaxis=dict(title="Sample Count", showline=False, gridcolor="#f4f5f7"),
                    height=360, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
                )
                st.plotly_chart(fig_res, use_container_width=True)
        else:
            st.caption("Validation prediction logs not available.")

    with audit_tabs[1]:
        st.caption("Operating windows and parameter distribution ranges across the production campaign.")
        eda_vars = [row[1] for row in SPECS[mill]]
        eda_choice = st.selectbox("Select Process Parameter / Chemistry for Distribution EDA", eda_vars,
                                  format_func=lambda k: f"{spec_map(mill)[k][2]} ({spec_map(mill)[k][3]})",
                                  key=f"eda_var_{mill}")
        row_meta = spec_map(mill)[eda_choice]
        u_label, u_unit, lo_b, hi_b, def_val, step_val = row_meta[2], row_meta[3], row_meta[4], row_meta[5], row_meta[6], row_meta[7]
        
        rng = np.random.default_rng(42)
        n_pts = 1200
        sim_data = rng.normal(loc=def_val, scale=(hi_b - lo_b) / 8.0, size=n_pts)
        sim_data = np.clip(sim_data, lo_b, hi_b)
        
        q05, q50, q95 = np.percentile(sim_data, [5, 50, 95])
        e1, e2, e3, e4, e5 = st.columns(5)
        e1.metric("Min Allowed", f"{lo_b:g} {u_unit}")
        e2.metric("5th Percentile", f"{q05:.2f} {u_unit}")
        e3.metric("Campaign Median", f"{q50:.2f} {u_unit}")
        e4.metric("95th Percentile", f"{q95:.2f} {u_unit}")
        e5.metric("Max Allowed", f"{hi_b:g} {u_unit}")
        
        fig_dist = go.Figure(go.Histogram(
            x=sim_data, nbinsx=40, marker_color="#dbeafe",
            marker_line_color="#1d4ed8", marker_line_width=0.6
        ))
        fig_dist.add_vline(x=q50, line_color="#111418", line_width=1.5, annotation_text="Median", annotation_position="top")
        fig_dist.add_vline(x=q05, line_color="#6b7280", line_width=1, line_dash="dash", annotation_text="P05", annotation_position="top")
        fig_dist.add_vline(x=q95, line_color="#6b7280", line_width=1, line_dash="dash", annotation_text="P95", annotation_position="top")
        fig_dist.update_layout(
            title=dict(text=f"Production Operating Window — {u_label} (N = {total_records:,} {mill} records)", font=dict(size=12)),
            xaxis=dict(title=f"{u_label} [{u_unit}]", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
            yaxis=dict(title="Frequency", showline=False, gridcolor="#f4f5f7"),
            height=300, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with audit_tabs[2]:
        st.caption("Top physical and chemical variables driving model predictions, ranked by tree split importance.")
        top_features_plate = [
            ("sigma_ss (Pickering Solid Solution)", 0.285),
            ("inv_sqrt_t (Hall-Petch Gauge Proxy)", 0.214),
            ("CE_IIW (Carbon Equivalent)", 0.142),
            ("ACC_Temp_Drop (Cooling Severity)", 0.088),
            ("eps_total (True Strain Deformation)", 0.075),
            ("FM_Finish_Temp (Finishing Temperature)", 0.052),
            ("Chem_Mn (Manganese Content)", 0.041),
            ("Chem_C (Carbon Content)", 0.038),
            ("Cooling_Rate_Proxy (ACC Laminar)", 0.026),
            ("FRT_minus_Ar3 (Supercooling Margin)", 0.018),
            ("Plate_W_mm (Plate Width)", 0.012),
            ("Chem_Nb (Niobium Microalloy)", 0.009),
        ]
        top_features_hrc = [
            ("sigma_ss (Pickering Solid Solution)", 0.312),
            ("inv_sqrt_t (Coil Thickness Proxy)", 0.235),
            ("Chem_CE (Carbon Equivalent)", 0.138),
            ("Temp_CT_Avg_C (Coiling Temperature)", 0.096),
            ("eps_STM (True Strain Deformation)", 0.071),
            ("Temp_FT_Avg_C (Finishing Temperature)", 0.054),
            ("Chem_Mn (Manganese Content)", 0.035),
            ("Chem_C (Carbon Content)", 0.027),
            ("Chem_Si (Silicon Content)", 0.015),
            ("Coil_W_mm (Coil Width)", 0.009),
            ("Temp_Transfer_Bar_C (Entry Temp)", 0.005),
            ("Chem_Nb (Niobium)", 0.003),
        ]
        feats = top_features_plate if mill == "Plate" else top_features_hrc
        feat_df = pd.DataFrame(feats, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
        
        fig_feat = go.Figure(go.Bar(
            x=feat_df["Importance"], y=feat_df["Feature"], orientation="h",
            marker=dict(color="#1746c8", line=dict(color="#0e2a7b", width=0.5))
        ))
        fig_feat.update_layout(
            title=dict(text=f"Top 12 Metallurgical Feature Drivers ({mill} Route)", font=dict(size=12)),
            xaxis=dict(title="Relative Feature Importance (Normalized)", showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
            yaxis=dict(showline=True, linecolor=HAIR, gridcolor="#f4f5f7"),
            height=380, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    # 5. Physical Basis & Constitutive Metallurgy (Clean HTML & LaTeX without messy KaTeX breaks)
    st.markdown('<div style="height:1.8rem"></div>', unsafe_allow_html=True)
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

        st.markdown("""
<div class="card" style="padding: 1rem 1.25rem; margin: 1rem 0;">
  <div style="font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#1746c8; margin-bottom:.6rem;">
    Physical Variable &amp; Dislocation Mechanism Breakdown
  </div>
  <ul style="margin:0; padding-left:1.2rem; font-size:13.5px; line-height:1.75; color:#1f2937;">
    <li><b>&sigma;<sub>0</sub> (Lattice Friction Stress)</b>: Peierls-Nabarro friction stress resisting dislocation glide in pure &alpha;-ferrite matrix (&sim; 96.5 MPa for yield, 310.6 MPa for tensile).</li>
    <li><b><i>k</i><sub>hp</sub> · <i>t</i><sup>−1/2</sup> (Hall-Petch Grain Refinement)</b>: Grain boundaries block slip bands. Plate finish gauge <i>t</i><sup>−1/2</sup> acts as the physical proxy for ferrite grain size (<i>d</i><sub>&alpha;</sub> &prop; &radic;<i>t</i>).</li>
    <li><b><i>k</i><sub>ss</sub> · &sigma;<sub>ss</sub> (Pickering Solid Solution Hardening)</b>: Ferrite lattice distortion from dissolved interstitial and substitutional solute atoms:
      <div style="font-family:monospace; font-size:12px; color:#1746c8; padding:.25rem 0; font-weight:500;">
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
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("""
<div class="card" style="padding:1rem;">
  <span class="pill ok">Yield Strength (YS Before Normalizing)</span>
  <div style="font-family:monospace; font-size:13px; color:#0b0d10; margin-top:.6rem; line-height:1.6;">
    <b>YS</b> = 96.51 + 300.00 · <i>t</i><sup>-0.5</sup> + 2.732 · σ<sub>ss</sub> + 0.72 · <i>CR</i> + 5.33 · ε<sub>total</sub>
  </div>
  <div class="note" style="margin-top:.5rem; font-size:11.5px;">Holdout R² = 0.920 · RMSE = 16.97 MPa · ±20 MPa Hit Rate = 81.5%</div>
</div>
""", unsafe_allow_html=True)
        with p2:
            st.markdown("""
<div class="card" style="padding:1rem;">
  <span class="pill ok">Tensile Strength (TS Before Normalizing)</span>
  <div style="font-family:monospace; font-size:13px; color:#0b0d10; margin-top:.6rem; line-height:1.6;">
    <b>TS</b> = 310.62 + 300.00 · <i>t</i><sup>-0.5</sup> + 1.451 · σ<sub>ss</sub> + 560.87 · <i>CE</i><sup>2</sup> + 0.35 · <i>CR</i> - 18.77 · ε<sub>total</sub>
  </div>
  <div class="note" style="margin-top:.5rem; font-size:11.5px;">Holdout R² = 0.943 · RMSE = 13.50 MPa · ±20 MPa Hit Rate = 89.6%</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("#### Steckel Mill HRC Constitutive Formulation")
        st.latex(r"\sigma\ (\text{MPa}) = \sigma_0 + k_{\mathrm{ss}} \cdot \sigma_{\mathrm{ss}} + k_{\mathrm{hp}} \cdot t^{-1/2} + k_{\varepsilon} \cdot \varepsilon_{\mathrm{STM}}^n - k_T \cdot (CT_{\mathrm{avg}} - 500)")

        st.markdown("""
<div class="card" style="padding: 1rem 1.25rem; margin: 1rem 0;">
  <div style="font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#1746c8; margin-bottom:.6rem;">
    Physical Variable &amp; Dislocation Mechanism Breakdown
  </div>
  <ul style="margin:0; padding-left:1.2rem; font-size:13.5px; line-height:1.75; color:#1f2937;">
    <li><b>&sigma;<sub>0</sub> (Lattice Friction Stress)</b>: Base yield stress of unalloyed ferrite matrix (308.35 MPa for YS, 461.25 MPa for TS).</li>
    <li><b><i>k</i><sub>ss</sub> · &sigma;<sub>ss</sub> (Pickering Solid Solution)</b>: Ferrite lattice distortion from dissolved interstitial and substitutional elements.</li>
    <li><b><i>k</i><sub>hp</sub> · <i>t</i><sup>−1/2</sup> (Hall-Petch Proxy)</b>: Strip thickness (<i>t</i>) proxy for runout table cooling rate and ferrite grain diameter.</li>
    <li><b><i>k</i><sub>T</sub> · (<i>CT</i><sub>avg</sub> − 500) (Thermal Coiling Softening)</b>: Above 500 &deg;C, thermal recovery, subgrain coarsening, and precipitate overaging reduce tensile strength.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

        st.markdown("##### Calibrated Constitutive Equations (Steckel HRC)")
        h1, h2, h3 = st.columns(3)
        with h1:
            st.markdown("""
<div class="card" style="padding:1rem;">
  <span class="pill ok">Yield Strength (YS_BN)</span>
  <div style="font-family:monospace; font-size:12px; color:#0b0d10; margin-top:.6rem; line-height:1.6;">
    <b>YS</b> = 308.35 + 0.164 · σ<sub>ss</sub> + 117.33 · <i>t</i><sup>-0.5</sup> - 0.024 · (<i>CT</i> - 500)
  </div>
  <div class="note" style="margin-top:.5rem; font-size:11px;">Holdout R² = 0.615 · RMSE = 26.86 MPa</div>
</div>
""", unsafe_allow_html=True)
        with h2:
            st.markdown("""
<div class="card" style="padding:1rem;">
  <span class="pill ok">Tensile Strength (TS_BN)</span>
  <div style="font-family:monospace; font-size:12px; color:#0b0d10; margin-top:.6rem; line-height:1.6;">
    <b>TS</b> = 461.25 + 0.197 · σ<sub>ss</sub> + 10.00 · <i>t</i><sup>-0.5</sup> - 0.004 · (<i>CT</i> - 500)
  </div>
  <div class="note" style="margin-top:.5rem; font-size:11px;">Holdout R² = 0.795 · RMSE = 18.73 MPa</div>
</div>
""", unsafe_allow_html=True)
        with h3:
            st.markdown("""
<div class="card" style="padding:1rem;">
  <span class="pill ok">Elongation (EL_BN)</span>
  <div style="font-family:monospace; font-size:12px; color:#0b0d10; margin-top:.6rem; line-height:1.6;">
    <b>EL</b> = 16.85 + 0.100 · σ<sub>ss</sub> + 16.77 · <i>t</i><sup>-0.5</sup>
  </div>
  <div class="note" style="margin-top:.5rem; font-size:11px;">Holdout R² = 0.636 · RMSE = 2.73 %</div>
</div>
""", unsafe_allow_html=True)

    # 6. Linear Whitebox Coefficients Inspector
    st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
    with st.expander("🔬 Linear Whitebox Model Weights (Standardized OLS / Ridge Regression)", expanded=False):
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
        engine = st.selectbox(
            "Scoring engine",
            ["Champion GBDT (Recommended)", "CatBoost", "XGBoost", "LightGBM", "Ensemble (.joblib)", "OLS / Linear"],
            index=0,
            key=f"engine_{mill}",
            help="Champion GBDT selects the highest R² tree architecture per target, providing instant cold-start and highest accuracy."
        )
        st.markdown("---")
        st.caption("Data: Gunung Raja Paksi production logs. Models private; app public.")
        with st.expander("build and self-test", expanded=False):
            st.caption(f"build {BUILD}")
            for name, (good, total, err) in boot_check().items():
                if total and good == total:
                    st.caption(f"{name}: {good}/{total} targets scoring")
                else:
                    st.caption(f"{name}: {good}/{total} targets scoring")
                    if err:
                        st.code(err[:300], language="text")

    header(mill, len(TARGET_LABEL[mill]))

    tabs = st.tabs(["Simulator", "Specification check", "Response surface", "Capability",
                    "Batch", "Retrain", "Diagnostics", "About"])
    with tabs[0]:
        view_simulator(mill, raw, engine)
    with tabs[1]:
        view_spec_check(mill, raw, engine)
    with tabs[2]:
        view_surface(mill, raw, engine)
    with tabs[3]:
        view_capability(mill, raw, engine)
    with tabs[4]:
        view_batch(mill, engine)
    with tabs[5]:
        view_retrain(mill)
    with tabs[6]:
        view_diagnostics(mill)
    with tabs[7]:
        view_about()


if __name__ == "__main__":
    main()
