"""Dual-mill mechanical property platform — fresh technical white UI.

Heavy plate and Steckel mill coil, one model layer. Sections: simulator, response surface, batch,
retrain, diagnostics.
"""

import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import model_loader as ml  # noqa: E402

st.set_page_config(
    page_title="Dual-mill property platform",
    page_icon="▦",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#0b0d10"
MUTED = "#6b7280"
HAIR = "#e5e7eb"
ACCENT = "#1746c8"
PANEL = "#f7f8fa"


# ----------------------------------------------------------------------------- input specs
PLATE_SPEC = [
    ("Geometry", "Slab_T_mm", "Slab thickness", "mm", 120.0, 320.0, 218.0, 1.0),
    ("Geometry", "Slab_W_mm", "Slab width", "mm", 900.0, 2300.0, 1515.0, 5.0),
    ("Geometry", "Plate_T_mm", "Plate thickness", "mm", 6.0, 120.0, 16.0, 0.5),
    ("Geometry", "Plate_W_mm", "Plate width", "mm", 1000.0, 4000.0, 1889.0, 10.0),
    ("Geometry", "TB_Thickness_mm", "Transfer bar thickness", "mm", 15.0, 80.0, 41.0, 0.5),
    ("Finishing mill", "FM_Temp_Before_Rolling", "Entry temperature", "C", 850.0, 1150.0, 985.0, 1.0),
    ("Finishing mill", "FM_Finish_Temp", "Finishing temperature", "C", 700.0, 1050.0, 832.0, 1.0),
    ("Finishing mill", "FM_Number_of_Passes", "Number of passes", "count", 3.0, 15.0, 7.0, 1.0),
    ("Finishing mill", "FM_Last_Pass_Thick", "Last pass entry gauge", "mm", 8.0, 60.0, 18.0, 0.5),
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
    ("Chemistry", "Chem_B", "B", "wt%", 0.0, 0.005, 0.0, 0.0002),
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

TARGET_LABEL = {
    "Plate": {"YS_BN": ("yield, as rolled", "MPa"), "TS_BN": ("tensile, as rolled", "MPa"),
              "EL_BN": ("elongation, as rolled", "%"), "YS_AN": ("yield, normalised", "MPa"),
              "TS_AN": ("tensile, normalised", "MPa"), "EL_AN": ("elongation, normalised", "%"),
              "Impact_0C": ("Charpy 0 C", "J"), "Impact_m20C": ("Charpy -20 C", "J"),
              "Impact_m40C": ("Charpy -40 C", "J"), "Z_Test": ("Z-test", "%")},
    "HRC": {"YS_BN": ("yield", "MPa"), "TS_BN": ("tensile", "MPa"), "EL_BN": ("elongation", "%")},
}


# ----------------------------------------------------------------------------- styling
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background: #ffffff; }
.block-container { max-width: 1320px; padding-top: 2.2rem; padding-bottom: 3rem; }

h1, h2, h3, h4 { font-family: 'Inter', sans-serif; color: #0b0d10; letter-spacing: -.02em; }

.kicker { font-size: 10.5px; letter-spacing: .18em; text-transform: uppercase;
          color: #1746c8; font-weight: 600; }
.title { font-size: 40px; line-height: 1.05; font-weight: 600; margin: .35rem 0 .5rem;
         letter-spacing: -.035em; }
.lede { font-size: 14.5px; color: #4b5563; max-width: 62ch; line-height: 1.55; }
.hair { height: 1px; background: #e5e7eb; margin: 1.4rem 0 1.2rem; border: 0; }

.section { font-size: 10.5px; letter-spacing: .16em; text-transform: uppercase; color: #6b7280;
           font-weight: 600; display: flex; align-items: center; gap: .7rem; margin-bottom: .7rem; }
.section i { flex: 1; height: 1px; background: #e5e7eb; display: block; }

table.mt { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
table.mt th { font-size: 9.8px; letter-spacing: .12em; text-transform: uppercase; color: #6b7280;
              font-weight: 600; text-align: right; padding: 0 0 .55rem; border-bottom: 1px solid #e5e7eb; }
table.mt th:first-child { text-align: left; }
table.mt td { font-size: 13.5px; padding: .52rem 0; border-bottom: 1px solid #f1f2f4; text-align: right;
              color: #0b0d10; }
table.mt td:first-child { text-align: left; font-weight: 500; }
table.mt td.mono, table.mt th.mono { font-family: 'JetBrains Mono', monospace; }
table.mt td .unit { color: #9ca3af; font-size: 11px; margin-left: .25rem; }
table.mt td .ci { color: #6b7280; font-size: 11.5px; }
.tag { display: inline-block; font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
       padding: .18rem .45rem; border: 1px solid #d1d5db; color: #374151; border-radius: 2px; }

.facts { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0; border: 1px solid #e5e7eb; }
.facts div { padding: .8rem .9rem; border-right: 1px solid #e5e7eb; }
.facts div:last-child { border-right: 0; }
.facts span { display: block; font-size: 9.6px; letter-spacing: .12em; text-transform: uppercase;
              color: #6b7280; margin-bottom: .28rem; }
.facts b { font-family: 'JetBrains Mono', monospace; font-size: 16px; font-weight: 600; }
.facts em { font-style: normal; font-size: 10.5px; color: #9ca3af; margin-left: .2rem; }

.note { font-size: 12px; color: #6b7280; line-height: 1.5; border-left: 2px solid #e5e7eb;
        padding-left: .7rem; margin-top: .6rem; }

div[data-testid="stSidebar"] { background: #fbfbfc; border-right: 1px solid #e5e7eb; }
div[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
div[data-testid="stSidebar"] h2 { font-size: 13px; letter-spacing: .1em; text-transform: uppercase; }

.stTabs [data-baseweb="tab-list"] { gap: 1.4rem; border-bottom: 1px solid #e5e7eb; }
.stTabs [data-baseweb="tab"] { font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
                               color: #6b7280; padding: .5rem 0; }
.stTabs [aria-selected="true"] { color: #0b0d10 !important; font-weight: 600; }

div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 22px; }
#MainMenu, footer, [data-testid="stAppDeployButton"] { display: none; }
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
        out["FM_Last_Pass_Drafting"] = float(out.get("FM_Last_Pass_Thick", plate_t * 1.15)) - plate_t
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


def score_frame(mill, frame):
    """Vectorised scoring: features once, then every target."""
    frame = prepare_frame(mill, frame)
    X = feature_matrix(mill, frame)
    out = pd.DataFrame(index=frame.index)
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
    out = {}
    for key, bundle in bundles(mill).items():
        if bundle is None:
            continue
        value = np.nan
        for cols in candidate_columns(bundle, X):
            try:
                value = float(bundle["model"].predict(align(X, cols))[0])
                break
            except Exception:
                continue
        half = ml.TARGET_CONFIG[mill][key]["tol"] * 0.5
        if not ci.empty and "Target" in ci.columns:
            row = ci[ci["Target"] == key]
            if not row.empty and "Val_RMSE" in row.columns:
                half = 1.96 * (float(row["Val_RMSE"].values[0]) / np.sqrt(50))
        label, unit = TARGET_LABEL[mill].get(key, (key, ""))
        out[key] = {"name": label, "pred": round(value, 2) if np.isfinite(value) else float("nan"),
                    "unit": unit, "ci_low": round(value - half, 2) if np.isfinite(value) else float("nan"),
                    "ci_high": round(value + half, 2) if np.isfinite(value) else float("nan")}
    return out


def physical(raw, mill):
    merged = dict(raw)
    merged.setdefault("ACC_Headers_n", raw.get("ACC_Laminar_Header", 8.0))
    merged.setdefault("FM_Entry_Temp", raw.get("FM_Temp_Before_Rolling", 985.0))
    return ml.compute_physical_indicators(merged, mill)


def targets_table(mill, predictions):
    rows = []
    for key, res in predictions.items():
        label, unit = TARGET_LABEL[mill].get(key, (res["name"], res["unit"]))
        tol = ml.TARGET_CONFIG[mill][key]["tol"]
        inside = abs(res["pred"] - (res["ci_low"] + res["ci_high"]) / 2) <= tol
        rows.append(f"""<tr>
          <td>{label}<span style="color:#9ca3af;font-size:11px;margin-left:.4rem">{key}</span></td>
          <td class="mono">{res['pred']:.1f}<span class="unit">{unit}</span></td>
          <td class="mono ci">{res['ci_low']:.1f} – {res['ci_high']:.1f}</td>
          <td class="mono ci">±{tol:g}</td>
          <td><span class="tag">{'in tolerance' if inside else 'review'}</span></td>
        </tr>""")
    st.markdown(
        """<table class="mt"><thead><tr><th>property</th><th class="mono">predicted</th>
        <th class="mono">95% interval</th><th class="mono">tolerance</th><th></th></tr></thead>
        <tbody>""" + "".join(rows) + "</tbody></table>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- header
def header(mill, n_targets):
    route = ("Heavy plate · rougher, finisher, laminar ACC, optional normalising"
             if mill == "Plate" else
             "Steckel coil · rougher, reversing Steckel mill, runout table, downcoiler")
    st.markdown(f"""
<div class="kicker">Dual-mill property platform</div>
<div class="title">Mechanical properties, before the heat is rolled</div>
<div class="lede">{route}. {n_targets} certified properties, fitted on melt-grouped holdouts so
sister plates never straddle the split.</div>
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
                    values[key] = st.number_input(f"{label} [{unit}]", min_value=float(lo),
                                                  max_value=float(hi), value=float(default),
                                                  step=float(step), format=fmt, key=f"{mill}_{key}")
        if mill == "Plate":
            st.markdown('<div class="note">Delivery route</div>', unsafe_allow_html=True)
            values["Rolling_Type"] = st.selectbox(
                "Rolling type", ["AR", "TMCP", "N"], key=f"{mill}_rolling_type",
                help="AR = as rolled, TMCP = thermomechanically controlled, N = normalised delivery.")
        submitted = st.form_submit_button("Calculate", type="primary", width="stretch")
    if submitted or "schedule" not in st.session_state:
        st.session_state["schedule"] = dict(values)
        st.session_state["schedule_mill"] = mill
    if st.session_state.get("schedule_mill") != mill:
        st.session_state["schedule"] = dict(values)
        st.session_state["schedule_mill"] = mill
    return st.session_state["schedule"]


# ----------------------------------------------------------------------------- sections
def view_simulator(mill, raw):
    section("predicted properties")
    preds = predict_one(mill, raw)
    targets_table(mill, preds)

    ind = physical(raw, mill)
    st.markdown(f"""
<div class="facts">
  <div><span>Pickering σ_ss</span><b>{ind['sigma_ss']:.0f}</b><em>MPa</em></div>
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


def view_surface(mill, raw):
    left, mid, right = st.columns([1.2, 1.2, 1.6], vertical_alignment="bottom")
    keys = [k for k in SURFACE[mill] if k in spec_map(mill)]
    x_key = left.selectbox("Horizontal axis", keys, index=0, key=f"x_{mill}")
    y_key = mid.selectbox("Vertical axis", [k for k in keys if k != x_key], index=1, key=f"y_{mill}")
    target = right.selectbox("Property", list(TARGET_LABEL[mill]),
                             format_func=lambda k: TARGET_LABEL[mill][k][0], key=f"t_{mill}")

    c1, c2, c3 = st.columns([1.4, 1.4, 1.2], vertical_alignment="bottom")
    view = c1.radio("View", ["2D contour", "3D surface"], horizontal=True, key=f"v_{mill}")
    grid = c2.slider("Grid", 15, 41, 25, step=2, key=f"g_{mill}")
    c3.caption("The rest of the schedule is held at the sidebar values.")

    sx, sy = spec_map(mill)[x_key], spec_map(mill)[y_key]
    xs = np.linspace(sx[4], sx[5], grid)
    ys = np.linspace(sy[4], sy[5], grid)
    frame = pd.DataFrame([dict(raw) for _ in range(grid * grid)])
    frame[x_key] = np.repeat(xs, grid)
    frame[y_key] = np.tile(ys, grid)
    z = score_frame(mill, frame)[target].values.reshape(grid, grid)

    label, unit = TARGET_LABEL[mill][target]
    axes = dict(showline=True, linecolor=HAIR, gridcolor="#f1f2f4", zeroline=False,
                tickfont=dict(size=10, color=MUTED), title_font=dict(size=11, color=INK))
    here = predict_one(mill, raw)[target]["pred"]

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
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True})
    st.markdown(f'<div class="note">Surface held on the mill\'s own feature engineering, not a proxy: '
                f'every node is a full pass through the {mill} pipeline. The marker is the current '
                f'schedule at {here:.1f} {unit}.</div>', unsafe_allow_html=True)


def template_bytes(mill):
    cols = [row[1] for row in SPECS[mill]]
    blank = pd.DataFrame([{c: row[6] for c, row in zip(cols, SPECS[mill])}])
    buf = io.BytesIO()
    blank.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def view_batch(mill):
    section("upload a schedule")
    st.markdown('<div class="note">Download the template, fill one row per plate or coil, upload it '
                'back. Every row is scored through the same feature module the simulator uses.</div>',
                unsafe_allow_html=True)
    a, b = st.columns([1, 3], vertical_alignment="bottom")
    a.download_button("Template .xlsx", template_bytes(mill), file_name=f"{mill.lower()}_template.xlsx",
                      width="stretch")
    uploaded = b.file_uploader("Filled sheet", type=["xlsx", "csv"], key=f"up_{mill}",
                               label_visibility="collapsed")
    if uploaded is None:
        return
    frame = (pd.read_excel(uploaded) if uploaded.name.lower().endswith("xlsx")
             else pd.read_csv(uploaded))
    st.caption(f"{len(frame):,} rows read")
    try:
        scored = score_frame(mill, frame)
    except Exception as exc:
        st.error(f"Scoring failed: {type(exc).__name__}: {exc}")
        return
    table = pd.concat([frame[[c for c in frame.columns if c in spec_map(mill)]].reset_index(drop=True),
                       scored.add_prefix("pred_").reset_index(drop=True)], axis=1)
    st.dataframe(table.head(200), width="stretch", height=320)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="scored")
    st.download_button("Download scored sheet", buf.getvalue(), file_name=f"{mill.lower()}_scored.xlsx",
                       type="primary")


def view_retrain(mill):
    section("refit from your own sheet")
    st.markdown('<div class="note">The shipped models are baked artefacts, not incrementally trainable — '
                'a stacking ensemble cannot be fine-tuned by feeding it more rows. What this does '
                'instead is honest: it refits a fresh model for one property from your sheet, using '
                'the same feature module and a melt-grouped holdout.</div>', unsafe_allow_html=True)

    up = st.file_uploader("Sheet with a measured property column", type=["xlsx", "csv"], key=f"rt_{mill}")
    if up is None:
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
    ok = y.notna()
    if ok.sum() < 120:
        st.error(f"Only {int(ok.sum())} measured rows — too few to hold anything out.")
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
    st.download_button("Download fitted model (.joblib)", buf.getvalue(),
                       file_name=f"{mill.lower()}_{target}_refit.joblib")


def view_diagnostics(mill):
    section("leaderboard")
    lb = ml.get_leaderboard_data(mill)
    if not lb.empty:
        cols = [c for c in ("Target", "Model", "Val_R2", "Val_RMSE", "Final_Val_R2", "Final_Val_RMSE")
                if c in lb.columns]
        st.dataframe(lb[cols].sort_values(cols[2], ascending=False), width="stretch", height=280,
                     hide_index=True)
    else:
        st.caption("No leaderboard report found.")

    section("95% bootstrap intervals")
    ci = ml.get_ci_data(mill)
    if not ci.empty:
        fig = go.Figure()
        for row in ci.itertuples():
            fig.add_trace(go.Scatter(
                x=[row.R2_95_CI_Low, row.R2_95_CI_High], y=[row.Target, row.Target], mode="lines",
                line=dict(color=HAIR, width=6), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[row.Val_R2], y=[row.Target], mode="markers",
                                     marker=dict(size=9, color=ACCENT), showlegend=False,
                                     hovertemplate=f"{row.Target}<br>R² {row.Val_R2:.3f}<extra></extra>"))
        fig.update_layout(height=90 + 32 * len(ci), margin=dict(l=0, r=10, t=10, b=10),
                          plot_bgcolor="white",
                          xaxis=dict(title="held-out R²", showline=True, linecolor=HAIR,
                                     gridcolor="#f1f2f4", range=[0, 1]),
                          yaxis=dict(showline=False, tickfont=dict(size=10)))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    section("physical basis, as written")
    formulas = ml.get_nlr_formulas(mill)
    if formulas:
        st.code(formulas[:3000], language="text")


# ----------------------------------------------------------------------------- main
def main():
    inject_css()
    with st.sidebar:
        st.markdown("### Route")
        mill = st.radio("Route", ["Plate", "HRC"], horizontal=True, label_visibility="collapsed",
                        format_func=lambda m: "Heavy plate" if m == "Plate" else "Steckel coil")
        st.markdown("---")
        raw = schedule_inputs(mill)
        st.markdown("---")
        st.caption("Data: Gunung Raja Paksi production logs. Models private; app public.")

    header(mill, len(TARGET_LABEL[mill]))

    tabs = st.tabs(["Simulator", "Response surface", "Batch", "Retrain", "Diagnostics"])
    with tabs[0]:
        view_simulator(mill, raw)
    with tabs[1]:
        view_surface(mill, raw)
    with tabs[2]:
        view_batch(mill)
    with tabs[3]:
        view_retrain(mill)
    with tabs[4]:
        view_diagnostics(mill)


if __name__ == "__main__":
    main()
