"""Furnace Normalizing tab: 1D thermal cycle, precipitate dissolution/pinning & Modal-calibrated physics.

Provides a fast, zero-Modal-dispatch interactive normalizing simulator inside Streamlit:
- 1D through-thickness heating/cooling lag and effective core austenitic soak time
- Coupled Ti -> Nb -> Al -> V nitride-then-carbide dissolution, re-precipitation & JMAK kinetics
- Zener grain-boundary pinning with Ostwald coarsening and Sellars-Beynon ferrite grain size
- 2D (T_soak x residence_min) process window contours and grain-growth trajectories
- 4-way comparison (As-Rolled ML BN, Shipped ML AN, Modal-Tuned Physics AN, Hybrid Physics+ML),
  constitutive waterfall decomposition, sensitivity sweeps, and Modal Cloud calibration parity

Import invariant: NEVER import ``modal`` or ``streamlit_app`` at module level; resolve the
active app module lazily through ``app_ref.get_app_module()``.
"""

import html as html_lib
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import normalizing_physics as np_phys  # noqa: E402

# ----------------------------------------------------------------------------- palette & styling
INK = "#0b0d10"
ACCENT = "#1746c8"
HAIR = "#e5e7eb"
GRID = "#f1f2f4"
MUTED = "#6b7280"
SEMANTIC_OK = "#065f46"
SEMANTIC_WARN = "#92400e"
SEMANTIC_BAD = "#991b1b"
WARN_LINE = "#b45309"
WARN_STAR = "#f59e0b"

METAL_ORDER = ("Al", "Nb", "Ti", "V")
METAL_LABEL = {
    "Al": "AlN",
    "Nb": "Nb(C,N)",
    "Ti": "Ti(C,N)",
    "V": "V(C,N)",
}
METAL_COLOR = {
    "Al": WARN_LINE,
    "Nb": ACCENT,
    "Ti": SEMANTIC_BAD,
    "V": SEMANTIC_OK,
}

HEATING_PROFILES = {
    "Standard Roller Hearth (1.2 min/mm)": 1.2,
    "Fast Radiant Tube (0.9 min/mm)": 0.9,
    "Heavy Batch Furnace (1.8 min/mm)": 1.8,
}
COOLING_MODES = {
    "Still Air": "still_air",
    "Forced Air / Fan": "forced_air",
}

REGIME_PILL_STYLE = {
    "FULL_NORMALIZING": ("ok", SEMANTIC_OK, "#ecfdf5", "#a7f3d0"),
    "INTERCRITICAL": ("warn", SEMANTIC_WARN, "#fffbeb", "#fde68a"),
    "UNDER_SOAKED": ("warn", SEMANTIC_WARN, "#fffbeb", "#fde68a"),
    "OVERHEATED": ("bad", SEMANTIC_BAD, "#fef2f2", "#fecaca"),
    "SUBCRITICAL": ("bad", SEMANTIC_BAD, "#fef2f2", "#fecaca"),
}

TARGET_META = {
    "YS_AN": ("Yield Strength (YS)", "MPa", 20.0, "ys_phys_mpa", "ys_hybrid_mpa", "YS_BN"),
    "TS_AN": ("Tensile Strength (TS)", "MPa", 20.0, "ts_phys_mpa", "ts_hybrid_mpa", "TS_BN"),
    "EL_AN": ("Elongation (EL)", "%", 2.5, "el_phys_pct", "el_hybrid_pct", "EL_BN"),
}

WATERFALL_LABELS = [
    ("lattice_friction", "Base Lattice Friction (σ₀)"),
    ("solid_solution", "Solid Solution (σ_ss)"),
    ("hall_petch_grain", "Hall-Petch Grain (d_α)"),
    ("pearlite_ce", "Pearlite / CE & Cooling"),
    ("microalloy_ppt", "Microalloy Ppt (V / Nb)"),
    ("soak_regime_shift", "Soak / Regime Shift"),
]


def _app():
    """Resolve the active streamlit_app module lazily without circular imports."""
    import app_ref  # noqa: PLC0415

    return app_ref.get_app_module()


def _shade(hex_color, factor):
    """Lighter (factor > 0) or darker (factor < 0) step of a hex colour."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    if factor >= 0:
        r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    else:
        r, g, b = (round(c * (1 + factor)) for c in (r, g, b))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def _rgba(hex_color, alpha):
    """Plotly-friendly rgba() string of a hex colour token."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _section(label):
    st.markdown(f'<div class="section"><span>{label}</span><i></i></div>', unsafe_allow_html=True)


def _base_fig(title, x_title, y_title, height=390):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=12.5, color=INK), x=0, xanchor="left"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=54, r=20, t=46, b=46),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        height=height,
    )
    fig.update_xaxes(title_text=x_title, showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title_text=y_title, showline=True, linecolor=HAIR, gridcolor=GRID, zeroline=False)
    return fig


def _safe_ml_preds(mill, raw, engine):
    """Score the active schedule through streamlit_app without crashing on missing artifacts."""
    try:
        app_mod = _app()
        return app_mod.predict_selected_engine(mill, raw, engine) or {}
    except Exception:  # noqa: BLE001
        return {}


def _fmt_opt(val, fmt=".0f", suffix=""):
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    return f"{v:{fmt}}{suffix}"


def _regime_pill_html(regime):
    cls, fg, bg, bdr = REGIME_PILL_STYLE.get(regime, ("neutral", MUTED, "#f3f4f6", HAIR))
    return (
        f'<span class="pill {cls}" '
        f'style="background:{bg};color:{fg};border:1px solid {bdr};">{html_lib.escape(regime)}</span>'
    )


def _calib_cache_key(calib):
    return (
        str(calib.get("source", "")),
        str(calib.get("timestamp", "")),
        float(calib.get("YS_AN", {}).get("sigma_0", 0.0)),
        float(calib.get("TS_AN", {}).get("sigma_0", 0.0)),
        float(calib.get("EL_AN", {}).get("sigma_0", 0.0)),
    )


def _ml_subset_key(ml_preds):
    keys = ("YS_AN", "TS_AN", "EL_AN", "YS_BN", "TS_BN", "EL_BN", "Impact_0C")
    out = []
    for k in keys:
        entry = (ml_preds or {}).get(k)
        if isinstance(entry, dict):
            out.append((k, entry.get("pred"), entry.get("ci_low"), entry.get("ci_high")))
        else:
            out.append((k, entry, None, None))
    return tuple(out)


@st.cache_data(show_spinner=False, max_entries=16)
def _cached_process_window(raw_items, ml_key, calib_key, heat_rate, cool_mode):
    raw_dict = dict(raw_items)
    ml_dict = {
        k: {"pred": p, "ci_low": lo, "ci_high": hi}
        for k, p, lo, hi in ml_key
        if p is not None
    }
    calib = np_phys.load_calibration()
    return np_phys.normalizing_process_window(
        raw_dict,
        ml_preds=ml_dict,
        calib=calib,
        t_min=780.0,
        t_max=1020.0,
        r_min=10.0,
        r_max=180.0,
        nt=21,
        nr=21,
        heating_rate_min_per_mm=heat_rate,
        cooling_mode=cool_mode,
    )


# ----------------------------------------------------------------------------- KPI & banner
def _render_kpis_and_banner(mill, res, t_soak_c, residence_min):
    therm = res["thermal"]
    grain = res["grain"]
    hyb = res["hybrid"]
    ml_b = res["ml_baseline"]
    ac1 = res["ac1_c"]
    ac3 = res["ac3_c"]
    regime = res["regime"]

    superheat = t_soak_c - ac3
    t_eff = therm["t_eff_soak_min"]
    t_heat = therm["t_heat_min"]
    d_gamma = grain["d_gamma_final_um"]
    d_alpha = grain["d_alpha_um"]
    d_lim = grain["d_lim_um"]
    pin_ratio = grain["pinning_ratio"]

    pill_html = _regime_pill_html(regime)
    st.markdown(
        '<div class="facts">'
        f'<div><span>austenitization regime</span>{pill_html}'
        f'<br><em>{html_lib.escape(grain["regime_label"])}</em></div>'
        f'<div><span>Ac1 / Ac3 &amp; superheat</span><b>{superheat:+.0f} °C</b>'
        f'<em>Ac1 {ac1:.0f} · Ac3 {ac3:.0f} °C</em></div>'
        f'<div><span>core effective soak</span><b>{t_eff:.1f} min</b>'
        f'<em>heat-up {t_heat:.1f} min · peak {therm["t_core_peak_c"]:.0f} °C</em></div>'
        f'<div><span>grain d<sub>γ</sub> → d<sub>α</sub></span><b>{d_gamma:.1f} → {d_alpha:.1f} µm</b>'
        f'<em>Zener ratio {pin_ratio:.2f} (D<sub>lim</sub> {d_lim:.0f} µm)</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)

    ys_hyb = hyb["ys_hybrid_mpa"]
    ts_hyb = hyb["ts_hybrid_mpa"]
    el_hyb = hyb["el_hybrid_pct"]
    ys_ml_an = _fmt_opt(ml_b.get("YS_AN"), ".0f")
    ts_ml_an = _fmt_opt(ml_b.get("TS_AN"), ".0f")
    el_ml_an = _fmt_opt(ml_b.get("EL_AN"), ".1f", "%")
    ys_bn = _fmt_opt(ml_b.get("YS_BN"), ".0f")
    ts_bn = _fmt_opt(ml_b.get("TS_BN"), ".0f")
    el_bn = _fmt_opt(ml_b.get("EL_BN"), ".1f", "%")

    dbtt = res["dbtt_shift_c"]
    imp0_est = res.get("impact_0c_est_j")
    imp0_ml = ml_b.get("Impact_0C")
    if imp0_est is not None and imp0_ml is not None:
        imp_note = f"Charpy 0 °C ~{imp0_est:.0f} J (ML {imp0_ml:.0f} J)"
    else:
        imp_note = "Charpy shift vs 910 °C nominal soak"

    st.markdown(
        '<div class="facts">'
        f'<div class="lead"><span>Hybrid YS<sub>AN</sub></span><b>{ys_hyb:.0f} MPa</b>'
        f'<em>ML AN {ys_ml_an} · BN {ys_bn}</em></div>'
        f'<div class="lead"><span>Hybrid TS<sub>AN</sub></span><b>{ts_hyb:.0f} MPa</b>'
        f'<em>ML AN {ts_ml_an} · BN {ts_bn}</em></div>'
        f'<div class="lead"><span>Hybrid EL<sub>AN</sub></span><b>{el_hyb:.1f} %</b>'
        f'<em>ML AN {el_ml_an} · BN {el_bn}</em></div>'
        f'<div><span>ΔDBTT toughness shift</span><b>{dbtt:+.1f} °C</b>'
        f'<em>{imp_note}</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    _, fg, bg, bdr = REGIME_PILL_STYLE.get(regime, ("neutral", MUTED, "#f9fafb", HAIR))
    cal_meta = res.get("calibration_meta", {})
    cal_src = cal_meta.get("source", "default_literature_baseline")
    cal_tag = "Modal-calibrated AN physics" if "default" not in str(cal_src).lower() else "Literature baseline"
    anchor_note = (
        "Anchored to Shipped ML AN at nominal soak (910 °C)."
        if hyb.get("anchored_to_ml")
        else f"Standalone {cal_tag} (no Shipped ML AN target on {mill} route)."
    )
    if hyb.get("subcritical_tempered"):
        anchor_note += " Subcritical tempering interpolation applied between BN and AN."

    st.markdown(
        f'<div style="margin:.65rem 0 .85rem; padding:.65rem .9rem; border-radius:6px; '
        f'border:1px solid {bdr}; border-left:4px solid {fg}; background:{bg}; font-size:12.5px; color:{INK};">'
        f'<b>{html_lib.escape(grain["regime_label"])} ({html_lib.escape(regime)})</b>: '
        f'{html_lib.escape(grain["regime_note"])} '
        f'<span style="color:{MUTED};">· {html_lib.escape(anchor_note)} · Calibration: '
        f'<code>{html_lib.escape(str(cal_src))}</code>.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- Subtab 1: Thermal
def _render_subtab_thermal(res, t_soak_c, residence_min):
    therm = res["thermal"]
    ac1 = res["ac1_c"]
    ac3 = res["ac3_c"]
    t_arr = np.asarray(therm["time_min"], dtype=float)
    t_surf = np.asarray(therm["t_surface_c"], dtype=float)
    t_core = np.asarray(therm["t_core_c"], dtype=float)
    t_heat = float(therm["t_heat_min"])
    t_eff = float(therm["t_eff_soak_min"])
    lag_c = float(therm["t_surf_peak_c"] - therm["t_core_peak_c"])

    fig = _base_fig(
        f"Through-thickness thermal profile (Soak {t_soak_c:.0f} °C × {residence_min:.0f} min + 20 min air cool)",
        "Elapsed time (min)",
        "Temperature (°C)",
        height=430,
    )

    # Shaded Effective Austenitic Soak window
    if t_eff > 0.0 and residence_min > t_heat:
        is_aust = t_soak_c >= ac3
        shade_col = _rgba(SEMANTIC_OK, 0.10) if is_aust else _rgba(SEMANTIC_WARN, 0.10)
        shade_lbl = (
            f"Effective Austenitic Soak ({t_eff:.1f} min)"
            if is_aust
            else f"Core Equilibrated Sub-Ac3 Window ({t_eff:.1f} min)"
        )
        fig.add_vrect(
            x0=t_heat,
            x1=residence_min,
            fillcolor=shade_col,
            line_width=0,
            annotation_text=shade_lbl,
            annotation_position="top left",
            annotation_font=dict(size=10, color=SEMANTIC_OK if is_aust else SEMANTIC_WARN),
        )

    fig.add_scatter(
        x=t_arr,
        y=t_surf,
        mode="lines",
        name="Surface T(t)",
        line=dict(color=ACCENT, width=2.5),
        hovertemplate="t = %{x:.1f} min<br>Surface = %{y:.1f} °C<extra></extra>",
    )
    fig.add_scatter(
        x=t_arr,
        y=t_core,
        mode="lines",
        name="Core Mid-Thickness T(t)",
        line=dict(color=SEMANTIC_BAD, width=2.5),
        hovertemplate="t = %{x:.1f} min<br>Core = %{y:.1f} °C<extra></extra>",
    )

    fig.add_hline(
        y=ac1,
        line=dict(color=MUTED, width=1.3, dash="dot"),
        annotation_text=f"Ac1 = {ac1:.0f} °C",
        annotation_position="bottom right",
        annotation_font=dict(size=10, color=MUTED),
    )
    fig.add_hline(
        y=ac3,
        line=dict(color=WARN_LINE, width=1.5, dash="dash"),
        annotation_text=f"Ac3 = {ac3:.0f} °C",
        annotation_position="top right",
        annotation_font=dict(size=10, color=WARN_LINE),
    )
    fig.add_hline(
        y=ac3 + 15.0,
        line=dict(color=SEMANTIC_OK, width=1.1, dash="dot"),
        annotation_text=f"Full γ threshold Ac3+15 = {ac3 + 15.0:.0f} °C",
        annotation_position="top left",
        annotation_font=dict(size=9.5, color=SEMANTIC_OK),
    )

    fig.add_vline(
        x=residence_min,
        line=dict(color=INK, width=1.4, dash="dash"),
        annotation_text=f"Discharge ({residence_min:.0f} min)",
        annotation_position="bottom left",
        annotation_font=dict(size=10, color=INK),
    )
    if t_heat <= residence_min + 20.0:
        fig.add_vline(
            x=t_heat,
            line=dict(color=ACCENT, width=1.2, dash="dot"),
            annotation_text=f"t_heat ({t_heat:.1f} min)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=ACCENT),
        )

    fig.add_annotation(
        x=residence_min,
        y=therm["t_core_peak_c"],
        text=f"Core peak {therm['t_core_peak_c']:.1f} °C (lag ΔT = {lag_c:.1f} °C)",
        showarrow=True,
        arrowhead=2,
        ax=-55,
        ay=36,
        font=dict(size=10, color=INK),
        bgcolor="#ffffff",
        bordercolor=HAIR,
        borderwidth=1,
    )
    fig.update_layout(legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(
        '<div class="facts">'
        f'<div><span>through-thickness heat-up t<sub>heat</sub></span><b>{t_heat:.1f} min</b>'
        f'<em>{therm["heating_rate_min_per_mm"]:.1f} min/mm · {res["plate_t_mm"]:.1f} mm</em></div>'
        f'<div><span>effective soak t<sub>eff</sub></span><b>{t_eff:.1f} min</b>'
        f'<em>{"Adequate (>= 5 min)" if t_eff >= 5.0 else "Under-soaked (< 5 min)"}</em></div>'
        f'<div><span>discharge core lag ΔT</span><b>{lag_c:.1f} °C</b>'
        f'<em>surf {therm["t_surf_peak_c"]:.1f} vs core {therm["t_core_peak_c"]:.1f} °C</em></div>'
        f'<div><span>air cooling rate CR<sub>air</sub></span><b>{therm["cr_air_k_s"]:.2f} K/s</b>'
        f'<em>{therm["cooling_mode"].replace("_", " ")}</em></div>'
        f'<div><span>Hollomon-Jaffe HJ</span><b>{therm["hj_param"]:.2f}</b>'
        f'<em>×10³ tempering/soak index</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="note">Surface temperature reaches furnace setpoint quickly '
        f'(τ<sub>surf</sub> = {therm["tau_surf_min"]:.1f} min), while mid-thickness core lags '
        f'(τ<sub>core</sub> = {therm["tau_core_min"]:.1f} min). Effective soak time is '
        't<sub>eff</sub> = max(0, residence − t<sub>heat</sub>), followed by a 20-minute '
        'post-discharge air-cooling tail governing ferrite transformation.</div>',
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------- Subtab 2: Solubility & kinetics
def _render_subtab_precipitates(raw, res, t_soak_c, residence_min):
    precip = res["precipitates"]
    elems = precip["elements"]

    # 1. Horizontal stacked bar chart: Bound vs Dissolved microalloys
    y_labels = [METAL_LABEL[m] for m in METAL_ORDER]
    bound_pct = [elems[m]["bound_frac"] * 100.0 for m in METAL_ORDER]
    diss_pct = [elems[m]["dissolved_frac"] * 100.0 for m in METAL_ORDER]
    bound_wt = [elems[m]["bound_wt"] for m in METAL_ORDER]
    diss_wt = [elems[m]["dissolved_wt"] for m in METAL_ORDER]
    tot_wt = [elems[m]["total_wt"] for m in METAL_ORDER]

    fig_bar = _base_fig(
        f"Microalloy partitioning at soak ({t_soak_c:.0f} °C, t_eff = {res['thermal']['t_eff_soak_min']:.1f} min)",
        "Share of Total Element (%)",
        "Precipitate Family",
        height=300,
    )
    fig_bar.add_trace(go.Bar(
        y=y_labels,
        x=bound_pct,
        name="Bound (Precipitated at Soak)",
        orientation="h",
        marker=dict(color=ACCENT),
        text=[f"{p:.1f}% ({w:.4f}%)" if t > 0 else "0%" for p, w, t in zip(bound_pct, bound_wt, tot_wt)],
        textposition="inside",
        hovertemplate="%{y}<br>Bound: %{x:.1f}% (%{customdata:.5f} wt%)<extra></extra>",
        customdata=bound_wt,
    ))
    fig_bar.add_trace(go.Bar(
        y=y_labels,
        x=diss_pct,
        name="Dissolved (Free in Austenite)",
        orientation="h",
        marker=dict(color=_shade(SEMANTIC_OK, 0.35)),
        text=[f"{p:.1f}% ({w:.4f}%)" if t > 0 else "0%" for p, w, t in zip(diss_pct, diss_wt, tot_wt)],
        textposition="inside",
        hovertemplate="%{y}<br>Dissolved: %{x:.1f}% (%{customdata:.5f} wt%)<extra></extra>",
        customdata=diss_wt,
    ))
    fig_bar.update_layout(
        barmode="stack",
        xaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)),
        margin=dict(l=72, r=20, t=54, b=42),
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

    # 2. Free N / Free C KPI Cards + Table
    n_tot_ppm = precip["total_n_wt"] * 1e4
    n_free_ppm = precip["free_n_wt"] * 1e4
    n_bound_ppm = precip["bound_n_wt"] * 1e4
    n_cool_free_ppm = precip["post_cool_free_n_wt"] * 1e4
    v_re_ppm = precip["v_reprecip_wt"] * 1e4
    st.markdown(
        '<div class="facts">'
        f'<div><span>free N at soak</span><b>{n_free_ppm:.1f} ppm</b>'
        f'<em>{precip["free_n_wt"]:.4f} wt% of {n_tot_ppm:.0f} ppm</em></div>'
        f'<div><span>bound N (nitrides)</span><b>{n_bound_ppm:.1f} ppm</b>'
        f'<em>post-cool free N: {n_cool_free_ppm:.1f} ppm</em></div>'
        f'<div><span>free C at soak</span><b>{precip["free_c_wt"]:.4f} wt%</b>'
        f'<em>bound C: {precip["bound_c_wt"] * 1e4:.1f} ppm</em></div>'
        f'<div><span>V(C,N) cool re-precip</span><b>{v_re_ppm:.1f} ppm</b>'
        f'<em>precipitated during 700 °C cool</em></div>'
        f'<div><span>Zener pinning P<sub>z</sub></span><b>{precip["zener_pinning_mpa"]:.3f} MPa</b>'
        f'<em>f<sub>v</sub> {precip["f_v_total"]:.2e} · r {precip["r_ppt_nm"]:.1f} nm</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    table_rows = []
    for m in METAL_ORDER:
        info = elems[m]
        sys_parts = ", ".join(
            f"{lbl}: {wt:.4f}%" for lbl, wt in info["systems"].items() if wt > 1e-6
        ) or "dissolved / trace"
        t_sol_str = f"{info['t_sol_c']:.0f} °C" if info["t_sol_c"] is not None else "—"
        table_rows.append({
            "Family": METAL_LABEL[m],
            "Total (wt%)": round(info["total_wt"], 4),
            "Soak Bound (wt%)": round(info["bound_wt"], 5),
            "Soak Bound (%)": round(info["bound_frac"] * 100.0, 1),
            "Soak Dissolved (wt%)": round(info["dissolved_wt"], 5),
            "Post-Cool Bound (%)": round(info["post_cool_bound_frac"] * 100.0, 1),
            "T_sol (°C)": t_sol_str,
            "Active Phases (Soak)": sys_parts,
        })
    st.markdown('<div style="height:.55rem"></div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    # 3. Solubility Curves (700-1150 C) & Kinetic Evolution vs Residence Time
    sol = np_phys.solubility_curves(raw, t_min=700.0, t_max=1150.0, n_pts=91)
    kin = np_phys.kinetics_time_series(raw, t_soak_c, residence_min, n_pts=60)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_sol = _base_fig(
            "Equilibrium solubility curves (700–1150 °C)",
            "Temperature (°C)",
            "Equilibrium Precipitated Fraction (%)",
            height=380,
        )
        temps_c = sol["temps_c"]
        for m in METAL_ORDER:
            el_sol = sol["elements"][m]
            y_pct = np.asarray(el_sol["bound_frac"], dtype=float) * 100.0
            fig_sol.add_scatter(
                x=temps_c,
                y=y_pct,
                mode="lines",
                name=f"{METAL_LABEL[m]}",
                line=dict(color=METAL_COLOR[m], width=2.2),
                hovertemplate=f"{METAL_LABEL[m]}<br>T = %{{x:.0f}} °C<br>Bound eq = %{{y:.1f}}%<extra></extra>",
            )
            t_sol = el_sol.get("t_sol_c")
            if t_sol is not None and 700.0 <= t_sol <= 1150.0:
                fig_sol.add_scatter(
                    x=[t_sol],
                    y=[0.0],
                    mode="markers+text",
                    marker=dict(size=9, symbol="diamond", color=METAL_COLOR[m],
                                line=dict(width=1, color="#ffffff")),
                    text=[f"T_sol {m}: {t_sol:.0f}°C"],
                    textposition="top center",
                    textfont=dict(size=9, color=METAL_COLOR[m]),
                    showlegend=False,
                    hovertemplate=f"{METAL_LABEL[m]} T_sol = {t_sol:.1f} °C<extra></extra>",
                )
        fig_sol.add_vline(
            x=t_soak_c,
            line=dict(color=INK, width=1.4, dash="dash"),
            annotation_text=f"T_soak = {t_soak_c:.0f} °C",
            annotation_position="top right",
            annotation_font=dict(size=10, color=INK),
        )
        fig_sol.update_yaxes(range=[-4, 105])
        fig_sol.update_layout(legend=dict(orientation="h", y=-0.24, x=0, font=dict(size=10)))
        st.plotly_chart(fig_sol, use_container_width=True, config={"displayModeBar": False})
        sys_tsol_parts = [
            f"{lbl}: {info['t_sol_c']:.0f} °C"
            for lbl, info in sol["systems"].items()
            if info.get("t_sol_c") is not None
        ]
        if sys_tsol_parts:
            st.caption("Solution temperatures (T_sol): " + " · ".join(sys_tsol_parts) + ".")

    with col_r:
        fig_kin = _base_fig(
            f"JMAK kinetic evolution at {t_soak_c:.0f} °C vs residence time",
            "Residence time (min)",
            "Precipitated Metal Fraction (%)",
            height=380,
        )
        t_kin = kin["time_min"]
        for m in METAL_ORDER:
            frac_pct = np.asarray(kin["bound_frac"][m], dtype=float) * 100.0
            eq_pct = float(kin["eq_bound_frac"][m]) * 100.0
            fig_kin.add_scatter(
                x=t_kin,
                y=frac_pct,
                mode="lines",
                name=f"{m} kinetic (eq {eq_pct:.0f}%)",
                line=dict(color=METAL_COLOR[m], width=2.2),
                hovertemplate=f"{METAL_LABEL[m]}<br>t = %{{x:.1f}} min<br>Bound = %{{y:.1f}}%<extra></extra>",
            )
            fig_kin.add_scatter(
                x=[t_kin[0], t_kin[-1]],
                y=[eq_pct, eq_pct],
                mode="lines",
                line=dict(color=METAL_COLOR[m], width=1.0, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            )
        fig_kin.add_vline(
            x=residence_min,
            line=dict(color=INK, width=1.4, dash="dash"),
            annotation_text=f"Residence = {residence_min:.0f} min",
            annotation_position="top left",
            annotation_font=dict(size=10, color=INK),
        )
        fig_kin.update_yaxes(range=[-4, 105])
        fig_kin.update_layout(legend=dict(orientation="h", y=-0.24, x=0, font=dict(size=10)))
        st.plotly_chart(fig_kin, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------- Subtab 3: 2D Window & Pinning
def _render_subtab_process_window(mill, raw, res, ml_preds, calib, t_soak_c, residence_min, heat_rate, cool_mode):
    win = _cached_process_window(
        tuple(sorted((str(k), v) for k, v in raw.items())),
        _ml_subset_key(ml_preds),
        _calib_cache_key(calib),
        float(heat_rate),
        str(cool_mode),
    )

    layer_options = {
        "Hybrid YS_AN (MPa)": ("ys_grid", "MPa", "Blues"),
        "Hybrid TS_AN (MPa)": ("ts_grid", "MPa", "Blues"),
        "Ferrite Grain Size d_α (μm)": ("d_alpha_grid", "µm", "Blues_r"),
        "Austenite Grain Size d_γ (μm)": ("d_gamma_grid", "µm", "Blues_r"),
    }
    sel_col, info_col = st.columns([1.4, 2.6], vertical_alignment="bottom")
    layer = sel_col.selectbox(
        "Contour Response Layer",
        list(layer_options.keys()),
        index=0,
        key=f"norm_map_layer_{mill}",
    )
    grid_key, unit, cscale = layer_options[layer]
    z_grid = win[grid_key]
    ac3_c = float(win["ac3_c"])
    t_heat_min = float(win["t_heat_min"])

    info_col.caption(
        f"21 × 21 furnace window ({win['t_grid'][0]:.0f}–{win['t_grid'][-1]:.0f} °C × "
        f"{win['r_grid'][0]:.0f}–{win['r_grid'][-1]:.0f} min) · Ac3 = {ac3_c:.0f} °C · "
        f"Min core heat-up t_heat = {t_heat_min:.1f} min."
    )

    fig_win = _base_fig(
        f"2D Normalizing Process Window: {layer}",
        "Soaking Temperature (°C)",
        "Total Residence Time (min)",
        height=440,
    )
    fig_win.add_trace(go.Contour(
        x=win["t_grid"],
        y=win["r_grid"],
        z=z_grid,
        colorscale=cscale,
        contours=dict(showlabels=True, labelfont=dict(size=9, color="#ffffff")),
        line=dict(width=0.5, color="rgba(255,255,255,0.55)"),
        colorbar=dict(title=dict(text=unit, side="right"), thickness=12, len=0.82),
        hovertemplate=(
            "T_soak = %{x:.0f} °C<br>Residence = %{y:.0f} min<br>"
            f"{layer}: %{{z:.1f}} {unit}<extra></extra>"
        ),
    ))

    # Zener pinning breakdown contour (d_gamma / D_lim = 0.85)
    pr_grid = np.asarray(win["pinning_ratio_grid"], dtype=float)
    has_breakdown = bool(np.nanmin(pr_grid) <= 0.85 <= np.nanmax(pr_grid))
    if has_breakdown:
        fig_win.add_trace(go.Contour(
            x=win["t_grid"],
            y=win["r_grid"],
            z=pr_grid,
            contours=dict(
                coloring="none",
                start=0.85,
                end=0.85,
                size=1.0,
                showlabels=True,
                labelfont=dict(size=9, color=SEMANTIC_BAD),
            ),
            line=dict(color=SEMANTIC_BAD, width=2.2, dash="dash"),
            showscale=False,
            name="Zener breakdown (d_γ/D_lim = 0.85)",
            hoverinfo="skip",
        ))

    fig_win.add_vline(
        x=ac3_c,
        line=dict(color=WARN_LINE, width=1.8, dash="dash"),
        annotation_text=f"Ac3 ({ac3_c:.0f} °C)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=WARN_LINE),
    )
    if win["r_grid"][0] <= t_heat_min <= win["r_grid"][-1]:
        fig_win.add_hline(
            y=t_heat_min,
            line=dict(color=INK, width=1.6, dash="dot"),
            annotation_text=f"Min core heat-up ({t_heat_min:.1f} min)",
            annotation_position="bottom right",
            annotation_font=dict(size=10, color=INK),
        )

    # Current setpoint gold star
    fig_win.add_scatter(
        x=[t_soak_c],
        y=[residence_min],
        mode="markers+text",
        marker=dict(size=16, color=WARN_STAR, symbol="star", line=dict(color=INK, width=1.2)),
        text=[f"Setpoint ({t_soak_c:.0f} °C, {residence_min:.0f} min)"],
        textposition="top center",
        textfont=dict(size=10, color=INK),
        name="Current Setpoint",
    )
    x_lo = min(float(win["t_grid"][0]), float(t_soak_c) - 10.0)
    x_hi = max(float(win["t_grid"][-1]), float(t_soak_c) + 10.0)
    y_lo = min(float(win["r_grid"][0]), float(residence_min) - 5.0)
    y_hi = max(float(win["r_grid"][-1]), float(residence_min) + 5.0)
    fig_win.update_xaxes(range=[x_lo, x_hi])
    fig_win.update_yaxes(range=[y_lo, y_hi])
    fig_win.update_layout(legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)))
    st.plotly_chart(fig_win, use_container_width=True, config={"displayModeBar": False})
    if not has_breakdown:
        st.caption(
            f"Zener pinning holds across the entire 780–1020 °C window "
            f"(max d_γ/D_lim = {float(np.nanmax(pr_grid)):.2f} < 0.85)."
        )

    # Grain growth d_gamma(t) vs Zener limit D_lim(t) + secondary axis r_ppt(t)
    grain = res["grain"]
    t_eff = float(res["thermal"]["t_eff_soak_min"])
    s_time = np.asarray(grain["soak_time_min"], dtype=float)
    d_g = np.asarray(grain["d_gamma_um_series"], dtype=float)
    d_l = np.asarray(grain["d_lim_um_series"], dtype=float)
    r_s = np.asarray(grain["r_ppt_nm_series"], dtype=float)

    fig_pin = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pin.add_trace(
        go.Scatter(
            x=s_time,
            y=d_g,
            mode="lines",
            name="Austenite Grain Size d_γ (µm)",
            line=dict(color=ACCENT, width=2.6),
            hovertemplate="t_soak = %{x:.1f} min<br>d_γ = %{y:.1f} µm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_pin.add_trace(
        go.Scatter(
            x=s_time,
            y=d_l,
            mode="lines",
            name="Zener Limiting Diameter D_lim (µm)",
            line=dict(color=SEMANTIC_BAD, width=2.0, dash="dash"),
            hovertemplate="t_soak = %{x:.1f} min<br>D_lim = %{y:.1f} µm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_pin.add_trace(
        go.Scatter(
            x=s_time,
            y=0.85 * d_l,
            mode="lines",
            name="0.85 · D_lim Overheating Bound (µm)",
            line=dict(color=WARN_LINE, width=1.4, dash="dot"),
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig_pin.add_trace(
        go.Scatter(
            x=s_time,
            y=r_s,
            mode="lines",
            name="Mean Precipitate Radius r_ppt (nm)",
            line=dict(color=SEMANTIC_OK, width=2.0, dash="dashdot"),
            hovertemplate="t_soak = %{x:.1f} min<br>r_ppt = %{y:.1f} nm<extra></extra>",
        ),
        secondary_y=True,
    )
    fig_pin.add_vline(
        x=t_eff,
        line=dict(color=INK, width=1.3, dash="dash"),
        annotation_text=f"Current t_eff = {t_eff:.1f} min (d_γ = {grain['d_gamma_final_um']:.1f} µm)",
        annotation_position="top right",
        annotation_font=dict(size=10, color=INK),
    )
    fig_pin.update_layout(
        title=dict(
            text="Austenite grain growth d_γ(t) vs Zener pinning limit D_lim(t) & Ostwald precipitate coarsening r(t)",
            font=dict(size=12.5, color=INK),
            x=0,
            xanchor="left",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=370,
        margin=dict(l=54, r=54, t=54, b=46),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=10)),
    )
    fig_pin.update_xaxes(title_text="Effective isothermal soak time (min)", showline=True, linecolor=HAIR, gridcolor=GRID)
    fig_pin.update_yaxes(title_text="Grain diameter (µm)", showline=True, linecolor=HAIR, gridcolor=GRID, secondary_y=False)
    fig_pin.update_yaxes(title_text="Precipitate radius r_ppt (nm)", showline=True, linecolor=HAIR, gridcolor=GRID, secondary_y=True)
    st.plotly_chart(fig_pin, use_container_width=True, config={"displayModeBar": False})


# ----------------------------------------------------------------------------- Subtab 4: ML vs Physics & Modal
def _err_hw(ci_pair, val):
    if val is None or not ci_pair or ci_pair[0] is None or ci_pair[1] is None:
        return 0.0
    try:
        v = float(val)
        lo, hi = float(ci_pair[0]), float(ci_pair[1])
    except (TypeError, ValueError):
        return 0.0
    if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
        return 0.0
    return max(0.0, 0.5 * (hi - lo))


def _render_four_way_bar(res, ml_preds):
    phys = res["physics"]
    hyb = res["hybrid"]
    ml_b = res["ml_baseline"]
    ci_b = ml_b.get("ci", {})

    def _bn_hw(k):
        entry = (ml_preds or {}).get(k)
        if isinstance(entry, dict):
            return _err_hw((entry.get("ci_low"), entry.get("ci_high")), entry.get("pred"))
        return 0.0

    # Values and symmetric 95% conformal error bars across [YS, TS, EL]
    bn_vals = [ml_b.get("YS_BN"), ml_b.get("TS_BN"), ml_b.get("EL_BN")]
    bn_errs = [_bn_hw("YS_BN"), _bn_hw("TS_BN"), _bn_hw("EL_BN")]

    an_vals = [ml_b.get("YS_AN"), ml_b.get("TS_AN"), ml_b.get("EL_AN")]
    an_errs = [
        _err_hw(ci_b.get("YS_AN"), ml_b.get("YS_AN")),
        _err_hw(ci_b.get("TS_AN"), ml_b.get("TS_AN")),
        _err_hw(ci_b.get("EL_AN"), ml_b.get("EL_AN")),
    ]

    phys_vals = [phys["ys_phys_mpa"], phys["ts_phys_mpa"], phys["el_phys_pct"]]
    hyb_vals = [hyb["ys_hybrid_mpa"], hyb["ts_hybrid_mpa"], hyb["el_hybrid_pct"]]
    hyb_errs = [
        _err_hw((hyb.get("ys_ci_low"), hyb.get("ys_ci_high")), hyb["ys_hybrid_mpa"]),
        _err_hw((hyb.get("ts_ci_low"), hyb.get("ts_ci_high")), hyb["ts_hybrid_mpa"]),
        _err_hw((hyb.get("el_ci_low"), hyb.get("el_ci_high")), hyb["el_hybrid_pct"]),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.64, 0.36],
        subplot_titles=("Strength Properties (MPa)", "Elongation (%)"),
    )

    series = [
        ("As-Rolled ML BN", bn_vals, bn_errs, MUTED),
        ("Shipped ML AN", an_vals, an_errs, "#60a5fa"),
        ("Modal-Tuned Physics AN", phys_vals, [0.0, 0.0, 0.0], WARN_LINE),
        ("Hybrid Physics+ML", hyb_vals, hyb_errs, ACCENT),
    ]
    for name, vals, errs, color in series:
        if all(v is None for v in vals):
            continue
        s_x, s_y, s_e = ["YS", "TS"], vals[:2], errs[:2]
        fig.add_trace(
            go.Bar(
                x=s_x,
                y=[v if v is not None else 0.0 for v in s_y],
                error_y=dict(type="data", array=s_e, visible=any(e > 0 for e in s_e), thickness=1.4, width=4),
                name=name,
                marker_color=color,
                text=[f"{v:.0f}" if v is not None else "N/A" for v in s_y],
                textposition="auto",
            ),
            row=1,
            col=1,
        )
        el_v = vals[2]
        fig.add_trace(
            go.Bar(
                x=["EL"],
                y=[el_v if el_v is not None else 0.0],
                error_y=dict(type="data", array=[errs[2]], visible=errs[2] > 0, thickness=1.4, width=4),
                name=name,
                marker_color=color,
                text=[f"{el_v:.1f}%" if el_v is not None else "N/A"],
                textposition="auto",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        title=dict(
            text="4-way property comparison with 95% split-conformal error bars",
            font=dict(size=12.5, color=INK),
            x=0,
            xanchor="left",
        ),
        barmode="group",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=370,
        margin=dict(l=48, r=20, t=64, b=42),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        legend=dict(orientation="h", y=1.18, x=0, font=dict(size=10)),
    )
    fig.update_xaxes(showline=True, linecolor=HAIR)
    fig.update_yaxes(title_text="MPa", showline=True, linecolor=HAIR, gridcolor=GRID, row=1, col=1)
    fig.update_yaxes(title_text="%", showline=True, linecolor=HAIR, gridcolor=GRID, row=1, col=2)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_waterfall(mill, res):
    wf_choice = st.radio(
        "Waterfall Target",
        ["YS_AN (Yield Strength)", "TS_AN (Tensile Strength)"],
        horizontal=True,
        key=f"norm_wf_tgt_{mill}",
    )
    if wf_choice.startswith("YS"):
        comps = res["strength_components"]
        total_val = res["physics"]["ys_phys_mpa"]
        title_lbl = "Yield Strength YS_AN"
    else:
        comps = res["ts_strength_components"]
        total_val = res["physics"]["ts_phys_mpa"]
        title_lbl = "Tensile Strength TS_AN"

    labels = [lbl for _, lbl in WATERFALL_LABELS] + ["Modal-Tuned Physics Total"]
    vals = [float(comps.get(key, 0.0)) for key, _ in WATERFALL_LABELS] + [float(total_val)]
    measures = ["relative"] * len(WATERFALL_LABELS) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels,
        y=vals,
        measure=measures,
        text=[f"{v:+.1f}" if m == "relative" else f"{v:.1f} MPa" for v, m in zip(vals, measures)],
        textposition="outside",
        connector=dict(line=dict(color=HAIR, width=1.2)),
        increasing=dict(marker=dict(color=ACCENT)),
        decreasing=dict(marker=dict(color=SEMANTIC_BAD)),
        totals=dict(marker=dict(color=INK)),
    ))
    fig.update_layout(
        title=dict(
            text=f"Constitutive strengthening breakdown: {title_lbl} = {total_val:.1f} MPa",
            font=dict(size=12.5, color=INK),
            x=0,
            xanchor="left",
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        height=360,
        margin=dict(l=52, r=20, t=48, b=46),
        font=dict(family="Inter, -apple-system, 'Segoe UI', sans-serif", size=11, color=INK),
        showlegend=False,
    )
    fig.update_xaxes(showline=True, linecolor=HAIR)
    fig.update_yaxes(title_text="MPa contribution", showline=True, linecolor=HAIR, gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_sensitivity_curves(mill, raw, res, ml_preds, calib, t_soak_c, residence_min, heat_rate, cool_mode):
    sens_tgt = st.selectbox(
        "Sensitivity Curve Property",
        ["YS_AN", "TS_AN", "EL_AN"],
        format_func=lambda k: f"{TARGET_META[k][0]} [{TARGET_META[k][1]}]",
        key=f"norm_sens_tgt_{mill}",
    )
    t_label, t_unit, _tol, phys_k, hyb_k, _bn_k = TARGET_META[sens_tgt]
    ml_val = res["ml_baseline"].get(sens_tgt)
    ac1 = res["ac1_c"]
    ac3 = res["ac3_c"]
    t_heat = res["thermal"]["t_heat_min"]

    t_sweep = np.linspace(720.0, 1040.0, 33)
    t_hyb, t_phys, t_reg = [], [], []
    for ts in t_sweep:
        r = np_phys.predict_normalizing_properties(
            raw, float(ts), residence_min, ml_preds=ml_preds, calib=calib,
            heating_rate_min_per_mm=heat_rate, cooling_mode=cool_mode,
        )
        t_hyb.append(r["hybrid"][hyb_k])
        t_phys.append(r["physics"][phys_k])
        t_reg.append(r["regime"])

    r_sweep = np.linspace(10.0, 200.0, 25)
    r_hyb, r_phys, r_reg = [], [], []
    for rm in r_sweep:
        r = np_phys.predict_normalizing_properties(
            raw, t_soak_c, float(rm), ml_preds=ml_preds, calib=calib,
            heating_rate_min_per_mm=heat_rate, cooling_mode=cool_mode,
        )
        r_hyb.append(r["hybrid"][hyb_k])
        r_phys.append(r["physics"][phys_k])
        r_reg.append(r["regime"])

    c1, c2 = st.columns(2)
    with c1:
        fig_t = _base_fig(
            f"{t_label} vs soaking temperature (res = {residence_min:.0f} min)",
            "Soaking Temperature (°C)",
            f"{t_label} ({t_unit})",
            height=360,
        )
        safe_t = [t for t, reg in zip(t_sweep, t_reg) if reg == "FULL_NORMALIZING"]
        if safe_t:
            fig_t.add_vrect(
                x0=min(safe_t),
                x1=max(safe_t),
                fillcolor=_rgba(SEMANTIC_OK, 0.08),
                line_width=0,
                annotation_text="Safe normalizing band",
                annotation_position="top left",
                annotation_font=dict(size=9.5, color=SEMANTIC_OK),
            )
        fig_t.add_scatter(x=t_sweep, y=t_hyb, mode="lines", name="Hybrid Physics+ML",
                          line=dict(color=ACCENT, width=2.5))
        fig_t.add_scatter(x=t_sweep, y=t_phys, mode="lines", name="Modal-Tuned Physics",
                          line=dict(color=WARN_LINE, width=2.0, dash="dash"))
        if ml_val is not None:
            fig_t.add_hline(y=float(ml_val), line=dict(color=INK, width=1.3, dash="dot"),
                            annotation_text=f"Shipped ML {sens_tgt}: {ml_val:.1f}",
                            annotation_position="bottom right", annotation_font=dict(size=9.5))
        fig_t.add_vline(x=ac1, line=dict(color=MUTED, width=1.0, dash="dot"),
                        annotation_text="Ac1", annotation_position="bottom left", annotation_font=dict(size=9))
        fig_t.add_vline(x=ac3, line=dict(color=WARN_LINE, width=1.1, dash="dot"),
                        annotation_text="Ac3", annotation_position="bottom right", annotation_font=dict(size=9))
        fig_t.add_vline(x=t_soak_c, line=dict(color=ACCENT, width=1.3, dash="dash"),
                        annotation_text=f"Set {t_soak_c:.0f} °C", annotation_position="top right",
                        annotation_font=dict(size=9.5, color=ACCENT))
        fig_t.update_layout(legend=dict(orientation="h", y=-0.24, x=0, font=dict(size=9.5)))
        st.plotly_chart(fig_t, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_r = _base_fig(
            f"{t_label} vs total residence time (T_soak = {t_soak_c:.0f} °C)",
            "Total Residence Time (min)",
            f"{t_label} ({t_unit})",
            height=360,
        )
        safe_r = [rm for rm, reg in zip(r_sweep, r_reg) if reg == "FULL_NORMALIZING"]
        if safe_r:
            fig_r.add_vrect(
                x0=min(safe_r),
                x1=max(safe_r),
                fillcolor=_rgba(SEMANTIC_OK, 0.08),
                line_width=0,
                annotation_text="Safe normalizing band",
                annotation_position="top left",
                annotation_font=dict(size=9.5, color=SEMANTIC_OK),
            )
        fig_r.add_scatter(x=r_sweep, y=r_hyb, mode="lines", name="Hybrid Physics+ML",
                          line=dict(color=ACCENT, width=2.5))
        fig_r.add_scatter(x=r_sweep, y=r_phys, mode="lines", name="Modal-Tuned Physics",
                          line=dict(color=WARN_LINE, width=2.0, dash="dash"))
        if ml_val is not None:
            fig_r.add_hline(y=float(ml_val), line=dict(color=INK, width=1.3, dash="dot"),
                            annotation_text=f"Shipped ML {sens_tgt}: {ml_val:.1f}",
                            annotation_position="bottom right", annotation_font=dict(size=9.5))
        if 10.0 <= t_heat <= 200.0:
            fig_r.add_vline(x=t_heat, line=dict(color=MUTED, width=1.1, dash="dot"),
                            annotation_text="t_heat", annotation_position="bottom left",
                            annotation_font=dict(size=9))
        fig_r.add_vline(x=residence_min, line=dict(color=ACCENT, width=1.3, dash="dash"),
                        annotation_text=f"Set {residence_min:.0f} min", annotation_position="top right",
                        annotation_font=dict(size=9.5, color=ACCENT))
        fig_r.update_layout(legend=dict(orientation="h", y=-0.24, x=0, font=dict(size=9.5)))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})


def _extract_holdout_metrics(metrics_block, target):
    """Read Held-Out R2, RMSE, MAE, Bias, and HitRate flexibly across Modal calibration schemas."""
    block = (metrics_block or {}).get(target) or {}
    if isinstance(block.get("heldout"), dict):
        sub = block["heldout"]
    elif isinstance(block.get("holdout"), dict):
        sub = block["holdout"]
    elif isinstance(block.get("test"), dict):
        sub = block["test"]
    else:
        sub = block

    def _pick(*keys):
        for k in keys:
            if k in sub and sub[k] is not None:
                try:
                    v = float(sub[k])
                    if np.isfinite(v):
                        return k, v
                except (TypeError, ValueError):
                    continue
        return None, None

    _, r2 = _pick("R2", "r2", "holdout_r2")
    _, rmse = _pick("RMSE", "rmse", "holdout_rmse")
    _, mae = _pick("MAE", "mae", "holdout_mae")
    _, bias = _pick("Bias", "bias", "holdout_bias")
    hit_key, hit = _pick("HitRate_20MPa", "HitRate_2.5pct", "HitRate", "hit_rate_pct", "hit_rate")
    if hit is not None and hit_key == "hit_rate" and hit <= 1.0:
        hit *= 100.0
    train_sub = block.get("train") if isinstance(block.get("train"), dict) else {}
    train_r2 = train_sub.get("R2", train_sub.get("r2"))
    return {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias, "hit": hit, "train_r2": train_r2}


def _render_modal_parity(mill, calib):
    _section("modal cloud calibration parity & held-out audit")
    parity_tgt = st.selectbox(
        "Calibration Parity Target",
        ["YS_AN", "TS_AN", "EL_AN"],
        format_func=lambda k: f"{TARGET_META[k][0]} ({k})",
        key=f"norm_parity_tgt_{mill}",
    )
    t_label, t_unit, tol, *_ = TARGET_META[parity_tgt]
    metrics = _extract_holdout_metrics(calib.get("metrics"), parity_tgt)
    src = calib.get("source", "default_literature_baseline")
    ts = calib.get("timestamp") or "not calibrated yet"
    n_sched = calib.get("n_schedules")
    n_grades = calib.get("n_grades")
    n_txt = f"{int(n_sched):,} stratified schedules" if n_sched else "no Modal sample count recorded"
    if n_grades:
        n_txt += f" across {int(n_grades)} grades"

    st.caption(f"Calibration artifact: `{src}` · {n_txt} · updated {ts}")

    hit_lbl = f"±{tol:g} {t_unit}"
    hit_val = metrics["hit"]
    hit_str = f"{hit_val:.1f}%" if hit_val is not None else "—"
    train_r2_note = f" (train R² {metrics['train_r2']:.3f})" if metrics.get("train_r2") is not None else ""
    st.markdown(
        '<div class="facts">'
        f'<div><span>held-out R²</span><b>{_fmt_opt(metrics["r2"], ".3f")}</b>'
        f'<em>grade-grouped holdout{train_r2_note}</em></div>'
        f'<div><span>held-out RMSE</span><b>{_fmt_opt(metrics["rmse"], ".2f", f" {t_unit}")}</b>'
        f'<em>physics vs ML AN</em></div>'
        f'<div><span>held-out MAE</span><b>{_fmt_opt(metrics["mae"], ".2f", f" {t_unit}")}</b>'
        f'<em>bias {_fmt_opt(metrics["bias"], "+.2f", f" {t_unit}")}</em></div>'
        f'<div><span>hit rate ({hit_lbl})</span><b>{hit_str}</b>'
        f'<em>within certification tol</em></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    samples = calib.get("parity_samples") or []
    if not samples:
        st.info(
            "No Modal parity samples are present in `models/normalizing_calibration.json` yet. "
            "Run `python -m modal run scripts/tune_normalizing_modal.py` to populate the parity audit."
        )
        return

    df_p = pd.DataFrame(samples)
    col_map = {
        "YS_AN": ("ys_ml", "ys_phys"),
        "TS_AN": ("ts_ml", "ts_phys"),
        "EL_AN": ("el_ml", "el_phys"),
    }
    x_col, y_col = col_map[parity_tgt]
    if x_col not in df_p.columns or y_col not in df_p.columns:
        st.caption(f"Parity samples do not contain `{x_col}` / `{y_col}` columns.")
        return

    x_vals = pd.to_numeric(df_p[x_col], errors="coerce")
    y_vals = pd.to_numeric(df_p[y_col], errors="coerce")
    valid = x_vals.notna() & y_vals.notna()
    df_p = df_p.loc[valid].copy()
    x_arr = x_vals.loc[valid].to_numpy(dtype=float)
    y_arr = y_vals.loc[valid].to_numpy(dtype=float)

    lo = float(min(x_arr.min(), y_arr.min()))
    hi = float(max(x_arr.max(), y_arr.max()))
    pad = max(tol * 1.5, 0.05 * (hi - lo))
    p_lo, p_hi = lo - pad, hi + pad

    fig_par = _base_fig(
        f"Modal Cloud calibration parity: {t_label} (N = {len(df_p)} stratified schedules)",
        f"Shipped ML {parity_tgt} ({t_unit})",
        f"Modal-Tuned Physics {parity_tgt} ({t_unit})",
        height=410,
    )
    # +/- tolerance band around 1:1
    fig_par.add_trace(go.Scatter(
        x=[p_lo, p_hi, p_hi, p_lo, p_lo],
        y=[p_lo - tol, p_hi - tol, p_hi + tol, p_lo + tol, p_lo - tol],
        fill="toself",
        fillcolor=_rgba(ACCENT, 0.08),
        line=dict(width=0),
        name=f"±{tol:g} {t_unit} tolerance band",
        hoverinfo="skip",
    ))
    fig_par.add_trace(go.Scatter(
        x=[p_lo, p_hi],
        y=[p_lo, p_hi],
        mode="lines",
        line=dict(color=WARN_LINE, width=1.6, dash="dash"),
        name="1:1 Parity",
    ))

    ce_vals = pd.to_numeric(df_p.get("ce_iiw"), errors="coerce").fillna(0.35).to_numpy(dtype=float)
    hover_txt = [
        f"Grade: {row.get('grade', '—')} ({row.get('split', '—')})<br>"
        f"Gauge: {row.get('plate_t_mm', 0):.1f} mm · CE_IIW: {row.get('ce_iiw', 0):.3f}<br>"
        f"d_α: {row.get('d_alpha_um', 0):.1f} µm<br>"
        f"Shipped ML: {xv:.1f} {t_unit}<br>Physics AN: {yv:.1f} {t_unit} (Δ {yv - xv:+.1f})"
        for (_, row), xv, yv in zip(df_p.iterrows(), x_arr, y_arr)
    ]
    fig_par.add_trace(go.Scatter(
        x=x_arr,
        y=y_arr,
        mode="markers",
        marker=dict(
            size=7,
            color=ce_vals,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=dict(text="CE_IIW", side="right"), thickness=12, len=0.8),
            line=dict(width=0.6, color="#ffffff"),
            opacity=0.85,
        ),
        text=hover_txt,
        hoverinfo="text",
        name="Stratified Schedules",
    ))
    fig_par.update_xaxes(range=[p_lo, p_hi])
    fig_par.update_yaxes(range=[p_lo, p_hi])
    fig_par.update_layout(legend=dict(orientation="h", y=1.10, x=0, font=dict(size=10)))
    st.plotly_chart(fig_par, use_container_width=True, config={"displayModeBar": False})

    plant_audit = calib.get("plant_audit")
    if isinstance(plant_audit, dict) and plant_audit:
        n_plant = plant_audit.get("n_rows") or plant_audit.get("n")
        pa_m = (plant_audit.get("metrics") or {}).get(parity_tgt) or {}
        if pa_m:
            pa_hit = pa_m.get("HitRate_20MPa", pa_m.get("HitRate_2.5pct"))
            pa_hit_str = f", HitRate {pa_hit:.1f}%" if pa_hit is not None else ""
            st.caption(
                f"Plant normalized-plate audit ({int(pa_m.get('n', n_plant or 0)):,} measured plates): "
                f"{parity_tgt} RMSE {pa_m.get('RMSE', 0):.2f} {t_unit}, "
                f"MAE {pa_m.get('MAE', 0):.2f} {t_unit}, Bias {pa_m.get('Bias', 0):+.2f} {t_unit}{pa_hit_str}."
            )
        else:
            st.caption(
                "Plant normalized-plate audit included in calibration"
                + (f" ({int(n_plant):,} measured rows)" if n_plant else "")
                + "."
            )


# ----------------------------------------------------------------------------- main view
def view_normalizing(mill, raw, engine="Champion GBDT (Recommended)"):
    """Furnace Normalizing tab: thermal cycle, precipitate binding, 2D window & Modal parity."""
    raw = dict(raw or {})

    _section("furnace normalizing simulator")
    st.markdown(
        '<div class="note">Simulates post-rolling furnace normalizing from first-principles '
        '1D through-thickness heating, sequential Ti/Nb/Al/V nitride-carbide dissolution &amp; '
        'JMAK kinetics, Zener grain pinning, and Modal-calibrated constitutive predictions anchored '
        'to the shipped After-Normalizing (AN) models. Runs locally in milliseconds.</div>',
        unsafe_allow_html=True,
    )

    if mill == "HRC":
        st.info(
            "Normalizing furnaces apply to Heavy Plate delivery; Steckel coil (HRC) carries no "
            "shipped AN models, so thermal cycles, precipitate solubility, grain evolution, and "
            "Modal-calibrated physics AN estimates below evaluate on the current strip chemistry & gauge."
        )

    plate_t = float(raw.get("Plate_T_mm", raw.get("Coil_T_mm", 16.0)) or 16.0)
    default_res = float(min(240.0, max(5.0, max(20.0, round(1.8 * plate_t / 5.0) * 5.0))))

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.35, 1.05], vertical_alignment="bottom")
    t_soak_c = float(c1.slider(
        "Soaking Temperature [°C]",
        min_value=650.0,
        max_value=1050.0,
        value=910.0,
        step=5.0,
        key=f"norm_t_soak_{mill}",
    ))
    residence_min = float(c2.slider(
        "Total Residence Time [min]",
        min_value=5.0,
        max_value=240.0,
        value=default_res,
        step=5.0,
        key=f"norm_res_min_{mill}",
    ))
    heat_label = c3.selectbox(
        "Furnace Heating Profile",
        list(HEATING_PROFILES.keys()),
        index=0,
        key=f"norm_heat_prof_{mill}",
    )
    cool_label = c4.selectbox(
        "Post-Discharge Cooling",
        list(COOLING_MODES.keys()),
        index=0,
        key=f"norm_cool_mode_{mill}",
    )
    heat_rate = HEATING_PROFILES[heat_label]
    cool_mode = COOLING_MODES[cool_label]

    calib = np_phys.load_calibration()
    ml_preds = _safe_ml_preds(mill, raw, engine)
    res = np_phys.predict_normalizing_properties(
        raw,
        t_soak_c=t_soak_c,
        residence_min=residence_min,
        ml_preds=ml_preds,
        calib=calib,
        heating_rate_min_per_mm=heat_rate,
        cooling_mode=cool_mode,
    )

    _render_kpis_and_banner(mill, res, t_soak_c, residence_min)

    subtabs = st.tabs([
        "1. Thermal Cycle & Core Soaking",
        "2. Element Binding, Solubility & Kinetics",
        "3. 2D Process Window & Grain Pinning",
        "4. ML vs. Physics & Modal Fine-Tuning",
    ])
    with subtabs[0]:
        _render_subtab_thermal(res, t_soak_c, residence_min)
    with subtabs[1]:
        _render_subtab_precipitates(raw, res, t_soak_c, residence_min)
    with subtabs[2]:
        _render_subtab_process_window(
            mill, raw, res, ml_preds, calib, t_soak_c, residence_min, heat_rate, cool_mode
        )
    with subtabs[3]:
        _render_four_way_bar(res, ml_preds)
        _render_waterfall(mill, res)
        _render_sensitivity_curves(
            mill, raw, res, ml_preds, calib, t_soak_c, residence_min, heat_rate, cool_mode
        )
        _render_modal_parity(mill, calib)
