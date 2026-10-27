"""
Interactive Mill Simulator View for Streamlit.
Enables real-time property prediction, metallurgical parameter readouts, and customer spec checks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model_loader import TARGET_CONFIG, compute_physical_indicators, predict_single_sample

PRESETS = {
    "Plate": {
        "Low-Carbon Structural (A36 / SS400)": {
            "Slab_T_mm": 200.0, "Plate_T_mm": 20.0, "Plate_W_mm": 2000.0, "TB_Ratio": 2.2, "FM_Last_Pass_Drafting": 2.5,
            "FM_Temp_Before_Rolling": 950.0, "FM_Finish_Temp": 840.0, "ACC_Headers_n": 8.0, "ACC_Temp_Post_Laminar": 600.0,
            "Rolling_Type": "AR",
            "Chem_C": 0.16, "Chem_Mn": 0.75, "Chem_Si": 0.22, "Chem_P": 0.015, "Chem_S": 0.005,
            "Chem_Nb": 0.001, "Chem_V": 0.002, "Chem_Ti": 0.002, "Chem_Cr": 0.02, "Chem_Ni": 0.02,
            "Chem_Cu": 0.02, "Chem_Mo": 0.002, "Chem_Al": 0.030, "Chem_N": 0.0035, "Chem_B": 0.0
        },
        "Microalloyed HSLA (SM490 / S355 / X52)": {
            "Slab_T_mm": 220.0, "Plate_T_mm": 25.0, "Plate_W_mm": 2200.0, "TB_Ratio": 2.5, "FM_Last_Pass_Drafting": 3.0,
            "FM_Temp_Before_Rolling": 920.0, "FM_Finish_Temp": 810.0, "ACC_Headers_n": 12.0, "ACC_Temp_Post_Laminar": 550.0,
            "Rolling_Type": "TMCP",
            "Chem_C": 0.12, "Chem_Mn": 1.35, "Chem_Si": 0.28, "Chem_P": 0.012, "Chem_S": 0.003,
            "Chem_Nb": 0.035, "Chem_V": 0.040, "Chem_Ti": 0.018, "Chem_Cr": 0.03, "Chem_Ni": 0.05,
            "Chem_Cu": 0.03, "Chem_Mo": 0.005, "Chem_Al": 0.035, "Chem_N": 0.0040, "Chem_B": 0.0
        },
        "High-Strength Offshore (S460 / X70 TMCP)": {
            "Slab_T_mm": 250.0, "Plate_T_mm": 32.0, "Plate_W_mm": 2500.0, "TB_Ratio": 3.0, "FM_Last_Pass_Drafting": 3.5,
            "FM_Temp_Before_Rolling": 890.0, "FM_Finish_Temp": 780.0, "ACC_Headers_n": 16.0, "ACC_Temp_Post_Laminar": 480.0,
            "Rolling_Type": "TMCP",
            "Chem_C": 0.08, "Chem_Mn": 1.55, "Chem_Si": 0.30, "Chem_P": 0.008, "Chem_S": 0.002,
            "Chem_Nb": 0.050, "Chem_V": 0.065, "Chem_Ti": 0.020, "Chem_Cr": 0.15, "Chem_Ni": 0.35,
            "Chem_Cu": 0.20, "Chem_Mo": 0.120, "Chem_Al": 0.040, "Chem_N": 0.0035, "Chem_B": 0.0002
        }
    },
    "HRC": {
        "Commercial Drawing / Forming (SPHC / SAE 1008)": {
            "Coil_T_mm": 3.2, "Coil_W_mm": 1250.0, "TB_Thick_mm": 28.0,
            "Temp_Transfer_Bar_C": 980.0, "Temp_FT_Avg_C": 880.0, "Temp_CT_Top_C": 620.0, "Temp_CT_Bot_C": 615.0,
            "Chem_C": 0.06, "Chem_Mn": 0.35, "Chem_Si": 0.02, "Chem_P": 0.012, "Chem_S": 0.006,
            "Chem_Nb": 0.0, "Chem_V": 0.0, "Chem_Ti": 0.0, "Chem_Cr": 0.02, "Chem_Ni": 0.01,
            "Chem_Cu": 0.02, "Chem_Mo": 0.0, "Chem_Al": 0.035, "Chem_N": 0.0030, "Chem_B": 0.0
        },
        "Structural Steckel Strip (SS400 / A36 equivalent)": {
            "Coil_T_mm": 6.0, "Coil_W_mm": 1500.0, "TB_Thick_mm": 32.0,
            "Temp_Transfer_Bar_C": 950.0, "Temp_FT_Avg_C": 850.0, "Temp_CT_Top_C": 580.0, "Temp_CT_Bot_C": 575.0,
            "Chem_C": 0.15, "Chem_Mn": 0.70, "Chem_Si": 0.18, "Chem_P": 0.015, "Chem_S": 0.005,
            "Chem_Nb": 0.001, "Chem_V": 0.002, "Chem_Ti": 0.001, "Chem_Cr": 0.02, "Chem_Ni": 0.01,
            "Chem_Cu": 0.02, "Chem_Mo": 0.0, "Chem_Al": 0.028, "Chem_N": 0.0035, "Chem_B": 0.0
        },
        "Microalloyed High-Strength Strip (S355MC / X52)": {
            "Coil_T_mm": 8.0, "Coil_W_mm": 1500.0, "TB_Thick_mm": 35.0,
            "Temp_Transfer_Bar_C": 930.0, "Temp_FT_Avg_C": 820.0, "Temp_CT_Top_C": 540.0, "Temp_CT_Bot_C": 535.0,
            "Chem_C": 0.10, "Chem_Mn": 1.25, "Chem_Si": 0.22, "Chem_P": 0.012, "Chem_S": 0.003,
            "Chem_Nb": 0.035, "Chem_V": 0.030, "Chem_Ti": 0.015, "Chem_Cr": 0.03, "Chem_Ni": 0.02,
            "Chem_Cu": 0.02, "Chem_Mo": 0.0, "Chem_Al": 0.035, "Chem_N": 0.0035, "Chem_B": 0.0
        }
    }
}


def render_simulator_view():
    st.markdown("### Interactive Mill Mechanical Property Simulator")
    st.markdown(
        "Simulate room-temperature mechanical tensile properties and toughness regimes by varying slab chemistry, "
        "pass drafting, and thermal runout cooling conditions in real time."
    )

    # Top Mill Selector
    col_mill, col_preset = st.columns([1, 2])
    with col_mill:
        mill_type = st.radio("Select Steel Rolling Route", ["Plate", "HRC"], horizontal=True,
                             help="Plate: Heavy plate sequential rolling (6-120mm). HRC: Steckel Mill continuous strip (2.5-16mm).")
    
    preset_dict = PRESETS[mill_type]
    with col_preset:
        chosen_preset = st.selectbox("Load Standard Grade Preset", list(preset_dict.keys()))

    preset_vals = preset_dict[chosen_preset]

    st.markdown("---")

    # 3-Column Parameter Control Panels
    col_geom, col_chem, col_therm = st.columns(3)

    with col_geom:
        st.subheader("1. Geometry & Drafting")
        if mill_type == "Plate":
            slab_t = st.number_input("Slab Thickness (mm)", 120.0, 320.0, float(preset_vals["Slab_T_mm"]), 5.0)
            plate_t = st.number_input("Finished Plate Thickness (mm)", 5.0, 120.0, float(preset_vals["Plate_T_mm"]), 1.0)
            plate_w = st.number_input("Finished Plate Width (mm)", 900.0, 4200.0, float(preset_vals["Plate_W_mm"]), 50.0)
            tb_ratio = st.slider("Transfer Bar Ratio (h_TB / h_final)", 1.2, 5.0, float(preset_vals["TB_Ratio"]), 0.1)
            last_draft = st.slider("FM Last Pass Drafting (mm)", 0.5, 12.0, float(preset_vals["FM_Last_Pass_Drafting"]), 0.5)
            rolling_type = st.selectbox("Rolling Process Route", ["AR", "TMCP", "CR"], index=1 if preset_vals.get("Rolling_Type") == "TMCP" else 0)
        else:
            tb_thick = st.number_input("Transfer Bar Thickness (mm)", 18.0, 60.0, float(preset_vals["TB_Thick_mm"]), 1.0)
            coil_t = st.number_input("Finished Strip Thickness (mm)", 2.0, 20.0, float(preset_vals["Coil_T_mm"]), 0.5)
            coil_w = st.number_input("Finished Strip Width (mm)", 800.0, 2100.0, float(preset_vals["Coil_W_mm"]), 50.0)

    with col_chem:
        st.subheader("2. Melt Chemistry (% wt)")
        c1, c2 = st.columns(2)
        with c1:
            chem_c = st.number_input("C (%)", 0.02, 0.45, float(preset_vals["Chem_C"]), 0.01, format="%.3f")
            chem_mn = st.number_input("Mn (%)", 0.20, 2.00, float(preset_vals["Chem_Mn"]), 0.05, format="%.2f")
            chem_si = st.number_input("Si (%)", 0.01, 0.80, float(preset_vals["Chem_Si"]), 0.02, format="%.2f")
            chem_p = st.number_input("P (%)", 0.002, 0.040, float(preset_vals["Chem_P"]), 0.002, format="%.3f")
            chem_s = st.number_input("S (%)", 0.001, 0.030, float(preset_vals["Chem_S"]), 0.001, format="%.3f")
            chem_al = st.number_input("Al (%)", 0.010, 0.080, float(preset_vals["Chem_Al"]), 0.005, format="%.3f")
        with c2:
            chem_nb = st.number_input("Nb (%)", 0.0, 0.10, float(preset_vals["Chem_Nb"]), 0.005, format="%.3f")
            chem_v = st.number_input("V (%)", 0.0, 0.12, float(preset_vals["Chem_V"]), 0.005, format="%.3f")
            chem_ti = st.number_input("Ti (%)", 0.0, 0.06, float(preset_vals["Chem_Ti"]), 0.002, format="%.3f")
            chem_cr = st.number_input("Cr (%)", 0.0, 0.60, float(preset_vals["Chem_Cr"]), 0.02, format="%.2f")
            chem_ni = st.number_input("Ni (%)", 0.0, 0.80, float(preset_vals["Chem_Ni"]), 0.05, format="%.2f")
            chem_cu = st.number_input("Cu (%)", 0.0, 0.50, float(preset_vals["Chem_Cu"]), 0.02, format="%.2f")
            chem_mo = st.number_input("Mo (%)", 0.0, 0.30, float(preset_vals["Chem_Mo"]), 0.01, format="%.3f")
            chem_n = st.number_input("N (%)", 0.001, 0.015, float(preset_vals["Chem_N"]), 0.001, format="%.4f")
            chem_b = st.number_input("B (%)", 0.0, 0.005, float(preset_vals["Chem_B"]), 0.0005, format="%.4f")

    with col_therm:
        st.subheader("3. Thermal & Cooling Schedule")
        if mill_type == "Plate":
            fm_start = st.slider("FM Start Rolling Temp (°C)", 800.0, 1050.0, float(preset_vals["FM_Temp_Before_Rolling"]), 5.0)
            frt = st.slider("Finish Rolling Temp FRT (°C)", 720.0, 950.0, float(preset_vals["FM_Finish_Temp"]), 5.0)
            headers = st.slider("Active ACC Laminar Headers", 0.0, 24.0, float(preset_vals["ACC_Headers_n"]), 1.0)
            t_post = st.slider("Post-Laminar Water Stop Temp (°C)", 420.0, frt, float(preset_vals["ACC_Temp_Post_Laminar"]), 5.0)
        else:
            ttb = st.slider("Transfer Bar Temp (°C)", 850.0, 1080.0, float(preset_vals["Temp_Transfer_Bar_C"]), 5.0)
            ft = st.slider("Finish Rolling Temp FT (°C)", 750.0, 950.0, float(preset_vals["Temp_FT_Avg_C"]), 5.0)
            ct_top = st.slider("Coiling Temp CT Top (°C)", 450.0, 720.0, float(preset_vals["Temp_CT_Top_C"]), 5.0)
            ct_bot = st.slider("Coiling Temp CT Bottom (°C)", 450.0, 720.0, float(preset_vals["Temp_CT_Bot_C"]), 5.0)

    # Assemble raw input dictionary
    raw_inputs = {
        "Chem_C": chem_c, "Chem_Mn": chem_mn, "Chem_Si": chem_si, "Chem_P": chem_p, "Chem_S": chem_s,
        "Chem_Al": chem_al, "Chem_Nb": chem_nb, "Chem_V": chem_v, "Chem_Ti": chem_ti, "Chem_Cr": chem_cr,
        "Chem_Ni": chem_ni, "Chem_Cu": chem_cu, "Chem_Mo": chem_mo, "Chem_N": chem_n, "Chem_B": chem_b
    }
    if mill_type == "Plate":
        raw_inputs.update({
            "Slab_T_mm": slab_t, "Plate_T_mm": plate_t, "Plate_W_mm": plate_w, "TB_Ratio": tb_ratio,
            "FM_Last_Pass_Drafting": last_draft, "Rolling_Type": rolling_type,
            "FM_Temp_Before_Rolling": fm_start, "FM_Finish_Temp": frt,
            "ACC_Headers_n": headers, "ACC_Temp_Post_Laminar": t_post,
            "ACC_Active": 1.0 if headers > 0 else 0.0
        })
    else:
        raw_inputs.update({
            "Coil_T_mm": coil_t, "Coil_W_mm": coil_w, "TB_Thick_mm": tb_thick,
            "Temp_Transfer_Bar_C": ttb, "Temp_FT_Avg_C": ft,
            "Temp_CT_Top_C": ct_top, "Temp_CT_Bot_C": ct_bot
        })

    # 4. Real-time Physical Metallurgy Indicators
    st.markdown("---")
    st.subheader("Physical Metallurgy Fundamentals")
    phys = compute_physical_indicators(raw_inputs, mill_type)

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    p_col1.metric("Pickering Solid Solution (σ_ss)", f"{phys['sigma_ss']:.1f} MPa", help="Solid solution lattice distortion from Mn, Si, Cu, Ni, N, Ti")
    p_col2.metric("Carbon Equiv (CE_IIW)", f"{phys['ce_iiw']:.3f}", help="International weldability index: C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15")
    p_col3.metric("Ito-Bessyo (P_cm)", f"{phys['pcm']:.3f}", help="Cold cracking sensitivity parameter for modern low-carbon steels")
    p_col4.metric("Austenite Ar3 Boundary", f"{phys['ar3']:.1f} °C", help="Transformation start temperature: ferrite nucleates below Ar3")
    p_col5.metric("Cooling Rate Proxy", f"{phys['cooling_rate']:.1f} °C/mm", help="Thermal driving force across Runout Table / Laminar Headers")

    # Physical Regimes Alert
    rolling_temp = frt if mill_type == "Plate" else ft
    if rolling_temp < phys["ar3"]:
        st.warning(f"⚠️ **Intercritical Rolling Regime**: Finish Rolling Temp ({rolling_temp:.1f} °C) is below Ar3 ({phys['ar3']:.1f} °C). Expect mixed deformed-ferrite microstructure and higher yield ratio.")
    else:
        st.success(f"✓ **Austenite Phase Rolling**: Finish Rolling Temp ({rolling_temp:.1f} °C) is above Ar3 ({phys['ar3']:.1f} °C). Dynamic/static recrystallization and subsequent fine ferrite transformation are favored.")

    # 5. Prediction Execution
    st.markdown("---")
    st.subheader("Predicted Mechanical Properties & 95% Confidence Intervals")

    preds = predict_single_sample(mill_type, raw_inputs)

    if not preds:
        st.error("No predictions returned. Ensure models exist in Plate/models or HRC/models.")
        return

    # Render Prediction Metric Cards
    pred_cols = st.columns(min(len(preds), 4))
    idx = 0
    for tgt_key, res in preds.items():
        curr_col = pred_cols[idx % len(pred_cols)]
        ci_str = f"95% CI: [{res['ci_low']}, {res['ci_high']}]"
        curr_col.metric(
            label=f"{res['name']} ({tgt_key})",
            value=f"{res['pred']} {res['unit']}",
            delta=f"± {round((res['ci_high'] - res['ci_low'])/2, 1)} {res['unit']} CI",
            help=f"Model: {res['model_name']} | {ci_str}"
        )
        idx += 1

    # Yield Ratio Check
    ys_key = "YS_BN"
    ts_key = "TS_BN"
    if ys_key in preds and ts_key in preds and preds[ts_key]["pred"] > 0:
        yr = preds[ys_key]["pred"] / preds[ts_key]["pred"]
        st.caption(f"**Yield-to-Tensile Ratio (YS/TS)**: `{yr:.3f}` {'(Complies with seismic/ductility standard < 0.85)' if yr <= 0.85 else '(Caution: High Y/T ratio > 0.85)'}")
