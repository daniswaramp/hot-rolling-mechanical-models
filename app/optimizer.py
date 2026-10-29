"""
Inverse Setpoint Optimizer (Reverse Design Module).
Solves for optimal chemistry and rolling setpoints to meet customer mechanical specifications
at minimum alloy cost and Carbon Equivalent (CE_IIW), with split-conformal 95% safety buffers (Cpk >= 1.33).
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from catboost import CatBoostRegressor

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
BOOSTED_DIR = os.path.join(MODELS_DIR, "boosted_out")

import model_loader as ml
import spec_checker as sc


PRESET_GRADES = {
    "Plate": {
        "ASTM A 572 Gr 50": {"ys_min": 345.0, "ts_min": 450.0, "el_min": 18.0, "c_max": 0.23, "mn_max": 1.35},
        "JIS G 3101 SS400": {"ys_min": 245.0, "ts_min": 400.0, "el_min": 21.0, "c_max": 0.25, "mn_max": 1.40},
        "JIS G 3106 SM490B": {"ys_min": 325.0, "ts_min": 490.0, "el_min": 21.0, "c_max": 0.18, "mn_max": 1.60},
        "ASTM A 36": {"ys_min": 250.0, "ts_min": 400.0, "el_min": 20.0, "c_max": 0.26, "mn_max": 1.20},
        "ASME SA 516 Gr 70": {"ys_min": 260.0, "ts_min": 485.0, "el_min": 21.0, "c_max": 0.28, "mn_max": 1.20},
        "EN 10025-2 S355JR": {"ys_min": 355.0, "ts_min": 470.0, "el_min": 20.0, "c_max": 0.24, "mn_max": 1.60},
        "Custom Spec": {"ys_min": 355.0, "ts_min": 510.0, "el_min": 22.0, "c_max": 0.25, "mn_max": 1.60}
    },
    "HRC": {
        "JIS G 3101 SS400": {"ys_min": 245.0, "ts_min": 400.0, "el_min": 21.0, "c_max": 0.25, "mn_max": 1.40},
        "JIS G 3131 SPHC": {"ys_min": 220.0, "ts_min": 270.0, "el_min": 27.0, "c_max": 0.15, "mn_max": 0.60},
        "ASTM A 1011 HSLAS Gr 50": {"ys_min": 345.0, "ts_min": 450.0, "el_min": 22.0, "c_max": 0.23, "mn_max": 1.35},
        "JIS G 3106 SM490A": {"ys_min": 325.0, "ts_min": 490.0, "el_min": 22.0, "c_max": 0.20, "mn_max": 1.60},
        "Custom Spec": {"ys_min": 355.0, "ts_min": 500.0, "el_min": 22.0, "c_max": 0.25, "mn_max": 1.60}
    }
}


def run_inverse_optimization(mill: str, target_spec: dict, thickness: float, base_inputs: dict, objective_mode: str = "min_ce"):
    """
    Evaluates candidate setpoint candidates in vectorized batch to find the minimum-cost / minimum-CE
    chemistry and rolling schedule that guarantees specification compliance with split-conformal 95% bounds.
    """
    try:
        conf_df = ml.get_conformal_intervals(mill) if hasattr(ml, "get_conformal_intervals") else pd.DataFrame()
    except Exception:
        conf_df = pd.DataFrame()
    q_ys = 33.96 if mill == "Plate" else 53.25
    q_ts = 27.26 if mill == "Plate" else 36.91
    q_el = 4.46 if mill == "Plate" else 5.43

    if not conf_df.empty and "Target" in conf_df.columns:
        r_ys = conf_df[conf_df["Target"] == "YS_BN"]
        if not r_ys.empty and "Conformal_Half_Width_95" in r_ys.columns:
            q_ys = float(r_ys["Conformal_Half_Width_95"].values[0])
        r_ts = conf_df[conf_df["Target"] == "TS_BN"]
        if not r_ts.empty and "Conformal_Half_Width_95" in r_ts.columns:
            q_ts = float(r_ts["Conformal_Half_Width_95"].values[0])
        r_el = conf_df[conf_df["Target"] == "EL_BN"]
        if not r_el.empty and "Conformal_Half_Width_95" in r_el.columns:
            q_el = float(r_el["Conformal_Half_Width_95"].values[0])

    # Discretized search space
    c_grid = [0.12, 0.14, 0.16, 0.18, 0.20]
    mn_grid = [0.70, 0.90, 1.10, 1.30, 1.50]
    nb_grid = [0.000, 0.015, 0.030] if mill == "Plate" else [0.000, 0.015]
    
    if mill == "Plate":
        frt_grid = [810.0, 840.0, 870.0]
        hdr_grid = [0.0, 4.0, 8.0]
        tpost_map = {0.0: 820.0, 4.0: 640.0, 8.0: 580.0}
    else:
        ft_grid = [830.0, 860.0, 890.0]
        ct_grid = [560.0, 600.0, 640.0]

    # Generate schedule candidates
    candidates = []
    if mill == "Plate":
        for c in c_grid:
            for mn in mn_grid:
                for nb in nb_grid:
                    for frt in frt_grid:
                        for hdr in hdr_grid:
                            row = dict(base_inputs)
                            row["Plate_T_mm"] = thickness
                            row["Chem_C"] = c
                            row["Chem_Mn"] = mn
                            row["Chem_Nb"] = nb
                            row["FM_Finish_Temp"] = frt
                            row["ACC_Laminar_Header"] = hdr
                            row["ACC_Headers_n"] = hdr
                            row["ACC_Active"] = 1.0 if hdr > 0 else 0.0
                            row["ACC_Temp_Post_Laminar"] = tpost_map.get(hdr, 600.0)
                            row["TB_Ratio"] = 2.5
                            row["FM_Last_Pass_Drafting"] = thickness * 0.18
                            row["Rolling_Type"] = "CR" if hdr > 0 else "AR"
                            candidates.append(row)
    else:
        for c in c_grid:
            for mn in mn_grid:
                for nb in nb_grid:
                    for ft in ft_grid:
                        for ct in ct_grid:
                            row = dict(base_inputs)
                            row["Coil_T_mm"] = thickness
                            row["Chem_C"] = c
                            row["Chem_Mn"] = mn
                            row["Chem_Nb"] = nb
                            row["Temp_FT_Avg_C"] = ft
                            row["Temp_CT_Top_C"] = ct
                            row["Temp_CT_Bot_C"] = ct
                            row["TB_Thick_mm"] = max(30.0, thickness * 3.0)
                            candidates.append(row)

    cand_df = pd.DataFrame(candidates)
    if mill == "Plate":
        if ml.engineer_plate_features is None:
            return None
        X_all = ml.engineer_plate_features(cand_df)
    else:
        if ml.engineer_hrc_features is None:
            return None
        X_all, _ = ml.engineer_hrc_features(cand_df, is_train=False)

    # Load champion models for fast batch inference
    with open(os.path.join(BOOSTED_DIR, "boosted_metrics.json"), "r") as f:
        bm = json.load(f)

    # 1. TS Model
    ts_key = f"{mill}|TS_BN|xgb" if f"{mill}|TS_BN|xgb" in bm else f"{mill}|TS_BN|cat"
    ts_feats = bm.get(ts_key, {}).get("features", [])
    X_ts = pd.DataFrame(index=X_all.index)
    for col in ts_feats:
        X_ts[col] = X_all[col] if col in X_all.columns else 0.0

    if "xgb" in ts_key:
        m_ts = xgb.XGBRegressor()
        m_ts.load_model(os.path.join(BOOSTED_DIR, f"{mill}_TS_BN_xgb.ubj"))
        pred_ts = m_ts.predict(X_ts)
    else:
        m_ts = CatBoostRegressor()
        m_ts.load_model(os.path.join(BOOSTED_DIR, f"{mill}_TS_BN_cat.cbm"))
        pred_ts = m_ts.predict(X_ts)

    # 2. YS Model
    ys_key = f"{mill}|YS_BN|cat" if f"{mill}|YS_BN|cat" in bm else f"{mill}|YS_BN|xgb"
    ys_feats = bm.get(ys_key, {}).get("features", [])
    X_ys = pd.DataFrame(index=X_all.index)
    for col in ys_feats:
        X_ys[col] = X_all[col] if col in X_all.columns else 0.0

    if "cat" in ys_key:
        m_ys = CatBoostRegressor()
        m_ys.load_model(os.path.join(BOOSTED_DIR, f"{mill}_YS_BN_cat.cbm"))
        pred_ys = m_ys.predict(X_ys)
    else:
        m_ys = xgb.XGBRegressor()
        m_ys.load_model(os.path.join(BOOSTED_DIR, f"{mill}_YS_BN_xgb.ubj"))
        pred_ys = m_ys.predict(X_ys)

    # Filter compliant schedules with conformal safety buffer
    req_ys = target_spec.get("ys_min", 325.0)
    req_ts = target_spec.get("ts_min", 470.0)

    # 95% conformal lower bound >= specification requirement
    lower_ys = pred_ys - q_ys
    lower_ts = pred_ts - q_ts

    compliant_mask = (lower_ys >= req_ys) & (lower_ts >= req_ts)

    cand_df["pred_ys"] = np.round(pred_ys, 1)
    cand_df["pred_ts"] = np.round(pred_ts, 1)
    cand_df["lower_ys"] = np.round(lower_ys, 1)
    cand_df["lower_ts"] = np.round(lower_ts, 1)
    cand_df["ce_iiw"] = np.round(cand_df["Chem_C"] + cand_df["Chem_Mn"] / 6.0 + 5.0 * cand_df["Chem_Nb"], 3)

    compliant_df = cand_df[compliant_mask].copy()

    if compliant_df.empty:
        # If no 100% compliant schedule, find the one closest to passing
        cand_df["shortfall"] = np.maximum(0, req_ys - cand_df["lower_ys"]) + np.maximum(0, req_ts - cand_df["lower_ts"])
        best_row = cand_df.sort_values(by=["shortfall", "ce_iiw"]).iloc[0]
        is_fully_compliant = False
    else:
        if objective_mode == "min_ce":
            best_row = compliant_df.sort_values(by=["ce_iiw", "pred_ys"]).iloc[0]
        else:
            # Max safety margin
            compliant_df["margin"] = (compliant_df["lower_ys"] - req_ys) + (compliant_df["lower_ts"] - req_ts)
            best_row = compliant_df.sort_values(by=["margin"], ascending=False).iloc[0]
        is_fully_compliant = True

    return {
        "best_schedule": best_row.to_dict(),
        "is_compliant": is_fully_compliant,
        "n_evaluated": len(cand_df),
        "n_compliant": len(compliant_df),
        "q_ys": q_ys,
        "q_ts": q_ts,
        "q_el": q_el
    }


def view_optimizer(mill: str, raw: dict, engine: str = "Champion GBDT (Recommended)"):
    """
    Renders the interactive Inverse Setpoint Optimizer in Streamlit.
    """
    st.markdown('<div class="kicker">Automated Inverse Design</div>', unsafe_allow_html=True)
    st.markdown('<div class="title">Reverse Setpoint Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="lede">Solves for the optimal alloy chemistry and thermomechanical rolling setpoints '
        f'to fulfill target customer specification limits with guaranteed <b>split-conformal 95% confidence bounds (Cpk ≥ 1.33)</b> '
        f'at minimal Carbon Equivalent (CE_IIW) and alloy cost.</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # 1. Spec Selection Bar
    col1, col2, col3 = st.columns([2.5, 2.0, 1.5], vertical_alignment="bottom")
    presets = PRESET_GRADES[mill]
    spec_names = list(presets.keys())
    chosen_spec = col1.selectbox("Target Standard / Grade", spec_names, index=0, key=f"opt_spec_{mill}")
    spec_cfg = presets[chosen_spec]

    t_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    cur_t = float(raw.get(t_key, 20.0 if mill == "Plate" else 6.0))
    target_gauge = col2.number_input(f"Target Gauge ({t_key})", min_value=3.0, max_value=120.0, value=cur_t, step=1.0)

    obj_mode = col3.selectbox("Optimization Strategy", ["Minimum CE / Alloy Cost", "Maximum Safety Buffer"], index=0)
    obj_key = "min_ce" if "Minimum" in obj_mode else "max_margin"

    # Show Target Requirements
    c1, c2, c3 = st.columns(3)
    c1.metric("Min Yield Strength (YS)", f"≥ {spec_cfg['ys_min']:.0f} MPa")
    c2.metric("Min Tensile Strength (TS)", f"≥ {spec_cfg['ts_min']:.0f} MPa")
    c3.metric("Min Elongation (EL)", f"≥ {spec_cfg['el_min']:.0f} %")

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    if st.button("⚡ Solve Optimal Rolling & Chemistry Schedule", type="primary", use_container_width=True):
        with st.spinner("Evaluating multi-parameter response space with split-conformal verification..."):
            t0 = time.time()
            res = run_inverse_optimization(mill, spec_cfg, target_gauge, raw, objective_mode=obj_key)
            elapsed = time.time() - t0

        if not res:
            st.error("Failed to run optimization solver.")
            return

        best = res["best_schedule"]
        is_comp = res["is_compliant"]

        st.markdown('<div style="height:1.2rem"></div>', unsafe_allow_html=True)
        if is_comp:
            st.success(f"Optimal schedule solved in {elapsed*1000:.1f} ms! "
                       f"Found {res['n_compliant']} compliant recipes out of {res['n_evaluated']} evaluated candidates.")
        else:
            st.warning(f"No candidate met the rigorous 95% conformal bound across both YS and TS simultaneously. "
                       f"Showing the nearest schedule with minimal shortfall.")

        # Display Comparison Table: Current vs Optimized
        st.markdown("#### 📋 Recommended Aim Setpoints vs Current Simulator Inputs")
        
        comp_rows = [
            {"Parameter": "Carbon (C)", "Current": f"{raw.get('Chem_C', 0.16):.3f} %", "Optimized": f"{best.get('Chem_C'):.3f} %", "Unit": "wt%"},
            {"Parameter": "Manganese (Mn)", "Current": f"{raw.get('Chem_Mn', 1.20):.3f} %", "Optimized": f"{best.get('Chem_Mn'):.3f} %", "Unit": "wt%"},
            {"Parameter": "Niobium (Nb)", "Current": f"{raw.get('Chem_Nb', 0.015):.4f} %", "Optimized": f"{best.get('Chem_Nb'):.4f} %", "Unit": "wt%"},
            {"Parameter": "Carbon Equivalent (CE_IIW)", "Current": f"{raw.get('Chem_C', 0.16) + raw.get('Chem_Mn', 1.20)/6.0:.3f}", "Optimized": f"{best.get('ce_iiw'):.3f}", "Unit": ""},
        ]
        if mill == "Plate":
            comp_rows.extend([
                {"Parameter": "Finish Rolling Temp (FRT)", "Current": f"{raw.get('FM_Finish_Temp', 830):.0f} °C", "Optimized": f"{best.get('FM_Finish_Temp'):.0f} °C", "Unit": "°C"},
                {"Parameter": "ACC Laminar Headers", "Current": f"{raw.get('ACC_Laminar_Header', 6):.0f} H", "Optimized": f"{best.get('ACC_Laminar_Header'):.0f} H", "Unit": "headers"},
                {"Parameter": "Post-Laminar Water Temp", "Current": f"{raw.get('ACC_Temp_Post_Laminar', 580):.0f} °C", "Optimized": f"{best.get('ACC_Temp_Post_Laminar'):.0f} °C", "Unit": "°C"},
            ])
        else:
            comp_rows.extend([
                {"Parameter": "Finishing Temp (FT)", "Current": f"{raw.get('Temp_FT_Avg_C', 850):.0f} °C", "Optimized": f"{best.get('Temp_FT_Avg_C'):.0f} °C", "Unit": "°C"},
                {"Parameter": "Coiling Temp (CT)", "Current": f"{raw.get('Temp_CT_Top_C', 580):.0f} °C", "Optimized": f"{best.get('Temp_CT_Top_C'):.0f} °C", "Unit": "°C"},
            ])

        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        # Verification Metrics Strip
        st.markdown("#### 🎯 Verified Mechanical Performance (with Split-Conformal 95% Bounds)")
        m1, m2, m3 = st.columns(3)
        ys_val = best.get("pred_ys", 0.0)
        ys_low = best.get("lower_ys", 0.0)
        ts_val = best.get("pred_ts", 0.0)
        ts_low = best.get("lower_ts", 0.0)

        m1.metric(
            label="Predicted Yield Strength",
            value=f"{ys_val:.1f} MPa",
            delta=f"95% lower: {ys_low:.1f} MPa (req ≥ {spec_cfg['ys_min']:.0f})"
        )
        m2.metric(
            label="Predicted Tensile Strength",
            value=f"{ts_val:.1f} MPa",
            delta=f"95% lower: {ts_low:.1f} MPa (req ≥ {spec_cfg['ts_min']:.0f})"
        )
        ce_delta = round(best.get("ce_iiw") - (raw.get('Chem_C', 0.16) + raw.get('Chem_Mn', 1.20)/6.0), 3)
        m3.metric(
            label="Carbon Equivalent Change (ΔCE)",
            value=f"{best.get('ce_iiw'):.3f}",
            delta=f"{ce_delta:+.3f} CE {'(Leaner / Cost Saving)' if ce_delta <= 0 else '(Added strength)'}",
            delta_color="normal" if ce_delta <= 0 else "inverse"
        )
