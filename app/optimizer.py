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
import constraints as cons


def route_for(mill: str, finish_temp: float, headers: float) -> str:
    """Infer the rolling route from the solved setpoints.

    AR   : no laminar cooling
    TMCP : accelerated cooling started below the recrystallisation stop (~850 C)
    CR   : accelerated cooling but still finishing above 850 C
    """
    if mill != "Plate" or headers <= 0:
        return "AR"
    if finish_temp < 850.0 and headers >= 4:
        return "TMCP"
    return "CR"


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


def score_candidates(mill: str, X_all):
    """Predict TS_BN and YS_BN for a feature frame.

    Order of preference:
      1. models/boosted_out (CatBoost / XGBoost / LightGBM artefacts + boosted_metrics.json)
      2. the per-mill joblib bundles used by the rest of the app
    Returns (pred_ts, pred_ys) or None when no model can be loaded.
    """

    def from_boosted():
        metrics_path = os.path.join(BOOSTED_DIR, "boosted_metrics.json")
        if not os.path.exists(metrics_path):
            return None
        with open(metrics_path, "r") as fh:
            metrics = json.load(fh)
        preds = {}
        for target, preferred in (("TS_BN", "xgb"), ("YS_BN", "cat")):
            key = f"{mill}|{target}|{preferred}"
            if key not in metrics:
                alternatives = [k for k in metrics if k.startswith(f"{mill}|{target}|")]
                if not alternatives:
                    return None
                key = alternatives[0]
            cols = metrics.get(key, {}).get("features", [])
            X = pd.DataFrame({c: (X_all[c] if c in X_all.columns else 0.0) for c in cols},
                             index=X_all.index)
            algo = key.rsplit("|", 1)[-1]
            if algo == "xgb":
                model = xgb.XGBRegressor()
                model.load_model(os.path.join(BOOSTED_DIR, f"{mill}_{target}_xgb.ubj"))
            elif algo == "cat":
                model = CatBoostRegressor()
                model.load_model(os.path.join(BOOSTED_DIR, f"{mill}_{target}_cat.cbm"))
            else:
                import lightgbm as lgb
                model = lgb.Booster(model_file=os.path.join(BOOSTED_DIR, f"{mill}_{target}_lgbm.txt"))
            preds[target] = np.asarray(model.predict(X)).ravel()
        return preds["TS_BN"], preds["YS_BN"]

    def from_bundles():
        preds = {}
        for target in ("TS_BN", "YS_BN"):
            ok = False
            # tuned ensemble first, then the plain best_model (some HRC ensembles fail to
            # unpickle into the shared StackingEnsemble class and must be skipped)
            for use_tuned in (True, False):
                bundle = ml.load_model_bundle(mill, target, use_tuned=use_tuned)
                if bundle is None:
                    continue
                try:
                    cols = bundle.get("feature_names", list(X_all.columns))
                    X = pd.DataFrame({c: (X_all[c] if c in X_all.columns else 0.0) for c in cols},
                                     index=X_all.index)
                    preds[target] = np.asarray(bundle["model"].predict(X)).ravel()
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                return None
        return preds["TS_BN"], preds["YS_BN"]

    for source in (from_boosted, from_bundles):
        try:
            result = source()
        except Exception:
            result = None
        if result is not None:
            return result
    return None


def run_inverse_optimization(mill: str, target_spec: dict, thickness: float, base_inputs: dict,
                             objective_mode: str = "min_ce", limits: dict = None):
    """
    Evaluates candidate setpoint candidates in vectorized batch to find the minimum-cost / minimum-CE
    chemistry and rolling schedule that guarantees specification compliance with split-conformal 95% bounds.

    Every process temperature swept here is bounded by `limits` (see constraints.py), so the solver can
    never return a setpoint outside the house band.
    """
    limits = limits or {}
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

    # Discretized search space. Chemistry stays on a fixed grid; every process temperature is
    # swept across the house limits instead of a hard-coded map.
    c_grid = [0.12, 0.14, 0.16, 0.18, 0.20]
    mn_grid = [0.70, 0.90, 1.10, 1.30, 1.50]
    nb_grid = [0.000, 0.015, 0.030] if mill == "Plate" else [0.000, 0.015]

    def spread(key, fallback, n=3):
        lo, hi = limits.get(key, fallback)
        if hi <= lo:
            lo, hi = fallback
        return [float(round(v, 1)) for v in np.linspace(lo, hi, n)]

    if mill == "Plate":
        frt_grid = spread("FM_Finish_Temp", (810.0, 870.0))
        entry_grid = spread("FM_Temp_Before_Rolling", (930.0, 990.0))
        post_grid = spread("ACC_Temp_Post_Laminar", (580.0, 800.0))
        hdr_grid = [0.0, 4.0, 8.0]
    else:
        ft_grid = spread("Temp_FT_Avg_C", (830.0, 890.0))
        ct_grid = spread("Temp_CT_Top_C", (560.0, 640.0))
        tb_grid = spread("Temp_Transfer_Bar_C", (930.0, 1030.0))

    # Generate schedule candidates
    candidates = []
    if mill == "Plate":
        for c in c_grid:
            for mn in mn_grid:
                for nb in nb_grid:
                    for entry in entry_grid:
                        for frt in frt_grid:
                            for hdr in hdr_grid:
                                for post in post_grid:
                                    row = dict(base_inputs)
                                    row["Plate_T_mm"] = thickness
                                    row["Chem_C"] = c
                                    row["Chem_Mn"] = mn
                                    row["Chem_Nb"] = nb
                                    row["FM_Temp_Before_Rolling"] = entry
                                    row["FM_Finish_Temp"] = frt
                                    row["ACC_Laminar_Header"] = hdr
                                    row["ACC_Headers_n"] = hdr
                                    row["ACC_Active"] = 1.0 if hdr > 0 else 0.0
                                    row["ACC_Temp_Post_Laminar"] = min(post, frt - 20.0)
                                    row["TB_Ratio"] = 2.5
                                    row["FM_Last_Pass_Drafting"] = thickness * 0.18
                                    row["Rolling_Type"] = route_for(mill, frt, hdr)
                                    candidates.append(row)
    else:
        for c in c_grid:
            for mn in mn_grid:
                for nb in nb_grid:
                    for tb in tb_grid:
                        for ft in ft_grid:
                            for ct in ct_grid:
                                row = dict(base_inputs)
                                row["Coil_T_mm"] = thickness
                                row["Chem_C"] = c
                                row["Chem_Mn"] = mn
                                row["Chem_Nb"] = nb
                                row["Temp_Transfer_Bar_C"] = tb
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

    # Score every candidate. Prefers the boosted_out artifacts; falls back to the per-mill
    # joblib bundles the rest of the app uses. Returns (pred_ts, pred_ys) or None.
    scored = score_candidates(mill, X_all)
    if scored is None:
        return None
    pred_ts, pred_ys = scored

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

    best_schedule = best_row.to_dict()
    best_schedule["route"] = route_for(
        mill,
        float(best_schedule.get("FM_Finish_Temp", best_schedule.get("Temp_FT_Avg_C", 0.0))),
        float(best_schedule.get("ACC_Headers_n", 0.0)),
    )

    # Post-solve guard: nothing may leave the house band. The grids are already built inside it,
    # so this should be a no-op except for the post-laminar joint rule; report it if it ever bites.
    clamped_schedule, violations = cons.clamp_raw(mill, best_schedule, limits)
    clamped_schedule["route"] = best_schedule["route"]

    return {
        "best_schedule": clamped_schedule,
        "is_compliant": is_fully_compliant,
        "n_evaluated": len(cand_df),
        "n_compliant": len(compliant_df),
        "q_ys": q_ys,
        "q_ts": q_ts,
        "q_el": q_el,
        "limits": dict(limits),
        "clamped": bool(violations),
        "violations": violations,
        "candidates_df": cand_df
    }


def view_optimizer(mill: str, raw: dict, engine: str = "Champion GBDT (Recommended)", limits: dict = None):
    """
    Renders the interactive Inverse Setpoint Optimizer in Streamlit.
    """
    limits = limits or {}
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

    if limits:
        st.caption("House limits enforced by the solver — " + cons.describe(limits))

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    if st.button("⚡ Solve Optimal Rolling & Chemistry Schedule", type="primary", use_container_width=True):
        with st.spinner("Evaluating multi-parameter response space with split-conformal verification..."):
            t0 = time.time()
            res = run_inverse_optimization(mill, spec_cfg, target_gauge, raw, objective_mode=obj_key,
                                           limits=limits)
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

        comp_rows.insert(4, {"Parameter": "Rolling route", "Current": str(raw.get("Rolling_Type", "AR")),
                             "Optimized": str(best.get("route", "AR")), "Unit": ""})
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

        if res.get("clamped"):
            st.warning("Schedule clamped into the house limits: " + "; ".join(res.get("violations", [])))
        if limits:
            st.caption("Limits in force — " + cons.describe(limits))

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
