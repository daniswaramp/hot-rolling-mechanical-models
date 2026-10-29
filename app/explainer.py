"""
Explainability & Applicability Domain Module.
Provides native TreeSHAP feature attributions, metallurgical factor groupings,
and historical P01-P99 out-of-distribution (OOD) envelope monitoring.
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor, Pool

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
BOOSTED_DIR = os.path.join(MODELS_DIR, "boosted_out")


@st.cache_data
def get_applicability_domain():
    """Loads historical P01, P99, min, max envelopes for Plate and HRC."""
    path = os.path.join(MODELS_DIR, "applicability_domain.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_boosted_metrics():
    """Loads feature index and champion performance metadata."""
    path = os.path.join(BOOSTED_DIR, "boosted_metrics.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def check_applicability_domain(mill: str, raw_inputs: dict) -> tuple[list[dict], list[str]]:
    """
    Checks whether the input schedule extrapolates outside historical 98% mill envelope (P01 - P99).
    Also checks metallurgy domain constraints (e.g. SA 516 Gr 70 boiler domain for Normalized Plate).
    """
    domain = get_applicability_domain().get(mill, {})
    warnings = []
    notices = []

    # 1. Parameter Quantile Checks
    for param, bounds in domain.items():
        if param in raw_inputs:
            try:
                val = float(raw_inputs[param])
                p01 = bounds["p01"]
                p99 = bounds["p99"]
                if val < p01:
                    warnings.append({
                        "param": param,
                        "val": val,
                        "bound": p01,
                        "direction": "below historical 1st percentile"
                    })
                elif val > p99:
                    warnings.append({
                        "param": param,
                        "val": val,
                        "bound": p99,
                        "direction": "above historical 99th percentile"
                    })
            except (ValueError, TypeError):
                continue

    # 2. Metallurgy Physical Domain Notice: Plate Normalizing
    if mill == "Plate":
        c = float(raw_inputs.get("Chem_C", 0.15))
        mn = float(raw_inputs.get("Chem_Mn", 0.90))
        cr = float(raw_inputs.get("Chem_Cr", 0.02))
        mo = float(raw_inputs.get("Chem_Mo", 0.005))
        v = float(raw_inputs.get("Chem_V", 0.005))
        ni = float(raw_inputs.get("Chem_Ni", 0.02))
        cu = float(raw_inputs.get("Chem_Cu", 0.02))
        ce = c + (mn / 6.0) + ((cr + mo + v) / 5.0) + ((ni + cu) / 15.0)

        if ce < 0.38:
            notices.append(
                f"**Domain Notice (Alloy Class)**: Slab Carbon Equivalent (CE_IIW = {ce:.3f} < 0.38) "
                f"matches commodity mild structural steel (SS400 / A36 / SM400). "
                f"In plant practice, these grades are delivered strictly As-Rolled / TMCP. "
                f"Furnace normalizing is standard for pressure vessel boiler grades (SA 516 Gr 70, CE ≥ 0.38). "
                f"Predicted normalized properties show physical equilibrium softening if furnace-normalized."
            )

    return warnings, notices


def compute_tree_shap_attribution(mill: str, target_key: str, X_row: pd.DataFrame) -> dict:
    """
    Computes exact TreeSHAP additive feature attributions:
    Prediction = Base_Value + Sum(SHAP_values)
    """
    metrics = get_boosted_metrics()
    
    # Identify champion family
    best_fam = None
    best_rmse = 999999.0
    best_feats = None
    for fam in ["xgb", "lgbm", "cat"]:
        k = f"{mill}|{target_key}|{fam}"
        if k in metrics and metrics[k]["rmse"] < best_rmse:
            best_rmse = metrics[k]["rmse"]
            best_fam = fam
            best_feats = metrics[k].get("features", [])

    if not best_fam or not best_feats:
        return {}

    # Align features
    X_aligned = pd.DataFrame(index=X_row.index)
    for col in best_feats:
        X_aligned[col] = X_row[col] if col in X_row.columns else 0.0

    try:
        if best_fam == "xgb":
            m_path = os.path.join(BOOSTED_DIR, f"{mill}_{target_key}_xgb.ubj")
            if not os.path.exists(m_path):
                return {}
            model = xgb.XGBRegressor()
            model.load_model(m_path)
            contribs = model.get_booster().predict(xgb.DMatrix(X_aligned), pred_contribs=True)[0]
            base_val = float(contribs[-1])
            shap_vals = contribs[:-1]

        elif best_fam == "cat":
            m_path = os.path.join(BOOSTED_DIR, f"{mill}_{target_key}_cat.cbm")
            if not os.path.exists(m_path):
                return {}
            model = CatBoostRegressor()
            model.load_model(m_path)
            contribs = model.get_feature_importance(Pool(X_aligned), type="ShapValues")[0]
            base_val = float(contribs[-1])
            shap_vals = contribs[:-1]

        elif best_fam == "lgbm":
            m_path = os.path.join(BOOSTED_DIR, f"{mill}_{target_key}_lgbm.txt")
            if not os.path.exists(m_path):
                return {}
            booster = lgb.Booster(model_file=m_path)
            contribs = booster.predict(X_aligned, pred_contrib=True)[0]
            base_val = float(contribs[-1])
            shap_vals = contribs[:-1]

        else:
            return {}

        feature_series = pd.Series(shap_vals, index=best_feats)

        # Metallurgical Grouping
        groups = {
            "Chemistry & Solid Solution": ["Chem_C", "Chem_Mn", "Chem_Si", "Chem_Cu", "Chem_Ni", "Chem_Cr", "Chem_Mo", "Chem_N", "sigma_ss", "CE_IIW", "P_cm", "Chem_P", "Chem_S"],
            "Microalloy Precipitation": ["Chem_Nb", "Chem_V", "Chem_Ti", "Chem_Al", "Microalloy_Total", "Nb_eff", "Ti_to_N_ratio"],
            "Hall-Petch & Geometry": ["Plate_T_mm", "Coil_T_mm", "Plate_W_mm", "Coil_W_mm", "Slab_T_mm", "TB_Ratio", "TB_Thick_mm", "inv_sqrt_t", "W_over_T", "d_alpha_est", "HP_term"],
            "Thermomechanical & Cooling": ["FM_Finish_Temp", "Temp_FT_Avg_C", "ACC_Temp_Post_Laminar", "Temp_CT_Top_C", "Temp_CT_Bot_C", "ACC_Headers_n", "ACC_Active", "Cooling_Rate_Proxy", "Ar3", "Tnr", "FRT_minus_Ar3", "FRT_minus_Tnr", "Tpost_minus_Ar3", "eps_total", "eps_FM", "eps_last_pass", "FM_Last_Pass_Drafting"]
        }

        grouped_contribs = {}
        for grp_name, col_list in groups.items():
            val = float(feature_series[feature_series.index.isin(col_list)].sum())
            grouped_contribs[grp_name] = round(val, 2)

        return {
            "target": target_key,
            "family": best_fam,
            "base_val": round(base_val, 2),
            "pred_val": round(base_val + float(feature_series.sum()), 2),
            "features": feature_series.sort_values(ascending=False).to_dict(),
            "grouped": grouped_contribs
        }

    except Exception as e:
        return {"error": str(e)}


def render_feature_contributions_ui(mill: str, target_key: str, X_row: pd.DataFrame, unit: str = "MPa"):
    """
    Renders an interactive feature attribution breakdown card in Streamlit.
    """
    attr = compute_tree_shap_attribution(mill, target_key, X_row)
    if not attr or "error" in attr:
        st.info("Feature attribution unavailable for this configuration.")
        return

    st.markdown(f"#### 🔬 Metallurgical Feature Attributions ({target_key})")
    st.caption(f"Deconstructs prediction via TreeSHAP against baseline heat mean ({attr['family'].upper()} model). "
               f"Baseline: **{attr['base_val']:.1f} {unit}** → Final Predicted: **{attr['pred_val']:.1f} {unit}**")

    # 1. Grouped Metallurgical Drivers
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for idx, (grp, val) in enumerate(attr["grouped"].items()):
        color = "#065f46" if val >= 0 else "#991b1b"
        sign = "+" if val >= 0 else ""
        cols[idx].metric(label=grp, value=f"{sign}{val:.1f} {unit}")

    # 2. Top Individual Drivers (Waterfall / Table)
    top_pos = {k: v for k, v in attr["features"].items() if v > 0.5}
    top_neg = {k: v for k, v in attr["features"].items() if v < -0.5}

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top Positive Drivers (+)**")
        pos_items = list(top_pos.items())[:6]
        if pos_items:
            for feat, val in pos_items:
                st.markdown(f"- `{feat}`: **+{val:.2f} {unit}**")
        else:
            st.caption("No significant positive drivers.")

    with col_b:
        st.markdown("**Top Negative / Softening Drivers (−)**")
        neg_items = list(top_neg.items())[-6:]
        if neg_items:
            for feat, val in reversed(neg_items):
                st.markdown(f"- `{feat}`: **{val:.2f} {unit}**")
        else:
            st.caption("No significant softening drivers.")
