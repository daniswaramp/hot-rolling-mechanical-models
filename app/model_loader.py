"""
Model Loader and Domain Feature Pipeline Service for Streamlit.
Provides namespace-safe unpickling for Plate (script.ml.*) and HRC (ml.*) artifacts,
cached model singletons, feature computation, and real-time inference.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
PLATE_DIR = os.path.join(REPO_ROOT, "Plate")
HRC_SCRIPT_DIR = os.path.join(REPO_ROOT, "HRC", "script")

# Register path roots for unpickling:
# - Plate models pickled under 'script.ml.*' resolve from PLATE_DIR
# - HRC models pickled under 'ml.*' resolve from HRC_SCRIPT_DIR
for p in [PLATE_DIR, HRC_SCRIPT_DIR, REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from script.ml.features import compute_metallurgical_features as engineer_plate_features
except ImportError:
    engineer_plate_features = None

try:
    from ml.features import engineer_hrc_features
except ImportError:
    engineer_hrc_features = None

# Metadata definitions
TARGET_CONFIG = {
    "Plate": {
        "YS_BN": {"name": "Yield Strength (Before Normalizing)", "unit": "MPa", "tol": 20.0},
        "TS_BN": {"name": "Tensile Strength (Before Normalizing)", "unit": "MPa", "tol": 20.0},
        "EL_BN": {"name": "Elongation (Before Normalizing)", "unit": "%", "tol": 2.5},
        "YS_AN": {"name": "Yield Strength (After Normalizing)", "unit": "MPa", "tol": 20.0},
        "TS_AN": {"name": "Tensile Strength (After Normalizing)", "unit": "MPa", "tol": 20.0},
        "EL_AN": {"name": "Elongation (After Normalizing)", "unit": "%", "tol": 2.5},
        "Impact_0C": {"name": "Charpy Impact @ 0 °C", "unit": "J", "tol": 15.0},
        "Impact_m20C": {"name": "Charpy Impact @ -20 °C", "unit": "J", "tol": 15.0},
        "Impact_m40C": {"name": "Charpy Impact @ -40 °C", "unit": "J", "tol": 15.0},
        "Z_Test": {"name": "Through-Thickness Reduction of Area (Z-Test)", "unit": "%", "tol": 5.0},
    },
    "HRC": {
        "TS_BN": {"name": "Tensile Strength (Before Normalizing)", "unit": "MPa", "tol": 20.0},
        "YS_BN": {"name": "Yield Strength (Before Normalizing)", "unit": "MPa", "tol": 20.0},
        "EL_BN": {"name": "Elongation (Before Normalizing)", "unit": "%", "tol": 2.5},
    }
}


@st.cache_resource(show_spinner="Loading metallurgical models...")
def load_model_bundle(mill_type: str, target_key: str, use_tuned: bool = True):
    """
    Loads model bundle from disk.
    Tuned ensemble is preferred, falls back to best baseline model if tuned is unavailable.
    """
    base_folder = os.path.join(REPO_ROOT, mill_type, "models")
    preferred_suffix = "tuned_ensemble.joblib" if use_tuned else "best_model.joblib"
    model_path = os.path.join(base_folder, f"{target_key}_{preferred_suffix}")
    
    if not os.path.exists(model_path):
        model_path = os.path.join(base_folder, f"{target_key}_best_model.joblib")
        
    if not os.path.exists(model_path):
        return None
        
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.warning(f"Failed to load {mill_type} model for {target_key}: {e}")
        return None


@st.cache_data
def get_ci_data(mill_type: str) -> pd.DataFrame:
    """Reads 95% Bootstrap Confidence Interval report."""
    if mill_type == "Plate":
        csv_path = os.path.join(REPO_ROOT, "Plate", "reports", "bootstrap_confidence_intervals.csv")
    else:
        csv_path = os.path.join(REPO_ROOT, "HRC", "reports", "hrc_bootstrap_confidence_intervals.csv")
        
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def get_conformal_intervals(mill_type: str) -> pd.DataFrame:
    """Reads 95% Split-Conformal Prediction Interval report."""
    csv_path = os.path.join(REPO_ROOT, mill_type, "reports", "conformal_intervals.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def get_leaderboard_data(mill_type: str) -> pd.DataFrame:
    """Reads model comparison leaderboard."""
    if mill_type == "Plate":
        csv_path = os.path.join(REPO_ROOT, "Plate", "reports", "tuning_and_ensemble_leaderboard.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(REPO_ROOT, "Plate", "reports", "baseline_leaderboard.csv")
    else:
        csv_path = os.path.join(REPO_ROOT, "HRC", "reports", "hrc_tuning_leaderboard.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(REPO_ROOT, "HRC", "reports", "hrc_baseline_leaderboard.csv")
            
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def get_nlr_formulas(mill_type: str) -> str:
    """Reads fitted non-linear constitutive formulas."""
    fname = "nlr_constitutive_formulas.txt" if mill_type == "Plate" else "hrc_nlr_constitutive_formulas.txt"
    p = os.path.join(REPO_ROOT, mill_type, "reports", fname)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    return "Formulas not found."


def compute_physical_indicators(raw_dict: dict, mill_type: str) -> dict:
    """
    Computes real-time physical metallurgy indicators for user feedback.
    """
    c = float(raw_dict.get("Chem_C", 0.15))
    mn = float(raw_dict.get("Chem_Mn", 0.90))
    si = float(raw_dict.get("Chem_Si", 0.20))
    p = float(raw_dict.get("Chem_P", 0.015))
    s = float(raw_dict.get("Chem_S", 0.005))
    cu = float(raw_dict.get("Chem_Cu", 0.02))
    ni = float(raw_dict.get("Chem_Ni", 0.02))
    cr = float(raw_dict.get("Chem_Cr", 0.02))
    mo = float(raw_dict.get("Chem_Mo", 0.005))
    v = float(raw_dict.get("Chem_V", 0.005))
    nb = float(raw_dict.get("Chem_Nb", 0.002))
    ti = float(raw_dict.get("Chem_Ti", 0.002))
    n = float(raw_dict.get("Chem_N", 0.003))
    b = float(raw_dict.get("Chem_B", 0.0))
    al = float(raw_dict.get("Chem_Al", 0.025))

    # Pickering solid solution strengthening
    sigma_ss = (32.0 * mn) + (84.0 * si) + (43.0 * ni) - (30.0 * cr) + (38.0 * cu) + (80.0 * ti) + (11.0 * mo) + (354.0 * n)

    # Carbon equivalents
    ce_iiw = c + (mn / 6.0) + ((cr + mo + v) / 5.0) + ((ni + cu) / 15.0)
    pcm = c + (si / 30.0) + ((mn + cu + cr) / 20.0) + (ni / 60.0) + (mo / 15.0) + (v / 10.0) + (5.0 * b)

    # Critical transformation temperatures
    ar3 = 910.0 - (310.0 * c) - (80.0 * mn) - (20.0 * cu) - (15.0 * cr) - (55.0 * ni) - (80.0 * mo)
    ar1 = 723.0 - (10.7 * mn) - (16.9 * ni) + (29.1 * si) + (16.9 * cr)

    if mill_type == "Plate":
        frt = float(raw_dict.get("FM_Finish_Temp", 820.0))
        t_post = float(raw_dict.get("ACC_Temp_Post_Laminar", 580.0))
        plate_t = float(raw_dict.get("Plate_T_mm", 20.0))
        headers = float(raw_dict.get("ACC_Headers_n", 6.0))
        cooling_rate = ((frt - t_post) * (headers + 0.5)) / max(plate_t, 1.0)
        true_strain = np.log(float(raw_dict.get("Slab_T_mm", 200.0)) / max(plate_t, 1.0))
    else:
        ft = float(raw_dict.get("Temp_FT_Avg_C", 850.0))
        ct_top = float(raw_dict.get("Temp_CT_Top_C", 580.0))
        ct_bot = float(raw_dict.get("Temp_CT_Bot_C", 580.0))
        ct = (ct_top + ct_bot) / 2.0
        coil_t = float(raw_dict.get("Coil_T_mm", 6.0))
        tb_t = float(raw_dict.get("TB_Thick_mm", 32.0))
        cooling_rate = max(0.0, ft - ct) / max(coil_t, 1.0)
        true_strain = np.log(max(tb_t, coil_t * 1.05) / max(coil_t, 1.0))

    return {
        "sigma_ss": sigma_ss,
        "ce_iiw": ce_iiw,
        "pcm": pcm,
        "ar3": ar3,
        "ar1": ar1,
        "cooling_rate": cooling_rate,
        "true_strain": true_strain
    }



def compute_plate_normalizing_deltas(raw_inputs: dict):
    """
    Computes dynamic physical softening deltas for furnace-normalized Heavy Plate:
    Delta TS = 12.0 + 8.5 * eps_FM + 0.35 * CR_ACC
    Delta YS = 15.0 + 10.0 * eps_FM + 0.45 * CR_ACC
    Delta EL = 1.5 + 0.03 * CR_ACC
    """
    plate_t = max(float(raw_inputs.get("Plate_T_mm", 16.0)), 1.0)
    tb_t = float(raw_inputs.get("TB_Thickness_mm", plate_t * 2.0))
    eps_fm = max(0.0, float(np.log(max(tb_t, plate_t * 1.01) / plate_t)))

    frt = float(raw_inputs.get("FM_Finish_Temp", 830.0))
    t_post = float(raw_inputs.get("ACC_Temp_Post_Laminar", 600.0))
    headers = float(raw_inputs.get("ACC_Laminar_Header", raw_inputs.get("ACC_Headers_n", 0.0)))

    cr_acc = max(0.0, float((frt - t_post) / plate_t * (headers + 0.5)))

    delta_ts = 12.0 + (8.5 * eps_fm) + (0.35 * cr_acc)
    delta_ys = 15.0 + (10.0 * eps_fm) + (0.45 * cr_acc)
    delta_el = 1.5 + (0.03 * cr_acc)

    return delta_ts, delta_ys, delta_el


def apply_plate_normalizing_coupling(raw_inputs: dict, results: dict) -> dict:
    """
    Enforces physical metallurgy thermodynamic coupling:
    Normalizing at ~920 °C erases TMCP dislocation density and fast cooling grain refinement.
    Therefore:
      TS_AN <= TS_BN - delta_ts  (Always lower tensile strength)
      YS_AN <= YS_BN - delta_ys  (Always lower yield strength)
      EL_AN >= EL_BN + delta_el  (Always higher elongation / ductility)
    """
    if "TS_BN" not in results or "TS_AN" not in results:
        return results

    ts_bn = results["TS_BN"].get("pred", np.nan)
    ys_bn = results.get("YS_BN", {}).get("pred", np.nan)
    el_bn = results.get("EL_BN", {}).get("pred", np.nan)

    if not np.isfinite(ts_bn):
        return results

    delta_ts, delta_ys, delta_el = compute_plate_normalizing_deltas(raw_inputs)

    # 1. Tensile Strength Coupling
    ts_ceiling = round(ts_bn - delta_ts, 2)
    raw_ts_an = results["TS_AN"].get("pred", np.nan)
    if not np.isfinite(raw_ts_an) or raw_ts_an > ts_ceiling:
        diff_an = (
            results["TS_AN"]["ci_high"] - results["TS_AN"]["pred"]
            if np.isfinite(results["TS_AN"].get("ci_high", np.nan))
            else 10.0
        )
        results["TS_AN"]["pred"] = ts_ceiling
        results["TS_AN"]["ci_low"] = round(ts_ceiling - diff_an, 2)
        results["TS_AN"]["ci_high"] = round(ts_ceiling + diff_an, 2)
        results["TS_AN"]["note"] = f"Physically coupled to blackplate (-{delta_ts:.1f} MPa softening)"

    # 2. Yield Strength Coupling
    if np.isfinite(ys_bn) and "YS_AN" in results:
        ys_ceiling = round(ys_bn - delta_ys, 2)
        raw_ys_an = results["YS_AN"].get("pred", np.nan)
        if not np.isfinite(raw_ys_an) or raw_ys_an > ys_ceiling:
            diff_an = (
                results["YS_AN"]["ci_high"] - results["YS_AN"]["pred"]
                if np.isfinite(results["YS_AN"].get("ci_high", np.nan))
                else 10.0
            )
            results["YS_AN"]["pred"] = ys_ceiling
            results["YS_AN"]["ci_low"] = round(ys_ceiling - diff_an, 2)
            results["YS_AN"]["ci_high"] = round(ys_ceiling + diff_an, 2)
            results["YS_AN"]["note"] = f"Physically coupled to blackplate (-{delta_ys:.1f} MPa softening)"

    # 3. Elongation Coupling
    if np.isfinite(el_bn) and "EL_AN" in results:
        el_floor = round(el_bn + delta_el, 2)
        raw_el_an = results["EL_AN"].get("pred", np.nan)
        if not np.isfinite(raw_el_an) or raw_el_an < el_floor:
            diff_an = (
                results["EL_AN"]["ci_high"] - results["EL_AN"]["pred"]
                if np.isfinite(results["EL_AN"].get("ci_high", np.nan))
                else 1.2
            )
            results["EL_AN"]["pred"] = el_floor
            results["EL_AN"]["ci_low"] = round(max(0.0, el_floor - diff_an), 2)
            results["EL_AN"]["ci_high"] = round(el_floor + diff_an, 2)
            results["EL_AN"]["note"] = f"Physically coupled to blackplate (+{delta_el:.1f}% ductility gain)"

    return results


def predict_single_sample(mill_type: str, raw_inputs: dict) -> dict:
    """
    Transforms raw inputs through exact metallurgical feature engineering and
    predicts all active targets with 95% Confidence Intervals.
    """
    input_df = pd.DataFrame([raw_inputs])

    if mill_type == "Plate":
        if engineer_plate_features is None:
            raise RuntimeError("Plate feature module not found.")
        X = engineer_plate_features(input_df)
    else:
        if engineer_hrc_features is None:
            raise RuntimeError("HRC feature module not found.")
        X, _ = engineer_hrc_features(input_df, is_train=False)

    target_defs = TARGET_CONFIG[mill_type]
    ci_df = get_ci_data(mill_type)
    
    results = {}
    for tgt_key, tgt_meta in target_defs.items():
        bundle = load_model_bundle(mill_type, tgt_key, use_tuned=True)
        if bundle is None:
            continue
            
        model = bundle["model"]
        feature_cols = bundle.get("feature_names", X.columns.tolist())
        
        # Align features safely
        X_aligned = pd.DataFrame(index=X.index)
        for col in feature_cols:
            X_aligned[col] = X[col] if col in X.columns else 0.0
            
        try:
            pred_val = float(model.predict(X_aligned)[0])
        except Exception as e:
            pred_val = np.nan

        # Extract 95% Conformal Prediction Interval (finite-sample valid >= 95% coverage)
        ci_half = tgt_meta["tol"] * 0.5  # default fall-back
        conf_df = get_conformal_intervals(mill_type)
        if not conf_df.empty and "Target" in conf_df.columns:
            tgt_row = conf_df[conf_df["Target"] == tgt_key]
            if not tgt_row.empty and "Conformal_Half_Width_95" in tgt_row.columns:
                ci_half = float(tgt_row["Conformal_Half_Width_95"].values[0])
        elif not ci_df.empty and "Target" in ci_df.columns:
            tgt_row = ci_df[ci_df["Target"] == tgt_key]
            if not tgt_row.empty and "Val_RMSE" in tgt_row.columns:
                rmse = float(tgt_row["Val_RMSE"].values[0])
                ci_half = float(rmse * 1.96)
                
        results[tgt_key] = {
            "name": tgt_meta["name"],
            "pred": round(pred_val, 2),
            "ci_low": round(pred_val - ci_half, 2),
            "ci_high": round(pred_val + ci_half, 2),
            "unit": tgt_meta["unit"],
            "model_name": bundle.get("model_name", "Ensemble"),
            "val_r2": bundle.get("val_r2", None)
        }

    if mill_type == "Plate":
        results = apply_plate_normalizing_coupling(raw_inputs, results)
    return results
