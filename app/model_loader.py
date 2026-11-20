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

    Reads the shared formulas from app/physics.py and clips the inputs to the same bounds the
    mill's feature module applies, so the card agrees with what a model actually sees.
    """
    import physics as ph

    def num(key, default):
        try:
            value = float(raw_dict.get(key, default))
        except (TypeError, ValueError):
            value = float(default)
        if not np.isfinite(value):
            value = float(default)
        lo, hi = (ph.PLATE_BOUNDS if mill_type == "Plate" else ph.HRC_BOUNDS).get(key, (None, None))
        value = max(value, lo) if lo is not None else value
        value = min(value, hi) if hi is not None else value
        return value

    c = num("Chem_C", 0.15)
    mn = num("Chem_Mn", 0.90)
    si = num("Chem_Si", 0.20)
    cu = num("Chem_Cu", 0.02)
    ni = num("Chem_Ni", 0.02)
    cr = num("Chem_Cr", 0.02)
    mo = num("Chem_Mo", 0.005)
    v = num("Chem_V", 0.005)
    ti = num("Chem_Ti", 0.002)
    n = num("Chem_N", 0.003)
    b = num("Chem_B", 0.0)

    sigma_ss = ph.sigma_ss(mn, si, ni, cr, cu, ti, mo, n)
    ce_iiw = ph.ce_iiw(c, mn, cr, mo, v, ni, cu)
    pcm = ph.pcm(c, si, mn, cu, cr, ni, mo, v, b)
    ar3 = ph.ar3(c, mn, cu, cr, ni, mo)
    ar1 = ph.ar1(mn, ni, si, cr)

    if mill_type == "Plate":
        frt = num("FM_Finish_Temp", 820.0)
        t_post = num("ACC_Temp_Post_Laminar", frt)
        plate_t = num("Plate_T_mm", 20.0)
        headers = num("ACC_Headers_n", 0.0)
        cooling_rate = ph.plate_cooling_rate(frt, t_post, plate_t, headers)
        true_strain = ph.true_strain(num("Slab_T_mm", 200.0), plate_t)
    else:
        ft = num("Temp_FT_Avg_C", 850.0)
        ct_top = num("Temp_CT_Top_C", 580.0)
        ct_bot = num("Temp_CT_Bot_C", 580.0)
        coil_t = num("Coil_T_mm", 6.0)
        if "TB_Thick_mm" in raw_dict and raw_dict["TB_Thick_mm"] is not None:
            tb_t = num("TB_Thick_mm", 32.0)
        elif "TB_Ratio" in raw_dict and raw_dict["TB_Ratio"] is not None:
            try:
                ratio_val = float(raw_dict["TB_Ratio"])
            except (TypeError, ValueError):
                ratio_val = float("nan")
            if np.isfinite(ratio_val):
                lo, hi = ph.HRC_BOUNDS.get("TB_Thick_mm", (15.0, 105.0))
                tb_t = ratio_val * coil_t
                if lo is not None:
                    tb_t = max(tb_t, lo)
                if hi is not None:
                    tb_t = min(tb_t, hi)
            else:
                tb_t = 32.0
        else:
            tb_t = 32.0
        cooling_rate = ph.hrc_cooling_rate(ft, ct_top, ct_bot, coil_t)
        true_strain = ph.true_strain(tb_t, coil_t)

    return {
        "sigma_ss": sigma_ss,
        "ce_iiw": ce_iiw,
        "pcm": pcm,
        "ar3": ar3,
        "ar1": ar1,
        "cooling_rate": cooling_rate,
        "true_strain": true_strain
    }



def plate_softening_deltas_frame(frame):
    """Vectorised softening deltas for batch scoring; the single source for the physics.

    Normalizing at ~920 C erases TMCP dislocation density and fast-cooling grain
    refinement, so the after-normalising targets are bounded by these deltas.
    """
    plate_t = pd.to_numeric(
        frame["Plate_T_mm"] if "Plate_T_mm" in frame else pd.Series(16.0, index=frame.index),
        errors="coerce",
    ).fillna(16.0).clip(lower=1.0)
    tb_t = pd.to_numeric(
        frame["TB_Thickness_mm"] if "TB_Thickness_mm" in frame else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    ).fillna(plate_t * 2.0)
    ratio_from_mm = np.maximum(tb_t, plate_t * 1.01) / plate_t
    if "TB_Ratio" in frame:
        tb_ratio = pd.to_numeric(frame["TB_Ratio"], errors="coerce")
        eff_ratio = tb_ratio.where(np.isfinite(tb_ratio), ratio_from_mm)
    else:
        eff_ratio = ratio_from_mm
    eps_fm = np.maximum(0.0, np.log(np.maximum(eff_ratio, 1.01)))

    frt = pd.to_numeric(
        frame["FM_Finish_Temp"] if "FM_Finish_Temp" in frame else pd.Series(830.0, index=frame.index),
        errors="coerce",
    ).fillna(830.0)
    t_post = pd.to_numeric(
        frame["ACC_Temp_Post_Laminar"] if "ACC_Temp_Post_Laminar" in frame else pd.Series(600.0, index=frame.index),
        errors="coerce",
    ).fillna(600.0)
    headers = pd.to_numeric(
        frame["ACC_Laminar_Header"] if "ACC_Laminar_Header" in frame else pd.Series(np.nan, index=frame.index),
        errors="coerce",
    )
    if "ACC_Headers_n" in frame:
        headers = headers.fillna(pd.to_numeric(frame["ACC_Headers_n"], errors="coerce"))
    headers = headers.fillna(0.0)
    cr_acc = np.maximum(0.0, (frt - t_post) / plate_t * (headers + 0.5))

    delta_ts = 12.0 + (8.5 * eps_fm) + (0.35 * cr_acc)
    delta_ys = 15.0 + (10.0 * eps_fm) + (0.45 * cr_acc)
    delta_el = 1.5 + (0.03 * cr_acc)
    return delta_ts, delta_ys, delta_el


def compute_plate_normalizing_deltas(raw_inputs: dict):
    """
    Computes dynamic physical softening deltas for furnace-normalized Heavy Plate:
    Delta TS = 12.0 + 8.5 * eps_FM + 0.35 * CR_ACC
    Delta YS = 15.0 + 10.0 * eps_FM + 0.45 * CR_ACC
    Delta EL = 1.5 + 0.03 * CR_ACC
    """
    frame = pd.DataFrame([raw_inputs])
    delta_ts, delta_ys, delta_el = plate_softening_deltas_frame(frame)
    return float(delta_ts.iloc[0]), float(delta_ys.iloc[0]), float(delta_el.iloc[0])


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
    if np.isfinite(raw_ts_an) and raw_ts_an > ts_ceiling:
        diff_an = (
            results["TS_AN"]["ci_high"] - results["TS_AN"]["pred"]
            if np.isfinite(results["TS_AN"].get("ci_high", np.nan))
            else 10.0
        )
        results["TS_AN"]["pred"] = ts_ceiling
        results["TS_AN"]["ci_low"] = round(ts_ceiling - diff_an, 2)
        results["TS_AN"]["ci_high"] = round(ts_ceiling + diff_an, 2)
        results["TS_AN"]["note"] = f"Physically coupled to blackplate (-{delta_ts:.1f} MPa softening)"
    elif not np.isfinite(raw_ts_an):
        results["TS_AN"]["error"] = results["TS_AN"].get(
            "error") or "after-normalising model refused the feature frame; left unscored rather than substituted"

    # 2. Yield Strength Coupling
    if np.isfinite(ys_bn) and "YS_AN" in results:
        ys_ceiling = round(ys_bn - delta_ys, 2)
        raw_ys_an = results["YS_AN"].get("pred", np.nan)
        if np.isfinite(raw_ys_an) and raw_ys_an > ys_ceiling:
            diff_an = (
                results["YS_AN"]["ci_high"] - results["YS_AN"]["pred"]
                if np.isfinite(results["YS_AN"].get("ci_high", np.nan))
                else 10.0
            )
            results["YS_AN"]["pred"] = ys_ceiling
            results["YS_AN"]["ci_low"] = round(ys_ceiling - diff_an, 2)
            results["YS_AN"]["ci_high"] = round(ys_ceiling + diff_an, 2)
            results["YS_AN"]["note"] = f"Physically coupled to blackplate (-{delta_ys:.1f} MPa softening)"
        elif not np.isfinite(raw_ys_an):
            results["YS_AN"]["error"] = results["YS_AN"].get(
                "error") or "after-normalising model refused the feature frame; left unscored rather than substituted"

    # 3. Elongation Coupling
    if np.isfinite(el_bn) and "EL_AN" in results:
        el_floor = round(el_bn + delta_el, 2)
        raw_el_an = results["EL_AN"].get("pred", np.nan)
        if np.isfinite(raw_el_an) and raw_el_an < el_floor:
            diff_an = (
                results["EL_AN"]["ci_high"] - results["EL_AN"]["pred"]
                if np.isfinite(results["EL_AN"].get("ci_high", np.nan))
                else 1.2
            )
            results["EL_AN"]["pred"] = el_floor
            results["EL_AN"]["ci_low"] = round(max(0.0, el_floor - diff_an), 2)
            results["EL_AN"]["ci_high"] = round(el_floor + diff_an, 2)
            results["EL_AN"]["note"] = f"Physically coupled to blackplate (+{delta_el:.1f}% ductility gain)"
        elif not np.isfinite(raw_el_an):
            results["EL_AN"]["error"] = results["EL_AN"].get(
                "error") or "after-normalising model refused the feature frame; left unscored rather than substituted"

    return results


def bundle_feature_columns(bundle, X):
    """Ordered best guess at the feature list a shipped estimator expects.

    The normalised Plate targets were fitted on per-target subsets, and the bundled name list does
    not always match the fitted one, so the estimator's own record comes first. Plate bundles carry
    "features"; HRC bundles carry "feature_names"; older pickles expose neither.
    """
    model = bundle.get("model")
    for cols in (getattr(model, "feature_names_in_", None),
                 bundle.get("feature_names"), bundle.get("features")):
        # feature_names_in_ is a numpy array, so a bare `if cols` is ambiguous
        if cols is not None and len(cols):
            return list(cols)
    return list(X.columns)


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
        feature_cols = bundle_feature_columns(bundle, X)

        # Align features safely
        X_aligned = pd.DataFrame(index=X.index)
        for col in feature_cols:
            X_aligned[col] = X[col] if col in X.columns else 0.0

        err = ""
        try:
            pred_val = float(model.predict(X_aligned)[0])
        except Exception as e:
            pred_val, err = np.nan, f"{type(e).__name__}: {e}"

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
            "val_r2": bundle.get("val_r2", None),
            "error": err
        }

    if mill_type == "Plate":
        results = apply_plate_normalizing_coupling(raw_inputs, results)
    return results
