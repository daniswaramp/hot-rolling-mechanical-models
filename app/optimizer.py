"""
Inverse Setpoint Optimizer (Reverse Design Module).
Solves for optimal chemistry and rolling setpoints to meet customer mechanical specifications
at minimum alloy cost and Carbon Equivalent (CE_IIW), with split-conformal 95% safety buffers (Cpk >= 1.33).
"""

import os
import json
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st

# NOTE: xgboost / catboost are intentionally NOT imported at module level —
# they add hundreds of MB of committed private memory and only the boosted
# model loading inside score_targets.from_boosted needs them (lightgbm was
# already lazy there). Imports happen on first use; Python caches per process.

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(REPO_ROOT, "models")
BOOSTED_DIR = os.path.join(MODELS_DIR, "boosted_out")

import model_loader as ml
import spec_checker as sc
import constraints as cons
import grade_templates as gt
import physics as ph
# Heavy-calculation dispatch (Modal when configured, this process otherwise). Importing this
# module does not import modal; see app/remote_compute.py.
import remote_compute


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


# relative alloy cost per 0.01 wt%, used by the cost objective
ALLOY_COST = {"C": 1.0, "Mn": 1.0, "Si": 1.2, "Al": 2.0, "Nb": 25.0, "V": 20.0,
              "Mo": 30.0, "Ti": 15.0, "Cu": 5.0, "Ni": 8.0, "Cr": 6.0}

# each objective is scored on a compliant candidate and read as "lower is better"
# once its direction has been applied
OBJECTIVES = {
    "min_ce": {
        "label": "Minimum CE_IIW",
        "help": "leanest weldable chemistry",
        "col": "ce_iiw", "better": "low",
    },
    "min_pcm": {
        "label": "Minimum P_cm",
        "help": "lowest cold-cracking risk",
        "col": "pcm", "better": "low",
    },
    "min_cost": {
        "label": "Minimum alloy cost",
        "help": "C, Mn and microalloy spend at relative prices",
        "col": "cost_proxy", "better": "low",
    },
    "max_margin": {
        "label": "Maximum safety margin",
        "help": "room above the YS and TS requirement at the 95% bound",
        "col": "margin", "better": "high",
    },
    "min_overshoot": {
        "label": "Minimum TS overshoot",
        "help": "closest to the target without going under it",
        "col": "ts_headroom", "better": "low",
    },
    "max_finish_temp": {
        "label": "Maximize finish temperature",
        "help": "finish as hot as the window allows",
        "col": {"Plate": "FM_Finish_Temp", "HRC": "Temp_FT_Avg_C"}, "better": "high",
    },
    "max_entry_temp": {
        "label": "Maximize entry temperature",
        "help": "hotter entry into the finishing train",
        "col": {"Plate": "FM_Temp_Before_Rolling", "HRC": "Temp_Transfer_Bar_C"}, "better": "high",
    },
    "max_post_laminar": {
        "label": "Maximize post-laminar temperature",
        "help": "less water at the laminar headers",
        "col": "ACC_Temp_Post_Laminar", "better": "high", "mills": ["Plate"],
    },
    "max_coiling_temp": {
        "label": "Maximize coiling temperature",
        "help": "hotter coiler, softer coil",
        "col": "Temp_CT_Top_C", "better": "high", "mills": ["HRC"],
    },
}

DEFAULT_MILLS = ("Plate", "HRC")


def available_objectives(mill):
    """Objectives that make sense for this route."""
    return [k for k, spec in OBJECTIVES.items() if mill in spec.get("mills", DEFAULT_MILLS)]


def objective_column(spec, mill):
    col = spec["col"]
    return col.get(mill) if isinstance(col, dict) else col


def objective_scores(frame: pd.DataFrame, keys, mill) -> pd.Series:
    """Normalise each chosen objective to 0..1 across the compliant set and average them.

    Lower is better in every case, so the winner is the row with the smallest score.
    """
    parts = []
    for key in keys:
        spec = OBJECTIVES.get(key)
        if spec is None or mill not in spec.get("mills", DEFAULT_MILLS):
            continue
        col = objective_column(spec, mill)
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").astype(float)
        if values.notna().any():
            values = values.fillna(values.mean())
        lo, hi = float(values.min()), float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
            parts.append(pd.Series(0.0, index=frame.index))
            continue
        norm = (values - lo) / (hi - lo)
        if spec["better"] == "high":
            norm = 1.0 - norm
        parts.append(norm)
    if not parts:
        return None
    return pd.concat(parts, axis=1).mean(axis=1)


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


def score_targets(mill: str, X_all, targets):
    """Predict an arbitrary set of targets for a feature frame, one model per target.

    Order of preference (identical to the historical score_candidates):
      1. models/boosted_out (CatBoost / XGBoost / LightGBM artefacts + boosted_metrics.json)
      2. the per-mill joblib bundles used by the rest of the app
    Returns {target: np.ndarray} or None when any target cannot be scored.
    """
    metrics_cache = {}

    def from_boosted(mill, target):
        metrics_path = os.path.join(BOOSTED_DIR, "boosted_metrics.json")
        if not os.path.exists(metrics_path):
            return None
        if mill not in metrics_cache:
            with open(metrics_path, "r") as fh:
                metrics_cache[mill] = json.load(fh)
        metrics = metrics_cache[mill]
        # historical preference: xgb for TS_BN, cat for YS_BN, first available otherwise
        preferred = {"TS_BN": "xgb", "YS_BN": "cat"}.get(target, "xgb")
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
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(os.path.join(BOOSTED_DIR, f"{mill}_{target}_xgb.ubj"))
        elif algo == "cat":
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(os.path.join(BOOSTED_DIR, f"{mill}_{target}_cat.cbm"))
        else:
            import lightgbm as lgb
            model = lgb.Booster(model_file=os.path.join(BOOSTED_DIR, f"{mill}_{target}_lgbm.txt"))
        return np.asarray(model.predict(X)).ravel()

    def from_bundle(mill, target):
        # tuned ensemble first, then the plain best_model (some HRC ensembles fail to
        # unpickle into the shared StackingEnsemble class and must be skipped)
        for use_tuned in (True, False):
            bundle = ml.load_model_bundle(mill, target, use_tuned=use_tuned)
            if bundle is None:
                continue
            try:
                cols = ml.bundle_feature_columns(bundle, X_all)
                X = pd.DataFrame({c: (X_all[c] if c in X_all.columns else 0.0) for c in cols},
                                 index=X_all.index)
                return np.asarray(bundle["model"].predict(X)).ravel()
            except Exception:
                continue
        return None

    out = {}
    for target in targets:
        value = None
        for source in (from_boosted, from_bundle):
            try:
                value = source(mill, target)
            except Exception:
                value = None
            if value is not None:
                break
        if value is None:
            return None
        out[target] = value
    return out


def score_candidates(mill: str, X_all):
    """Predict TS_BN and YS_BN for a feature frame (see score_targets).

    Returns (pred_ts, pred_ys) or None when no model can be loaded.
    """
    preds = score_targets(mill, X_all, ("TS_BN", "YS_BN"))
    if preds is None:
        return None
    return preds["TS_BN"], preds["YS_BN"]


def _raw_ce_iiw(raw):
    """The full IIW carbon equivalent of the current simulator inputs.

    The comparison table and the ΔCE metric must use the same formula as the optimized column
    (cons.ce_iiw); a C + Mn/6 stand-in understates the current CE and makes every delta look
    like the solver added alloy.
    """
    return cons.ce_iiw(
        float(raw.get("Chem_C", 0.16)), float(raw.get("Chem_Mn", 0.90)),
        float(raw.get("Chem_Cr", 0.02)), float(raw.get("Chem_Mo", 0.005)),
        float(raw.get("Chem_V", 0.005)), float(raw.get("Chem_Ni", 0.02)),
        float(raw.get("Chem_Cu", 0.02)),
    )


def run_inverse_optimization(mill: str, target_spec: dict, thickness: float, base_inputs: dict,
                             objective_mode: str = None, limits: dict = None,
                             chem_caps: dict = None, objectives=None):
    """
    Scores candidate setpoints in one vectorised batch and returns the schedule that clears the
    specification with split-conformal 95% bounds while best satisfying the selected objectives.

    Every process temperature swept here is bounded by `limits` (see constraints.py), so the solver can
    never return a setpoint outside the house band. `objectives` takes any keys from OBJECTIVES; when
    more than one is chosen they are normalised across the compliant set and averaged.
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

    # The mill feature modules expect a complete chemistry row. Callers hand us only the
    # columns they care about, so fill the rest here rather than letting the module decide
    # what a missing element means.
    for el in ("C", "Si", "Mn", "P", "S", "Nb", "Cu", "Cr", "Ni", "Mo", "V", "Al", "Ti", "N", "B"):
        col = f"Chem_{el}"
        if col not in cand_df.columns:
            cand_df[col] = 0.0
    if "Chem_CE" not in cand_df.columns:
        cand_df["Chem_CE"] = (cand_df["Chem_C"] + cand_df["Chem_Mn"] / 6.0
                              + (cand_df["Chem_Cr"] + cand_df["Chem_Mo"] + cand_df["Chem_V"]) / 5.0
                              + (cand_df["Chem_Ni"] + cand_df["Chem_Cu"]) / 15.0)
    if mill == "Plate" and "Rolling_Type" not in cand_df.columns:
        cand_df["Rolling_Type"] = "AR"

    try:
        if mill == "Plate":
            if ml.engineer_plate_features is None:
                return None
            X_all = ml.engineer_plate_features(cand_df)
        else:
            if ml.engineer_hrc_features is None:
                return None
            X_all, _ = ml.engineer_hrc_features(cand_df, is_train=False)
    except Exception:
        # A feature-module failure should cost the panel, not the whole page.
        return None

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

    # chemistry caps, computed on the full formulae
    chem_caps = chem_caps or {}
    ce_col = cons.ce_iiw(cand_df["Chem_C"], cand_df["Chem_Mn"], cand_df.get("Chem_Cr", 0.0),
                         cand_df.get("Chem_Mo", 0.0), cand_df.get("Chem_V", 0.0),
                         cand_df.get("Chem_Ni", 0.0), cand_df.get("Chem_Cu", 0.0))
    pcm_col = cons.pcm(cand_df["Chem_C"], cand_df.get("Chem_Si", 0.0), cand_df["Chem_Mn"],
                       cand_df.get("Chem_Cu", 0.0), cand_df.get("Chem_Cr", 0.0),
                       cand_df.get("Chem_Ni", 0.0), cand_df.get("Chem_Mo", 0.0),
                       cand_df.get("Chem_V", 0.0), cand_df.get("Chem_B", 0.0))
    ce_col = np.asarray(ce_col, dtype=float)
    pcm_col = np.asarray(pcm_col, dtype=float)

    cap_mask = np.ones(len(cand_df), dtype=bool)
    if chem_caps.get("ceq") is not None:
        cap_mask &= ce_col <= float(chem_caps["ceq"])
    if chem_caps.get("pcm") is not None:
        cap_mask &= pcm_col <= float(chem_caps["pcm"])

    compliant_mask = (lower_ys >= req_ys) & (lower_ts >= req_ts) & cap_mask

    cand_df["pred_ys"] = np.round(pred_ys, 1)
    cand_df["pred_ts"] = np.round(pred_ts, 1)
    cand_df["lower_ys"] = np.round(lower_ys, 1)
    cand_df["lower_ts"] = np.round(lower_ts, 1)
    cand_df["ce_iiw"] = np.round(ce_col, 3)
    cand_df["pcm"] = np.round(pcm_col, 3)

    # columns the objective library reads
    cost_col = np.zeros(len(cand_df))
    for el, weight in ALLOY_COST.items():
        col = f"Chem_{el}"
        if col in cand_df.columns:
            cost_col += weight * pd.to_numeric(cand_df[col], errors="coerce").fillna(0.0).to_numpy() * 100.0
    cand_df["cost_proxy"] = np.round(cost_col, 2)

    selected = [k for k in (objectives or []) if k in available_objectives(mill)]
    if not selected:
        selected = ["max_margin"] if objective_mode == "max_margin" else ["min_ce"]

    # chemistry caps that exclude the whole grid must stop the solve here: returning the
    # "nearest" schedule would recommend a chemistry that breaks the user's hard cap
    caps_active = chem_caps.get("ceq") is not None or chem_caps.get("pcm") is not None
    if caps_active and not cap_mask.any():
        return {
            "best_schedule": None,
            "is_compliant": False,
            "cap_infeasible": True,
            "n_evaluated": int(len(cand_df)),
            "n_compliant": 0,
            "chem_caps": dict(chem_caps),
            "objectives": selected,
            "message": ("Every candidate schedule violates the chemistry caps "
                        f"(CE_IIW ≤ {chem_caps.get('ceq')}, P_cm ≤ {chem_caps.get('pcm')}). "
                        "The caps exclude the whole grid before strength is even considered: "
                        "relax a cap or widen the window."),
        }

    # what the permitted window can actually reach, once the caps are applied
    window = {}
    if cap_mask.any():
        ys_in = pred_ys[cap_mask]
        ts_in = pred_ts[cap_mask]
        window = {
            "ys_range": (float(np.min(ys_in)), float(np.max(ys_in))),
            "ts_range": (float(np.min(ts_in)), float(np.max(ts_in))),
            "n_inside": int(cap_mask.sum()),
            "reachable": bool(np.max(ys_in) >= req_ys and np.max(ts_in) >= req_ts),
        }

    compliant_df = cand_df[compliant_mask].copy()

    if compliant_df.empty:
        # If no 100% compliant schedule, find the one closest to passing
        cand_df["shortfall"] = np.maximum(0, req_ys - cand_df["lower_ys"]) + np.maximum(0, req_ts - cand_df["lower_ts"])
        best_row = cand_df.sort_values(by=["shortfall", "ce_iiw"]).iloc[0]
        is_fully_compliant = False
    else:
        compliant_df["margin"] = np.minimum(compliant_df["lower_ys"] - req_ys,
                                            compliant_df["lower_ts"] - req_ts)
        compliant_df["ts_headroom"] = compliant_df["pred_ts"] - req_ts
        score = objective_scores(compliant_df, selected, mill)
        if score is not None:
            best_row = compliant_df.loc[score.idxmin()]
        else:
            best_row = compliant_df.iloc[0]
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
        "chem_caps": dict(chem_caps),
        "objectives": selected,
        "window": window,
        "clamped": bool(violations),
        "violations": violations,
        "candidates_df": cand_df
    }


# ---------------------------------------------------------------------------
# Hottest-feasible-first recipe search (redesigned Inverse Optimizer)
# ---------------------------------------------------------------------------

# User-facing process codes -> the Rolling_Type values the Plate feature module
# consumes. Plate/script/ml/features.py one-hots AR / CR / TMCP; the "N" family
# is the normalised supply condition the spec checker reads.
PROCESS_CHOICES = ["AR", "NF", "NR", "TMCP", "CR"]
ROLLING_TYPE_MAP = {"AR": "AR", "TMCP": "TMCP", "CR": "TMCP", "N": "N", "NF": "N", "NR": "N"}
PROCESS_HELP = {
    "AR": "AR → Rolling_Type 'AR'",
    "NF": "NF → Rolling_Type 'N' (normalised supply condition)",
    "NR": "NR → Rolling_Type 'N' (normalised supply condition)",
    "TMCP": "TMCP → Rolling_Type 'TMCP'",
    "CR": "CR → Rolling_Type 'TMCP' (controlled rolled shares the TMCP encoding)",
}

# Chemistry maxima the optimizer enforces directly on the candidate chemistry.
CHEM_REQ_KEYS = ("ce_iiw", "pcm", "Chem_S", "Chem_P")

# A candidate whose margin exceeds this many conformal intervals on a property is
# deprioritised (never hidden): it over-delivers far past the comfortable band.
OVERPOWER_CAP_INTERVALS = 3.0

AUTO_CHEMISTRY = "auto (find leanest chemistry)"

FINISH_COL = {"Plate": "FM_Finish_Temp", "HRC": "Temp_FT_Avg_C"}
ENTRY_COL = {"Plate": "FM_Temp_Before_Rolling", "HRC": "Temp_Transfer_Bar_C"}
CT_COL = {"Plate": None, "HRC": "Temp_CT_Top_C"}
FINISH_SHORT = {"Plate": "FRT", "HRC": "FT"}

# The Inverse Optimizer's three calculations. One form, one submit button, three answers:
# "best" keeps the hottest-feasible search, "pareto" returns the non-dominated trade-off
# frontier, "robust" re-ranks a shortlist on a Monte-Carlo CVaR. The stored key is what a
# session-state outcome carries so a rerun replays the renderer the result belongs to.
BEST_MODE = "Best recipe"
PARETO_MODE = "Pareto frontier"
ROBUST_MODE = "Robust (CVaR)"
CALCULATION_MODES = (BEST_MODE, PARETO_MODE, ROBUST_MODE)
CALC_MODE_KEYS = {BEST_MODE: "best", PARETO_MODE: "pareto", ROBUST_MODE: "robust"}
CVAR_ALPHA_DEFAULT = 0.10


def interval_half_widths(mill, keys):
    """95% conformal interval half-width per target key.

    Reads the app's interval lookup (ml.get_conformal_intervals) and falls back to the
    TARGET_CONFIG tolerance halved, then to 15.0, so a missing report never stalls the tab.
    """
    try:
        conf = ml.get_conformal_intervals(mill)
    except Exception:
        conf = pd.DataFrame()
    out = {}
    for key in keys:
        value = None
        if not conf.empty and "Target" in conf.columns:
            row = conf[conf["Target"] == key]
            if (not row.empty and "Conformal_Half_Width_95" in row.columns
                    and pd.notna(row["Conformal_Half_Width_95"].values[0])):
                value = float(row["Conformal_Half_Width_95"].values[0])
        if value is None:
            try:
                value = float(ml.TARGET_CONFIG[mill][key]["tol"]) * 0.5
            except Exception:
                value = 15.0
        out[key] = value
    return out


def spec_requirements(spec, grade, thickness, mill, rolling_type="AR", chemistry=None):
    """Spec-required properties for the chosen tier, read through spec_checker.

    min-type YS/TS/EL/impact limits carry the 95% conformal half width of their target;
    max-type CE/S/P chemistry ceilings are deterministic on the candidate chemistry and
    carry half width 0. Returns (requirements, info); requirements is empty when the
    workbook or the tier cannot be matched.
    """
    row, info = sc.find_matching_spec_row(spec, grade, thickness)
    if row is None:
        return [], info

    chemistry = chemistry or {}
    reqs = []

    mech = sc.evaluate_mechanical(row, {}, mill, rolling_type)
    mech_keys = [c["key"] for c in mech["checks"] if c.get("key")]
    halves = interval_half_widths(mill, mech_keys)
    for chk in mech["checks"]:
        key = chk.get("key")
        if not key:
            continue  # impact at a temperature the models do not cover
        for direction, limit in (("min", chk.get("limit_min")), ("max", chk.get("limit_max"))):
            if limit is None:
                continue
            reqs.append({
                "key": key, "kind": "mech", "direction": direction,
                "required": float(limit), "half": float(halves.get(key, 15.0)),
                "label": chk.get("property", key), "unit": chk.get("unit", ""),
            })

    chem_inputs = {k: float(v) for k, v in chemistry.items() if str(k).startswith("Chem_")}
    chem_inputs["ce_iiw"] = chemistry_ce_iiw(chemistry)
    chem_inputs["pcm"] = chemistry_pcm(chemistry)
    chem_eval = sc.evaluate_chemistry(row, chem_inputs, {"ce_iiw": 0.0, "pcm": 0.0}, None)
    for chk in chem_eval["active_checks"]:
        if chk.get("max") is None or chk.get("key") not in CHEM_REQ_KEYS:
            continue
        reqs.append({
            "key": chk["key"], "kind": "chem", "direction": "max",
            "required": float(chk["max"]), "half": 0.0,
            "label": chk.get("label", chk["key"]),
            "unit": "" if chk["key"] in ("ce_iiw", "pcm") else "wt%",
        })
    return reqs, info


def preset_requirements(mill, spec_name):
    """Fallback requirements from the built-in preset grades when the GRP workbook is absent."""
    cfg = PRESET_GRADES[mill].get(spec_name) or next(iter(PRESET_GRADES[mill].values()))
    halves = interval_half_widths(mill, ["YS_BN", "TS_BN", "EL_BN"])
    return [
        {"key": "YS_BN", "kind": "mech", "direction": "min", "required": float(cfg["ys_min"]),
         "half": halves["YS_BN"], "label": "Yield strength (as rolled)", "unit": "MPa"},
        {"key": "TS_BN", "kind": "mech", "direction": "min", "required": float(cfg["ts_min"]),
         "half": halves["TS_BN"], "label": "Tensile strength (as rolled)", "unit": "MPa"},
        {"key": "EL_BN", "kind": "mech", "direction": "min", "required": float(cfg["el_min"]),
         "half": halves["EL_BN"], "label": "Elongation (as rolled)", "unit": "%"},
    ]


def chemistry_ce_iiw(chem):
    """Full IIW carbon equivalent of a chemistry dict (Chem_* keys), via the shared physics."""
    return float(ph.ce_iiw(
        chem.get("Chem_C", 0.0), chem.get("Chem_Mn", 0.0), chem.get("Chem_Cr", 0.0),
        chem.get("Chem_Mo", 0.0), chem.get("Chem_V", 0.0), chem.get("Chem_Ni", 0.0),
        chem.get("Chem_Cu", 0.0)))


def chemistry_pcm(chem):
    """Ito-Bessyo cracking parameter of a chemistry dict (Chem_* keys)."""
    return float(ph.pcm(
        chem.get("Chem_C", 0.0), chem.get("Chem_Si", 0.0), chem.get("Chem_Mn", 0.0),
        chem.get("Chem_Cu", 0.0), chem.get("Chem_Cr", 0.0), chem.get("Chem_Ni", 0.0),
        chem.get("Chem_Mo", 0.0), chem.get("Chem_V", 0.0), chem.get("Chem_B", 0.0)))


def chemistry_microalloy(chem):
    """Nb + V + Ti total of a chemistry dict (Chem_* keys)."""
    return float(chem.get("Chem_Nb", 0.0) + chem.get("Chem_V", 0.0) + chem.get("Chem_Ti", 0.0))


def lean_chemistries():
    """Every grade template sorted lean-first: lowest CE_IIW, then lowest microalloy total."""
    items = []
    for name in gt.names():
        chem = gt.chemistry(name)
        if not chem:
            continue
        items.append((chemistry_ce_iiw(chem), chemistry_microalloy(chem), name, chem))
    items.sort(key=lambda item: (item[0], item[1]))
    return [(name, chem) for _ce, _ma, name, chem in items]


def ood_mask(mill, frame, keys):
    """True where a swept parameter leaves the historical P01-P99 envelope.

    Uses app/explainer.get_applicability_domain; imported lazily so the pure selection
    path stays cheap, and a missing envelope excludes nothing.
    """
    try:
        import explainer
        domain = expl.get_applicability_domain().get(mill, {})
    except Exception:
        domain = {}
    mask = pd.Series(False, index=frame.index)
    if not domain:
        return mask
    for key in keys:
        block = domain.get(key)
        if not isinstance(block, dict):
            continue
        try:
            p01, p99 = float(block["p01"]), float(block["p99"])
        except (KeyError, TypeError, ValueError):
            continue
        values = pd.to_numeric(frame[key], errors="coerce")
        mask = mask | (values < p01) | (values > p99)
    return mask.fillna(False)


def _temp_grid(lo, hi, step=10.0):
    """Inclusive 10 °C (or `step`) grid between two bounds; swaps a reversed range."""
    try:
        lo, hi, step = float(lo), float(hi), float(step)
    except (TypeError, ValueError):
        return []
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(step)) or step <= 0:
        return []
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return [round(lo, 1)]
    return [round(v, 1) for v in np.arange(lo, hi + step / 2.0, step)]


def _schedule_grid(mill, raw, thickness, chemistries, entry_range, finish_range,
                   cooling="air", headers_range=(4, 8), ct_range=None, step=10.0):
    """Build the entry × finish × cooling × chemistry candidate grid, before any scoring.

    Pure and model-free: every calculation mode sweeps this same grid, so the recipe search,
    the Pareto frontier and the robust study describe one search space rather than three.
    Returns (cand_df, error_message); cand_df is None when the bands leave nothing to sweep.
    """
    base = dict(raw)
    if mill == "Plate":
        plate_t = max(float(base.get("Plate_T_mm", thickness)), 1.0)
        tb = float(base.get("TB_Thickness_mm", plate_t * 2.0))
        base["TB_Ratio"] = tb / max(float(thickness), 1.0)
        base.setdefault("FM_Last_Pass_Drafting", float(thickness) * 0.18)
        base["Rolling_Type"] = ROLLING_TYPE_MAP.get(str(base.get("Rolling_Type", "AR")), "AR")

    entry_grid = _temp_grid(*entry_range, step)
    finish_grid = _temp_grid(*finish_range, step)
    if not entry_grid or not finish_grid:
        return None, "empty temperature range"
    ct_grid = _temp_grid(*ct_range, step) if (mill == "HRC" and ct_range) else [None]

    if mill == "Plate":
        header_values = ([0] if cooling == "air"
                         else list(range(int(headers_range[0]), int(headers_range[1]) + 1)))
    else:
        header_values = [0]  # the Steckel runout table has no header choice

    chemistries = list(chemistries) or [("simulator chemistry", {})]
    rows = []
    for chem_label, chem in chemistries:
        for entry in entry_grid:
            for finish in finish_grid:
                for ct in ct_grid:
                    for hdr in header_values:
                        row = dict(base)
                        for chem_key, chem_val in chem.items():
                            row[chem_key] = float(chem_val)
                        if mill == "Plate":
                            row["Plate_T_mm"] = float(thickness)
                            row["FM_Temp_Before_Rolling"] = entry
                            row["FM_Finish_Temp"] = finish
                            row["ACC_Laminar_Header"] = float(hdr)
                            row["ACC_Headers_n"] = float(hdr)
                            row["ACC_Active"] = 1.0 if hdr > 0 else 0.0
                            row["ACC_Temp_Post_Laminar"] = float(
                                ph.post_laminar_temp(finish, hdr, thickness))
                        else:
                            row["Coil_T_mm"] = float(thickness)
                            row["Temp_Transfer_Bar_C"] = entry
                            row["Temp_FT_Avg_C"] = finish
                            if ct is not None:
                                row["Temp_CT_Top_C"] = ct
                                row["Temp_CT_Bot_C"] = ct
                        row["_chemistry"] = chem_label
                        rows.append(row)
    return pd.DataFrame(rows), None


def _score_schedule_frame(mill, cand_df, mech_keys):
    """Score one frame of schedules: the shared tail every calculation mode runs through.

    Fills the chemistry blanks the mill feature modules expect, engineers the features,
    predicts every mechanical requirement, applies the Plate normalising coupling, then
    appends the deterministic indicators (CE_IIW, P_cm, microalloy total) and the
    applicability-domain flag. Decorates cand_df in place and returns it, so the candidate
    frame a mode ranks is the very frame that was scored. Returns (None, message) when the
    feature module or the models cannot deliver.

    The Monte-Carlo passes reuse this unchanged: a frame of jittered schedules is scored by
    exactly the plumbing a single deterministic schedule goes through.
    """
    # The mill feature modules expect a complete chemistry row; fill the blanks rather
    # than letting the module decide what a missing element means.
    for el in ("C", "Si", "Mn", "P", "S", "Nb", "Cu", "Cr", "Ni", "Mo", "V", "Al", "Ti", "N", "B"):
        col = f"Chem_{el}"
        if col not in cand_df.columns:
            cand_df[col] = 0.0
    if "Chem_CE" not in cand_df.columns:
        cand_df["Chem_CE"] = (cand_df["Chem_C"] + cand_df["Chem_Mn"] / 6.0
                              + (cand_df["Chem_Cr"] + cand_df["Chem_Mo"] + cand_df["Chem_V"]) / 5.0
                              + (cand_df["Chem_Ni"] + cand_df["Chem_Cu"]) / 15.0)

    try:
        if mill == "Plate":
            if ml.engineer_plate_features is None:
                return None, "Plate feature module unavailable"
            X_all = ml.engineer_plate_features(cand_df)
        else:
            if ml.engineer_hrc_features is None:
                return None, "HRC feature module unavailable"
            X_all, _ = ml.engineer_hrc_features(cand_df, is_train=False)
    except Exception as exc:  # noqa: BLE001
        return None, f"feature engineering failed: {exc}"

    preds = score_targets(mill, X_all, mech_keys)
    if preds is None:
        return None, "no model could score the candidate grid"
    for key in mech_keys:
        cand_df[f"pred_{key}"] = np.round(preds[key], 2)

    if mill == "Plate" and any(key.endswith("_AN") for key in mech_keys):
        # twin of the frame-path normalising coupling in streamlit_app.score_frame
        try:
            delta_ts, delta_ys, delta_el = ml.plate_softening_deltas_frame(cand_df)
            for stem, delta in (("TS", delta_ts), ("YS", delta_ys)):
                bn, an = f"pred_{stem}_BN", f"pred_{stem}_AN"
                if bn in cand_df.columns and an in cand_df.columns:
                    cap = cand_df[bn] - delta
                    cand_df[an] = cand_df[an].mask(cand_df[an].notna() & (cand_df[an] > cap), cap)
            if "pred_EL_BN" in cand_df.columns and "pred_EL_AN" in cand_df.columns:
                floor = cand_df["pred_EL_BN"] + delta_el
                cand_df["pred_EL_AN"] = cand_df["pred_EL_AN"].mask(
                    cand_df["pred_EL_AN"].notna() & (cand_df["pred_EL_AN"] < floor), floor)
        except Exception:
            pass

    cand_df["ce_iiw"] = np.round(cons.ce_iiw(
        cand_df["Chem_C"], cand_df["Chem_Mn"], cand_df["Chem_Cr"], cand_df["Chem_Mo"],
        cand_df["Chem_V"], cand_df["Chem_Ni"], cand_df["Chem_Cu"]), 4)
    cand_df["pcm"] = np.round(cons.pcm(
        cand_df["Chem_C"], cand_df["Chem_Si"], cand_df["Chem_Mn"], cand_df["Chem_Cu"],
        cand_df["Chem_Cr"], cand_df["Chem_Ni"], cand_df["Chem_Mo"], cand_df["Chem_V"],
        cand_df["Chem_B"]), 4)
    cand_df["microalloy_total"] = (cand_df["Chem_Nb"] + cand_df["Chem_V"] + cand_df["Chem_Ti"]).round(4)

    ood_keys = [ENTRY_COL[mill], FINISH_COL[mill]]
    if mill == "Plate":
        ood_keys += ["ACC_Laminar_Header", "ACC_Temp_Post_Laminar"]
    else:
        ood_keys += ["Temp_CT_Top_C", "Temp_CT_Bot_C"]
    cand_df["ood"] = ood_mask(mill, cand_df, ood_keys)
    return cand_df, None


def _scored_candidates(mill, raw, thickness, requirements, chemistries, entry_range,
                       finish_range, cooling="air", headers_range=(4, 8), ct_range=None,
                       step=10.0):
    """Grid + scoring builder shared by the recipe search, the Pareto frontier and the
    robust study: the one place the search space and its predictions are defined.

    Returns (cand_df, mech_keys, error_message). cand_df is None, and the message says why,
    when the search space or the models are unusable; mech_keys lists the mechanical targets
    that were scored (`pred_<key>` columns on the frame).
    """
    mech_keys = sorted({r["key"] for r in requirements if r.get("kind") == "mech"})
    if not mech_keys:
        return None, [], "no verifiable mechanical requirements for this specification tier"
    cand_df, error = _schedule_grid(mill, raw, thickness, chemistries, entry_range,
                                    finish_range, cooling=cooling,
                                    headers_range=headers_range, ct_range=ct_range, step=step)
    if cand_df is None:
        return None, mech_keys, error
    cand_df, error = _score_schedule_frame(mill, cand_df, mech_keys)
    if cand_df is None:
        return None, mech_keys, error
    return cand_df, mech_keys, None


def run_recipe_search(mill, raw, thickness, requirements, chemistries,
                      entry_range, finish_range, cooling="air", headers_range=(4, 8),
                      ct_range=None, k=1.0, step=10.0):
    """Hottest-feasible-first recipe search. Pure: no Streamlit calls, safe to drive from tests.

    Builds the vectorised candidate grid entry × finish × cooling × chemistry (temperature
    step `step`, default 10 °C so ±20 °C aims stay representable), scores it through the
    mill feature module and the shared scoring plumbing (_scored_candidates), then hands it
    to select_recipe. The grid and its scoring live in that shared builder, so this search,
    the Pareto frontier and the robust study all rank the same candidate pool.
    """
    cand_df, mech_keys, error = _scored_candidates(
        mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
        cooling=cooling, headers_range=headers_range, ct_range=ct_range, step=step)
    if cand_df is None:
        return {"error": error, "chosen": None, "ladder": [], "rejected": [],
                "candidates_df": pd.DataFrame()}

    chosen, ladder, rejected = select_recipe(cand_df, requirements, k=k, mill=mill)
    return {
        "error": None,
        "chosen": chosen,
        "ladder": ladder,
        "rejected": rejected,
        "candidates_df": cand_df,
        "n_evaluated": int(len(cand_df)),
        "k": float(k),
        "requirements": requirements,
        "half_widths": interval_half_widths(mill, mech_keys),
    }


def requirement_margins(df, requirements, k=1.0):
    """Add one margin_<key>|<direction> column per verifiable requirement; flag feasibility.

    margin = pred - required for a min-type limit and required - pred for a max-type
    ceiling. A candidate is feasible when every margin clears k × that requirement's 95 %
    conformal half width (half width 0 for the deterministic chemistry ceilings), which is
    the rule select_recipe, the Pareto frontier and the robust study all apply.

    `df` is decorated in place. Returns (active, feasible): the (rid, req) pairs that could
    be scored against the frame, and the boolean Series of the margin rule — ([] , None) when
    not one requirement could be scored.
    """
    active = []
    for req in requirements:
        col = f"pred_{req['key']}" if req.get("kind") == "mech" else req["key"]
        if col not in df.columns:
            continue
        rid = f"{req['key']}|{req['direction']}"
        values = pd.to_numeric(df[col], errors="coerce")
        if req["direction"] == "min":
            df[f"margin_{rid}"] = values - float(req["required"])
        else:
            df[f"margin_{rid}"] = float(req["required"]) - values
        active.append((rid, req))

    if not active:
        return [], None

    feasible = pd.Series(True, index=df.index)
    for rid, req in active:
        need = float(k) * float(req.get("half", 0.0))
        feasible &= df[f"margin_{rid}"] >= need - 1e-9
    return active, feasible


def comfort_margins(df, active):
    """The deepest normalised margin each candidate keeps, in units of its 95 % interval.

    Only requirements that carry a conformal half width are normalised. A deterministic
    chemistry ceiling (CE_IIW, P_cm, S, P) has no interval to divide by, and its raw margin
    is an order of magnitude smaller than an interval ratio, so folding the two together
    would let the S/P ceiling alone decide a comfort objective. When no requirement carries
    an interval the raw margins are used instead.

    Returns (ratios, worst): ratios holds one column per normalised requirement, worst is
    the row-wise minimum — the "thin end" of the margin the comfort objective reads.
    """
    parts = {}
    for rid, req in active:
        half = float(req.get("half", 0.0))
        if req.get("kind") != "mech" or half <= 0.0:
            continue
        parts[rid] = df[f"margin_{rid}"] / half
    if not parts:
        parts = {rid: df[f"margin_{rid}"] for rid, _req in active}
    ratios = pd.DataFrame(parts, index=df.index)
    return ratios, ratios.min(axis=1)


def _ranked_candidates(df, requirements, k=1.0, mill="Plate"):
    """The margin-feasible candidates of a scored frame, in select_recipe's ranking order.

    `df` is decorated in place with the margin, overpower and feasibility columns the
    ranking reads. Returns (ranked, active, extrapolated): ranked is the ordered frame of
    feasible candidates (empty when none is feasible), active the scored requirements, and
    extrapolated says whether the ranking had to fall back on candidates outside the
    historical envelope because no in-domain candidate was feasible.
    """
    finish_col, entry_col = FINISH_COL[mill], ENTRY_COL[mill]
    active, feasible = requirement_margins(df, requirements, k)
    if not active:
        return df.iloc[0:0], [], False

    for col in ("ce_iiw", "microalloy_total"):
        if col not in df.columns:
            df[col] = 0.0
    if mill == "Plate" and "ACC_Headers_n" not in df.columns:
        df["ACC_Headers_n"] = 0.0

    over_parts = []
    for rid, req in active:
        half = float(req.get("half", 0.0))
        if req.get("kind") != "mech" or half <= 0.0:
            continue  # deterministic chemistry ceilings carry no interval and no overpower
        cap = OVERPOWER_CAP_INTERVALS * half
        over_parts.append(np.maximum(0.0, df[f"margin_{rid}"] - cap) / half)
    df["overpower"] = (pd.concat(over_parts, axis=1).mean(axis=1) if over_parts
                       else pd.Series(0.0, index=df.index))

    ood = df["ood"].astype(bool) if "ood" in df.columns else pd.Series(False, index=df.index)

    pool = df[feasible & ~ood]
    extrapolated = False
    if pool.empty:
        pool = df[feasible]
        extrapolated = not pool.empty
    if pool.empty:
        return df.iloc[0:0], active, False

    sort_cols = ["overpower", finish_col, entry_col]
    ascending = [True, False, False]
    if mill == "Plate":
        sort_cols.append("ACC_Headers_n")
        ascending.append(True)
    else:
        ct_col = CT_COL[mill]
        if ct_col and ct_col in pool.columns:
            sort_cols.append(ct_col)
            ascending.append(False)
    sort_cols += ["ce_iiw", "microalloy_total"]
    ascending += [True, True]

    return pool.sort_values(sort_cols, ascending=ascending, kind="mergesort"), active, extrapolated


def select_recipe(candidates_df, requirements, k=1.0, mill="Plate"):
    """Pure hottest-feasible-first selection over a scored candidate frame.

    Feasibility: every spec-required property must clear margin >= k * half_width, where
    margin = pred - required for min-type and required - pred for max-type, and half_width
    is the 95% conformal interval of that target (0 for deterministic chemistry ceilings).

    Overpower: mean over the interval-backed properties of max(0, margin - 3*half)/half.
    Candidates far above the comfortable band are deprioritised, not hidden.

    Selection among feasible candidates, lexicographic:
      1. minimal overpower
      2. hottest finish temperature
      3. hottest entry temperature
      4. fewest headers (air beats any headers)
      5. (HRC only) hottest coiling temperature
      6. leanest chemistry: lowest CE_IIW, then lowest microalloy total

    Candidates outside the historical envelope (an "ood" column of True) are excluded from
    selection unless nothing else is feasible; the chosen row then carries _extrapolated
    so the caller can warn.

    The margin rule and the ranking itself live in requirement_margins and
    _ranked_candidates, which the Pareto frontier and the robust study reuse.

    Returns (chosen_row_dict_or_None, ladder_rows, rejected_notes).
    """
    finish_col = FINISH_COL[mill]
    df = candidates_df.copy()
    if df.empty:
        return None, [], ["no candidates were evaluated"]

    ranked, active, extrapolated = _ranked_candidates(df, requirements, k, mill)
    if not active:
        return None, [], ["no requirement could be scored against the candidates"]
    if ranked.empty:
        return None, [], _rejected_band_notes(df, active, k, mill, finish_col)

    chosen = ranked.iloc[0].to_dict()
    chosen["_extrapolated"] = bool(extrapolated)
    chosen["_k"] = float(k)
    ladder = [ranked.iloc[i].to_dict() for i in range(1, min(4, len(ranked)))]
    return chosen, ladder, _rejected_band_notes(df, active, k, mill, finish_col)


def _rejected_band_notes(df, active, k, mill, finish_col, max_bands=3):
    """For the hottest few rejected finish-temperature bands, name the first failing property.

    Within a band the candidate that came closest to passing (fewest failures, then the
    highest margin ratio on its first failure) speaks for the band.
    """
    notes = []
    if finish_col not in df.columns or not active:
        return notes

    margin_mat = pd.concat([df[f"margin_{rid}"] for rid, _ in active], axis=1)
    margin_mat.columns = [rid for rid, _ in active]
    need = pd.Series({rid: float(k) * float(req.get("half", 0.0)) for rid, req in active})
    fail = margin_mat.lt(need - 1e-9, axis=1)
    n_fail = fail.sum(axis=1)
    rej = df[n_fail > 0]
    if rej.empty:
        return notes

    first_fail = fail.idxmax(axis=1)  # first failing requirement per row
    ratio = margin_mat.div(need.replace(0.0, np.nan), axis=1)

    short = FINISH_SHORT.get(mill, finish_col)
    for band in sorted(rej[finish_col].unique(), reverse=True)[:max_bands]:
        band_rows = rej[rej[finish_col] == band]
        order = sorted(band_rows.index,
                       key=lambda i: (int(n_fail.loc[i]), -float(ratio.loc[i, first_fail.loc[i]]
                                                                 if pd.notna(ratio.loc[i, first_fail.loc[i]]) else -np.inf)))
        idx = order[0]
        rid = first_fail.loc[idx]
        req = next(item for item_id, item in active if item_id == rid)
        if req.get("kind") == "chem":
            value = pd.to_numeric(pd.Series([df.loc[idx, req["key"]]]), errors="coerce").iloc[0]
            notes.append(f"{short} {float(band):.0f} °C band rejected: {req['label']} "
                         f"{value:.4f} exceeds the {req['required']:.4f} limit")
        else:
            r = float(ratio.loc[idx, rid]) if pd.notna(ratio.loc[idx, rid]) else float("-inf")
            notes.append(f"{short} {float(band):.0f} °C band rejected: {req['label']} "
                         f"margin {r:.1f}×interval < required {float(k):.1f}×")
    return notes


# ---------------------------------------------------------------------------
# Pareto frontier: cheapest / least cooling / most comfortable, non-dominated
# ---------------------------------------------------------------------------

def nondominated_mask(f1, f2, f3):
    """Boolean mask of the non-dominated rows of three objectives, all "smaller is better".

    Vectorised sort-and-sweep, safe on a full candidate grid (~120k rows). Rows that share
    the first two objectives are collapsed first, because the leanest third objective speaks
    for the pair; what remains is grouped by the first objective and, inside a group, ordered
    by the second, so a running minimum of the third decides dominance. Across groups only
    the surviving 2-D staircase of the strictly leaner groups can dominate. Cost is the two
    sorts plus a sweep whose width is the surviving staircase, not the row count.
    """
    f1 = np.asarray(f1, dtype=float).ravel()
    f2 = np.asarray(f2, dtype=float).ravel()
    f3 = np.asarray(f3, dtype=float).ravel()
    n = int(min(f1.size, f2.size, f3.size))
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    f1, f2, f3 = f1[:n], f2[:n], f3[:n]
    # a non-finite objective can never win: push it to +inf so it dominates nothing
    f1 = np.where(np.isfinite(f1), f1, np.inf)
    f2 = np.where(np.isfinite(f2), f2, np.inf)
    f3 = np.where(np.isfinite(f3), f3, np.inf)

    order = np.lexsort((f3, f2, f1))  # objective 1, then 2, then 3
    a1, a2, a3 = f1[order], f2[order], f3[order]

    # collapse identical (objective 1, objective 2) pairs onto their best third objective
    rep = np.ones(n, dtype=bool)
    rep[1:] = ~((a1[1:] == a1[:-1]) & (a2[1:] == a2[:-1]))
    u1, u2, u3 = a1[rep], a2[rep], a3[rep]

    alive = np.zeros(u1.size, dtype=bool)
    starts = np.flatnonzero(np.r_[True, u1[1:] != u1[:-1]])
    ends = np.r_[starts[1:], u1.size]
    stair2 = np.empty(0, dtype=float)
    stair3 = np.empty(0, dtype=float)

    def _running_min_exclusive(values):
        out = np.full(values.size, np.inf, dtype=float)
        if values.size > 1:
            out[1:] = np.minimum.accumulate(values)[:-1]
        return out

    for start, end in zip(starts, ends):
        g2, g3 = u2[start:end], u3[start:end]
        # inside one first-objective group every earlier row has an equal or smaller second
        # objective, so a row survives only if it beats the running third-objective minimum
        idx = np.flatnonzero(g3 < _running_min_exclusive(g3))
        if idx.size == 0:
            continue
        if stair2.size:
            dominated = np.zeros(idx.size, dtype=bool)
            chunk = max(1, int(1_000_000 // max(stair2.size, 1)))
            for begin in range(0, idx.size, chunk):
                block = idx[begin:begin + chunk]
                dominated[begin:begin + block.size] = (
                    (stair2[None, :] <= g2[block, None])
                    & (stair3[None, :] <= g3[block, None])).any(axis=1)
            idx = idx[~dominated]
            if idx.size == 0:
                continue
        alive[start + idx] = True
        # fold the survivors into the 2-D staircase and re-prune it
        m2 = np.concatenate([stair2, g2[idx]])
        m3 = np.concatenate([stair3, g3[idx]])
        reorder = np.lexsort((m3, m2))
        m2, m3 = m2[reorder], m3[reorder]
        stair_alive = m3 < _running_min_exclusive(m3)
        stair2, stair3 = m2[stair_alive], m3[stair_alive]

    keep[order[rep][alive]] = True
    return keep


def pareto_rows(candidates_df, requirements, k=1.0, mill="Plate", max_points=40):
    """Non-dominated frontier of an already scored candidate frame (pure, no Streamlit).

    Three objectives, each read as "smaller is better" once its direction is applied:

      1. CE_IIW             — the leanest weldable chemistry
      2. cooling intensity  — laminar headers in play (air = 0; the Steckel table has none)
      3. −minimum margin    — comfort, i.e. the deepest normalised margin the row keeps

    Only candidates that clear the same margin rule as select_recipe take part (every
    required property margin >= k × its 95 % conformal half width): a frontier over
    infeasible schedules would be a menu of scrap. Rows come back lean-first, then
    hotter-first, cut to max_points.

    Returns {"frontier", "n_feasible", "n_evaluated", "n_frontier", "n_truncated"}.
    """
    finish_col, entry_col = FINISH_COL[mill], ENTRY_COL[mill]
    df = candidates_df.copy()
    n_evaluated = int(len(df))
    empty = {"frontier": [], "n_feasible": 0, "n_evaluated": n_evaluated,
             "n_frontier": 0, "n_truncated": False}
    if df.empty:
        return empty

    active, feasible = requirement_margins(df, requirements, k)
    if not active or feasible is None:
        return empty
    feasible = feasible.fillna(False)
    if not bool(feasible.any()):
        return empty

    ratios, worst = comfort_margins(df, active)
    headers = (pd.to_numeric(df["ACC_Headers_n"], errors="coerce").fillna(0.0)
               if "ACC_Headers_n" in df.columns
               else pd.Series(0.0, index=df.index))
    ce = (pd.to_numeric(df["ce_iiw"], errors="coerce") if "ce_iiw" in df.columns
          else pd.Series(0.0, index=df.index))

    # the filter only ever sees the feasible subset, in (CE_IIW, headers, -comfort) space
    subset = df[feasible]
    mask = nondominated_mask(ce[feasible].to_numpy(dtype=float),
                             headers[feasible].to_numpy(dtype=float),
                             -worst[feasible].to_numpy(dtype=float))
    picked = subset.loc[mask]
    if picked.empty:
        return empty

    records = []
    for idx, row in picked.iterrows():
        margins = {rid: float(row[f"margin_{rid}"]) for rid, _req in active}
        records.append({
            "entry": float(row[entry_col]),
            "finish": float(row[finish_col]),
            "ct": (float(row[CT_COL[mill]])
                   if CT_COL.get(mill) and CT_COL[mill] in df.columns else None),
            "headers": float(headers.loc[idx]),
            "chemistry": str(row.get("_chemistry", "")),
            "ce_iiw": float(row.get("ce_iiw", 0.0) or 0.0),
            "microalloy_total": float(row.get("microalloy_total", 0.0) or 0.0),
            "min_margin_interval": float(worst.loc[idx]),
            "min_margin_raw": float(min(margins.values())) if margins else float("nan"),
            "margins": margins,
            "margin_ratios": {rid: float(ratios.loc[idx, rid]) for rid in ratios.columns},
            "chem_c": float(row.get("Chem_C", 0.0) or 0.0),
            "chem_mn": float(row.get("Chem_Mn", 0.0) or 0.0),
            "chem_nb": float(row.get("Chem_Nb", 0.0) or 0.0),
            "ood": bool(row.get("ood", False)),
        })

    frontier = pd.DataFrame(records).sort_values(
        ["ce_iiw", "finish", "entry"], ascending=[True, False, False], kind="mergesort")
    n_frontier = int(len(frontier))
    cut = max(1, int(max_points))
    return {"frontier": frontier.head(cut).to_dict("records"),
            "n_feasible": int(feasible.sum()),
            "n_evaluated": n_evaluated,
            "n_frontier": n_frontier,
            "n_truncated": bool(n_frontier > cut)}


def _mode_meta(mill, thickness, k, step, cooling, headers_range, entry_range, finish_range,
               ct_range, chemistries):
    """The scalars a Pareto or robust result carries so it can describe itself on replay."""
    return {
        "mill": mill,
        "thickness": float(thickness),
        "k": float(k),
        "step": float(step),
        "cooling": str(cooling),
        "headers_range": (int(headers_range[0]), int(headers_range[1])),
        "entry_range": (float(entry_range[0]), float(entry_range[1])),
        "finish_range": (float(finish_range[0]), float(finish_range[1])),
        "ct_range": (float(ct_range[0]), float(ct_range[1])) if ct_range else None,
        "chemistries": [str(name) for name, _chem in (chemistries or [])],
        "entry_col": ENTRY_COL[mill],
        "finish_col": FINISH_COL[mill],
    }


def pareto_frontier(mill, raw, thickness, requirements, chemistries, entry_range,
                    finish_range, cooling="air", headers_range=(4, 8), ct_range=None,
                    k=1.0, step=10.0, max_points=40):
    """The non-dominated recipe frontier. Pure: no Streamlit calls, safe to drive from tests.

    Scores the same candidate grid run_recipe_search scores (through the shared
    _scored_candidates builder), keeps the candidates that clear the same margin rule, and
    returns the Pareto set over minimum CE_IIW, minimum cooling intensity (laminar headers,
    air = 0) and maximum comfort (the deepest margin kept, in units of the 95 % interval).

    Each frontier row carries its entry and finish setpoints, header count, chemistry name,
    CE_IIW, microalloy total, the comfort margin, the raw and normalised margin of every
    required property and the applicability-domain flag, so the UI can tabulate, plot and
    export it without another model call.
    """
    meta = _mode_meta(mill, thickness, k, step, cooling, headers_range, entry_range,
                      finish_range, ct_range, chemistries)
    cand_df, _mech_keys, error = _scored_candidates(
        mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
        cooling=cooling, headers_range=headers_range, ct_range=ct_range, step=step)
    if cand_df is None:
        return {"error": error, "frontier": [], "n_feasible": 0, "n_evaluated": 0,
                "n_frontier": 0, "n_truncated": False, "meta": meta}

    rows = pareto_rows(cand_df, requirements, k=k, mill=mill, max_points=max_points)
    meta.update({"objectives": ["min CE_IIW", "min cooling intensity (headers; air = 0)",
                                "max min-margin (comfort, × 95% interval)"],
                 "frontier_size": int(rows["n_frontier"]),
                 "shown": int(len(rows["frontier"]))})
    return {"error": None, **rows, "meta": meta}


# ---------------------------------------------------------------------------
# Robust recipe: deterministic ranking re-ranked on Monte-Carlo CVaR
# ---------------------------------------------------------------------------

# Monte-Carlo noise model. The temperature aims match the ±20 °C aiming band the recipe card
# already quotes; the chemistry band is a fraction of the grade's own element band width.
ENTRY_AIM_C = 20.0
FINISH_AIM_C = 20.0
CHEM_BAND_FRACTION = 0.10
CHEM_RELATIVE_FRACTION = 0.05

# Gauge row of the app's input spec table (lo, hi, step), used only for the Monte-Carlo gauge
# band on the paths that run without streamlit_app loaded (tests, the Modal container).
GAUGE_SPEC = {"Plate": (6.0, 120.0, 0.5), "HRC": (1.8, 16.0, 0.1)}


def _gauge_spec(mill):
    """(lo, hi, step) of the gauge row, taken from the app's own spec table when it is loaded.

    streamlit_app imports this module, so its SPECS table is in sys.modules by the time the
    UI gets here; reading the live row keeps the Monte-Carlo gauge band on the same step as
    the number input. GAUGE_SPEC is the same row for every path that has no app to ask.
    """
    name = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    module = sys.modules.get("streamlit_app")
    rows = getattr(module, "SPECS", {}).get(mill, []) if module is not None else []
    for row in rows:
        if len(row) >= 8 and row[1] == name:
            try:
                return float(row[4]), float(row[5]), float(row[7])
            except (TypeError, ValueError):
                break
    return GAUGE_SPEC[mill]


def cvar_worst_mean(values, alpha=0.10):
    """Mean of the worst alpha fraction of a sample: the empirical lower-tail CVaR.

    ceil(alpha × N) of the worst draws are averaged, at least one, so a small ensemble still
    answers instead of dividing by zero.
    """
    sample = np.asarray(values, dtype=float).ravel()
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return float("nan")
    sample.sort()
    n_worst = max(1, int(np.ceil(float(alpha) * sample.size)))
    return float(np.mean(sample[:n_worst]))


def robust_metrics(margin_per_sample, fail_flags, candidate_ids, alpha=0.10):
    """Per-candidate risk numbers from one long series of Monte-Carlo draws.

    mean_margin : average of margin_per_sample over that candidate's draws
    cvar_margin : mean of the worst alpha fraction of them (lower-tail CVaR)
    p_fail      : fraction of draws where a required property fails at k = 0
    n_samples   : number of draws behind the three numbers

    Returns a DataFrame indexed by candidate id (empty when there are no draws).
    """
    values = np.asarray(margin_per_sample, dtype=float).ravel()
    fails = np.asarray(fail_flags, dtype=bool).ravel()
    ids = np.asarray(candidate_ids).ravel()
    records = []
    for cid in np.unique(ids):
        pick = ids == cid
        sample = values[pick]
        records.append({
            "candidate": int(cid),
            "mean_margin": float(np.mean(sample)) if sample.size else float("nan"),
            "cvar_margin": cvar_worst_mean(sample, alpha),
            "p_fail": float(np.mean(fails[pick])) if sample.size else float("nan"),
            "n_samples": int(sample.size),
        })
    frame = pd.DataFrame(records)
    return frame.set_index("candidate") if not frame.empty else frame


def _perturbation_draws(mill, pool, thickness, n_samples, seed, cooling, headers_range):
    """One long frame of Monte-Carlo schedule draws around the ranked candidates.

    Every draw is a complete mill input row, so the whole ensemble is scored in a single
    vectorised pass instead of one model call per sample:

      * entry and finish temperature : ±20 °C, uniform around the candidate setpoint
      * laminar headers             : ±1, clamped to the swept header range (air stays 0)
      * chemistry                   : ±10 % of each element's grade band width where the
                                      chemistry is a grade template, ±5 % relative where it
                                      is the simulator chemistry or the element has no band
      * plate / coil gauge          : ±1 step of its mill spec row, clamped to that row

    The plate post-laminar temperature is re-derived from each draw's own finish and headers,
    so the joint physics the grid respects is not broken by the jitter.

    Returns (frame, candidate_ids, noise_description).
    """
    rng = np.random.default_rng(int(seed))
    reps = max(1, int(n_samples))
    base = pool.reset_index(drop=True)
    n_candidates = int(len(base))
    frame = base.loc[base.index.repeat(reps)].reset_index(drop=True)
    n = int(len(frame))
    ids = np.repeat(np.arange(n_candidates), reps)

    entry_col, finish_col = ENTRY_COL[mill], FINISH_COL[mill]
    entry = pd.to_numeric(frame[entry_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    finish = pd.to_numeric(frame[finish_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    entry = entry + rng.uniform(-ENTRY_AIM_C, ENTRY_AIM_C, n)
    finish = finish + rng.uniform(-FINISH_AIM_C, FINISH_AIM_C, n)
    frame[entry_col] = entry
    frame[finish_col] = finish

    gauge_lo, gauge_hi, gauge_step = _gauge_spec(mill)
    gauge = np.clip(float(thickness) + rng.uniform(-gauge_step, gauge_step, n),
                    gauge_lo, gauge_hi)
    noise = [f"entry ±{ENTRY_AIM_C:.0f} °C", f"finish ±{FINISH_AIM_C:.0f} °C"]

    if mill == "Plate":
        allowed = ([0] if str(cooling) == "air"
                   else list(range(int(headers_range[0]), int(headers_range[1]) + 1)))
        headers = pd.to_numeric(frame.get("ACC_Headers_n", 0.0), errors="coerce").fillna(0.0)
        headers = np.clip(headers.to_numpy(dtype=float) + np.rint(rng.uniform(-1.0, 1.0, n)),
                          float(min(allowed)), float(max(allowed)))
        if str(cooling) == "air":
            headers = np.zeros(n, dtype=float)
        frame["ACC_Headers_n"] = headers
        frame["ACC_Laminar_Header"] = headers
        frame["ACC_Active"] = (headers > 0).astype(float)
        frame["Plate_T_mm"] = gauge
        frame["ACC_Temp_Post_Laminar"] = ph.post_laminar_temp(finish, headers, gauge)
        noise.append(f"headers ±1 inside {int(min(allowed))}–{int(max(allowed))}"
                     + (" (air: no headers)" if str(cooling) == "air" else ""))
    else:
        frame["Coil_T_mm"] = gauge
        noise.append("no header choice on the Steckel runout table")
    noise.append(f"gauge {float(thickness):.2f} ± {gauge_step:g} mm")

    # stale prediction/margin columns from the deterministic pass must not survive the jitter
    stale = [c for c in frame.columns if str(c).startswith(("pred_", "margin_"))]
    frame = frame.drop(columns=stale + [c for c in ("Chem_CE", "overpower", "shortfall", "ood")
                                        if c in frame.columns], errors="ignore")

    labels = (base["_chemistry"].astype(str).tolist() if "_chemistry" in base.columns
              else [""] * n_candidates)
    banded = 0
    for col in [c for c in frame.columns if str(c).startswith("Chem_")]:
        element = str(col)[len("Chem_"):]
        sigma = np.zeros(n_candidates, dtype=float)
        for i, name in enumerate(labels):
            lows, highs = gt.band(name)
            lo, hi = lows.get(element), highs.get(element)
            if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and float(hi) > float(lo):
                sigma[i] = CHEM_BAND_FRACTION * (float(hi) - float(lo))
                banded += 1
            else:
                value = base.iloc[i].get(col, 0.0)
                try:
                    sigma[i] = CHEM_RELATIVE_FRACTION * abs(float(value))
                except (TypeError, ValueError):
                    sigma[i] = 0.0
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        frame[col] = np.maximum(0.0, values + rng.uniform(-1.0, 1.0, n) * np.repeat(sigma, reps))
    noise.append(f"chemistry ±{CHEM_BAND_FRACTION:.0%} of each grade band width"
                 + (f" ({banded} element bands)" if banded else "")
                 + f", ±{CHEM_RELATIVE_FRACTION:.0%} relative where no band is defined")
    noise.append(f"{reps} draws per candidate, seed {int(seed)}")
    return frame, ids, " · ".join(noise)


def robust_recipe(mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
                  cooling="air", headers_range=(4, 8), ct_range=None, k=1.0, step=10.0,
                  alpha=0.10, n_samples=200, seed=42, n_candidates=40):
    """Robust recipe: the deterministic shortlist re-ranked on downside risk. Pure.

    Scores the deterministic grid exactly as run_recipe_search scores it, takes the top
    `n_candidates` margin-feasible schedules in select_recipe's order, and surrounds each
    one with the Monte-Carlo band _perturbation_draws documents. Every draw is a row of one
    long frame, so the ensemble goes through the same feature module and models in a single
    vectorised pass.

    Per candidate it reports the mean margin, the lower-tail CVaR at `alpha` and the fraction
    of draws that fail a required property at k = 0, and selects the candidate with the best
    CVaR — ties go to the hotter finish, then the leaner CE_IIW. The deterministic pick
    travels along as "deterministic" so the UI can show what robustness cost.

    margin_per_sample is the minimum normalised margin across the interval-backed required
    properties (margin / its 95 % conformal half width). The deterministic chemistry ceilings
    (CE_IIW, P_cm, S, P) carry no interval to normalise by, so they enter P(fail) but not the
    comfort series: mixing them would let a 0.02 wt% S headroom outvote a 1.5 × YS margin.
    """
    cand_df, mech_keys, error = _scored_candidates(
        mill, raw, thickness, requirements, chemistries, entry_range, finish_range,
        cooling=cooling, headers_range=headers_range, ct_range=ct_range, step=step)
    if cand_df is None:
        return {"error": error, "chosen": None, "runners": [], "deterministic": None,
                "noise_description": "", "n_evaluated": 0, "n_candidates": 0, "n_draws": 0,
                "alpha": float(alpha), "n_samples": int(n_samples), "seed": int(seed)}

    n_evaluated = int(len(cand_df))
    ranked, active, extrapolated = _ranked_candidates(cand_df, requirements, k, mill)
    if not active or ranked.empty:
        return {"error": None, "chosen": None, "runners": [], "deterministic": None,
                "noise_description": "", "n_evaluated": n_evaluated,
                "n_candidates": 0, "n_draws": 0, "alpha": float(alpha),
                "n_samples": int(n_samples), "seed": int(seed),
                "message": (f"no candidate clears the margin rule at k = {k:.1f}× the 95% "
                            "interval, so there is no shortlist to stress-test")}

    pool = ranked.head(max(1, int(n_candidates))).reset_index(drop=True)
    frame, candidate_ids, noise = _perturbation_draws(
        mill, pool, thickness, n_samples, seed, cooling, headers_range)
    n_draws = int(len(frame))
    scored, error = _score_schedule_frame(mill, frame, mech_keys)
    if scored is None:
        return {"error": error, "chosen": None, "runners": [], "deterministic": None,
                "noise_description": noise, "n_evaluated": n_evaluated,
                "n_candidates": int(len(pool)), "n_draws": n_draws,
                "alpha": float(alpha), "n_samples": int(n_samples), "seed": int(seed)}

    metrics_active, pass_at_zero = requirement_margins(scored, requirements, 0.0)
    _ratios, worst = comfort_margins(scored, metrics_active)
    metrics = robust_metrics(worst.to_numpy(dtype=float),
                             (~pass_at_zero).to_numpy(dtype=bool), candidate_ids, alpha=alpha)

    finish_col = FINISH_COL[mill]

    def _risk(i):
        if metrics.empty or i not in metrics.index:
            return {"mean_margin": float("nan"), "cvar_margin": float("nan"),
                    "p_fail": float("nan"), "n_samples": 0}
        row = metrics.loc[i]
        return {"mean_margin": float(row["mean_margin"]),
                "cvar_margin": float(row["cvar_margin"]),
                "p_fail": float(row["p_fail"]),
                "n_samples": int(row["n_samples"])}

    def _row(i):
        row = pool.iloc[i].to_dict()
        row.update(_risk(i))
        row["_extrapolated"] = bool(extrapolated)
        row["_k"] = float(k)
        return row

    def _sort_key(i):
        risk = _risk(i)
        cvar = risk["cvar_margin"]
        hot = float(pool.iloc[i].get(finish_col, 0.0) or 0.0)
        ce = float(pool.iloc[i].get("ce_iiw", 0.0) or 0.0)
        return (-cvar if np.isfinite(cvar) else np.inf, -hot, ce)

    order = sorted(range(len(pool)), key=_sort_key)
    return {
        "error": None,
        "chosen": _row(order[0]),
        "runners": [_row(i) for i in order[1:6]],
        "deterministic": _row(0),
        "det_is_chosen": bool(order[0] == 0),
        "noise_description": noise,
        "n_evaluated": n_evaluated,
        "n_candidates": int(len(pool)),
        "n_draws": n_draws,
        "alpha": float(alpha),
        "k": float(k),
        "n_samples": int(n_samples),
        "seed": int(seed),
        "requirements": requirements,
    }


def _render_recipe_card(mill, chosen, requirements, k, meta, elapsed_text, n_evaluated):
    """The setpoint, chemistry and per-property card for one chosen recipe row.

    Shared by the deterministic search and the robust study, which rank rows carrying the
    same scored columns (pred_<key>, margin_<key>|<direction>, ce_iiw, microalloy_total), so
    the two modes cannot drift apart in how they describe a recipe. ``meta`` carries the few
    scalars that are not part of a row (spec, grade, thickness, rolling type).
    """
    entry_col, finish_col = ENTRY_COL[mill], FINISH_COL[mill]
    thickness = float(meta.get("thickness", 0.0))
    rolling_type = meta.get("rolling_type", "AR")

    st.markdown(f"#### Recipe card: solved in {elapsed_text} "
                f"({int(n_evaluated):,} candidates scored)")
    if mill == "Plate":
        entry, frt = float(chosen[entry_col]), float(chosen[finish_col])
        headers = float(chosen.get("ACC_Headers_n", 0.0))
        cooling_text = (f"**{headers:.0f} headers** (0.5 start bonus per the physics)"
                        if headers > 0 else "**air cool**")
        st.markdown(
            f"- Rolling process: **{rolling_type}** (route {chosen.get('route', 'AR')}) · "
            f"thickness **{thickness:.1f} mm**\n"
            f"- FM entry: **{entry:.0f} °C** (aim ± 20 °C)\n"
            f"- Finish rolling: **{frt:.0f} °C** (aim ± 20 °C)\n"
            f"- Cooling: {cooling_text} · post-laminar aim "
            f"**{float(chosen.get('ACC_Temp_Post_Laminar', frt)):.0f} °C**")
    else:
        entry, frt = float(chosen[entry_col]), float(chosen[finish_col])
        ct = float(chosen.get("Temp_CT_Top_C", 0.0))
        st.markdown(
            f"- Thickness **{thickness:.1f} mm** coil · no process selector on this route\n"
            f"- Transfer bar: **{entry:.0f} °C** (aim ± 20 °C)\n"
            f"- Finishing: **{frt:.0f} °C** (aim ± 20 °C)\n"
            f"- Coiling: **{ct:.0f} °C** (aim ± 20 °C, top and bottom symmetric)")

    st.markdown(
        f"- Slab grade: **{chosen.get('_chemistry', '?')}** · "
        f"C {chosen.get('Chem_C', 0.0):.3f} % · Mn {chosen.get('Chem_Mn', 0.0):.3f} % · "
        f"Nb {chosen.get('Chem_Nb', 0.0):.4f} % · "
        f"CE_IIW {chosen.get('ce_iiw', 0.0):.3f} · microalloys "
        f"{chosen.get('microalloy_total', 0.0):.4f} %")

    pred_rows = []
    for req in requirements:
        rid = f"{req['key']}|{req['direction']}"
        req_symbol = "≥" if req["direction"] == "min" else "≤"
        if req["kind"] == "mech":
            pred = float(chosen.get(f"pred_{req['key']}", float("nan")))
            half = float(req["half"])
            margin = float(chosen.get(f"margin_{rid}", float("nan")))
            pred_rows.append({
                "Property": req["label"],
                "Required": f"{req_symbol} {req['required']:.1f} {req['unit']}",
                "Predicted": f"{pred:.1f} {req['unit']}",
                "95% interval": f"{pred - half:.1f} – {pred + half:.1f}",
                "Margin": f"{margin:+.1f} {req['unit']}",
                "Margin (×interval)": f"{margin / half:.2f}×" if half > 0 else "—",
                "Verdict": "PASS" if margin >= k * half - 1e-9 else "FAIL",
            })
        else:
            value = float(chosen.get(req["key"], float("nan")))
            margin = float(chosen.get(f"margin_{rid}", float("nan")))
            pred_rows.append({
                "Property": req["label"],
                "Required": f"{req_symbol} {req['required']:.4f}",
                "Predicted": f"{value:.4f}",
                "95% interval": "deterministic",
                "Margin": f"{margin:+.4f}",
                "Margin (×interval)": "—",
                "Verdict": "PASS" if margin >= -1e-9 else "FAIL",
            })
    st.dataframe(pd.DataFrame(pred_rows), hide_index=True, use_container_width=True)
    st.caption(f"Feasibility rule: margin ≥ k × 95% conformal interval with k = {k:.1f}. "
               f"Overpower deprioritises (never hides) any candidate whose margin on a property "
               f"exceeds {OVERPOWER_CAP_INTERVALS:.0f} × its interval (cap = "
               f"{OVERPOWER_CAP_INTERVALS:.0f} × conformal half-width, fixed constant).")


def _recipe_csv_row(mill, chosen, requirements, k):
    """One recipe row reduced to the columns a CSV export should carry."""
    entry_col, finish_col = ENTRY_COL[mill], FINISH_COL[mill]
    recipe_keys = (["_chemistry"] + [c for c in chosen if str(c).startswith("Chem_")]
                   + [entry_col, finish_col]
                   + (["ACC_Laminar_Header", "ACC_Headers_n", "ACC_Active",
                       "ACC_Temp_Post_Laminar"] if mill == "Plate" else
                      ["Temp_CT_Top_C", "Temp_CT_Bot_C"])
                   + ["ce_iiw", "pcm", "microalloy_total",
                      "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"]
                   + (["route"] if mill == "Plate" else [])
                   + [f"pred_{r['key']}" for r in requirements if r.get("kind") == "mech"]
                   + [f"margin_{r['key']}|{r['direction']}" for r in requirements]
                   + ["mean_margin", "cvar_margin", "p_fail", "n_samples"])
    csv_row = {key: chosen.get(key) for key in dict.fromkeys(recipe_keys) if key in chosen}
    csv_row["k"] = k
    csv_row["ood"] = bool(chosen.get("ood", False))
    return csv_row


def _render_optimizer_result(mill: str, res: dict, requirements, k: float, elapsed,
                             source: str = "local"):
    """
    Renders one solved (or failed) best-recipe outcome: the extrapolation warning, recipe
    card, alternative-recipe ladder, rejection notes and the CSV download.

    It renders from the search result alone and never recomputes, so the submit path and a
    plain rerun that replays st.session_state[f"opt_result_{mill}"] produce identical output.
    The failure states (res["error"] and chosen is None) are handled here too, so a stored
    failure replays exactly as it was reported.

    ``elapsed`` is the short elapsed string ("812 ms"); a float number of seconds is accepted.
    The few scalars the card needs that are not part of the search result (spec name, grade,
    thickness, rolling type) travel in res["_meta"], which keeps this signature free of the
    widget state.

    ``source`` is where the search actually ran ("modal" or "local"): it is captioned under
    the recipe card so the operator can see whether those numbers came off Modal.
    """
    entry_col, finish_col = ENTRY_COL[mill], FINISH_COL[mill]

    if res.get("error"):
        st.error(res["error"])
        st.caption(remote_compute.source_label(source, "recipe"))
        return

    meta = res.get("_meta") or {}
    chosen_spec = meta.get("spec", "")
    chosen_grade = meta.get("grade", "")
    elapsed_text = elapsed if isinstance(elapsed, str) else f"{float(elapsed) * 1000:.0f} ms"

    chosen = res.get("chosen")
    if chosen is None:
        st.error(f"No feasible recipe in {res['n_evaluated']:,} evaluated candidates at "
                 f"k = {k:.1f}× the 95% interval. Nothing is proposed: widen a temperature "
                 "band, raise k's tolerance via a lower safety factor, pick a richer slab "
                 "grade, or relax the specification.")
        for note in res["rejected"]:
            st.markdown(f"- {note}")
        st.caption(remote_compute.source_label(source, "recipe"))
        return

    if chosen.get("_extrapolated"):
        st.warning("The selected recipe sits outside the historical P01–P99 envelope on at "
                   "least one parameter. Every in-domain candidate failed the margin rule, so "
                   "an extrapolated candidate was used: treat the predictions as extrapolation.")

    # ---- recipe card ---------------------------------------------------------------------
    _render_recipe_card(mill, chosen, requirements, k, meta, elapsed_text,
                        res.get("n_evaluated", 0))
    st.caption(remote_compute.source_label(source, "recipe"))

    # ---- rejection ladder -----------------------------------------------------------------
    st.markdown("#### Rejection ladder: next hottest feasible alternatives")
    if res["ladder"]:
        ladder_rows = []
        for rank, alt in enumerate(res["ladder"], start=2):
            ratios = [float(alt.get(f"margin_{r['key']}|{r['direction']}", float("nan"))) /
                      float(r["half"])
                      for r in requirements
                      if r.get("kind") == "mech" and float(r.get("half", 0.0)) > 0]
            ratios = [v for v in ratios if np.isfinite(v)]
            worst = min(ratios) if ratios else float("nan")
            if mill == "Plate":
                thermal = (f"entry {float(alt[entry_col]):.0f} °C · FRT {float(alt[finish_col]):.0f} °C · "
                           + ("air" if float(alt.get('ACC_Headers_n', 0)) == 0
                              else f"{float(alt['ACC_Headers_n']):.0f} headers"))
            else:
                thermal = (f"entry {float(alt[entry_col]):.0f} °C · FT {float(alt[finish_col]):.0f} °C · "
                           f"CT {float(alt.get('Temp_CT_Top_C', 0)):.0f} °C")
            ladder_rows.append({
                "Rank": rank,
                "Chemistry": alt.get("_chemistry", "?"),
                "Setpoints": thermal,
                "CE_IIW": f"{float(alt.get('ce_iiw', 0)):.3f}",
                "Worst margin": f"{worst:.2f}×interval" if np.isfinite(worst) else "—",
                "Overpower": f"{float(alt.get('overpower', 0)):.2f}",
            })
        st.dataframe(pd.DataFrame(ladder_rows), hide_index=True, use_container_width=True)
    else:
        st.caption("No hotter feasible alternative: the selected recipe is the only feasible point.")

    st.markdown("#### Why the solver stepped down: hottest rejected bands")
    if res["rejected"]:
        for note in res["rejected"]:
            st.markdown(f"- {note}")
    else:
        st.caption("Nothing was rejected: every evaluated candidate cleared the margin rule.")

    # ---- CSV download -----------------------------------------------------------------------
    csv_row = _recipe_csv_row(mill, chosen, requirements, k)
    st.download_button(
        "Download recipe (CSV)",
        data=pd.DataFrame([csv_row]).to_csv(index=False).encode("utf-8"),
        file_name=f"recipe_{mill}_{chosen_spec.replace(' ', '_')}_{chosen_grade}.csv",
        mime="text/csv",
        use_container_width=True)


def _hover_label(row):
    """One hover line for a frontier point: the recipe it stands for."""
    headers = f"{row['headers']:.0f} hdr" if row.get("headers") else "air"
    return (f"{row.get('chemistry', '?')}<br>entry {row['entry']:.0f} °C · "
            f"finish {row['finish']:.0f} °C · {headers}<br>"
            f"CE_IIW {row['ce_iiw']:.3f} · min margin "
            f"{row['min_margin_interval']:.2f}× interval")


def _render_pareto_result(mill: str, res: dict, requirements, k: float, elapsed,
                          source: str = "local"):
    """Render one stored Pareto frontier: note, frontier table, scatter and CSV export.

    Renders from the stored result alone and never recomputes, so the submit path and a plain
    rerun that replays st.session_state[f"opt_result_{mill}"] produce identical output.
    """
    if res.get("error"):
        st.error(res["error"])
        st.caption(remote_compute.source_label(source, "pareto"))
        return

    meta = res.get("_meta") or {}
    elapsed_text = elapsed if isinstance(elapsed, str) else f"{float(elapsed) * 1000:.0f} ms"
    frontier = list(res.get("frontier") or [])
    n_feasible = int(res.get("n_feasible") or 0)
    n_evaluated = int(res.get("n_evaluated") or 0)

    if not frontier:
        st.error(f"No margin-feasible schedule in {n_evaluated:,} evaluated candidates at "
                 f"k = {k:.1f}× the 95% interval, so there is no frontier to draw. Widen a "
                 "temperature band, lower the margin factor, pick a richer slab grade or "
                 "relax the specification.")
        st.caption(remote_compute.source_label(source, "pareto"))
        return

    st.markdown(f"#### Pareto frontier: {len(frontier)} of {n_feasible:,} feasible schedules "
                f"({n_evaluated:,} scored in {elapsed_text})")
    st.markdown(
        "Three objectives, all read as **smaller is better**: minimum CE_IIW, minimum cooling "
        "intensity (laminar headers in play; air = 0) and maximum comfort (the deepest margin "
        "kept, in units of the 95% interval). Every row below is non-dominated: no evaluated "
        "schedule is at least as lean, at least as lightly cooled *and* at least as "
        "comfortable at once. Read it as a price list — the frontier never hides a cheaper or "
        "hotter option behind one blended score.")

    tail = ""
    if res.get("n_truncated"):
        tail = (f" — the full frontier holds {int(res.get('n_frontier') or 0)} rows, "
                f"the {len(frontier)} leanest are shown here and exported")
    st.markdown(f"#### Frontier, leanest first{tail}")

    table = []
    for rank, row in enumerate(frontier[:15], start=1):
        if mill == "Plate":
            setpoints = (f"entry {row['entry']:.0f} °C · FRT {row['finish']:.0f} °C · "
                         + ("air" if not row.get("headers")
                            else f"{row['headers']:.0f} headers"))
        else:
            setpoints = f"entry {row['entry']:.0f} °C · FT {row['finish']:.0f} °C"
        table.append({
            "Rank": rank,
            "Chemistry": row.get("chemistry", ""),
            "Setpoints": setpoints,
            "CE_IIW": f"{row['ce_iiw']:.3f}",
            "Microalloys %": f"{row['microalloy_total']:.4f}",
            "Min margin": f"{row['min_margin_interval']:.2f}×",
            "Envelope": "extrapolated" if row.get("ood") else "in-domain",
        })
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
    st.caption("Min margin is the thinnest margin any interval-backed required property keeps, "
               "in units of its own 95% conformal half width. 'extrapolated' marks a schedule "
               "outside the historical P01–P99 envelope on at least one parameter.")

    try:
        import plotly.graph_objects as go

        fig = go.Figure(go.Scatter(
            x=[row["ce_iiw"] for row in frontier],
            y=[row["finish"] for row in frontier],
            mode="markers",
            marker=dict(size=10,
                        color=[row["min_margin_interval"] for row in frontier],
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="min margin<br>(× interval)"),
                        line=dict(color="white", width=0.8)),
            text=[_hover_label(row) for row in frontier],
            hoverinfo="text"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=24, b=10),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(title="CE_IIW", showline=True, linecolor="#d7dbe0",
                                     gridcolor="#f4f5f7"),
                          yaxis=dict(title=f"{FINISH_SHORT.get(mill, 'finish')} [°C]",
                                     showline=False, gridcolor="#f4f5f7"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("One point per non-dominated schedule: further left is leaner, higher is a "
                   "hotter finish, brighter keeps more margin. Hover for the recipe.")
    except Exception:  # noqa: BLE001 - a missing plotly must not cost the whole panel
        st.caption("Chart unavailable: plotly is not importable in this environment.")

    export = []
    for row in frontier:
        record = {"chemistry": row.get("chemistry", ""), "entry": row["entry"],
                  "finish": row["finish"], "headers": row.get("headers"),
                  "ce_iiw": row["ce_iiw"], "microalloy_total": row["microalloy_total"],
                  "min_margin_interval": row["min_margin_interval"],
                  "min_margin_raw": row.get("min_margin_raw"),
                  "ood": bool(row.get("ood"))}
        if row.get("ct") is not None:
            record["ct"] = row["ct"]
        for rid, value in (row.get("margins") or {}).items():
            record[f"margin_{rid}"] = value
        for rid, value in (row.get("margin_ratios") or {}).items():
            record[f"margin_ratio_{rid}"] = value
        export.append(record)
    spec_name = str(meta.get("spec", "spec")).replace(" ", "_")
    st.download_button(
        "Download frontier (CSV)",
        data=pd.DataFrame(export).to_csv(index=False).encode("utf-8"),
        file_name=f"pareto_{mill}_{spec_name}.csv",
        mime="text/csv",
        use_container_width=True)
    st.caption(remote_compute.source_label(source, "pareto"))


def _render_robust_result(mill: str, res: dict, requirements, k: float, elapsed,
                          source: str = "local"):
    """Render one stored robust (CVaR) outcome: card, risk strip, runners and CSV export.

    Renders from the stored result alone and never recomputes, exactly like the deterministic
    renderer, so a rerun replays it without spending another Modal call.
    """
    if res.get("error"):
        st.error(res["error"])
        st.caption(remote_compute.source_label(source, "robust"))
        return

    meta = res.get("_meta") or {}
    elapsed_text = elapsed if isinstance(elapsed, str) else f"{float(elapsed) * 1000:.0f} ms"
    chosen = res.get("chosen")
    if chosen is None:
        st.error(f"No feasible candidate in {int(res.get('n_evaluated') or 0):,} evaluated "
                 f"schedules at k = {k:.1f}× the 95% interval, so there is nothing to "
                 f"stress-test. {res.get('message') or ''}".strip())
        st.caption(remote_compute.source_label(source, "robust"))
        return

    if chosen.get("_extrapolated"):
        st.warning("The selected recipe sits outside the historical P01–P99 envelope on at "
                   "least one parameter. Every in-domain candidate failed the margin rule, so "
                   "an extrapolated candidate was used: treat the predictions as extrapolation.")

    _render_recipe_card(mill, chosen, requirements, k, meta, elapsed_text,
                        res.get("n_evaluated", 0))

    alpha = float(res.get("alpha", 0.10))
    st.markdown(f"#### Downside risk: {alpha:.0%} CVaR over the noise band")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"CVaR {alpha:.0%}", f"{float(chosen.get('cvar_margin', float('nan'))):.2f}×")
    c2.metric("P(fail)", f"{float(chosen.get('p_fail', float('nan'))) * 100:.1f} %")
    c3.metric("Mean margin", f"{float(chosen.get('mean_margin', float('nan'))):.2f}×")
    c4.metric("Draws / candidate", f"{int(res.get('n_samples') or chosen.get('n_samples') or 0)}")
    st.caption(f"{int(res.get('n_candidates') or 0)} shortlisted candidates × "
               f"{int(res.get('n_samples') or 0)} draws = "
               f"{int(res.get('n_draws') or 0):,} scored schedules (seed {res.get('seed', '?')}).")
    st.caption(f"Noise model: {res.get('noise_description', '')}")

    det = res.get("deterministic") or {}
    if res.get("det_is_chosen"):
        st.caption("The deterministic winner and the robust winner are the same point: the "
                   "stress test did not buy anything here, and the plain recipe stands.")
    elif det:
        finish_col = FINISH_COL[mill]
        d_cvar = float(det.get("cvar_margin", float("nan")))
        c_cvar = float(chosen.get("cvar_margin", float("nan")))
        d_pf = float(det.get("p_fail", float("nan")))
        c_pf = float(chosen.get("p_fail", float("nan")))
        st.caption(
            f"Deterministic pick ({det.get('_chemistry', '?')} · "
            f"{FINISH_SHORT.get(mill, 'finish')} {float(det.get(finish_col, 0.0) or 0.0):.0f} °C)"
            f" keeps CVaR {d_cvar:.2f}× and P(fail) {d_pf * 100:.1f} %; the robust pick keeps "
            f"CVaR {c_cvar:.2f}× and P(fail) {c_pf * 100:.1f} % — that difference is what the "
            f"stress test traded for.")

    st.markdown("#### Runners: the same shortlist, ranked by CVaR")
    runners = list(res.get("runners") or [])
    if runners:
        finish_col, entry_col = FINISH_COL[mill], ENTRY_COL[mill]
        rows = []
        for rank, alt in enumerate(runners, start=2):
            setpoints = (f"entry {float(alt.get(entry_col, 0.0) or 0.0):.0f} °C · "
                         f"{FINISH_SHORT.get(mill, 'finish')} "
                         f"{float(alt.get(finish_col, 0.0) or 0.0):.0f} °C")
            if mill == "Plate":
                headers = float(alt.get("ACC_Headers_n", 0.0) or 0.0)
                setpoints += " · " + ("air" if headers <= 0 else f"{headers:.0f} headers")
            rows.append({
                "Rank": rank,
                "Chemistry": alt.get("_chemistry", "?"),
                "Setpoints": setpoints,
                f"CVaR {alpha:.0%}": f"{float(alt.get('cvar_margin', float('nan'))):.2f}×",
                "P(fail)": f"{float(alt.get('p_fail', float('nan'))) * 100:.1f} %",
                "Mean margin": f"{float(alt.get('mean_margin', float('nan'))):.2f}×",
                "CE_IIW": f"{float(alt.get('ce_iiw', 0.0) or 0.0):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.caption("The shortlist held a single feasible candidate, so there are no runners.")

    export = [_recipe_csv_row(mill, chosen, requirements, k)]
    for alt in runners:
        export.append(_recipe_csv_row(mill, alt, requirements, k))
    for position, record in enumerate(export):
        record["role"] = "chosen" if position == 0 else "runner"
    spec_name = str(meta.get("spec", "spec")).replace(" ", "_")
    st.download_button(
        "Download robust recipe (CSV)",
        data=pd.DataFrame(export).to_csv(index=False).encode("utf-8"),
        file_name=f"robust_{mill}_{spec_name}.csv",
        mime="text/csv",
        use_container_width=True)
    st.caption(remote_compute.source_label(source, "robust"))


def _render_stored_result(mill, stored):
    """Dispatch one stored optimizer outcome to the renderer its mode belongs to.

    Nothing here recomputes: every mode keeps what it needs in
    st.session_state[f"opt_result_{mill}"] (a "mode" marker plus the result, requirements,
    k, elapsed and source), so a rerun replays the last calculation verbatim.
    """
    mode = stored.get("mode", "best")
    res = stored.get("res") or {}
    requirements = stored.get("requirements") or []
    k = float(stored.get("k", 1.0))
    elapsed = stored.get("elapsed", "")
    source = stored.get("source", "local")
    if mode == "pareto":
        _render_pareto_result(mill, res, requirements, k, elapsed, source=source)
    elif mode == "robust":
        _render_robust_result(mill, res, requirements, k, elapsed, source=source)
    else:
        _render_optimizer_result(mill, res, requirements, k, elapsed, source=source)


def _slab_band_text(name):
    """One-line min-max band for a grade template, or "" when the record is unusable.

    gt.band() returns two element-keyed dicts (min, max) and the min side may carry
    fewer elements than the max side, so the caption iterates their intersection
    instead of zipping the dicts themselves.
    """
    mins, maxs = gt.band(name)
    parts = []
    for element, lo in mins.items():
        hi = maxs.get(element)
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            continue
        if hi > lo:
            parts.append(f"{element}: {lo:.3f}–{hi:.3f}")
    return " · ".join(parts)


def view_optimizer(mill: str, raw: dict, engine: str = "Champion GBDT (Recommended)",
                   spec_rows=None):
    """
    Renders the redesigned Inverse Optimizer: one form, one submit, three calculations.

    The user picks the end specification, target thickness and rolling process, then
    constrains the slab grade, entry/finish temperature bands and cooling; the app searches
    the constraint space and answers whichever question the "Calculation" radio asks:

      * **Best recipe** — the hottest recipe that keeps every spec-required property
        non-marginal (margin >= k × 95% conformal interval), with a rejection ladder.
      * **Pareto frontier** — the non-dominated trade-off between minimum CE_IIW, minimum
        cooling intensity and maximum comfort, as a table, a scatter and a CSV.
      * **Robust (CVaR)** — the deterministic shortlist stress-tested over a Monte-Carlo
        noise band, re-ranked on lower-tail CVaR, with P(fail) and the runners.

    Every input lives inside one st.form, so touching a widget never reruns the script and
    nothing is scored until "Calculate" is submitted. The submitted outcome (with the mode it
    belongs to and the source it came from - Modal or this process) is kept in
    st.session_state[f"opt_result_{mill}"] and simply re-rendered on later reruns, so a
    rerun never spends another Modal call and never recomputes.
    """
    raw = dict(raw)
    st.markdown('<div class="section"><span>inverse setpoint search</span><i></i></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="note">Recipe search against the spec at the 95% conformal bound. '
                '<b>Best recipe</b> returns the hottest entry and finish that keep every '
                'spec-required property non-marginal, with the leanest chemistry that gets '
                'there. <b>Pareto frontier</b> returns the non-dominated CE_IIW / cooling / '
                'margin trade-off instead of one blended winner. <b>Robust (CVaR)</b> re-ranks '
                'the shortlist on its worst-case margin over a Monte-Carlo noise band.</div>',
                unsafe_allow_html=True)

    result_key = f"opt_result_{mill}"

    # Every input sits inside one st.form: widget changes stay client-side (no rerun, no
    # mid-search interaction) and nothing is scored until the operator submits the form.
    with st.form(key=f"opt_form_{mill}"):
        # ---- 0. which calculation -------------------------------------------------------
        # One submit button drives all three modes, so the choice is part of the form: the
        # mode is read on submit and stored with the result, never inferred on a rerun.
        calc_mode = st.radio("Calculation", CALCULATION_MODES, index=0, horizontal=True,
                             key=f"opt_calc_{mill}",
                             help="Best recipe returns the single hottest feasible schedule. "
                                  "Pareto frontier returns the non-dominated trade-off between "
                                  "CE_IIW, cooling and margin. Robust (CVaR) stress-tests the "
                                  "shortlist over a Monte-Carlo noise band and picks the "
                                  "schedule whose worst cases hold up.")
        alpha = CVAR_ALPHA_DEFAULT
        if calc_mode == ROBUST_MODE:
            alpha = st.slider("CVaR level α (Robust only)", min_value=0.05, max_value=0.50,
                              value=CVAR_ALPHA_DEFAULT, step=0.05, key=f"opt_alpha_{mill}",
                              help="Fraction of the worst Monte-Carlo draws averaged into the "
                                   "risk number. α = 0.10 means 'the average of the worst 10 % "
                                   "of outcomes'.")

        # ---- 1. end specification -------------------------------------------------------
        labels, lookup = sc.spec_grade_options()
        if labels:
            preferred_key = ("JISG3101SS400" if mill == "HRC" else "ASTMA57250")
            preferred = next((label for label in labels
                              if sc.norm_key(label) == preferred_key),
                             next((label for label in labels
                                   if ("JIS G 3101" in label if mill == "HRC" else "A 572" in label)),
                                  labels[0]))
            chosen_label = st.selectbox("End specification + grade", labels,
                                        index=labels.index(preferred), key=f"opt_sg_{mill}")
            chosen_spec, chosen_grade = lookup[chosen_label]
        else:
            st.warning("The GRP specification workbook is unavailable: using the built-in preset "
                       "grades (YS/TS/EL minima only).")
            preset_names = list(PRESET_GRADES[mill].keys())
            chosen_spec = st.selectbox("End specification standard (preset)", preset_names,
                                       key=f"opt_spec_{mill}")
            chosen_grade = chosen_spec

        # ---- 2. target thickness + rolling process --------------------------------------
        t_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
        spec_row = {row[1]: row for row in (spec_rows or [])}.get(t_key)
        gauge_lo = cons.spec_float(spec_row, 4, 6.0 if mill == "Plate" else 1.8)
        gauge_hi = cons.spec_float(spec_row, 5, 120.0 if mill == "Plate" else 16.0)
        if not gauge_hi > gauge_lo:
            gauge_lo, gauge_hi = (6.0, 120.0) if mill == "Plate" else (1.8, 16.0)
        cur_t = float(raw.get(t_key, 16.0 if mill == "Plate" else 6.0))
        cur_t = min(max(cur_t, gauge_lo), gauge_hi)
        thickness = st.number_input("Target thickness", min_value=gauge_lo,
                                    max_value=gauge_hi, value=cur_t, step=1.0,
                                    help="Finish gauge in millimetres.")

        rolling_type = "AR"
        if mill == "Plate":
            process = st.radio("Rolling process", PROCESS_CHOICES, index=0, horizontal=True)
            rolling_type = ROLLING_TYPE_MAP[process]
            st.caption("Process mapping: " + " · ".join(PROCESS_HELP[p] for p in PROCESS_CHOICES))
        else:
            st.caption("Rolling_Type is not a feature of the Steckel HRC model, so the coil route "
                       "offers no process selector.")

        # ---- 3. constraints -------------------------------------------------------------
        start_choices = [AUTO_CHEMISTRY] + gt.names()
        slab_choice = st.selectbox(
            "Slab grade", start_choices, key=f"opt_slab_{mill}",
            help="Pick an internal grade template to roll exactly that chemistry, or 'auto' to "
                 "search every template lean-first (lowest CE_IIW, then lowest microalloy total) "
                 "and return the leanest chemistry whose recipe stays feasible.")
        if slab_choice != AUTO_CHEMISTRY:
            band_text = _slab_band_text(slab_choice)
            if band_text:
                st.caption(band_text)

        def band_inputs(key, label):
            """lo/hi number inputs bounded by the mill spec row; default the row's full range."""
            row = {r[1]: r for r in (spec_rows or [])}.get(key)
            lo = cons.spec_float(row, 4, None)
            hi = cons.spec_float(row, 5, None)
            if lo is None or hi is None or not hi > lo:
                lo, hi = (850.0, 1150.0) if key == ENTRY_COL[mill] else (700.0, 1050.0)
            c1, c2 = st.columns(2)
            band_lo = c1.number_input(f"{label} min (°C)", min_value=lo, max_value=hi,
                                      value=float(lo), step=10.0, key=f"opt_{key}_lo_{mill}")
            band_hi = c2.number_input(f"{label} max (°C)", min_value=lo, max_value=hi,
                                      value=float(hi), step=10.0, key=f"opt_{key}_hi_{mill}")
            if band_hi < band_lo:
                st.warning(f"{label}: max below min, using the full row band.")
                band_lo, band_hi = lo, hi
            return float(band_lo), float(band_hi)

        k = st.slider("required margin: k × 95% interval (non-marginal without overpower)",
                      min_value=0.5, max_value=2.0, value=1.0, step=0.1, key=f"opt_k_{mill}")

        if mill == "Plate":
            entry_range = band_inputs("FM_Temp_Before_Rolling", "FM entry temperature")
            finish_range = band_inputs("FM_Finish_Temp", "Finish rolling temperature")
            cooling_label = st.radio("Cooling", ["Air cool", "Laminar headers"], horizontal=True)
            cooling = "air" if cooling_label == "Air cool" else "laminar"
            headers_range = (4, 8)
            if cooling == "laminar":
                headers_range = st.slider("Laminar header count range", 0, 20, (4, 8), step=1,
                                          key=f"opt_hdr_{mill}")
            ct_range = None
        else:
            entry_range = band_inputs("Temp_Transfer_Bar_C", "Transfer bar temperature")
            finish_range = band_inputs("Temp_FT_Avg_C", "Finishing temperature")
            ct_range = band_inputs("Temp_CT_Top_C", "Coiling temperature")
            cooling = "rot"
            headers_range = (0, 0)
            st.caption("No header choice on the Steckel runout table: the coiling temperature "
                       "carries the cooling, applied symmetrically to top and bottom.")

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Calculate", type="primary",
                                          use_container_width=True)

    # ---- idle path: replay the last submitted outcome, or do nothing ---------------------
    if not submitted:
        stored = st.session_state.get(result_key)
        if not stored:
            st.caption("Set the constraints, then press Calculate.")
            return
        st.caption("Shown from your last calculation. Press Calculate to run again with "
                   "the current inputs.")
        _render_stored_result(mill, stored)
        return

    if slab_choice == AUTO_CHEMISTRY:
        chemistries = lean_chemistries()
        if not chemistries:
            chemistries = [("simulator chemistry",
                            {c: float(raw.get(c, 0.0)) for c in raw if str(c).startswith("Chem_")})]
    else:
        chem = gt.chemistry(slab_choice) or {}
        if chem:
            chemistries = [(slab_choice, chem)]
        else:
            chemistries = [("simulator chemistry",
                            {c: float(raw.get(c, 0.0)) for c in raw if str(c).startswith("Chem_")})]

    if labels:
        requirements, info = spec_requirements(chosen_spec, chosen_grade, thickness, mill,
                                               rolling_type, chemistries[0][1])
        if info.get("status") == "out_of_gauge":
            st.warning(info.get("message", "Thickness outside the standard range."))
        if not requirements:
            st.error("This specification tier carries no readable YS/TS/EL/CE limits: "
                     "pick another specification + grade.")
            return
    else:
        requirements = preset_requirements(mill, chosen_spec)

    search_raw = dict(raw)
    search_raw["Rolling_Type"] = rolling_type

    # The submit above is the only gate for all three modes: each calculation runs on Modal
    # when it is configured and falls back to this process when it is not, and
    # remote_compute reports which one answered.
    where = " on Modal" if remote_compute.remote_ready() else ""
    mode_meta = {"spec": chosen_spec, "grade": chosen_grade, "thickness": float(thickness),
                 "rolling_type": rolling_type}

    if calc_mode == PARETO_MODE:
        with st.spinner(f"Scoring the entry × finish × cooling × chemistry grid and building "
                        f"the Pareto frontier against the specification at the conformal "
                        f"bound{where}..."):
            t0 = time.time()
            res, source = remote_compute.pareto_search(
                mill, search_raw, thickness, requirements, chemistries,
                entry_range, finish_range, cooling=cooling, headers_range=headers_range,
                ct_range=ct_range, k=k)
            elapsed = time.time() - t0
        elapsed_text = f"{elapsed * 1000:.0f} ms"
        # The frontier rows are small dicts, so the whole frontier fits in session state:
        # the replay needs no candidate frame and no second model call.
        stored_res = {
            "error": res.get("error"),
            "frontier": list(res.get("frontier") or []),
            "n_feasible": int(res.get("n_feasible") or 0),
            "n_evaluated": int(res.get("n_evaluated") or 0),
            "n_frontier": int(res.get("n_frontier") or 0),
            "n_truncated": bool(res.get("n_truncated")),
            "meta": res.get("meta") or {},
            "_meta": mode_meta,
        }
    elif calc_mode == ROBUST_MODE:
        with st.spinner(f"Scoring the deterministic grid{where}, then stress-testing the "
                        f"shortlist over the Monte-Carlo noise band..."):
            t0 = time.time()
            res, source = remote_compute.robust_recipe(
                mill, search_raw, thickness, requirements, chemistries,
                entry_range, finish_range, cooling=cooling, headers_range=headers_range,
                ct_range=ct_range, k=k, alpha=float(alpha))
            elapsed = time.time() - t0
        elapsed_text = f"{elapsed * 1000:.0f} ms"
        stored_res = {
            "error": res.get("error"),
            "message": res.get("message"),
            "chosen": res.get("chosen"),
            "runners": list(res.get("runners") or []),
            "deterministic": res.get("deterministic"),
            "det_is_chosen": bool(res.get("det_is_chosen")),
            "noise_description": res.get("noise_description", ""),
            "n_evaluated": int(res.get("n_evaluated") or 0),
            "n_candidates": int(res.get("n_candidates") or 0),
            "n_draws": int(res.get("n_draws") or 0),
            "alpha": float(res.get("alpha", alpha)),
            "n_samples": int(res.get("n_samples") or 0),
            "seed": res.get("seed"),
            "_meta": mode_meta,
        }
    else:
        with st.spinner(f"Scoring the entry × finish × cooling × chemistry grid against the "
                        f"specification at the conformal bound{where}..."):
            t0 = time.time()
            res, source = remote_compute.recipe_search(
                mill, search_raw, thickness, requirements, chemistries,
                entry_range, finish_range, cooling=cooling, headers_range=headers_range,
                ct_range=ct_range, k=k)
            elapsed = time.time() - t0
        elapsed_text = f"{elapsed * 1000:.0f} ms"

        # Keep only what the card renders: the scored candidate frame and the chemistries are
        # dropped, so a stored outcome stays small.
        stored_res = {
            "error": res.get("error"),
            "chosen": res.get("chosen"),
            "ladder": list(res.get("ladder") or []),
            "rejected": list(res.get("rejected") or []),
            "n_evaluated": int(res.get("n_evaluated") or 0),
            "_meta": mode_meta,
        }

    st.session_state[result_key] = {"mode": CALC_MODE_KEYS[calc_mode], "res": stored_res,
                                    "requirements": requirements, "k": float(k),
                                    "elapsed": elapsed_text, "source": source}

    # the submit path and the replay path render through the same helper
    _render_stored_result(mill, st.session_state[result_key])
