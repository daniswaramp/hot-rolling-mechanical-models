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


def run_recipe_search(mill, raw, thickness, requirements, chemistries,
                      entry_range, finish_range, cooling="air", headers_range=(4, 12),
                      ct_range=None, k=1.0, step=10.0):
    """Hottest-feasible-first recipe search. Pure: no Streamlit calls, safe to drive from tests.

    Builds the vectorised candidate grid entry × finish × cooling × chemistry (temperature
    step `step`, default 10 °C so ±20 °C aims stay representable), scores it through the
    mill feature module and the shared scoring plumbing, then hands it to select_recipe.
    """
    finish_col, entry_col = FINISH_COL[mill], ENTRY_COL[mill]
    mech_keys = sorted({r["key"] for r in requirements if r.get("kind") == "mech"})
    if not mech_keys:
        return {"error": "no verifiable mechanical requirements for this specification tier",
                "chosen": None, "ladder": [], "rejected": [], "candidates_df": pd.DataFrame()}

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
        return {"error": "empty temperature range", "chosen": None, "ladder": [],
                "rejected": [], "candidates_df": pd.DataFrame()}
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
    cand_df = pd.DataFrame(rows)

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
                return {"error": "Plate feature module unavailable", "chosen": None,
                        "ladder": [], "rejected": [], "candidates_df": pd.DataFrame()}
            X_all = ml.engineer_plate_features(cand_df)
        else:
            if ml.engineer_hrc_features is None:
                return {"error": "HRC feature module unavailable", "chosen": None,
                        "ladder": [], "rejected": [], "candidates_df": pd.DataFrame()}
            X_all, _ = ml.engineer_hrc_features(cand_df, is_train=False)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"feature engineering failed: {exc}", "chosen": None, "ladder": [],
                "rejected": [], "candidates_df": pd.DataFrame()}

    preds = score_targets(mill, X_all, mech_keys)
    if preds is None:
        return {"error": "no model could score the candidate grid", "chosen": None,
                "ladder": [], "rejected": [], "candidates_df": pd.DataFrame()}
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

    ood_keys = [entry_col, finish_col]
    if mill == "Plate":
        ood_keys += ["ACC_Laminar_Header", "ACC_Temp_Post_Laminar"]
    else:
        ood_keys += ["Temp_CT_Top_C", "Temp_CT_Bot_C"]
    cand_df["ood"] = ood_mask(mill, cand_df, ood_keys)

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

    Returns (chosen_row_dict_or_None, ladder_rows, rejected_notes).
    """
    finish_col, entry_col = FINISH_COL[mill], ENTRY_COL[mill]
    df = candidates_df.copy()
    if df.empty:
        return None, [], ["no candidates were evaluated"]

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
        return None, [], ["no requirement could be scored against the candidates"]

    for col in ("ce_iiw", "microalloy_total"):
        if col not in df.columns:
            df[col] = 0.0
    if mill == "Plate" and "ACC_Headers_n" not in df.columns:
        df["ACC_Headers_n"] = 0.0

    feasible = pd.Series(True, index=df.index)
    for rid, req in active:
        need = float(k) * float(req.get("half", 0.0))
        feasible &= df[f"margin_{rid}"] >= need - 1e-9

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
        return None, [], _rejected_band_notes(df, active, k, mill, finish_col)

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

    ranked = pool.sort_values(sort_cols, ascending=ascending, kind="mergesort")
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


def _render_optimizer_result(mill: str, res: dict, requirements, k: float, elapsed,
                             source: str = "local"):
    """
    Renders one solved (or failed) optimizer outcome: the extrapolation warning, recipe
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
        st.caption(remote_compute.source_label(source))
        return

    meta = res.get("_meta") or {}
    chosen_spec = meta.get("spec", "")
    chosen_grade = meta.get("grade", "")
    thickness = float(meta.get("thickness", 0.0))
    rolling_type = meta.get("rolling_type", "AR")
    elapsed_text = elapsed if isinstance(elapsed, str) else f"{float(elapsed) * 1000:.0f} ms"

    chosen = res.get("chosen")
    if chosen is None:
        st.error(f"No feasible recipe in {res['n_evaluated']:,} evaluated candidates at "
                 f"k = {k:.1f}× the 95% interval. Nothing is proposed: widen a temperature "
                 "band, raise k's tolerance via a lower safety factor, pick a richer slab "
                 "grade, or relax the specification.")
        for note in res["rejected"]:
            st.markdown(f"- {note}")
        st.caption(remote_compute.source_label(source))
        return

    if chosen.get("_extrapolated"):
        st.warning("The selected recipe sits outside the historical P01–P99 envelope on at "
                   "least one parameter. Every in-domain candidate failed the margin rule, so "
                   "an extrapolated candidate was used: treat the predictions as extrapolation.")

    # ---- recipe card ---------------------------------------------------------------------
    st.markdown(f"#### Recipe card: solved in {elapsed_text} "
                f"({res['n_evaluated']:,} candidates scored)")
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
    st.caption(remote_compute.source_label(source))

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
    recipe_keys = (["_chemistry"] + [c for c in chosen if str(c).startswith("Chem_")]
                   + [entry_col, finish_col]
                   + (["ACC_Laminar_Header", "ACC_Headers_n", "ACC_Active",
                       "ACC_Temp_Post_Laminar"] if mill == "Plate" else
                      ["Temp_CT_Top_C", "Temp_CT_Bot_C"])
                   + ["ce_iiw", "pcm", "microalloy_total", "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"]
                   + (["route"] if mill == "Plate" else [])
                   + [f"pred_{r['key']}" for r in requirements if r.get("kind") == "mech"]
                   + [f"margin_{r['key']}|{r['direction']}" for r in requirements])
    csv_row = {key: chosen.get(key) for key in dict.fromkeys(recipe_keys) if key in chosen}
    csv_row["k"] = k
    csv_row["ood"] = bool(chosen.get("ood", False))
    st.download_button(
        "Download recipe (CSV)",
        data=pd.DataFrame([csv_row]).to_csv(index=False).encode("utf-8"),
        file_name=f"recipe_{mill}_{chosen_spec.replace(' ', '_')}_{chosen_grade}.csv",
        mime="text/csv",
        use_container_width=True)


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
    Renders the redesigned Inverse Optimizer: a hottest-feasible-first recipe search.

    The user picks the end specification, target thickness and rolling process, then
    constrains the slab grade, entry/finish temperature bands and cooling; the app searches
    the constraint space and proposes the hottest recipe that keeps every spec-required
    property non-marginal (margin >= k × 95% conformal interval), with a rejection ladder.

    Every input lives inside one st.form, so touching a widget never reruns the script and
    nothing is scored until "Calculate recipe" is submitted. The submitted outcome (with the
    source it came from - Modal or this process) is kept in
    st.session_state[f"opt_result_{mill}"] and simply re-rendered on later reruns, so a
    rerun never spends another Modal call.
    """
    raw = dict(raw)
    st.markdown('<div class="section"><span>inverse setpoint search</span><i></i></div>',
                unsafe_allow_html=True)
    st.markdown('<div class="note">Hottest feasible recipe search against the spec at the 95% '
                'conformal bound: the hottest entry and finish temperatures that keep every '
                'spec-required property non-marginal, with the leanest chemistry that gets there.</div>',
                unsafe_allow_html=True)

    result_key = f"opt_result_{mill}"

    # Every input sits inside one st.form: widget changes stay client-side (no rerun, no
    # mid-search interaction) and nothing is scored until the operator submits the form.
    with st.form(key=f"opt_form_{mill}"):
        # ---- 1. end specification -------------------------------------------------------
        specs = sc.get_available_specs()
        use_workbook = bool(specs)
        if use_workbook:
            preferred = [s for s in specs if "JIS" in s.upper()]
            default_idx = specs.index(preferred[0]) if (mill == "HRC" and preferred) else 0
            chosen_spec = st.selectbox("End specification standard", specs, index=default_idx,
                                       key=f"opt_spec_{mill}")
            grades = sc.get_available_grades(chosen_spec) or ["-"]
            grade_idx = 0
            if mill == "HRC":
                jis_grades = [g for g in grades if "SS400" in g.upper()]
                if jis_grades:
                    grade_idx = grades.index(jis_grades[0])
            chosen_grade = st.selectbox("Grade", grades, index=grade_idx,
                                        key=f"opt_grade_{mill}")
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
            headers_range = (4, 12)
            if cooling == "laminar":
                headers_range = st.slider("Laminar header count range", 0, 20, (4, 12), step=1,
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
        submitted = st.form_submit_button("Calculate recipe", type="primary",
                                          use_container_width=True)

    # ---- idle path: replay the last submitted outcome, or do nothing ---------------------
    if not submitted:
        stored = st.session_state.get(result_key)
        if not stored:
            st.caption("Set the constraints, then press Calculate recipe.")
            return
        st.caption("Shown from your last calculation. Press Calculate recipe to run again with "
                   "the current inputs.")
        _render_optimizer_result(mill, stored["res"], stored["requirements"],
                                 stored["k"], stored["elapsed"],
                                 source=stored.get("source", "local"))
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

    if use_workbook:
        requirements, info = spec_requirements(chosen_spec, chosen_grade, thickness, mill,
                                               rolling_type, chemistries[0][1])
        if info.get("status") == "out_of_gauge":
            st.warning(info.get("message", "Thickness outside the standard range."))
        if not requirements:
            st.error("This specification tier carries no readable YS/TS/EL/CE limits: "
                     "pick another standard/grade.")
            return
    else:
        requirements = preset_requirements(mill, chosen_spec)

    search_raw = dict(raw)
    search_raw["Rolling_Type"] = rolling_type

    # The submit above is the only gate: this search runs on Modal when it is configured and
    # falls back to this process when it is not, and remote_compute reports which one answered.
    where = " on Modal" if remote_compute.remote_ready() else ""
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
        "_meta": {"spec": chosen_spec, "grade": chosen_grade,
                  "thickness": float(thickness), "rolling_type": rolling_type},
    }
    st.session_state[result_key] = {"res": stored_res, "requirements": requirements,
                                    "k": float(k), "elapsed": elapsed_text, "source": source}

    # the submit path and the replay path render through the same helper
    _render_optimizer_result(mill, stored_res, requirements, float(k), elapsed_text,
                             source=source)
