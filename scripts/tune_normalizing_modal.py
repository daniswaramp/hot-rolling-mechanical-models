"""Modal Cloud calibration of normalizing physics against shipped Plate Champion GBDT models.

Runs strictly on Modal Cloud (never locally):

    python -m modal run scripts/tune_normalizing_modal.py

Workflow executed remotely inside Modal:
1. Build a 1,800-schedule stratified benchmark spanning ALL shipped grades in
   ``models/grade_templates.json``, plate gauges 8-80 mm, finishing rolling temperatures
   760-900 C, entry temperatures 920-1060 C, ACC headers 0/4/8, and ``Rolling_Type="N"``.
2. Score every benchmark schedule through the shipped Plate Champion GBDT models
   (``models/boosted_out/`` + ``Plate/script/ml/features.py`` +
   ``plate_softening_deltas_frame`` coupling, matching ``streamlit_app.score_frame``).
3. Extract physical constitutive predictors per schedule using ``app/normalizing_physics.py``
   at nominal plant normalizing (``t_soak_ref_c = 910.0``,
   ``t_res_ref = t_heat + 0.6 * plate_t_mm``).
4. Perform a zero-leakage grade-grouped split (``GroupShuffleSplit`` on ``grade``,
   80% train grades / 20% held-out grades).
5. Optimize constitutive weights (``YS_AN``, ``TS_AN``, ``EL_AN``) on train grades via
   bounded ``scipy.optimize.least_squares`` enforcing metallurgical sign discipline.
6. Compute Train and Held-Out metrics (``R2``, ``RMSE``, ``MAE``, ``Bias``,
   ``HitRate_20MPa`` / ``HitRate_2.5pct``), per-grade summary, a 240-point stratified
   ``parity_samples`` list, and an optional ``grp-plate-data`` plant-data audit block.
7. Write the resulting JSON payload locally to ``models/normalizing_calibration.json``.
"""

from datetime import datetime, timezone
import json
import os
import sys

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_REPO = "/root/repo"
OUT_JSON_PATH = os.path.join(REPO_ROOT, "models", "normalizing_calibration.json")

IMAGE_IGNORE = [
    ".git", ".git/**",
    "**/.git/**",
    "**/__pycache__/**", "**/*.pyc", "**/*.pyo",
    ".pytest_cache/**", "**/.pytest_cache/**",
    "tests", "tests/**",
    "notebooks", "notebooks/**",
    "docs", "docs/**",
    "graphify-out", "graphify-out/**",
    "push-to-GH", "push-to-GH/**",
    ".devcontainer", ".devcontainer/**",
    ".github", ".github/**",
    ".commandcode", ".commandcode/**",
]

app = modal.App("hrcm-normalizing-tune")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(os.path.join(REPO_ROOT, "requirements.txt"))
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, ignore=IMAGE_IGNORE)
)

data_volume = modal.Volume.from_name("grp-plate-data", create_if_missing=True)

PARAM_ORDER = ("sigma_0", "k_ss", "k_hp", "k_ce", "k_c", "k_v", "k_nb", "k_cr")

# Physical sign & magnitude bounds matching app/normalizing_physics.py constitutive terms:
#   y = sigma_0 + k_ss*sigma_ss + k_hp*inv_sqrt_d_mm + k_ce*ce_iiw + k_c*c_wt
#       + k_v*v_reprecip_wt + k_nb*nb_dissolved_wt + k_cr*cr_air_k_s + regime_adj
PARAM_BOUNDS = {
    "YS_AN": {
        "sigma_0": (10.0, 220.0),
        "k_ss": (0.20, 1.80),
        "k_hp": (8.0, 30.0),
        "k_ce": (40.0, 500.0),
        "k_c": (20.0, 500.0),
        "k_v": (100.0, 4000.0),
        "k_nb": (100.0, 4000.0),
        "k_cr": (0.0, 40.0),
    },
    "TS_AN": {
        "sigma_0": (80.0, 420.0),
        "k_ss": (0.20, 1.80),
        "k_hp": (4.0, 22.0),
        "k_ce": (50.0, 700.0),
        "k_c": (50.0, 900.0),
        "k_v": (50.0, 4000.0),
        "k_nb": (50.0, 4000.0),
        "k_cr": (0.0, 40.0),
    },
    "EL_AN": {
        "sigma_0": (20.0, 55.0),
        "k_ss": (-0.20, 0.0),
        "k_hp": (0.10, 2.5),
        "k_ce": (-45.0, 0.0),
        "k_c": (-45.0, 0.0),
        "k_v": (-150.0, -5.0),
        "k_nb": (-150.0, -5.0),
        "k_cr": (-5.0, 0.0),
    },
}


def _enter_repo():
    """Put the mounted repo, app/, Plate/, and HRC/script/ on sys.path and chdir to repo root."""
    root = REMOTE_REPO if os.path.isdir(REMOTE_REPO) else REPO_ROOT
    for p in (
        root,
        os.path.join(root, "app"),
        os.path.join(root, "Plate"),
        os.path.join(root, "HRC", "script"),
    ):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)
    return root


def _build_benchmark_frame(templates: dict, n_schedules: int = 1800, seed: int = 42):
    """Build a 1,800-schedule stratified benchmark spanning every shipped grade."""
    import numpy as np
    import pandas as pd
    import physics as ph

    rng = np.random.default_rng(seed)
    grades_dict = templates.get("grades", {})
    ordered_grades = sorted(
        grades_dict.keys(),
        key=lambda g: (float(grades_dict[g].get("ce_median") or 0.0), g),
    )
    n_grades = len(ordered_grades)
    if n_grades == 0:
        raise RuntimeError("models/grade_templates.json contains no grades")

    base_count = n_schedules // n_grades
    remainder = n_schedules % n_grades
    extra_idx = (
        set(np.linspace(0, n_grades - 1, remainder, dtype=int).tolist())
        if remainder
        else set()
    )

    elements = templates.get(
        "elements",
        ["C", "Mn", "Si", "Al", "S", "P", "Nb", "V", "Mo", "Ti", "Cu", "Ni", "Cr", "N"],
    )

    rows = []
    gauge_bins = [(8.0, 18.0), (18.0, 34.0), (34.0, 56.0), (56.0, 80.0)]
    acc_choices = (0.0, 4.0, 8.0)

    schedule_idx = 0
    for g_idx, grade_name in enumerate(ordered_grades):
        rec = grades_dict[grade_name]
        g_min = rec.get("min", {})
        g_max = rec.get("max", {})
        g_med = rec.get("median", {})
        count = base_count + (1 if g_idx in extra_idx else 0)

        for k in range(count):
            row = {"grade": grade_name}
            for el in elements:
                med = float(g_med.get(el, 0.0) or 0.0)
                lo = float(g_min.get(el, 0.0) if g_min.get(el) is not None else 0.0)
                hi = float(g_max.get(el, med) if g_max.get(el) is not None else med)
                if hi < lo:
                    lo, hi = hi, lo
                if k % 2 == 0 and hi > lo:
                    span = 0.15 * (hi - lo)
                    val = float(rng.uniform(max(lo, med - span), min(hi, med + span)))
                elif hi > lo:
                    val = float(rng.uniform(lo, hi))
                else:
                    val = med

                b_lo, b_hi = ph.PLATE_BOUNDS.get(f"Chem_{el}", (0.0, None))
                if b_lo is not None:
                    val = max(val, b_lo)
                if b_hi is not None:
                    val = min(val, b_hi)
                row[f"Chem_{el}"] = round(val, 5)

            g_lo, g_hi = gauge_bins[(schedule_idx + k) % len(gauge_bins)]
            plate_t = float(rng.uniform(g_lo, g_hi))
            slab_t = float(
                np.clip(max(218.0, plate_t * 2.1 + 55.0) + rng.uniform(-12.0, 14.0), 180.0, 265.0)
            )
            tb_t = float(np.clip(plate_t * rng.uniform(1.6, 2.5), plate_t * 1.25, 140.0))
            plate_w = float(rng.uniform(1400.0, 2900.0))
            last_draft = float(np.clip(plate_t * rng.uniform(0.12, 0.24), 1.5, 12.0))

            frt = float(rng.uniform(760.0, 900.0))
            entry_lo = max(920.0, frt + 40.0)
            entry_t = float(rng.uniform(entry_lo, 1060.0))
            headers = float(acc_choices[(schedule_idx + k) % len(acc_choices)])
            t_post = float(np.clip(ph.post_laminar_temp(frt, headers, plate_t), 400.0, frt))

            row.update({
                "Slab_T_mm": round(slab_t, 2),
                "Plate_T_mm": round(plate_t, 2),
                "Plate_W_mm": round(plate_w, 1),
                "TB_Thickness_mm": round(tb_t, 2),
                "FM_Last_Pass_Drafting": round(last_draft, 2),
                "FM_Temp_Before_Rolling": round(entry_t, 2),
                "FM_Finish_Temp": round(frt, 2),
                "ACC_Laminar_Header": headers,
                "ACC_Temp_Post_Laminar": round(t_post, 2),
                "Rolling_Type": "N",
            })
            rows.append(row)
            schedule_idx += 1

    return pd.DataFrame(rows)


def _extract_physics_matrix(frame, t_soak_ref_c: float = 910.0, soak_min_per_mm_ref: float = 0.6):
    """Evaluate normalizing_physics at nominal plant soak and build feature matrices."""
    import numpy as np
    import normalizing_physics as np_phys

    n = len(frame)
    # Columns matching PARAM_ORDER[1:]: sigma_ss, inv_sqrt_d_mm, ce_iiw, c_wt, v_reprecip_wt, nb_dissolved_wt, cr_air_k_s
    x_mat = np.zeros((n, 7), dtype=float)
    adj_ys = np.zeros(n, dtype=float)
    adj_ts = np.zeros(n, dtype=float)
    adj_el = np.zeros(n, dtype=float)
    d_alpha_um = np.zeros(n, dtype=float)
    ce_iiw = np.zeros(n, dtype=float)

    base_calib = dict(np_phys.DEFAULT_CALIBRATION)
    base_calib["t_soak_ref_c"] = float(t_soak_ref_c)
    base_calib["soak_min_per_mm_ref"] = float(soak_min_per_mm_ref)

    records = frame.to_dict(orient="records")
    for idx, row in enumerate(records):
        plate_t = float(row.get("Plate_T_mm", 16.0))
        t_heat = 1.2 * max(plate_t, 1.0) * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
        res_min = t_heat + soak_min_per_mm_ref * max(plate_t, 1.0)

        pred = np_phys.predict_normalizing_properties(
            row,
            t_soak_ref_c,
            res_min,
            ml_preds=None,
            calib=base_calib,
            heating_rate_min_per_mm=1.2,
            cooling_mode="still_air",
        )
        feats = pred["physics"]["features"]
        x_mat[idx, 0] = feats["sigma_ss"]
        x_mat[idx, 1] = feats["inv_sqrt_d_mm"]
        x_mat[idx, 2] = feats["ce_iiw"]
        x_mat[idx, 3] = feats["c_wt"]
        x_mat[idx, 4] = feats["v_reprecip_wt"]
        x_mat[idx, 5] = feats["nb_dissolved_wt"]
        x_mat[idx, 6] = feats["cr_air_k_s"]

        d_alpha_um[idx] = float(feats["d_alpha_um"])
        ce_iiw[idx] = float(feats["ce_iiw"])

        adj = np_phys._regime_adjustments(
            pred["regime"],
            t_soak_ref_c,
            pred["thermal"]["t_eff_soak_min"],
            pred["ac1_c"],
            pred["ac3_c"],
            pred["grain"]["d_gamma_final_um"],
            pred["grain"]["d_lim_um"],
        )
        adj_ys[idx] = float(adj["YS_AN"])
        adj_ts[idx] = float(adj["TS_AN"])
        adj_el[idx] = float(adj["EL_AN"])

    return {
        "X": x_mat,
        "adj": {"YS_AN": adj_ys, "TS_AN": adj_ts, "EL_AN": adj_el},
        "d_alpha_um": d_alpha_um,
        "ce_iiw": ce_iiw,
    }


def _eval_constitutive(w_vec, x_mat, adj_vec, clip_el: bool = False):
    """Evaluate constitutive equation identically to app/normalizing_physics.py."""
    import numpy as np

    raw = float(w_vec[0]) + x_mat @ np.asarray(w_vec[1:], dtype=float) + adj_vec
    if clip_el:
        return np.clip(raw, 5.0, 55.0)
    return raw


def _fit_target_weights(target_key: str, x_tr, y_tr, adj_tr, defaults: dict):
    """Fit bounded constitutive weights on training grades using scipy.optimize.least_squares."""
    import numpy as np
    from scipy.optimize import least_squares

    bounds_map = PARAM_BOUNDS[target_key]
    lb = np.array([bounds_map[k][0] for k in PARAM_ORDER], dtype=float)
    ub = np.array([bounds_map[k][1] for k in PARAM_ORDER], dtype=float)
    x0 = np.clip(np.array([float(defaults[k]) for k in PARAM_ORDER], dtype=float), lb + 1e-9, ub - 1e-9)
    clip_el = target_key == "EL_AN"

    def residuals(w):
        return _eval_constitutive(w, x_tr, adj_tr, clip_el=clip_el) - y_tr

    res = least_squares(
        residuals,
        x0,
        bounds=(lb, ub),
        method="trf",
        loss="linear",
        x_scale="jac",
        max_nfev=4000,
    )
    return {k: round(float(val), 6) for k, val in zip(PARAM_ORDER, res.x)}


def _regression_metrics(y_true, y_pred, hit_tol: float, hit_key: str) -> dict:
    """Compute R2, RMSE, MAE, Bias (pred - true), and within-tolerance HitRate (%)."""
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return {"n": 0, "R2": None, "RMSE": None, "MAE": None, "Bias": None, hit_key: None}

    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    hit_rate = 100.0 * float(np.mean(np.abs(err) <= hit_tol))
    return {
        "n": int(yt.size),
        "R2": round(r2, 4),
        "RMSE": round(rmse, 3),
        "MAE": round(mae, 3),
        "Bias": round(bias, 3),
        hit_key: round(hit_rate, 2),
    }


def _select_parity_samples(df, ce_arr, d_alpha_arr, phys_preds, is_heldout, n_samples: int = 240):
    """Build a compact 240-point stratified parity sample list across grades and splits."""
    import numpy as np

    n_total = len(df)
    held_idx = np.where(is_heldout)[0]
    train_idx = np.where(~is_heldout)[0]

    # Allocate half from held-out grades (up to available) and remainder from train grades, stratified by CE
    n_held_pick = min(len(held_idx), n_samples // 2)
    n_train_pick = min(len(train_idx), n_samples - n_held_pick)

    def _pick_stratified(indices, count):
        if count <= 0 or len(indices) == 0:
            return []
        order = indices[np.argsort(ce_arr[indices], kind="mergesort")]
        sel_pos = np.linspace(0, len(order) - 1, count, dtype=int)
        return order[sel_pos].tolist()

    chosen = _pick_stratified(held_idx, n_held_pick) + _pick_stratified(train_idx, n_train_pick)
    # Preserve deterministic CE ordering
    chosen = sorted(set(chosen), key=lambda i: (0 if is_heldout[i] else 1, float(ce_arr[i]), i))
    # Top up if duplicate rounding reduced length below n_samples
    if len(chosen) < min(n_samples, n_total):
        chosen_set = set(chosen)
        for i in np.argsort(ce_arr, kind="mergesort"):
            if int(i) not in chosen_set:
                chosen.append(int(i))
                chosen_set.add(int(i))
            if len(chosen) >= min(n_samples, n_total):
                break

    samples = []
    for i in chosen[:n_samples]:
        row = df.iloc[i]
        samples.append({
            "grade": str(row["grade"]),
            "split": "heldout" if bool(is_heldout[i]) else "train",
            "plate_t_mm": round(float(row["Plate_T_mm"]), 2),
            "ce_iiw": round(float(ce_arr[i]), 4),
            "ys_ml": round(float(row["YS_AN_ml"]), 2),
            "ys_phys": round(float(phys_preds["YS_AN"][i]), 2),
            "ts_ml": round(float(row["TS_AN_ml"]), 2),
            "ts_phys": round(float(phys_preds["TS_AN"][i]), 2),
            "el_ml": round(float(row["EL_AN_ml"]), 2),
            "el_phys": round(float(phys_preds["EL_AN"][i]), 2),
            "ys_bn": round(float(row["YS_BN_ml"]), 2),
            "ts_bn": round(float(row["TS_BN_ml"]), 2),
            "d_alpha_um": round(float(d_alpha_arr[i]), 2),
        })
    return samples


def _build_per_grade_summary(df, ce_arr, phys_preds, is_heldout):
    """Per-grade summary across all 91 shipped grades."""
    import numpy as np

    summary = []
    for grade, grp in df.groupby("grade", sort=True):
        idx = grp.index.to_numpy()
        split = "heldout" if bool(is_heldout[idx[0]]) else "train"
        summary.append({
            "grade": str(grade),
            "split": split,
            "n": int(len(idx)),
            "ce_iiw_mean": round(float(np.mean(ce_arr[idx])), 4),
            "ys_ml_mean": round(float(np.mean(df.loc[idx, "YS_AN_ml"])), 2),
            "ys_phys_mean": round(float(np.mean(phys_preds["YS_AN"][idx])), 2),
            "ys_mae": round(float(np.mean(np.abs(phys_preds["YS_AN"][idx] - df.loc[idx, "YS_AN_ml"]))), 2),
            "ts_ml_mean": round(float(np.mean(df.loc[idx, "TS_AN_ml"])), 2),
            "ts_phys_mean": round(float(np.mean(phys_preds["TS_AN"][idx])), 2),
            "ts_mae": round(float(np.mean(np.abs(phys_preds["TS_AN"][idx] - df.loc[idx, "TS_AN_ml"]))), 2),
            "el_ml_mean": round(float(np.mean(df.loc[idx, "EL_AN_ml"])), 2),
            "el_phys_mean": round(float(np.mean(phys_preds["EL_AN"][idx])), 2),
            "el_mae": round(float(np.mean(np.abs(phys_preds["EL_AN"][idx] - df.loc[idx, "EL_AN_ml"]))), 2),
        })
    return summary


def _audit_plant_data(tuned_calib: dict, t_soak_ref_c: float, soak_min_per_mm_ref: float):
    """Read-only check against /data/plate_rolling_2026_cleaned.csv if mounted with AN rows."""
    import numpy as np
    import pandas as pd
    import normalizing_physics as np_phys

    csv_path = "/data/plate_rolling_2026_cleaned.csv"
    if not os.path.exists(csv_path):
        return {
            "available": False,
            "n_rows": 0,
            "note": "plate_rolling_2026_cleaned.csv not present in grp-plate-data volume",
        }

    raw_df = pd.read_csv(csv_path, low_memory=False)
    target_cols = {
        "YS_AN": ("YS_After_N_MPa", 180.0, 800.0, 20.0, "HitRate_20MPa"),
        "TS_AN": ("TS_After_N_MPa", 280.0, 900.0, 20.0, "HitRate_20MPa"),
        "EL_AN": ("EL_After_N_pct", 10.0, 55.0, 2.5, "HitRate_2.5pct"),
    }

    any_valid = np.zeros(len(raw_df), dtype=bool)
    for tgt, (col, lo, hi, _, _) in target_cols.items():
        if col in raw_df.columns:
            vals = pd.to_numeric(raw_df[col], errors="coerce")
            any_valid |= vals.between(lo, hi).fillna(False).to_numpy()

    an_df = raw_df.loc[any_valid].copy().reset_index(drop=True)
    if an_df.empty:
        return {
            "available": True,
            "n_rows": 0,
            "note": "CSV mounted but contains 0 valid After-Normalizing (AN) rows",
        }

    # Normalize column aliases if cleaned sheet uses TB_Ratio without TB_Thickness_mm
    if "Plate_T_mm" not in an_df.columns:
        an_df["Plate_T_mm"] = 16.0

    ys_phys = np.full(len(an_df), np.nan, dtype=float)
    ts_phys = np.full(len(an_df), np.nan, dtype=float)
    el_phys = np.full(len(an_df), np.nan, dtype=float)

    for idx, row in enumerate(an_df.to_dict(orient="records")):
        raw_t = pd.to_numeric(row.get("Plate_T_mm"), errors="coerce")
        plate_t = float(raw_t) if np.isfinite(raw_t) and raw_t > 0.0 else 16.0
        t_heat = 1.2 * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
        res_min = t_heat + soak_min_per_mm_ref * plate_t
        pred = np_phys.predict_normalizing_properties(
            row,
            t_soak_ref_c,
            res_min,
            ml_preds=None,
            calib=tuned_calib,
            heating_rate_min_per_mm=1.2,
            cooling_mode="still_air",
        )
        ys_phys[idx] = pred["physics"]["ys_phys_mpa"]
        ts_phys[idx] = pred["physics"]["ts_phys_mpa"]
        el_phys[idx] = pred["physics"]["el_phys_pct"]

    pred_map = {"YS_AN": ys_phys, "TS_AN": ts_phys, "EL_AN": el_phys}
    audit_metrics = {}
    for tgt, (col, lo, hi, tol, hit_key) in target_cols.items():
        if col not in an_df.columns:
            continue
        yv = pd.to_numeric(an_df[col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(yv) & (yv >= lo) & (yv <= hi)
        if ok.sum() > 0:
            audit_metrics[tgt] = _regression_metrics(yv[ok], pred_map[tgt][ok], tol, hit_key)

    return {
        "available": True,
        "n_rows": int(len(an_df)),
        "metrics": audit_metrics,
    }


@app.function(
    image=image,
    cpu=4.0,
    memory=8192,
    timeout=1800,
    volumes={"/data": data_volume},
)
def tune_against_an_models():
    """Run full Modal-cloud benchmark scoring, grade-grouped optimization, and audit."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    root = _enter_repo()
    import normalizing_physics as np_phys
    import streamlit_app

    t_soak_ref_c = 910.0
    soak_min_per_mm_ref = 0.6

    with open(os.path.join(root, "models", "grade_templates.json"), "r", encoding="utf-8") as fh:
        templates = json.load(fh)

    bench_df = _build_benchmark_frame(templates, n_schedules=1800, seed=42)
    scored_df = streamlit_app.score_frame("Plate", bench_df, engine="Champion GBDT (Recommended)")

    for col in ("YS_AN", "TS_AN", "EL_AN", "YS_BN", "TS_BN", "EL_BN"):
        bench_df[f"{col}_ml"] = np.asarray(scored_df[col], dtype=float)

    phys_data = _extract_physics_matrix(
        bench_df,
        t_soak_ref_c=t_soak_ref_c,
        soak_min_per_mm_ref=soak_min_per_mm_ref,
    )
    x_all = phys_data["X"]
    adj_map = phys_data["adj"]
    d_alpha_arr = phys_data["d_alpha_um"]
    ce_arr = phys_data["ce_iiw"]

    groups = bench_df["grade"].to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, heldout_idx = next(splitter.split(x_all, bench_df["YS_AN_ml"].to_numpy(), groups))
    is_heldout = np.zeros(len(bench_df), dtype=bool)
    is_heldout[heldout_idx] = True

    train_grades = sorted(set(groups[train_idx].tolist()))
    heldout_grades = sorted(set(groups[heldout_idx].tolist()))
    assert set(train_grades).isdisjoint(set(heldout_grades)), "Grade leakage across train/heldout!"

    tuned_weights = {}
    metrics_out = {}
    phys_preds = {}

    target_specs = (
        ("YS_AN", "YS_AN_ml", 20.0, "HitRate_20MPa", False),
        ("TS_AN", "TS_AN_ml", 20.0, "HitRate_20MPa", False),
        ("EL_AN", "EL_AN_ml", 2.5, "HitRate_2.5pct", True),
    )

    for tgt, ml_col, hit_tol, hit_key, clip_el in target_specs:
        y_all = bench_df[ml_col].to_numpy(dtype=float)
        adj_all = adj_map[tgt]
        defaults = np_phys.DEFAULT_CALIBRATION[tgt]

        w_dict = _fit_target_weights(tgt, x_all[train_idx], y_all[train_idx], adj_all[train_idx], defaults)
        tuned_weights[tgt] = w_dict

        w_vec = [w_dict[k] for k in PARAM_ORDER]
        y_pred_all = _eval_constitutive(w_vec, x_all, adj_all, clip_el=clip_el)
        phys_preds[tgt] = y_pred_all

        metrics_out[tgt] = {
            "train": _regression_metrics(y_all[train_idx], y_pred_all[train_idx], hit_tol, hit_key),
            "heldout": _regression_metrics(y_all[heldout_idx], y_pred_all[heldout_idx], hit_tol, hit_key),
        }

    tuned_calib = {
        "source": "Modal Cloud Calibration (hrcm-normalizing-tune)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "t_soak_ref_c": t_soak_ref_c,
        "soak_min_per_mm_ref": soak_min_per_mm_ref,
        "n_schedules": int(len(bench_df)),
        "n_grades": int(len(set(groups.tolist()))),
        "n_train_grades": int(len(train_grades)),
        "n_heldout_grades": int(len(heldout_grades)),
        "n_train_schedules": int(len(train_idx)),
        "n_heldout_schedules": int(len(heldout_idx)),
        "split_strategy": "GroupShuffleSplit(test_size=0.20, random_state=42) grouped by grade",
        "YS_AN": tuned_weights["YS_AN"],
        "TS_AN": tuned_weights["TS_AN"],
        "EL_AN": tuned_weights["EL_AN"],
        "metrics": metrics_out,
        "per_grade": _build_per_grade_summary(bench_df, ce_arr, phys_preds, is_heldout),
        "parity_samples": _select_parity_samples(bench_df, ce_arr, d_alpha_arr, phys_preds, is_heldout, 240),
    }
    tuned_calib["plant_audit"] = _audit_plant_data(
        tuned_calib,
        t_soak_ref_c=t_soak_ref_c,
        soak_min_per_mm_ref=soak_min_per_mm_ref,
    )
    return tuned_calib


@app.local_entrypoint()
def main():
    payload = tune_against_an_models.remote()
    os.makedirs(os.path.dirname(OUT_JSON_PATH), exist_ok=True)
    with open(OUT_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Saved {OUT_JSON_PATH} ({payload['n_schedules']} schedules across {payload['n_grades']} grades)")
    print(f"Split: {payload['n_train_grades']} train grades ({payload['n_train_schedules']} rows), "
          f"{payload['n_heldout_grades']} held-out grades ({payload['n_heldout_schedules']} rows)")
    for tgt in ("YS_AN", "TS_AN", "EL_AN"):
        tr = payload["metrics"][tgt]["train"]
        ho = payload["metrics"][tgt]["heldout"]
        hit_key = "HitRate_20MPa" if tgt != "EL_AN" else "HitRate_2.5pct"
        print(
            f"  {tgt} | Train: R2={tr['R2']:.4f} RMSE={tr['RMSE']:.3f} MAE={tr['MAE']:.3f} "
            f"Bias={tr['Bias']:+.3f} {hit_key}={tr[hit_key]:.2f}% | "
            f"Held-Out: R2={ho['R2']:.4f} RMSE={ho['RMSE']:.3f} MAE={ho['MAE']:.3f} "
            f"Bias={ho['Bias']:+.3f} {hit_key}={ho[hit_key]:.2f}%"
        )
    pa = payload.get("plant_audit", {})
    print(f"Plant audit: available={pa.get('available')} n_rows={pa.get('n_rows')}")
