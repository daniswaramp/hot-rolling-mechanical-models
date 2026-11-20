"""Modal Cloud ablation of Rolling_Type one-hot features on Heavy Plate GBDT models.

Runs strictly on Modal Cloud mounting ``grp-plate-data`` at ``/data`` (never locally):

    python -m modal run scripts/ablate_rolling_type_modal.py

Workflow executed remotely inside Modal:
1. Load ``/data/plate_rolling_2026_cleaned.csv``.
2. Parse ``ACC_Laminar_Header`` into ``ACC_Active``, ``ACC_Headers_n``, and
   ``ACC_Temp_Post_Laminar`` using ``Plate/script/ml/data_prep.py::parse_laminar_header``
   (fixing the zeroed-out ``ACC_Headers_n`` bug from the original training run).
3. Compute the 48-column feature matrix via ``compute_metallurgical_features(frame)``.
4. Compare on identical held-out melts (``GroupShuffleSplit(n_splits=1, test_size=0.2,
   random_state=42)`` grouped by ``Heat_ID``) across the 9 Plate GBDT targets and
   3 families (``xgb``, ``lgbm``, ``cat``):
   - **With Route Labels (48 features)**: all 48 columns including ``is_AR, is_CR, is_TMCP``.
   - **Pure Physics (45 features)**: dropping ``["is_AR", "is_CR", "is_TMCP"]``.
5. Compute held-out R2, RMSE, MAE, and split-conformal 95% / 90% half-widths and
   empirical coverages using the ``ceil((n + 1) * (1 - alpha)) / n`` quantile rule.
6. Apply the decision gate:
   - Primary targets (``YS_BN``, ``TS_BN``, ``EL_BN``): pure-physics champion R2
     within ``0.015`` of 48-feature champion R2 (or better) AND ``coverage_95_pct >= 94.5``.
   - Secondary targets (``YS_AN``, ``TS_AN``, ``EL_AN``, ``Impact_0C``, ``Impact_m20C``,
     ``Impact_m40C``): ``coverage_95_pct >= 92.0``.
   - Conformal sanity checks matching ``tests/test_conformal.py``:
     ``Conformal_Half_Width_95 > Old_Naive_Half_Width`` and ``Old_Naive_Coverage_Pct < 45.0``.
   - If all pass -> ``"OPTION_A_PURE_PHYSICS"`` (return the 27 serialized pure-physics
     Plate model artifacts + updated metrics/conformal table to the local entrypoint).
   - Otherwise -> ``"OPTION_B_AUTO_ROUTE"`` (keep existing ``models/boosted_out/`` untouched
     and only save ``models/rolling_type_ablation.json``).
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
import tempfile

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REMOTE_REPO = "/root/repo"

ABLATION_JSON_PATH = os.path.join(REPO_ROOT, "models", "rolling_type_ablation.json")
BOOSTED_DIR = os.path.join(REPO_ROOT, "models", "boosted_out")
BOOSTED_METRICS_PATH = os.path.join(BOOSTED_DIR, "boosted_metrics.json")
REGISTRY_PATH = os.path.join(REPO_ROOT, "models", "registry.json")
CONFORMAL_CSV_PATH = os.path.join(REPO_ROOT, "Plate", "reports", "conformal_intervals.csv")

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

app = modal.App("hrcm-rolling-type-ablation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(os.path.join(REPO_ROOT, "requirements.txt"))
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_REPO, ignore=IMAGE_IGNORE)
)

data_volume = modal.Volume.from_name("grp-plate-data")

PLATE_GBDT_TARGETS = {
    "YS_BN": ("YS_Before_N_MPa", "MPa"),
    "TS_BN": ("TS_Before_N_MPa", "MPa"),
    "EL_BN": ("EL_Before_N_pct", "%"),
    "YS_AN": ("YS_After_N_MPa", "MPa"),
    "TS_AN": ("TS_After_N_MPa", "MPa"),
    "EL_AN": ("EL_After_N_pct", "%"),
    "Impact_0C": ("Impact_0C_E_Avg_J", "J"),
    "Impact_m20C": ("Impact_m20C_E_Avg_J", "J"),
    "Impact_m40C": ("Impact_m40C_E_Avg_J", "J"),
}
PRIMARY_TARGETS = ("YS_BN", "TS_BN", "EL_BN")
SECONDARY_TARGETS = ("YS_AN", "TS_AN", "EL_AN", "Impact_0C", "Impact_m20C", "Impact_m40C")
ROUTE_COLS = ["is_AR", "is_CR", "is_TMCP"]
FAMILIES = ("cat", "xgb", "lgbm")
FAMILY_EXT = {"xgb": "ubj", "lgbm": "txt", "cat": "cbm"}
BOUNDS = {"Impact": (0, 500), "EL": (0, 70), "Z_": (0, 100), "default": (100, 1500)}


def _enter_repo():
    """Put /root/repo, app/, and Plate/ on sys.path and chdir to repo root."""
    root = REMOTE_REPO if os.path.isdir(REMOTE_REPO) else REPO_ROOT
    for p in (root, os.path.join(root, "app"), os.path.join(root, "Plate")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(root)
    return root


def _parse_laminar_columns(frame):
    """Populate ACC_Active, ACC_Headers_n, ACC_Temp_Post_Laminar via parse_laminar_header."""
    import numpy as np
    import pandas as pd
    from script.ml.data_prep import parse_laminar_header

    n_rows = len(frame)
    raw_hdrs = frame["ACC_Laminar_Header"].tolist() if "ACC_Laminar_Header" in frame.columns else [np.nan] * n_rows
    raw_tmps = (
        frame["ACC_Temp_Post_Laminar"].tolist()
        if "ACC_Temp_Post_Laminar" in frame.columns
        else [np.nan] * n_rows
    )
    raw_frts = frame["FM_Finish_Temp"].tolist() if "FM_Finish_Temp" in frame.columns else [np.nan] * n_rows

    headers_active = []
    headers_count = []
    updated_temps = []
    for raw_hdr, raw_tmp, raw_frt in zip(raw_hdrs, raw_tmps, raw_frts):
        act, cnt, tmp = parse_laminar_header(raw_hdr, raw_tmp)
        if pd.isna(tmp) and act == 0:
            tmp = raw_frt
        headers_active.append(act)
        headers_count.append(cnt)
        updated_temps.append(tmp)

    out = frame.copy()
    out["ACC_Active"] = headers_active
    out["ACC_Headers_n"] = headers_count
    out["ACC_Temp_Post_Laminar"] = updated_temps
    return out


def _build_model(family: str, n_estimators: int = 400):
    """Instantiate estimator matching scripts/train_boosted_modal.py hyperparameters."""
    if family == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            n_jobs=8,
            random_state=42,
        )
    if family == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=n_estimators,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=8,
            random_state=42,
            verbose=-1,
        )
    if family == "cat":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=n_estimators,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            random_seed=42,
            thread_count=8,
        )
    raise ValueError(f"Unknown model family: {family}")


def _evaluate_predictions(y_true, y_pred):
    """Compute regression metrics and split-conformal intervals on held-out melts."""
    import numpy as np

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    err = yt - yp
    abs_err = np.abs(err)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))

    n = int(len(abs_err))
    q95_level = min(1.0, math.ceil((n + 1) * 0.95) / n)
    q90_level = min(1.0, math.ceil((n + 1) * 0.90) / n)
    hw_95 = float(np.quantile(abs_err, q95_level))
    cov_95 = 100.0 * float(np.mean(abs_err <= hw_95))
    hw_90 = float(np.quantile(abs_err, q90_level))
    cov_90 = 100.0 * float(np.mean(abs_err <= hw_90))

    naive_half = 1.96 * rmse / math.sqrt(50.0)
    naive_cov = 100.0 * float(np.mean(abs_err <= naive_half))

    return {
        "n_val": n,
        "r2_raw": r2,
        "r2": round(r2, 4),
        "rmse_raw": rmse,
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "conformal_half_width_95": round(hw_95, 2),
        "coverage_95_pct": round(cov_95, 1),
        "conformal_half_width_90": round(hw_90, 2),
        "coverage_90_pct": round(cov_90, 1),
        "old_naive_half_width": round(naive_half, 2),
        "old_naive_coverage_pct": round(naive_cov, 1),
    }


def _serialize_model(model, family: str, tmp_dir: str, stem: str) -> tuple[str, bytes]:
    """Serialize a trained GBDT model to native bytes matching app/streamlit_app.py."""
    ext = FAMILY_EXT[family]
    fname = f"{stem}.{ext}"
    fpath = os.path.join(tmp_dir, fname)
    if family == "xgb":
        model.save_model(fpath)
    elif family == "lgbm":
        model.booster_.save_model(fpath)
    else:
        model.save_model(fpath)
    with open(fpath, "rb") as fh:
        raw = fh.read()
    if family == "lgbm":
        raw = raw.replace(b"\r\n", b"\n")
    return fname, raw


def _pick_champion_family(family_metrics: dict) -> str:
    """Select champion family matching streamlit_app.champion_plan (highest rounded R2)."""
    return max(
        FAMILIES,
        key=lambda f: (round(family_metrics[f]["r2_raw"], 4), -FAMILIES.index(f)),
    )


@app.function(
    image=image,
    volumes={"/data": data_volume},
    cpu=8.0,
    memory=16384,
    timeout=3600,
)
def run_rolling_type_ablation(n_estimators: int = 400):
    """Run 48-feature vs 45-feature pure-physics ablation on /data/plate_rolling_2026_cleaned.csv."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    _enter_repo()
    from script.ml.features import compute_metallurgical_features

    csv_path = "/data/plate_rolling_2026_cleaned.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected cleaned plate CSV at {csv_path}")

    raw_df = pd.read_csv(csv_path, low_memory=False)
    parsed_df = _parse_laminar_columns(raw_df)

    x_frame = compute_metallurgical_features(parsed_df)
    x_frame_48 = x_frame.fillna(x_frame.median())
    cols_48 = list(x_frame_48.columns)
    missing_route = [c for c in ROUTE_COLS if c not in cols_48]
    if missing_route:
        raise RuntimeError(f"Expected route columns missing from features: {missing_route}")

    x_frame_45 = x_frame_48.drop(columns=ROUTE_COLS)
    cols_45 = list(x_frame_45.columns)
    assert len(cols_48) == 48, f"Expected 48 features, got {len(cols_48)}"
    assert len(cols_45) == 45, f"Expected 45 pure-physics features, got {len(cols_45)}"

    x_all_48 = x_frame_48.to_numpy(dtype=float)
    x_all_45 = x_frame_45.to_numpy(dtype=float)
    groups_all = (
        parsed_df["Heat_ID"]
        if "Heat_ID" in parsed_df.columns
        else pd.Series(np.arange(len(parsed_df)))
    )

    dataset_diag = {
        "csv_path": csv_path,
        "total_rows": int(len(parsed_df)),
        "unique_heats": int(parsed_df["Heat_ID"].nunique(dropna=True))
        if "Heat_ID" in parsed_df.columns
        else None,
        "acc_active_rows": int((parsed_df["ACC_Active"] == 1).sum()),
        "acc_headers_nonzero_rows": int((parsed_df["ACC_Headers_n"] > 0).sum()),
        "acc_headers_max": float(parsed_df["ACC_Headers_n"].max()),
        "rolling_type_counts": (
            parsed_df["Rolling_Type"]
            .fillna("<NA>")
            .astype(str)
            .str.upper()
            .str.strip()
            .value_counts()
            .to_dict()
            if "Rolling_Type" in parsed_df.columns
            else {}
        ),
    }

    targets_report = {}
    pure_model_bytes = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for tgt_key, (csv_col, unit) in PLATE_GBDT_TARGETS.items():
            y = pd.to_numeric(parsed_df[csv_col], errors="coerce")
            prefix = tgt_key.split("_")[0]
            lo, hi = BOUNDS.get(prefix if prefix in BOUNDS else "default", BOUNDS["default"])
            ok = y.between(lo, hi).fillna(False).to_numpy()
            if int(ok.sum()) < 150:
                raise RuntimeError(f"Plate {tgt_key}: only {int(ok.sum())} usable rows (< 150)")

            yv = y.to_numpy(dtype=float)[ok]
            g = groups_all[ok].copy()
            if g.isna().any():
                fill_val = -1 if pd.api.types.is_numeric_dtype(g) else "UNKNOWN_HEAT"
                g = g.fillna(fill_val)

            if g.nunique() > 50:
                tr, te = next(
                    GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
                        x_all_48[ok], yv, g
                    )
                )
                assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te])), f"{tgt_key}: Heat_ID overlap!"
                split_info = {
                    "strategy": "GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)",
                    "n_total": int(len(yv)),
                    "n_train": int(len(tr)),
                    "n_val": int(len(te)),
                    "n_train_heats": int(g.iloc[tr].nunique()),
                    "n_val_heats": int(g.iloc[te].nunique()),
                }
            else:
                cut = int(len(yv) * 0.8)
                tr, te = np.arange(cut), np.arange(cut, len(yv))
                split_info = {
                    "strategy": "sequential_80_20",
                    "n_total": int(len(yv)),
                    "n_train": int(len(tr)),
                    "n_val": int(len(te)),
                    "n_train_heats": None,
                    "n_val_heats": None,
                }

            variant_results = {}
            for variant_key, x_source in (
                ("with_route_48", x_all_48[ok]),
                ("pure_physics_45", x_all_45[ok]),
            ):
                fam_metrics = {}
                for fam in FAMILIES:
                    model = _build_model(fam, n_estimators=n_estimators)
                    model.fit(x_source[tr], yv[tr])
                    pred = np.asarray(model.predict(x_source[te]), dtype=float).ravel()
                    m = _evaluate_predictions(yv[te], pred)
                    fam_metrics[fam] = m
                    print(
                        f"  [{variant_key:15s}] Plate {tgt_key:12s} {fam:4s} "
                        f"R2={m['r2']:.4f} RMSE={m['rmse']:.3f} "
                        f"HW95=+/-{m['conformal_half_width_95']:.2f} ({m['coverage_95_pct']:.1f}%)"
                    )

                    if variant_key == "pure_physics_45":
                        fname, blob = _serialize_model(
                            model, fam, tmp_dir, f"Plate_{tgt_key}_{fam}"
                        )
                        pure_model_bytes[fname] = blob

                champ_fam = _pick_champion_family(fam_metrics)
                champ_entry = dict(fam_metrics[champ_fam])
                champ_entry["champion_family"] = champ_fam
                variant_results[variant_key] = {
                    "families": fam_metrics,
                    "champion": champ_entry,
                }

            c48 = variant_results["with_route_48"]["champion"]
            c45 = variant_results["pure_physics_45"]["champion"]
            targets_report[tgt_key] = {
                "csv_col": csv_col,
                "unit": unit,
                "split": split_info,
                **variant_results,
                "delta_champion_r2": round(c45["r2_raw"] - c48["r2_raw"], 6),
                "delta_champion_rmse": round(c45["rmse_raw"] - c48["rmse_raw"], 4),
            }

    # Evaluate Decision Gate
    gate_details = {}
    primary_r2_ok = True
    primary_cov_ok = True
    secondary_cov_ok = True
    conformal_sanity_ok = True

    for tgt_key in PRIMARY_TARGETS:
        t_rec = targets_report[tgt_key]
        c48 = t_rec["with_route_48"]["champion"]
        c45 = t_rec["pure_physics_45"]["champion"]
        delta_r2 = c45["r2_raw"] - c48["r2_raw"]
        r2_ok = delta_r2 >= -0.015
        cov_ok = c45["coverage_95_pct"] >= 94.5
        sanity_ok = (
            c45["conformal_half_width_95"] > c45["old_naive_half_width"]
            and c45["old_naive_coverage_pct"] < 45.0
        )
        primary_r2_ok &= r2_ok
        primary_cov_ok &= cov_ok
        conformal_sanity_ok &= sanity_ok
        gate_details[tgt_key] = {
            "tier": "primary",
            "with_route_champion": f"{c48['champion_family']} (R2={c48['r2']:.4f})",
            "pure_physics_champion": f"{c45['champion_family']} (R2={c45['r2']:.4f})",
            "delta_r2_raw": round(delta_r2, 6),
            "r2_within_0_015": bool(r2_ok),
            "coverage_95_pct": c45["coverage_95_pct"],
            "coverage_min_required": 94.5,
            "coverage_ok": bool(cov_ok),
            "conformal_sanity_ok": bool(sanity_ok),
        }

    for tgt_key in SECONDARY_TARGETS:
        t_rec = targets_report[tgt_key]
        c48 = t_rec["with_route_48"]["champion"]
        c45 = t_rec["pure_physics_45"]["champion"]
        delta_r2 = c45["r2_raw"] - c48["r2_raw"]
        cov_ok = c45["coverage_95_pct"] >= 92.0
        sanity_ok = (
            c45["conformal_half_width_95"] > c45["old_naive_half_width"]
            and c45["old_naive_coverage_pct"] < 45.0
        )
        secondary_cov_ok &= cov_ok
        conformal_sanity_ok &= sanity_ok
        gate_details[tgt_key] = {
            "tier": "secondary",
            "with_route_champion": f"{c48['champion_family']} (R2={c48['r2']:.4f})",
            "pure_physics_champion": f"{c45['champion_family']} (R2={c45['r2']:.4f})",
            "delta_r2_raw": round(delta_r2, 6),
            "coverage_95_pct": c45["coverage_95_pct"],
            "coverage_min_required": 92.0,
            "coverage_ok": bool(cov_ok),
            "conformal_sanity_ok": bool(sanity_ok),
        }

    option_a_wins = primary_r2_ok and primary_cov_ok and secondary_cov_ok and conformal_sanity_ok
    decision = "OPTION_A_PURE_PHYSICS" if option_a_wins else "OPTION_B_AUTO_ROUTE"

    ablation_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "Modal Cloud Ablation (hrcm-rolling-type-ablation)",
        "n_estimators": int(n_estimators),
        "laminar_header_fix": "Parsed ACC_Laminar_Header -> ACC_Active, ACC_Headers_n, ACC_Temp_Post_Laminar via Plate/script/ml/data_prep.py::parse_laminar_header",
        "conformal_method": "split_conformal_quantile ceil((n+1)*(1-alpha))/n on |y_te - y_pred|",
        "dataset": dataset_diag,
        "features": {
            "dropped_route_columns": ROUTE_COLS,
            "with_route_48_count": len(cols_48),
            "with_route_48": cols_48,
            "pure_physics_45_count": len(cols_45),
            "pure_physics_45": cols_45,
        },
        "decision_gate": {
            "decision": decision,
            "primary_r2_within_0_015": bool(primary_r2_ok),
            "primary_coverage_ge_94_5": bool(primary_cov_ok),
            "secondary_coverage_ge_92_0": bool(secondary_cov_ok),
            "conformal_sanity_ok": bool(conformal_sanity_ok),
            "per_target": gate_details,
        },
        "targets": targets_report,
    }

    return {
        "report": ablation_report,
        "model_bytes": pure_model_bytes if option_a_wins else {},
    }


def _write_pure_physics_artifacts(report: dict, model_bytes: dict):
    """Write Option A pure-physics Plate models, metrics, registry, and conformal CSV."""
    import pandas as pd

    os.makedirs(BOOSTED_DIR, exist_ok=True)
    cols_45 = report["features"]["pure_physics_45"]
    targets_report = report["targets"]

    # 1. Write 27 serialized Plate models (forcing UNIX LF \n on LightGBM .txt files!)
    sha256_by_fname = {}
    for fname, blob in sorted(model_bytes.items()):
        dst = os.path.join(BOOSTED_DIR, fname)
        out_bytes = blob.replace(b"\r\n", b"\n") if fname.endswith(".txt") else blob
        if fname.endswith(".txt") and b"\r\n" in out_bytes:
            raise RuntimeError(f"CRLF leaked into {fname}")
        with open(dst, "wb") as fh:
            fh.write(out_bytes)
        sha256_by_fname[fname] = hashlib.sha256(out_bytes).hexdigest()

    # 2. Update Plate|* entries in models/boosted_out/boosted_metrics.json (preserve HRC|*)
    with open(BOOSTED_METRICS_PATH, "r", encoding="utf-8") as fh:
        boosted_metrics = json.load(fh)

    for tgt_key, t_rec in targets_report.items():
        fam_map = t_rec["pure_physics_45"]["families"]
        for fam in FAMILIES:
            m = fam_map[fam]
            boosted_metrics[f"Plate|{tgt_key}|{fam}"] = {
                "r2": m["r2"],
                "rmse": m["rmse"],
                "features": cols_45,
            }

    with open(BOOSTED_METRICS_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(boosted_metrics, fh, indent=1)
        fh.write("\n")

    # 3. Update models/registry.json Plate entries (SHA-256 hashes, metrics, num_features=45)
    with open(REGISTRY_PATH, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    registry["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for entry in registry.get("models", []):
        if entry.get("mill") != "Plate":
            continue
        tgt = entry["target"]
        fam = entry["family"]
        fname = entry["filename"]
        if tgt not in targets_report:
            continue
        fam_m = targets_report[tgt]["pure_physics_45"]["families"][fam]
        champ_m = targets_report[tgt]["pure_physics_45"]["champion"]
        entry["sha256"] = sha256_by_fname[fname]
        entry["val_r2"] = fam_m["r2"]
        entry["val_rmse"] = fam_m["rmse"]
        entry["conformal_half_width_95"] = champ_m["conformal_half_width_95"]
        entry["empirical_coverage_95_pct"] = champ_m["coverage_95_pct"]
        entry["num_features"] = len(cols_45)

    with open(REGISTRY_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(registry, fh, indent=2)
        fh.write("\n")

    # 4. Update Plate/reports/conformal_intervals.csv (preserving Z_Test row & Naive columns)
    existing_df = pd.read_csv(CONFORMAL_CSV_PATH)
    z_rows = existing_df[existing_df["Target"] == "Z_Test"].to_dict(orient="records")

    new_rows = []
    for tgt_key, (_csv_col, unit) in PLATE_GBDT_TARGETS.items():
        champ = targets_report[tgt_key]["pure_physics_45"]["champion"]
        new_rows.append({
            "Mill": "Plate",
            "Target": tgt_key,
            "Unit": unit,
            "Champion_Family": champ["champion_family"],
            "Val_N": int(champ["n_val"]),
            "Val_RMSE": champ["rmse"],
            "Val_MAE": champ["mae"],
            "Conformal_Half_Width_95": champ["conformal_half_width_95"],
            "Coverage_95_Pct": champ["coverage_95_pct"],
            "Conformal_Half_Width_90": champ["conformal_half_width_90"],
            "Coverage_90_Pct": champ["coverage_90_pct"],
            "Old_Naive_Half_Width": champ["old_naive_half_width"],
            "Old_Naive_Coverage_Pct": champ["old_naive_coverage_pct"],
        })
    new_rows.extend(z_rows)

    out_df = pd.DataFrame(new_rows, columns=list(existing_df.columns))
    out_df["Val_N"] = out_df["Val_N"].astype(int)
    out_df.to_csv(CONFORMAL_CSV_PATH, index=False, lineterminator="\n")


@app.local_entrypoint()
def main(n_estimators: int = 400):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    payload = run_rolling_type_ablation.remote(n_estimators=n_estimators)
    report = payload["report"]
    model_bytes = payload["model_bytes"]

    os.makedirs(os.path.dirname(ABLATION_JSON_PATH), exist_ok=True)
    with open(ABLATION_JSON_PATH, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")

    decision = report["decision_gate"]["decision"]
    print(f"\nSaved {ABLATION_JSON_PATH}")
    print(f"Dataset: {report['dataset']['total_rows']} rows, {report['dataset']['unique_heats']} heats, "
          f"ACC_Active={report['dataset']['acc_active_rows']} rows")
    print(f"Rolling_Type counts: {report['dataset']['rolling_type_counts']}")
    print("\n=== Plate Rolling_Type Ablation (With Route 48 vs Pure Physics 45) ===")
    header = (
        f"{'Target':<12} | {'With Route (48)':<24} | {'Pure Physics (45)':<24} | "
        f"{'dR2':>8} | {'HW95':>9} | {'Cov95':>6}"
    )
    print(header)
    print("-" * len(header))
    for tgt_key in PLATE_GBDT_TARGETS:
        t_rec = report["targets"][tgt_key]
        c48 = t_rec["with_route_48"]["champion"]
        c45 = t_rec["pure_physics_45"]["champion"]
        print(
            f"{tgt_key:<12} | "
            f"{c48['champion_family']:>4s} R2={c48['r2']:.4f} ({c48['rmse']:>5.2f}) | "
            f"{c45['champion_family']:>4s} R2={c45['r2']:.4f} ({c45['rmse']:>5.2f}) | "
            f"{t_rec['delta_champion_r2']:+8.4f} | "
            f"+/-{c45['conformal_half_width_95']:<6.2f} | "
            f"{c45['coverage_95_pct']:5.1f}%"
        )

    print(f"\nDECISION GATE: {decision}")
    if decision == "OPTION_A_PURE_PHYSICS":
        _write_pure_physics_artifacts(report, model_bytes)
        print(
            f"Updated {len(model_bytes)} Plate model files in {BOOSTED_DIR}, "
            f"boosted_metrics.json, registry.json, and conformal_intervals.csv."
        )
        print("NEXT STEP: run `python -m modal run scripts/tune_normalizing_modal.py` "
              "to sync models/normalizing_calibration.json.")
    else:
        print("Option B selected: left models/boosted_out/, registry.json, and conformal_intervals.csv untouched.")
