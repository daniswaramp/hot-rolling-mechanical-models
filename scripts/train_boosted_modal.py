"""Train XGBoost, LightGBM and CatBoost per target and ship the artifacts.

NLR is not trained here: its fitted equations already live in `reports/nlr_constitutive_formulas.txt`
and are parsed into the app instead.

    modal run scripts/train_boosted_modal.py
    modal volume get grp-dual-mill-models / ./boosted_out
"""

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Plate"))
sys.path.insert(0, os.path.join(ROOT, "HRC", "script"))

PLATE_TARGETS = {
    "YS_BN": "YS_Before_N_MPa", "TS_BN": "TS_Before_N_MPa", "EL_BN": "EL_Before_N_pct",
    "YS_AN": "YS_After_N_MPa", "TS_AN": "TS_After_N_MPa", "EL_AN": "EL_After_N_pct",
    "Impact_0C": "Impact_0C_E_Avg_J", "Impact_m20C": "Impact_m20C_E_Avg_J",
    "Impact_m40C": "Impact_m40C_E_Avg_J", "Z_Test": "Z_Test_Avg_pct",
}
HRC_TARGETS = {"YS_BN": "YS_BN_MPa", "TS_BN": "TS_BN_MPa", "EL_BN": "EL_BN_pct"}
BOUNDS = {"Impact": (0, 500), "EL": (0, 70), "Z_": (0, 100), "default": (100, 1500)}


def features(mill, frame):
    if mill == "Plate":
        from script.ml.features import compute_metallurgical_features
        return compute_metallurgical_features(frame)
    from ml.features import engineer_hrc_features
    x, _ = engineer_hrc_features(frame, is_train=True)
    return x


def train_all(data_dir, out_dir, n_estimators=400, plate_csv=None, hrc_csv=None):
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import GroupShuffleSplit
    from xgboost import XGBRegressor
    from catboost import CatBoostRegressor

    os.makedirs(out_dir, exist_ok=True)
    summary = {}

    for mill, csv_name, targets in (
        ("Plate", "plate_rolling_2026_cleaned.csv", PLATE_TARGETS),
        ("HRC", "steckel_mill_cleaned.csv", HRC_TARGETS),
    ):
        source = (plate_csv if mill == "Plate" else hrc_csv) or os.path.join(data_dir, csv_name)
        frame = pd.read_csv(source, low_memory=False)
        x_frame = features(mill, frame)
        cols = list(x_frame.columns)
        x_all = x_frame.fillna(x_frame.median()).to_numpy(dtype=float)
        groups = frame["Heat_ID"] if "Heat_ID" in frame else pd.Series(np.arange(len(frame)))

        for key, col in targets.items():
            y = pd.to_numeric(frame[col], errors="coerce")
            lo, hi = BOUNDS.get(key.split("_")[0] if key.split("_")[0] in BOUNDS else "default",
                                BOUNDS["default"])
            ok = y.between(lo, hi).fillna(False).to_numpy()
            if ok.sum() < 150:
                print(f"  {mill} {key}: {int(ok.sum())} usable rows, skipped")
                continue
            x, yv = x_all[ok], y.to_numpy(dtype=float)[ok]
            g = groups[ok] if groups is not None else None
            if g is not None and g.nunique() > 50:
                tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                              .split(x, yv, g))
            else:
                cut = int(len(yv) * .8)
                tr, te = np.arange(cut), np.arange(cut, len(yv))

            for name, model in (
                ("xgb", XGBRegressor(n_estimators=n_estimators, max_depth=5, learning_rate=.05,
                                     subsample=.85, colsample_bytree=.85, tree_method="hist",
                                     n_jobs=8, random_state=42)),
                ("lgbm", LGBMRegressor(n_estimators=n_estimators, num_leaves=31, learning_rate=.05,
                                       subsample=.85, colsample_bytree=.85, n_jobs=8,
                                       random_state=42, verbose=-1)),
                ("cat", CatBoostRegressor(iterations=n_estimators, depth=6, learning_rate=.05,
                                          verbose=False, random_seed=42, thread_count=8)),
            ):
                model.fit(x[tr], yv[tr])
                pred = model.predict(x[te])
                r2 = 1 - ((yv[te] - pred) ** 2).sum() / ((yv[te] - yv[te].mean()) ** 2).sum()
                rmse = float(np.sqrt(((yv[te] - pred) ** 2).mean()))
                path = os.path.join(out_dir, f"{mill}_{key}_{name}")
                if name == "xgb":
                    model.save_model(path + ".ubj")
                elif name == "lgbm":
                    model.booster_.save_model(path + ".txt")
                else:
                    model.save_model(path + ".cbm")
                summary[f"{mill}|{key}|{name}"] = {"r2": round(float(r2), 4),
                                                   "rmse": round(rmse, 3), "features": cols}
                print(f"  {mill} {key:12} {name:5} R2 {r2:.3f}")

    with open(os.path.join(out_dir, "boosted_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"trained {len(summary)} models")
    return summary


try:
    import modal

    app = modal.App("grp-dual-mill-boosted")
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("pandas>=2.1", "numpy>=1.26", "scikit-learn>=1.4", "xgboost>=2.0",
                          "lightgbm>=4.0", "catboost>=1.2"))
    data_volume = modal.Volume.from_name("grp-plate-data", create_if_missing=True)
    out_volume = modal.Volume.from_name("grp-dual-mill-models", create_if_missing=True)

    @app.function(image=image, volumes={"/data": data_volume, "/out": out_volume},
                  timeout=7200, memory=32768, cpu=8)
    def train_remote(n_estimators: int = 400):
        train_all("/data", "/out", n_estimators)
        out_volume.commit()

    @app.local_entrypoint()
    def main(n_estimators: int = 400):
        train_remote.remote(n_estimators)
        print("pull with: modal volume get grp-dual-mill-models / ./boosted_out")
except Exception as exc:  # modal is optional
    print(f"modal unavailable: {exc}")


if __name__ == "__main__":
    if "--local" in sys.argv:
        train_all(os.path.join(ROOT, "Plate", "csv"), os.path.join(ROOT, "boosted_out"),
                  plate_csv=os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
                  hrc_csv=os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"))
