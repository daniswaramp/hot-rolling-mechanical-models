"""Fit plain OLS and Ridge per target so the app can offer genuine model choice.

The shipped artifacts are stacking/bagged ensembles. They are good but opaque, and a
metallurgist often wants the readable equation beside them. These two families are tiny enough to
commit as coefficients, so the choice does not cost a 30 MB artifact per target.

    python scripts/fit_linear_families.py
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

from script.ml.features import compute_metallurgical_features  # noqa: E402
from ml.features import engineer_hrc_features  # noqa: E402

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PLATE_TARGETS = {
    "YS_BN": "YS_Before_N_MPa", "TS_BN": "TS_Before_N_MPa", "EL_BN": "EL_Before_N_pct",
    "YS_AN": "YS_After_N_MPa", "TS_AN": "TS_After_N_MPa", "EL_AN": "EL_After_N_pct",
    "Impact_0C": "Impact_0C_E_Avg_J", "Impact_m20C": "Impact_m20C_E_Avg_J",
    "Impact_m40C": "Impact_m40C_E_Avg_J", "Z_Test": "Z_Test_Avg_pct",
}
HRC_TARGETS = {"YS_BN": "YS_BN_MPa", "TS_BN": "TS_BN_MPa", "EL_BN": "EL_BN_pct"}


def features(mill, frame):
    if mill == "Plate":
        return compute_metallurgical_features(frame)
    x, _ = engineer_hrc_features(frame, is_train=False)
    return x


def fit_one(x, y, groups):
    if groups is not None and groups.nunique() > 50:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
                      .split(x, y, groups))
    else:
        idx = np.arange(len(y))
        rng = np.random.default_rng(42)
        rng.shuffle(idx)
        cut = int(len(idx) * .8)
        tr, te = idx[:cut], idx[cut:]

    # standardise before solving: raw temperatures and chemistry differ by four orders of
    # magnitude and unscaled least squares is numerically useless on this feature set
    mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0)
    sd[sd == 0] = 1.0
    xs = (x - mu) / sd
    fit_s = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), xs[tr]]), y[tr], rcond=None)[0]
    coef = fit_s[1:] / sd
    ols_fit = np.concatenate([[fit_s[0] - float(np.dot(coef, mu))], coef])
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20)))
    ridge.fit(x[tr], y[tr])

    def score(pred):
        r2 = 1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum()
        return round(float(r2), 4), round(float(np.sqrt(((y[te] - pred) ** 2).mean())), 3)

    ols_pred = np.column_stack([np.ones(len(te)), x[te]]) @ ols_fit
    return {
        "ols": {"intercept": float(ols_fit[0]), "coef": [round(float(c), 10) for c in ols_fit[1:]],
                "r2": score(ols_pred)[0], "rmse": score(ols_pred)[1]},
        "ridge": {"scaler_mean": [round(float(v), 6) for v in ridge[0].mean_],
                  "scaler_scale": [round(float(v), 6) for v in ridge[0].scale_],
                  "coef": [round(float(v), 8) for v in ridge[1].coef_],
                  "intercept": float(ridge[1].intercept_),
                  "alpha": float(ridge[1].alpha_),
                  "r2": score(ridge.predict(x[te]))[0], "rmse": score(ridge.predict(x[te]))[1]},
    }


def run(mill, csv, targets, out):
    frame = pd.read_csv(csv, low_memory=False)
    x_frame = features(mill, frame)
    cols = list(x_frame.columns)
    x_all = x_frame.fillna(x_frame.median()).to_numpy(dtype=float)
    groups = frame["Heat_ID"] if "Heat_ID" in frame else None

    for key, col in targets.items():
        y = pd.to_numeric(frame[col], errors="coerce")
        # production logs carry decimal shifts; a single 41,205 MPa entry destroys least squares
        if key.startswith(("Impact",)):
            lo, hi = 0.0, 500.0
        elif key.startswith("EL"):
            lo, hi = 0.0, 70.0
        elif key.startswith("Z_"):
            lo, hi = 0.0, 100.0
        else:
            lo, hi = 100.0, 1500.0
        in_bounds = y.between(lo, hi).fillna(False)
        dropped = int((y.notna() & ~in_bounds).sum())
        if dropped:
            print(f"  {key}: dropped {dropped} implausible measured values")
        ok = in_bounds.to_numpy()
        if ok.sum() < 150:
            print(f"  {key}: {int(ok.sum())} rows, skipped")
            continue
        fitted = fit_one(x_all[ok], y.to_numpy(dtype=float)[ok],
                         groups[ok] if groups is not None else None)
        out.setdefault(mill, {})[key] = {"features": cols, **fitted}
        print(f"  {mill} {key:12} OLS {fitted['ols']['r2']:.3f}  Ridge {fitted['ridge']['r2']:.3f}")


def main():
    out = {}
    print("plate")
    run("Plate", os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
        PLATE_TARGETS, out)
    print("coil")
    run("HRC", os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"), HRC_TARGETS, out)

    path = os.path.join(ROOT, "models", "linear_families.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"wrote {path} ({os.path.getsize(path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
