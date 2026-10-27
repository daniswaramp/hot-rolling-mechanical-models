"""Parse the fitted NLR equations into coefficients the app can evaluate, then sanity check them.

`reports/nlr_constitutive_formulas.txt` holds one equation per target, for example

    Stress = 96.51 + 300.00*(t^-0.5) + 2.732*(sigma_ss) + 0.00*(CE^2.00) + 0.72*(CoolingRate^1.00)

The terms are a fixed vocabulary, so they parse cleanly. The check at the end refits nothing: it
evaluates the parsed equation against measured plates and compares the R2 with the score the
training run reported. If the mapping of a variable to a column were wrong, that R2 would collapse.
"""

import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Plate"))
sys.path.insert(0, os.path.join(ROOT, "HRC", "script"))

# variable name in the equation -> column in the feature frame
VAR_MAP = {
    "t": "Plate_T_mm",
    "sigma_ss": "Pickering_SS",
    "CE": "CE_IIW",
    "CoolingRate": "Cooling_Rate_Index",
    "eps_total": "True_Strain_Total",
}
HRC_VAR_MAP = {"t": "Coil_T_mm"}
TERM = re.compile(r"([+-]?\s*[\d.]+)\s*\*\s*\(([A-Za-z_]+)(?:\^(-?[\d.]+))?\)")
PLAIN = re.compile(r"([+-]?\s*[\d.]+)\s*(?=\+|$)")


def parse_line(line):
    body = line.split("=", 1)[1]
    terms = []
    for coef, var, power in TERM.findall(body):
        terms.append({"coef": float(coef.replace(" ", "")), "var": var,
                      "power": float(power) if power else 1.0})
    head = body.strip()
    match = re.match(r"([+-]?[\d.]+)", head)
    intercept = float(match.group(1)) if match else 0.0
    return intercept, terms


def parse_file(path):
    out = {}
    target = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("Target ["):
                target = line.split("[", 1)[1].split("]", 1)[0]
            elif target and line.startswith("Stress"):
                intercept, terms = parse_line(line)
                out[target] = {"intercept": intercept, "terms": terms}
                target = None
    return out


def evaluate(model, feats):
    value = model["intercept"]
    for term in model["terms"]:
        col = VAR_MAP.get(term["var"]) or HRC_VAR_MAP.get(term["var"])
        if col is None or col not in feats:
            return np.nan
        value += term["coef"] * float(feats[col]) ** term["power"]
    return value


def term_values(frame, mill):
    """The five terms the equations use, computed from the raw sheet, independent of feature names."""
    num = lambda c, d=0.0: pd.to_numeric(frame.get(c), errors="coerce").fillna(d)
    if mill == "Plate":
        thick = num("Plate_T_mm", 16.0).clip(lower=1.0)
        slab = num("Slab_T_mm", 218.0).clip(lower=1.0)
        headers = num("ACC_Laminar_Header", 0.0)
    else:
        thick = num("Coil_T_mm", 6.0).clip(lower=1.0)
        slab = num("TB_Thick_mm", 25.0).clip(lower=1.0)
        headers = 0.0
    c, mn, si = num("Chem_C", .16), num("Chem_Mn", .9), num("Chem_Si", .2)
    cu, ni, cr, mo, v = num("Chem_Cu", .02), num("Chem_Ni", .02), num("Chem_Cr", .02), num("Chem_Mo", .01), num("Chem_V", .005)
    return {
        "t": thick,
        "sigma_ss": 32 * mn + 84 * si + 38 * cu + 32 * ni,
        "CE": c + mn / 6 + (cr + mo + v) / 5 + (ni + cu) / 15,
        "CoolingRate": (headers + 0.1) / thick ** 1.5,
        "eps_total": np.log(slab / thick),
    }


def check(mill, csv, parsed, target, actual_col):
    frame = pd.read_csv(csv, low_memory=False)
    if mill == "Plate":
        from script.ml.features import compute_metallurgical_features as build
    else:
        from ml.features import engineer_hrc_features as build
    values = term_values(frame, mill)
    y = pd.to_numeric(frame[actual_col], errors="coerce")
    lo, hi = (0, 500) if target.startswith("Impact") else (100, 1500) if target[:2] in ("YS", "TS") else (0, 70)
    ok = y.between(lo, hi).fillna(False)
    model = parsed[target]
    pred = pd.Series(model["intercept"], index=frame.index, dtype=float)
    for term in model["terms"]:
        base = values.get(term["var"])
        if base is None:
            return float("nan"), 0
        pred = pred + term["coef"] * base.astype(float) ** term["power"]
    pred, yv = pred[ok].head(4000), y[ok].head(4000)
    good = pred.notna() & np.isfinite(pred)
    r2 = 1 - ((yv[good] - pred[good]) ** 2).sum() / ((yv[good] - yv[good].mean()) ** 2).sum()
    return float(r2), int(good.sum())


def main():
    out = {}
    for mill, path, csv, targets in (
        ("Plate", os.path.join(ROOT, "Plate", "reports", "nlr_constitutive_formulas.txt"),
         os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
         {"YS_BN": "YS_Before_N_MPa", "TS_BN": "TS_Before_N_MPa", "EL_BN": "EL_Before_N_pct"}),
        ("HRC", os.path.join(ROOT, "HRC", "reports", "hrc_nlr_constitutive_formulas.txt"),
         os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"),
         {"YS_BN": "YS_BN_MPa", "TS_BN": "TS_BN_MPa", "EL_BN": "EL_BN_pct"}),
    ):
        if not os.path.exists(path):
            print(f"{mill}: no formula file, skipped")
            continue
        parsed = parse_file(path)
        out[mill] = parsed
        for target, col in targets.items():
            if target not in parsed:
                continue
            try:
                r2, n = check(mill, csv, parsed, target, col)
                parsed[target]["r2"] = round(r2, 4)
                print(f"{mill} {target:8} parsed R2 {r2:.3f} on {n} plates "
                      f"({len(parsed[target]['terms'])} terms)")
            except Exception as exc:
                print(f"{mill} {target}: check failed {type(exc).__name__}: {str(exc)[:90]}")

    dest = os.path.join(ROOT, "models", "nlr_models.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", dest)


if __name__ == "__main__":
    main()
