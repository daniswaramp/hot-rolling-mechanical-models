"""Transfer-bar input range tests.

The input widgets used to clamp transfer-bar thickness at the historical envelope
(Plate 80 mm, HRC 70 mm) and the feature modules silently saturated above it, so a thicker
schedule scored exactly like one at the old ceiling. These tests pin the widened ranges,
the lifted saturation clips and the applicability-domain extrapolation warning in place.
"""

import json
import os
import sys

import pandas as pd
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")
PLATE_DIR = os.path.join(REPO_ROOT, "Plate")
HRC_DIR = os.path.join(REPO_ROOT, "HRC")

for p in [REPO_ROOT, APP_DIR, PLATE_DIR, os.path.join(HRC_DIR, "script")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import model_loader as ml  # noqa: E402
import streamlit_app as app  # noqa: E402
import explainer  # noqa: E402
from ml.features import engineer_hrc_features  # noqa: E402  (HRC/script/ml on the path)
from Plate.script.ml.features import compute_metallurgical_features  # noqa: E402


# --------------------------------------------------------------------------- widget bounds
def _tb_row(mill, key):
    row = app.spec_map(mill)[key]
    assert row[1] == key
    return row


def test_plate_tb_widget_ceiling_is_120():
    row = _tb_row("Plate", "TB_Ratio")
    assert row[3] == "×"
    assert row[4] == 1.1 and row[5] == 8.0 and row[6] == 2.56 and row[7] == 0.05


def test_hrc_tb_widget_ceiling_is_105():
    row = _tb_row("HRC", "TB_Ratio")
    assert row[3] == "×"
    assert row[4] == 1.2 and row[5] == 15.0 and row[6] == 4.13 and row[7] == 0.1


# --------------------------------------------------------------------------- feature clips
def test_hrc_feature_no_longer_saturates_at_70():
    """TB 90 mm must reach the model as 90, not the old 70 mm training ceiling."""
    frame = pd.DataFrame([{"TB_Thick_mm": 90.0, "Coil_T_mm": 6.0}])
    X, _ = engineer_hrc_features(frame, is_train=False)
    assert float(X["TB_Thick_mm"].iloc[0]) == pytest.approx(90.0)


def test_plate_tb_ratio_no_longer_saturates_at_6():
    """120 mm transfer bar on 16 mm plate gives TB_Ratio 7.5, not the old 6.0 ceiling."""
    frame = app.prepare_frame("Plate", pd.DataFrame(
        [{"Plate_T_mm": 16.0, "TB_Thickness_mm": 120.0}]))
    X = compute_metallurgical_features(frame)
    assert float(X["TB_Ratio"].iloc[0]) == pytest.approx(7.5)


def test_plate_tb_ratio_unchanged_in_distribution():
    """In-range values are numerically unchanged: the default 41/16 still gives 2.5625."""
    frame = app.prepare_frame("Plate", pd.DataFrame(
        [{"Plate_T_mm": 16.0, "TB_Thickness_mm": 41.0}]))
    X = compute_metallurgical_features(frame)
    assert float(X["TB_Ratio"].iloc[0]) == pytest.approx(41.0 / 16.0)


# --------------------------------------------------------------------------- OOD envelope
def _domain():
    with open(os.path.join(REPO_ROOT, "models", "applicability_domain.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def test_plate_domain_defines_tb_thickness_envelope():
    block = _domain()["Plate"]["TB_Thickness_mm"]
    assert block["p99"] == 80.0
    assert block["p01"] <= block["p99"]


def test_plate_tb_above_envelope_warns():
    """Plate transfer bar never used to warn; 110 mm is outside the full historical range."""
    raw = dict(app.defaults("Plate"), Rolling_Type="AR", TB_Thickness_mm=110.0)
    warnings, _notices = explainer.check_applicability_domain("Plate", raw)
    tb_flags = [w for w in warnings if w.get("param") == "TB_Thickness_mm"]
    assert tb_flags, "no applicability-domain warning for TB_Thickness_mm=110"
    assert tb_flags[0]["bound"] == 80.0


def test_plate_tb_default_stays_in_distribution():
    raw = app.defaults("Plate")
    warnings, _notices = explainer.check_applicability_domain("Plate", raw)
    assert all(w.get("param") != "TB_Thickness_mm" for w in warnings)


def test_tb_ratio_propagates_to_features_indicators_and_ood():
    """TB_Ratio propagates to Plate X['TB_Ratio'], HRC X['TB_Thick_mm'], indicators, and OOD."""
    plate_frame = app.prepare_frame("Plate", pd.DataFrame([{"Plate_T_mm": 16.0, "TB_Ratio": 7.5}]))
    assert float(plate_frame["TB_Thickness_mm"].iloc[0]) == pytest.approx(120.0)
    X_plate = compute_metallurgical_features(plate_frame)
    assert float(X_plate["TB_Ratio"].iloc[0]) == pytest.approx(7.5)

    hrc_frame = app.prepare_frame("HRC", pd.DataFrame([{"Coil_T_mm": 6.0, "TB_Ratio": 15.0}]))
    assert float(hrc_frame["TB_Thick_mm"].iloc[0]) == pytest.approx(90.0)
    X_hrc, _ = engineer_hrc_features(hrc_frame, is_train=False)
    assert float(X_hrc["TB_Thick_mm"].iloc[0]) == pytest.approx(90.0)

    # Physical indicators derive TB thickness from TB_Ratio when TB_Thick_mm is absent
    import math
    hrc_raw = dict(app.defaults("HRC"), TB_Ratio=10.0, Coil_T_mm=6.0)
    hrc_raw.pop("TB_Thick_mm", None)
    ind_hrc = ml.compute_physical_indicators(hrc_raw, "HRC")
    assert ind_hrc["true_strain"] == pytest.approx(math.log(60.0 / 6.0), rel=1e-6)

    # Plate softening deltas use TB_Ratio directly when present
    dts, dys, del_ = ml.plate_softening_deltas_frame(
        pd.DataFrame([{"Plate_T_mm": 16.0, "TB_Ratio": 2.56, "FM_Finish_Temp": 832.0,
                       "ACC_Temp_Post_Laminar": 744.0, "ACC_Laminar_Header": 8.0}]))
    assert float(dts.iloc[0]) > 12.0 and float(dys.iloc[0]) > 15.0 and float(del_.iloc[0]) > 1.5

    # OOD alerts fire from TB_Ratio * gauge when TB_Thickness_mm / TB_Thick_mm is omitted
    plate_ood = dict(app.defaults("Plate"), TB_Ratio=7.0, Plate_T_mm=16.0)
    plate_ood.pop("TB_Thickness_mm", None)
    p_warn, _ = explainer.check_applicability_domain("Plate", plate_ood)
    assert any(w.get("param") == "TB_Thickness_mm" for w in p_warn)

    hrc_ood = dict(app.defaults("HRC"), TB_Ratio=12.0, Coil_T_mm=6.0)
    hrc_ood.pop("TB_Thick_mm", None)
    h_warn, _ = explainer.check_applicability_domain("HRC", hrc_ood)
    assert any(w.get("param") == "TB_Thick_mm" for w in h_warn)

    # Default HRC TB_Ratio stays in distribution (4.13 * 6.0 = 24.78 inside 24.0..26.0)
    h_def_warn, _ = explainer.check_applicability_domain("HRC", app.defaults("HRC"))
    assert all(w.get("param") != "TB_Thick_mm" for w in h_def_warn)
