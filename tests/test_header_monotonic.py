"""
Header-sweep monotonicity tests: post-laminar physical coupling and the isotonic guardrail.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import isotonic_regression

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import physics
import streamlit_app as sa


def test_post_laminar_temp_calibration_and_monotonicity():
    """Default schedule: frt=832, 8 headers, 16 mm -> exactly 744 C; monotone in headers."""
    assert physics.post_laminar_temp(832, 8, 16) == pytest.approx(744.0)
    temps = [physics.post_laminar_temp(832, h, 16) for h in range(0, 21)]
    assert all(b <= a for a, b in zip(temps, temps[1:])), "post-laminar T must fall as headers rise"
    assert min(temps) >= 200.0
    # deep-drop floor even for absurd header counts
    assert physics.post_laminar_temp(832, 100, 16) >= 200.0
    # thin plate cools deeper for the same header count
    assert physics.post_laminar_temp(832, 8, 8) <= physics.post_laminar_temp(832, 8, 16)


def test_guardrail_ts_non_decreasing():
    df = pd.DataFrame({"ACC_Laminar_Header": [0.0, 1.0, 2.0, 3.0],
                       "TS_BN": [372.0, 380.0, 374.0, 378.0]})
    out = sa.monotone_guardrail(df, "ACC_Laminar_Header", "Plate")
    y = out["TS_BN"].to_numpy(dtype=float)
    assert (np.diff(y) >= -1e-12).all(), "TS_BN header sweep must be non-decreasing"
    expected = isotonic_regression(df["TS_BN"].to_numpy(dtype=float), increasing=True).x
    assert np.allclose(y, expected)
    # input frame untouched
    assert (df["TS_BN"].to_numpy() == [372.0, 380.0, 374.0, 378.0]).all()


def test_guardrail_el_non_increasing():
    df = pd.DataFrame({"ACC_Laminar_Header": [0.0, 1.0, 2.0, 3.0],
                       "EL_BN": [20.0, 19.5, 21.0, 18.0]})
    out = sa.monotone_guardrail(df, "ACC_Laminar_Header", "Plate")
    y = out["EL_BN"].to_numpy(dtype=float)
    assert (np.diff(y) <= 1e-12).all(), "EL_BN header sweep must be non-increasing"
    expected = isotonic_regression(df["EL_BN"].to_numpy(dtype=float), increasing=False).x
    assert np.allclose(y, expected)


def test_guardrail_leaves_unmapped_target_and_other_axes_untouched():
    df = pd.DataFrame({"ACC_Laminar_Header": [0.0, 1.0, 2.0, 3.0],
                       "Z_Test": [90.0, 80.0, 95.0, 70.0],
                       "TS_BN": [372.0, 380.0, 374.0, 378.0]})
    # Z_Test is not in the sign map: untouched
    out = sa.monotone_guardrail(df, "ACC_Laminar_Header", "Plate")
    assert (out["Z_Test"] == df["Z_Test"]).all()
    # wrong mill: untouched
    out_hrc = sa.monotone_guardrail(df, "ACC_Laminar_Header", "HRC")
    assert (out_hrc["TS_BN"] == df["TS_BN"]).all()
    # wrong sweep axis: untouched
    out_x = sa.monotone_guardrail(df, "FM_Finish_Temp", "Plate")
    assert (out_x["TS_BN"] == df["TS_BN"]).all()


def _plate_default_schedule():
    values = {row[1]: row[6] for row in sa.PLATE_SPEC}
    values["Rolling_Type"] = "AR"
    values["ACC_Temp_Post_Laminar"] = physics.post_laminar_temp(
        values["FM_Finish_Temp"], values["ACC_Laminar_Header"], values["Plate_T_mm"])
    return values


def test_header_sweep_end_to_end_monotone():
    boosted = os.path.join(REPO_ROOT, "models", "boosted_out")
    if not os.path.isdir(boosted) or not os.listdir(boosted):
        pytest.skip("models/boosted_out artifacts missing")
    raw_tuple = tuple(sorted(_plate_default_schedule().items()))
    try:
        scores = sa._slice_scores("Plate", "ACC_Laminar_Header",
                                  tuple(np.arange(0, 11, 1.0)),
                                  "Champion GBDT (Recommended)", raw_tuple,
                                  couple_post=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model artifacts failed to load: {exc}")
    assert "TS_BN" in scores.columns
    ts = scores["TS_BN"].to_numpy(dtype=float)
    assert (np.diff(ts) >= -1e-9).all(), "coupled header sweep must be non-decreasing in TS_BN"
