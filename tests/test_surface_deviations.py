"""Tests for app/surface_deviations.py: defaults, vectorized MC perturbation scoring,
Plate & HRC physical couplings, header monotonicity, and bare-mode + AppTest UI controls.
"""

import ast
import os
import sys

import numpy as np
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

import physics  # noqa: E402
import streamlit_app as app  # noqa: E402
import surface_deviations as sd  # noqa: E402

MILLS = ["Plate", "HRC"]
HAS_BOOSTED = os.path.isdir(os.path.join(REPO_ROOT, "models", "boosted_out"))
NEEDS_MODELS = pytest.mark.skipif(not HAS_BOOSTED, reason="models/boosted_out is not on disk")


def _sample_raw(mill, with_tb_ratio=True):
    raw = dict(app.defaults(mill))
    if mill == "Plate":
        raw["Rolling_Type"] = "AR"
        raw["ACC_Temp_Post_Laminar"] = physics.post_laminar_temp(
            raw["FM_Finish_Temp"], raw["ACC_Laminar_Header"], raw["Plate_T_mm"]
        )
        if with_tb_ratio:
            raw["TB_Ratio"] = 2.56
            raw["TB_Thickness_mm"] = round(raw["TB_Ratio"] * raw["Plate_T_mm"], 2)
    else:
        if with_tb_ratio:
            raw["TB_Ratio"] = 4.13
            raw["TB_Thick_mm"] = round(raw["TB_Ratio"] * raw["Coil_T_mm"], 2)
    return raw


# --------------------------------------------------------------------------- 1. import hygiene & defaults
def test_surface_deviations_never_imports_modal_or_streamlit_app_at_module_level():
    path = os.path.join(APP_DIR, "surface_deviations.py")
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    forbidden = {"modal", "streamlit_app"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden


def test_default_deviation_tables_and_active_filter():
    assert sd.DEFAULT_CHEM_DEVIATIONS["Chem_C"] == pytest.approx(0.01)
    assert sd.DEFAULT_CHEM_DEVIATIONS["Chem_B"] == pytest.approx(0.0002)
    assert len(sd.DEFAULT_CHEM_DEVIATIONS) == 15

    assert sd.DEFAULT_PROCESS_DEVIATIONS["Plate"]["FM_Finish_Temp"] == pytest.approx(15.0)
    assert sd.DEFAULT_PROCESS_DEVIATIONS["Plate"]["ACC_Laminar_Header"] == pytest.approx(0.0)
    assert sd.DEFAULT_PROCESS_DEVIATIONS["Plate"]["TB_Ratio"] == pytest.approx(0.15)

    assert sd.DEFAULT_PROCESS_DEVIATIONS["HRC"]["Temp_FT_Avg_C"] == pytest.approx(15.0)
    assert sd.DEFAULT_PROCESS_DEVIATIONS["HRC"]["Coil_T_mm"] == pytest.approx(0.2)
    assert sd.DEFAULT_PROCESS_DEVIATIONS["HRC"]["TB_Ratio"] == pytest.approx(0.15)

    filtered = sd.active_deviations({
        "Chem_C": 0.01,
        "ACC_Laminar_Header": 0.0,
        "Neg": -1.0,
        "Bad": "not-a-number",
        "NaN": float("nan"),
        "Inf": float("inf"),
        "None": None,
        "Plate_T_mm": "0.5",
    })
    assert filtered == {"Chem_C": 0.01, "Plate_T_mm": 0.5}
    assert sd.active_deviations(None) == {}


# --------------------------------------------------------------------------- 2. perturbation physics & clipping
def test_perturbed_frame_clips_rounds_and_propagates_tb_ratio_and_post_laminar():
    # Plate: clipping, header integer rounding, TB_Ratio -> TB_Thickness_mm, post-laminar coupling
    raw_p = _sample_raw("Plate", with_tb_ratio=True)
    raw_p["Chem_Nb"] = 0.0
    raw_p["Plate_T_mm"] = 1.2
    raw_p["TB_Ratio"] = 1.1

    devs_p = {
        "Chem_Nb": 0.02,
        "Plate_T_mm": 0.5,
        "TB_Ratio": 0.3,
        "ACC_Laminar_Header": 2.0,
        "FM_Finish_Temp": 15.0,
    }
    xs = np.array([0.0, 4.0, 8.0])
    frame_p = sd._build_perturbed_frame(
        "Plate", raw_p, {"ACC_Laminar_Header": xs}, devs_p,
        n_draws=16, couple_post=True, seed=42, spec_defaults={},
    )
    assert len(frame_p) == 16 * 3
    assert (frame_p["Chem_Nb"] >= 0.0).all()
    assert (frame_p["Plate_T_mm"] >= 1.0).all()
    assert (frame_p["TB_Ratio"] >= 1.0).all()
    assert np.allclose(frame_p["TB_Thickness_mm"], frame_p["TB_Ratio"] * frame_p["Plate_T_mm"])
    hdrs = frame_p["ACC_Laminar_Header"].to_numpy()
    assert (hdrs >= 0.0).all() and np.allclose(hdrs, np.rint(hdrs))
    assert np.allclose(frame_p["ACC_Headers_n"], hdrs)
    assert np.allclose(
        frame_p["ACC_Temp_Post_Laminar"],
        physics.post_laminar_temp(frame_p["FM_Finish_Temp"], hdrs, frame_p["Plate_T_mm"]),
    )

    # HRC: TB_Ratio -> TB_Thick_mm even when raw only started with TB_Thick_mm
    raw_h = _sample_raw("HRC", with_tb_ratio=False)
    devs_h = {"TB_Ratio": 0.5, "Coil_T_mm": 0.5}
    frame_h = sd._build_perturbed_frame(
        "HRC", raw_h, {}, devs_h,
        n_draws=12, couple_post=True, seed=42, spec_defaults={},
    )
    assert (frame_h["TB_Ratio"] >= 1.0).all()
    assert (frame_h["Coil_T_mm"] >= 1.0).all()
    assert np.allclose(frame_h["TB_Thick_mm"], frame_h["TB_Ratio"] * frame_h["Coil_T_mm"])
    assert frame_h["TB_Thick_mm"].nunique() > 1


# --------------------------------------------------------------------------- 3. scoring invariants (Plate & HRC)
@NEEDS_MODELS
@pytest.mark.parametrize("mill", MILLS)
def test_zero_deviations_give_zero_spread_across_slice_grid_and_point(mill):
    raw = _sample_raw(mill, with_tb_ratio=True)
    x_key = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    y_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    xs = np.linspace(780.0, 880.0, 5)
    ys = np.linspace(10.0, 30.0, 4) if mill == "Plate" else np.linspace(3.0, 10.0, 4)

    zero_devs = {k: 0.0 for k in sd.DEFAULT_CHEM_DEVIATIONS}
    zero_devs.update({k: 0.0 for k in sd.DEFAULT_PROCESS_DEVIATIONS[mill]})

    slc = sd.score_slice_with_deviations(mill, raw, x_key, xs, ["YS_BN", "TS_BN", "EL_BN"], zero_devs)
    for tgt in ("YS_BN", "TS_BN", "EL_BN"):
        res = slc[tgt]
        assert res["nominal"].shape == (5,)
        assert np.allclose(res["p05"], res["nominal"])
        assert np.allclose(res["p50"], res["nominal"])
        assert np.allclose(res["p95"], res["nominal"])
        assert np.allclose(res["std"], 0.0)

    grd = sd.score_grid_with_deviations(mill, raw, x_key, y_key, xs, ys, "TS_BN", zero_devs)
    for k in ("nominal", "p05", "p50", "p95", "spread"):
        assert grd[k].shape == (5, 4)
    assert np.allclose(grd["p05"], grd["nominal"])
    assert np.allclose(grd["p50"], grd["nominal"])
    assert np.allclose(grd["p95"], grd["nominal"])
    assert np.allclose(grd["spread"], 0.0)

    pt = sd.evaluate_point_with_deviations(mill, raw, "TS_BN", {})
    assert pt["p05"] == pytest.approx(pt["nominal"])
    assert pt["p50"] == pytest.approx(pt["nominal"])
    assert pt["p95"] == pytest.approx(pt["nominal"])
    assert pt["spread"] == pytest.approx(0.0)


@NEEDS_MODELS
@pytest.mark.parametrize("mill", MILLS)
def test_chem_and_process_deviations_order_and_nonzero_spread(mill):
    raw = _sample_raw(mill, with_tb_ratio=True)
    x_key = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    y_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    xs = np.linspace(790.0, 870.0, 5)
    ys = np.linspace(12.0, 28.0, 4) if mill == "Plate" else np.linspace(4.0, 10.0, 4)

    devs = dict(sd.DEFAULT_CHEM_DEVIATIONS)
    devs.update(sd.DEFAULT_PROCESS_DEVIATIONS[mill])

    # Slice
    slc = sd.score_slice_with_deviations(
        mill, raw, x_key, xs, ["YS_BN", "TS_BN", "EL_BN"], devs, n_draws=24, seed=42
    )
    for tgt in ("YS_BN", "TS_BN", "EL_BN"):
        r = slc[tgt]
        assert np.all(np.isfinite(r["p05"])) and np.all(np.isfinite(r["p95"]))
        assert np.all(r["p05"] <= r["p50"] + 1e-12)
        assert np.all(r["p50"] <= r["p95"] + 1e-12)
        assert np.all(r["std"] > 0.0), f"{mill}/{tgt}: expected positive std"
        assert np.all(r["p95"] - r["p05"] > 0.0)

    # Grid (non-square shape check + quantile ordering)
    grd = sd.score_grid_with_deviations(
        mill, raw, x_key, y_key, xs, ys, "TS_BN", devs, n_draws=12, seed=42
    )
    assert grd["nominal"].shape == (5, 4)
    assert np.all(grd["p05"] <= grd["p50"] + 1e-12)
    assert np.all(grd["p50"] <= grd["p95"] + 1e-12)
    assert np.allclose(grd["spread"], grd["p95"] - grd["p05"])
    assert np.all(grd["spread"] > 0.0)

    # Point
    pt = sd.evaluate_point_with_deviations(mill, raw, "TS_BN", devs, n_draws=48, seed=42)
    assert pt["p05"] <= pt["p50"] <= pt["p95"]
    assert pt["spread"] == pytest.approx(pt["p95"] - pt["p05"])
    assert pt["spread"] > 1.0

    # Seed determinism
    pt2 = sd.evaluate_point_with_deviations(mill, raw, "TS_BN", devs, n_draws=48, seed=42)
    assert pt == pt2


@NEEDS_MODELS
@pytest.mark.parametrize("mill", MILLS)
def test_individual_perturbation_families_produce_nonzero_spread(mill):
    """Verify chemistry alone, temperature alone, thickness alone, and TB_Ratio alone all move TS_BN."""
    raw = _sample_raw(mill, with_tb_ratio=True)
    # Use a wider gauge/TB perturbation so GBDT splits register even on a single variable
    temp_key = "FM_Finish_Temp" if mill == "Plate" else "Temp_FT_Avg_C"
    thick_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
    groups = [
        {"Chem_C": 0.03, "Chem_Mn": 0.15},
        {temp_key: 30.0},
        {thick_key: 2.0},
        {"TB_Ratio": 0.6},
    ]
    for grp in groups:
        pt = sd.evaluate_point_with_deviations(mill, raw, "TS_BN", grp, n_draws=32, seed=7)
        assert pt["p05"] <= pt["p50"] <= pt["p95"]
        assert pt["spread"] > 0.0, f"{mill} group {grp} yielded zero spread"


@NEEDS_MODELS
def test_tb_ratio_and_gauge_sweeps_update_model_transfer_bar():
    for mill in MILLS:
        raw = _sample_raw(mill, with_tb_ratio=True)
        xs_ratio = np.linspace(1.5, 5.0, 5)
        res_r = sd.score_slice_with_deviations(
            mill, raw, "TB_Ratio", xs_ratio, ["TS_BN"], {"Chem_C": 0.01}, n_draws=12
        )
        assert np.ptp(res_r["TS_BN"]["nominal"]) > 0.0, f"{mill}: sweeping TB_Ratio did not move TS_BN"

        gauge_key = "Plate_T_mm" if mill == "Plate" else "Coil_T_mm"
        xs_g = np.linspace(10.0, 40.0, 5) if mill == "Plate" else np.linspace(3.0, 12.0, 5)
        res_g = sd.score_slice_with_deviations(
            mill, raw, gauge_key, xs_g, ["TS_BN"], {}, n_draws=12
        )
        assert np.ptp(res_g["TS_BN"]["nominal"]) > 0.0


@NEEDS_MODELS
def test_plate_header_sweep_applies_coupled_post_laminar_and_monotone_guardrail():
    raw = _sample_raw("Plate", with_tb_ratio=True)
    xs = np.arange(0.0, 9.0, 1.0)
    devs = {
        "Chem_C": 0.01,
        "FM_Finish_Temp": 15.0,
        "ACC_Laminar_Header": 1.0,
        "ACC_Temp_Post_Laminar": 20.0,
    }
    slc = sd.score_slice_with_deviations(
        "Plate", raw, "ACC_Laminar_Header", xs, ["TS_BN", "EL_BN"], devs, n_draws=16, couple_post=True
    )
    for bound in ("nominal", "p05", "p50", "p95"):
        assert np.all(np.diff(slc["TS_BN"][bound]) >= -1e-9), f"TS_BN {bound} not non-decreasing"
        assert np.all(np.diff(slc["EL_BN"][bound]) <= 1e-9), f"EL_BN {bound} not non-increasing"

    # Grid with ACC_Laminar_Header on y_key
    xs_frt = np.linspace(780.0, 860.0, 4)
    grd = sd.score_grid_with_deviations(
        "Plate", raw, "FM_Finish_Temp", "ACC_Laminar_Header", xs_frt, xs, "TS_BN", devs, n_draws=10
    )
    assert grd["nominal"].shape == (4, 9)
    for bound in ("nominal", "p05", "p50", "p95"):
        assert np.all(np.diff(grd[bound], axis=1) >= -1e-9)


@NEEDS_MODELS
def test_scoring_is_single_vectorized_batch(monkeypatch):
    calls = []
    orig = app.score_frame

    def spy_score(mill, frame, engine="Champion GBDT (Recommended)"):
        calls.append(len(frame))
        return orig(mill, frame, engine=engine)

    monkeypatch.setattr(app, "score_frame", spy_score)
    raw = _sample_raw("Plate", with_tb_ratio=True)
    sd.score_slice_with_deviations(
        "Plate", raw, "FM_Finish_Temp", np.linspace(780.0, 860.0, 5), ["TS_BN"], {"Chem_C": 0.01}, n_draws=12
    )
    assert calls == [5, 5 * 12]

    calls.clear()
    sd.score_grid_with_deviations(
        "HRC", _sample_raw("HRC"), "Temp_FT_Avg_C", "Coil_T_mm",
        np.linspace(800.0, 860.0, 4), np.linspace(4.0, 10.0, 3), "TS_BN", {"Chem_C": 0.01}, n_draws=8
    )
    assert calls == [12, 12 * 8]


# --------------------------------------------------------------------------- 4. UI controls (bare mode & AppTest)
@pytest.mark.parametrize("mill", MILLS)
def test_render_deviation_controls_bare_mode(mill):
    enabled, devs, mode, n_draws = sd.render_deviation_controls(mill, app.SPECS[mill])
    assert enabled is False
    assert mode == "Conservative lower bound (P05)"
    assert n_draws == 12
    chem_in_spec = [r[1] for r in app.SPECS[mill] if r[1].startswith("Chem_")]
    for ck in chem_in_spec:
        assert ck in devs
        assert devs[ck] == pytest.approx(sd.DEFAULT_CHEM_DEVIATIONS[ck])
    for pk, pv in sd.DEFAULT_PROCESS_DEVIATIONS[mill].items():
        assert pk in devs
        assert devs[pk] == pytest.approx(pv)


def test_render_deviation_controls_interactive_presets_and_toggles():
    from streamlit.testing.v1 import AppTest

    script = f"""
import sys
sys.path.insert(0, {APP_DIR!r})
import streamlit as st
import streamlit_app as app
import surface_deviations as sd

out = sd.render_deviation_controls("Plate", app.SPECS["Plate"])
st.session_state["sdev_test_result"] = out
"""
    at = AppTest.from_string(script, default_timeout=20).run()
    assert not at.exception, at.exception
    enabled, devs, mode, n_draws = at.session_state["sdev_test_result"]
    assert enabled is False
    assert devs["Chem_C"] == pytest.approx(0.01)
    assert devs["TB_Ratio"] == pytest.approx(0.15)
    assert mode == "Conservative lower bound (P05)"
    assert n_draws == 12

    # Switch preset -> Tight (±50%), enable checkbox, change display mode & draws
    at.checkbox(key="sdev_on_Plate").check().run()
    at.selectbox(key="sdev_preset_Plate").select("Tight (±50%)").run()
    at.selectbox(key="sdev_mode_Plate").select("Uncertainty spread (P95 − P05)").run()
    at.slider(key="sdev_draws_Plate").set_value(20).run()
    assert not at.exception, at.exception

    enabled2, devs2, mode2, n_draws2 = at.session_state["sdev_test_result"]
    assert enabled2 is True
    assert devs2["Chem_C"] == pytest.approx(0.005)
    assert devs2["FM_Finish_Temp"] == pytest.approx(7.5)
    assert devs2["TB_Ratio"] == pytest.approx(0.075)
    assert mode2 == "Uncertainty spread (P95 − P05)"
    assert n_draws2 == 20

    # Switch preset -> Zero (manual)
    at.selectbox(key="sdev_preset_Plate").select("Zero (manual)").run()
    assert not at.exception, at.exception
    _en3, devs3, _m3, _n3 = at.session_state["sdev_test_result"]
    assert sd.active_deviations(devs3) == {}
