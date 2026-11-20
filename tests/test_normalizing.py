"""Furnace Normalizing physics, Modal calibration, UI bare-mode & tab-registration tests."""

import ast
import inspect
import json
import os
import sys
import textwrap

import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import normalizing as norm_ui  # noqa: E402
import normalizing_physics as np_phys  # noqa: E402
import streamlit_app as app  # noqa: E402

SAMPLE_PLATE_CHEM = {
    "Plate_T_mm": 20.0,
    "Chem_C": 0.17,
    "Chem_Mn": 1.35,
    "Chem_Si": 0.28,
    "Chem_P": 0.012,
    "Chem_S": 0.004,
    "Chem_Al": 0.032,
    "Chem_Nb": 0.022,
    "Chem_V": 0.035,
    "Chem_Ti": 0.012,
    "Chem_Cu": 0.02,
    "Chem_Ni": 0.02,
    "Chem_Cr": 0.03,
    "Chem_Mo": 0.01,
    "Chem_N": 0.005,
}


def _clear_norm_state():
    import streamlit as st

    for key in [k for k in list(st.session_state.keys()) if str(k).startswith("norm_")]:
        del st.session_state[key]


# ------------------------------------------------------------------ 1. Ac1 & Ac3
def test_ac1_and_ac3_andrews_and_key_variants():
    short_chem = {k.replace("Chem_", ""): v for k, v in SAMPLE_PLATE_CHEM.items() if k.startswith("Chem_")}
    ac1_pref = np_phys.ac1_temp(SAMPLE_PLATE_CHEM)
    ac1_short = np_phys.ac1_temp(short_chem)
    assert ac1_pref == pytest.approx(ac1_short, abs=1e-9)
    expected_ac1 = 723.0 - 10.7 * 1.35 - 16.9 * 0.02 + 29.1 * 0.28 + 16.9 * 0.03
    assert ac1_pref == pytest.approx(expected_ac1, abs=1e-6)

    ac3_pref = np_phys.ac3_temp(SAMPLE_PLATE_CHEM)
    ac3_short = np_phys.ac3_temp(short_chem)
    assert ac3_pref == pytest.approx(ac3_short, abs=1e-9)
    assert 760.0 <= ac3_pref <= 960.0
    assert ac3_pref > ac1_pref
    # Clip guards
    assert np_phys.ac3_temp({"C": 0.80, "Mn": 2.0}) == 760.0
    assert np_phys.ac3_temp({"C": 0.005, "Si": 1.0, "Al": 0.15}) == 960.0


# ------------------------------------------------------------------ 2. Thermal profile
def test_thermal_profile_heating_lag_and_cooling():
    still = np_phys.thermal_profile(25.0, 910.0, 45.0, heating_rate_min_per_mm=1.2, cooling_mode="still_air")
    forced = np_phys.thermal_profile(25.0, 910.0, 45.0, heating_rate_min_per_mm=1.2, cooling_mode="forced_air")

    expected_t_heat = 1.2 * 25.0 * (1.0 + 0.004 * 5.0)
    assert still["t_heat_min"] == pytest.approx(expected_t_heat, abs=1e-6)
    assert still["t_eff_soak_min"] == pytest.approx(45.0 - expected_t_heat, abs=1e-6)
    assert forced["cr_air_k_s"] == pytest.approx(2.2 * still["cr_air_k_s"], rel=1e-6)
    assert still["t_surf_peak_c"] >= still["t_core_peak_c"]
    assert still["time_min"][0] == pytest.approx(0.0)
    assert still["time_min"][-1] == pytest.approx(45.0 + 20.0)
    assert len(still["time_min"]) == len(still["t_surface_c"]) == len(still["t_core_c"])

    # Under-soaked when residence < t_heat
    short = np_phys.thermal_profile(50.0, 910.0, 20.0, heating_rate_min_per_mm=1.2)
    assert short["t_eff_soak_min"] == 0.0
    assert short["t_core_peak_c"] < 910.0 - 20.0

    # Hollomon-Jaffe increases with soak temperature and residence time
    hotter = np_phys.thermal_profile(25.0, 950.0, 60.0)
    assert hotter["hj_param"] > still["hj_param"]


# ------------------------------------------------------------------ 3. Precipitate conservation
def test_normalizing_precipitates_mass_and_interstitial_conservation():
    ppt_state = np_phys.normalizing_precipitates(SAMPLE_PLATE_CHEM, 910.0, 15.0)
    for metal in ("Al", "Nb", "Ti", "V"):
        el = ppt_state["elements"][metal]
        assert el["bound_wt"] >= 0.0 and el["dissolved_wt"] >= 0.0
        assert el["bound_wt"] + el["dissolved_wt"] == pytest.approx(el["total_wt"], abs=1e-12)
        assert el["bound_frac"] + el["dissolved_frac"] == pytest.approx(1.0, abs=1e-9)
        assert el["post_cool_bound_wt"] + el["post_cool_dissolved_wt"] == pytest.approx(el["total_wt"], abs=1e-12)

    # Free + bound nitrogen and carbon conservation at soak and post-cool
    assert ppt_state["free_n_wt"] >= 0.0 and ppt_state["bound_n_wt"] >= 0.0
    assert ppt_state["free_n_wt"] + ppt_state["bound_n_wt"] == pytest.approx(
        ppt_state["total_n_wt"], abs=1e-12
    )
    assert ppt_state["post_cool_free_n_wt"] + ppt_state["post_cool_bound_n_wt"] == pytest.approx(
        ppt_state["total_n_wt"], abs=1e-12
    )
    assert ppt_state["free_c_wt"] >= 0.0 and ppt_state["bound_c_wt"] >= 0.0
    assert ppt_state["free_c_wt"] + ppt_state["bound_c_wt"] == pytest.approx(
        ppt_state["total_c_wt"], abs=1e-12
    )
    assert ppt_state["zener_pinning_mpa"] > 0.0
    assert 8.0 <= ppt_state["d_lim_um"] <= 250.0

    # Higher soak temperature dissolves more Nb/Al and lowers equilibrium bound fraction
    sol = np_phys.solubility_curves(SAMPLE_PLATE_CHEM, t_min=750.0, t_max=1150.0, n_pts=41)
    assert sol["temps_c"][0] == pytest.approx(750.0)
    assert sol["temps_c"][-1] == pytest.approx(1150.0)
    assert sol["elements"]["Nb"]["bound_frac"][0] >= sol["elements"]["Nb"]["bound_frac"][-1]

    kin = np_phys.kinetics_time_series(SAMPLE_PLATE_CHEM, 910.0, 40.0, n_pts=25)
    assert len(kin["time_min"]) == 25
    assert kin["bound_frac"]["V"].shape == (25,)


# ------------------------------------------------------------------ 4. Grain evolution & regimes
def test_grain_evolution_regimes_and_coarsening():
    ac1 = np_phys.ac1_temp(SAMPLE_PLATE_CHEM)
    ac3 = np_phys.ac3_temp(SAMPLE_PLATE_CHEM)
    cr = 0.75

    # Subcritical (< Ac1)
    p_sub = np_phys.normalizing_precipitates(SAMPLE_PLATE_CHEM, 680.0, 15.0)
    g_sub = np_phys.grain_evolution(20.0, 680.0, 15.0, ac1, ac3, p_sub, cr)
    assert g_sub["regime"] == "SUBCRITICAL"

    # Intercritical (Ac1 <= T < Ac3 + 15)
    t_inter = 0.5 * (ac1 + ac3)
    p_int = np_phys.normalizing_precipitates(SAMPLE_PLATE_CHEM, t_inter, 15.0)
    g_int = np_phys.grain_evolution(20.0, t_inter, 15.0, ac1, ac3, p_int, cr)
    assert g_int["regime"] == "INTERCRITICAL"

    # Under-soaked (T >= Ac3 + 15 but t_eff < 5 min)
    p_und = np_phys.normalizing_precipitates(SAMPLE_PLATE_CHEM, 910.0, 2.0)
    g_und = np_phys.grain_evolution(20.0, 910.0, 2.0, ac1, ac3, p_und, cr)
    assert g_und["regime"] == "UNDER_SOAKED"

    # Full normalizing (910 C, 15 min effective soak)
    p_full = np_phys.normalizing_precipitates(SAMPLE_PLATE_CHEM, 910.0, 15.0)
    g_full = np_phys.grain_evolution(20.0, 910.0, 15.0, ac1, ac3, p_full, cr)
    assert g_full["regime"] == "FULL_NORMALIZING"
    assert 4.0 <= g_full["d_alpha_um"] <= 45.0

    # Overheated (high temp & long soak coarsens precipitates and austenite grains)
    lean_chem = dict(SAMPLE_PLATE_CHEM, Chem_Nb=0.0, Chem_Ti=0.0, Chem_Al=0.005, Chem_V=0.0)
    p_over = np_phys.normalizing_precipitates(lean_chem, 1050.0, 180.0)
    g_over = np_phys.grain_evolution(20.0, 1050.0, 180.0, ac1, ac3, p_over, cr)
    assert g_over["regime"] == "OVERHEATED"
    assert g_over["d_gamma_final_um"] > g_full["d_gamma_final_um"]
    assert g_over["d_alpha_um"] > g_full["d_alpha_um"]
    assert g_over["r_ppt_nm_series"][-1] > g_over["r_ppt_nm_series"][0]


# ------------------------------------------------------------------ 5. Hybrid anchoring & sensitivity
def test_predict_normalizing_properties_anchoring_and_sensitivity():
    cal = np_phys.load_calibration()
    plate_t = SAMPLE_PLATE_CHEM["Plate_T_mm"]
    t_heat = 1.2 * plate_t * (1.0 + 0.004 * max(0.0, plate_t - 20.0))
    t_soak_ref = float(cal["t_soak_ref_c"])
    t_res_ref = t_heat + float(cal["soak_min_per_mm_ref"]) * plate_t

    ml_preds = {
        "YS_BN": {"pred": 410.0, "ci_low": 385.0, "ci_high": 435.0},
        "TS_BN": {"pred": 560.0, "ci_low": 535.0, "ci_high": 585.0},
        "EL_BN": {"pred": 22.0, "ci_low": 19.5, "ci_high": 24.5},
        "YS_AN": {"pred": 355.0, "ci_low": 335.0, "ci_high": 375.0},
        "TS_AN": {"pred": 510.0, "ci_low": 490.0, "ci_high": 530.0},
        "EL_AN": {"pred": 26.5, "ci_low": 24.0, "ci_high": 29.0},
        "Impact_0C": {"pred": 120.0, "ci_low": 95.0, "ci_high": 145.0},
    }

    at_ref = np_phys.predict_normalizing_properties(
        SAMPLE_PLATE_CHEM, t_soak_ref, t_res_ref, ml_preds=ml_preds, calib=cal
    )
    assert at_ref["hybrid"]["anchored_to_ml"] is True
    assert at_ref["hybrid"]["ys_hybrid_mpa"] == pytest.approx(355.0, abs=1e-6)
    assert at_ref["hybrid"]["ts_hybrid_mpa"] == pytest.approx(510.0, abs=1e-6)
    assert at_ref["hybrid"]["el_hybrid_pct"] == pytest.approx(26.5, abs=1e-6)
    assert at_ref["dbtt_shift_c"] == pytest.approx(0.0, abs=1e-9)

    # Waterfall components sum identically to physics predictions
    assert sum(at_ref["strength_components"].values()) == pytest.approx(
        at_ref["physics"]["ys_phys_mpa"], abs=1e-9
    )
    assert sum(at_ref["ts_strength_components"].values()) == pytest.approx(
        at_ref["physics"]["ts_phys_mpa"], abs=1e-9
    )

    # Overheating coarsens d_alpha, lowers YS vs nominal, and degrades DBTT (+shift)
    over = np_phys.predict_normalizing_properties(
        SAMPLE_PLATE_CHEM, 1040.0, 200.0, ml_preds=ml_preds, calib=cal
    )
    assert over["grain"]["d_alpha_um"] > at_ref["grain"]["d_alpha_um"]
    assert over["hybrid"]["ys_hybrid_mpa"] < at_ref["hybrid"]["ys_hybrid_mpa"]
    assert over["dbtt_shift_c"] > 0.0

    # Subcritical soak triggers tempering between BN and AN
    sub = np_phys.predict_normalizing_properties(
        SAMPLE_PLATE_CHEM, 680.0, 60.0, ml_preds=ml_preds, calib=cal
    )
    assert sub["regime"] == "SUBCRITICAL"
    assert sub["hybrid"]["subcritical_tempered"] is True
    assert 355.0 < sub["hybrid"]["ys_hybrid_mpa"] < 410.0


# ------------------------------------------------------------------ 6. 2D process window
def test_normalizing_process_window_shapes_and_fineness():
    win = np_phys.normalizing_process_window(SAMPLE_PLATE_CHEM, nt=11, nr=9)
    assert win["t_grid"].shape == (11,)
    assert win["r_grid"].shape == (9,)
    for key in ("ys_grid", "ts_grid", "el_grid", "d_alpha_grid", "d_gamma_grid", "pinning_ratio_grid"):
        arr = win[key]
        assert arr.shape == (9, 11)
        assert np.isfinite(arr).all()
    assert win["regime_grid"].shape == (9, 11)
    assert win["t_heat_min"] > 0.0
    assert 760.0 <= win["ac3_c"] <= 960.0


# ------------------------------------------------------------------ 7. Modal calibration JSON
def test_modal_calibration_json_is_valid_and_loaded():
    cal_path = os.path.join(REPO_ROOT, "models", "normalizing_calibration.json")
    assert os.path.exists(cal_path), "models/normalizing_calibration.json was not generated by Modal"
    with open(cal_path, "r", encoding="utf-8") as fh:
        raw_cal = json.load(fh)

    for key in ("YS_AN", "TS_AN", "EL_AN", "metrics", "parity_samples"):
        assert key in raw_cal, f"missing {key} in normalizing_calibration.json"

    loaded = np_phys.load_calibration(cal_path)
    assert "default" not in str(loaded.get("source", "")).lower()
    assert len(loaded["parity_samples"]) >= 100

    for tgt in ("YS_AN", "TS_AN", "EL_AN"):
        m = norm_ui._extract_holdout_metrics(loaded["metrics"], tgt)
        assert m["r2"] is not None and m["r2"] > 0.5, f"{tgt} holdout R2 unexpectedly low: {m}"
        assert m["rmse"] is not None and m["rmse"] > 0.0
        assert m["mae"] is not None and m["mae"] > 0.0
        assert m["hit"] is not None and m["hit"] > 50.0, f"{tgt} hit rate unexpectedly low: {m}"


# ------------------------------------------------------------------ 8. Bare-mode render
@pytest.mark.parametrize("mill", ["Plate", "HRC"])
def test_view_normalizing_renders_bare_mode(mill):
    raw = app.defaults(mill)
    _clear_norm_state()
    try:
        # Call with default engine (2-arg form) and explicit engine (3-arg form)
        norm_ui.view_normalizing(mill, raw)
        norm_ui.view_normalizing(mill, raw, "Champion GBDT (Recommended)")
    finally:
        _clear_norm_state()


# ------------------------------------------------------------------ 9. Tab registration & isolation
def test_tab_registers_after_pass_schedule_and_before_about():
    source = textwrap.dedent(inspect.getsource(app.main))
    tree = ast.parse(source)
    labels = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "tabs":
            labels = ast.literal_eval(node.args[0])
    assert labels, "main() declares no st.tabs list"
    assert "Furnace Normalizing" in labels
    assert labels.index("Atlas") + 1 == labels.index("Shadow")
    assert labels.index("Pass schedule") == labels.index("Shadow") + 1
    assert labels.index("Furnace Normalizing") == labels.index("Pass schedule") + 1
    assert labels[-1] == "About"
    assert "normalizing.view_normalizing(" in source
    assert app.BUILD == "2026-11-20 · Furnace Normalizing module · Modal-calibrated physics vs AN models"


def test_normalizing_modules_respect_import_boundaries():
    for filename, forbidden in (
        ("normalizing.py", ("modal", "remote_compute", "remote_jobs", "streamlit_app")),
        ("normalizing_physics.py", ("modal", "remote_compute", "remote_jobs", "streamlit", "streamlit_app")),
    ):
        path = os.path.join(APP_DIR, filename)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden, f"{filename} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                assert mod not in forbidden, f"{filename} imports from {mod}"
