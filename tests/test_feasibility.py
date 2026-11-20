"""Ranged Process Feasibility (Monte Carlo) engine and bare-mode UI tests."""

import ast
import os
import sys

import numpy as np
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import feasibility as feas  # noqa: E402
import physics  # noqa: E402
import spec_checker as sc  # noqa: E402
import streamlit_app as app  # noqa: E402


def _clear_feas_state():
    import streamlit as st
    from streamlit.delta_generator_singletons import context_dg_stack

    for key in [k for k in list(st.session_state.keys()) if str(k).startswith("feas_")]:
        del st.session_state[key]
    # Bare-mode st.form() mutates st._main._form_data in place without resetting on exit.
    if hasattr(st, "_main") and hasattr(st._main, "_form_data"):
        st._main._form_data = None
        context_dg_stack.set((st._main,))


def _default_ranges(mill, wide=False):
    raw = app.defaults(mill)
    if mill == "Plate":
        if not wide:
            proc = {
                "FM_Temp_Before_Rolling": (975.0, 995.0),
                "FM_Finish_Temp": (825.0, 840.0),
                "ACC_Laminar_Header": (6, 8),
            }
            geom = {
                "Plate_T_mm": (14.0, 18.0),
                "Plate_W_mm": (1800.0, 2000.0),
                "TB_Ratio": (2.3, 2.8),
            }
        else:
            proc = {
                "FM_Temp_Before_Rolling": (900.0, 1080.0),
                "FM_Finish_Temp": (760.0, 930.0),
                "ACC_Laminar_Header": (0, 8),
            }
            geom = {
                "Plate_T_mm": (10.0, 60.0),
                "Plate_W_mm": (1200.0, 3200.0),
                "TB_Ratio": (1.5, 4.5),
            }
    else:
        if not wide:
            proc = {
                "Temp_Transfer_Bar_C": (925.0, 950.0),
                "Temp_FT_Avg_C": (825.0, 845.0),
                "Temp_CT_Top_C": (575.0, 605.0),
            }
            geom = {
                "Coil_T_mm": (5.5, 6.5),
                "Coil_W_mm": (1100.0, 1300.0),
                "TB_Ratio": (3.8, 4.5),
            }
        else:
            proc = {
                "Temp_Transfer_Bar_C": (870.0, 1040.0),
                "Temp_FT_Avg_C": (770.0, 910.0),
                "Temp_CT_Top_C": (500.0, 690.0),
            }
            geom = {
                "Coil_T_mm": (2.5, 12.0),
                "Coil_W_mm": (800.0, 1500.0),
                "TB_Ratio": (2.0, 7.0),
            }
    return raw, proc, geom


# ------------------------------------------------------------------ 1. Import boundary
def test_feasibility_respects_import_boundaries():
    path = os.path.join(APP_DIR, "feasibility.py")
    with open(path, "r", encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    forbidden = ("modal", "remote_compute", "remote_jobs", "streamlit_app")
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in forbidden, f"forbidden from-import: {mod}"


# ------------------------------------------------------------------ 2. TB_Ratio & physics coupling
def test_plate_tb_ratio_headers_and_post_laminar_coupling():
    raw, proc, geom = _default_ranges("Plate", wide=True)
    targets = {"YS_BN_min": 280.0, "TS_BN_min": 420.0, "TS_BN_max": None, "EL_BN_min": 20.0}
    res = feas.run_ranged_feasibility("Plate", raw, proc, geom, targets, samples=180, couple_post=True, seed=7)
    assert res["error"] is None
    assert res["mill"] == "Plate"
    assert res["n_samples"] == 180

    df = res["samples_df"]
    assert len(df) == 180
    assert "TB_Thickness_mm" in df.columns
    assert np.allclose(
        df["TB_Thickness_mm"].to_numpy(dtype=float),
        df["TB_Ratio"].to_numpy(dtype=float) * df["Plate_T_mm"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )

    # Integer uniform headers spanning [0, 8]
    hdrs = df["ACC_Laminar_Header"].to_numpy()
    assert np.all(np.equal(hdrs, np.round(hdrs)))
    assert hdrs.min() == 0 and hdrs.max() == 8
    assert len(np.unique(hdrs)) >= 7

    # Physical post-laminar coupling
    expected_post = physics.post_laminar_temp(
        df["FM_Finish_Temp"].to_numpy(dtype=float),
        hdrs,
        df["Plate_T_mm"].to_numpy(dtype=float),
    )
    assert np.allclose(df["ACC_Temp_Post_Laminar"].to_numpy(dtype=float), expected_post, rtol=1e-9, atol=1e-9)


def test_hrc_tb_ratio_and_symmetric_coiling_coupling():
    raw, proc, geom = _default_ranges("HRC", wide=True)
    targets = {"YS_BN_min": 310.0, "TS_BN_min": 440.0, "TS_BN_max": None, "EL_BN_min": 20.0}
    res = feas.run_ranged_feasibility("HRC", raw, proc, geom, targets, samples=180, seed=11)
    assert res["error"] is None
    assert res["mill"] == "HRC"

    df = res["samples_df"]
    assert len(df) == 180
    assert "TB_Thick_mm" in df.columns
    assert np.allclose(
        df["TB_Thick_mm"].to_numpy(dtype=float),
        df["TB_Ratio"].to_numpy(dtype=float) * df["Coil_T_mm"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.allclose(
        df["Temp_CT_Bot_C"].to_numpy(dtype=float),
        df["Temp_CT_Top_C"].to_numpy(dtype=float),
        rtol=1e-9,
        atol=1e-9,
    )

    # Explicit Temp_CT_Bot_C overrides symmetric coupling
    proc_asym = dict(proc, Temp_CT_Bot_C=(450.0, 520.0))
    res_asym = feas.run_ranged_feasibility("HRC", raw, proc_asym, geom, targets, samples=80, seed=11)
    df_asym = res_asym["samples_df"]
    assert not np.allclose(df_asym["Temp_CT_Bot_C"], df_asym["Temp_CT_Top_C"])


# ------------------------------------------------------------------ 3. Tight vs wide windows & failure modes
@pytest.mark.parametrize("mill", ["Plate", "HRC"])
def test_tight_vs_wide_windows_and_failure_accounting(mill):
    raw_t, proc_t, geom_t = _default_ranges(mill, wide=False)
    raw_w, proc_w, geom_w = _default_ranges(mill, wide=True)
    if mill == "Plate":
        # Tight high-cooling thin-plate window passes comfortably; wide thick/low-cooling fails often
        targets = {"YS_BN_min": 290.0, "TS_BN_min": 445.0, "TS_BN_max": 520.0, "EL_BN_min": 24.0}
    else:
        targets = {"YS_BN_min": 340.0, "TS_BN_min": 470.0, "TS_BN_max": 550.0, "EL_BN_min": 22.0}

    res_tight = feas.run_ranged_feasibility(mill, raw_t, proc_t, geom_t, targets, samples=250, seed=42)
    res_wide = feas.run_ranged_feasibility(mill, raw_w, proc_w, geom_w, targets, samples=250, seed=42)
    assert res_tight["error"] is None and res_wide["error"] is None

    s_t, s_w = res_tight["summary"], res_wide["summary"]
    assert s_t["joint_pass_rate_pct"] > s_w["joint_pass_rate_pct"], (
        f"{mill} tight pass {s_t['joint_pass_rate_pct']:.1f}% should beat wide {s_w['joint_pass_rate_pct']:.1f}%"
    )
    assert sum(s_w["failure_counts"].values()) == 250
    assert s_w["failure_counts"]["PASS"] == int(res_wide["samples_df"]["joint_pass"].sum())

    # Every row satisfies worst_margin_ratio >= 0 iff joint_pass is True
    df_w = res_wide["samples_df"]
    assert np.array_equal(df_w["worst_margin_ratio"].to_numpy() >= 0.0, df_w["joint_pass"].to_numpy(dtype=bool))

    # Dominant failure matches max failure mode count
    fail_modes = ("YS fail", "TS fail", "EL fail", "Multi fail")
    max_fail = max(s_w["failure_counts"][m] for m in fail_modes)
    if max_fail == 0:
        assert s_w["dominant_failure"] == "None"
    else:
        assert s_w["failure_counts"][s_w["dominant_failure"]] == max_fail

    # TS_BN_max = 0 means unconstrained upper bound
    res_unbounded = feas.run_ranged_feasibility(
        mill, raw_w, proc_w, geom_w,
        {"YS_BN_min": targets["YS_BN_min"], "TS_BN_min": targets["TS_BN_min"], "TS_BN_max": 0.0, "EL_BN_min": targets["EL_BN_min"]},
        samples=100, seed=42,
    )
    assert res_unbounded["targets"]["TS_BN_max"] is None


# ------------------------------------------------------------------ 4. Conformal buffer stricter than point
@pytest.mark.parametrize("mill", ["Plate", "HRC"])
def test_conformal_pass_rate_strictly_stricter_than_point(mill):
    raw, proc, geom = _default_ranges(mill, wide=False)
    # Probe median predictions so targets sit right near median - 0.3 * half_95 -> point passes often, conformal fails!
    probe = feas.run_ranged_feasibility(
        mill, raw, proc, geom,
        {"YS_BN_min": 100.0, "TS_BN_min": 200.0, "TS_BN_max": None, "EL_BN_min": 5.0},
        samples=120, seed=42,
    )
    q = probe["summary"]["target_quantiles"]
    borderline_targets = {
        "YS_BN_min": round(q["YS_BN"]["p50"] - 10.0, 1),
        "TS_BN_min": round(q["TS_BN"]["p50"] - 10.0, 1),
        "TS_BN_max": None,
        "EL_BN_min": round(q["EL_BN"]["p50"] - 1.0, 1),
    }

    res_pt = feas.run_ranged_feasibility(
        mill, raw, proc, geom, borderline_targets, samples=250, include_conformal=False, seed=42
    )
    res_cf = feas.run_ranged_feasibility(
        mill, raw, proc, geom, borderline_targets, samples=250, include_conformal=True, seed=42
    )

    pt_rate = res_pt["summary"]["joint_pass_rate_pct"]
    cf_rate = res_pt["summary"]["conformal_pass_rate_pct"]
    assert cf_rate < pt_rate, f"{mill}: expected conformal {cf_rate:.1f}% < point {pt_rate:.1f}%"
    assert res_cf["summary"]["joint_pass_rate_pct"] == pytest.approx(cf_rate, abs=1e-9)
    assert res_cf["summary"]["joint_pass_rate_pct"] < pt_rate
    assert (res_pt["samples_df"]["conformal_joint_pass"] <= res_pt["samples_df"]["joint_pass"]).all()


# ------------------------------------------------------------------ 5. Sensitivity direction & edge guards
def test_sensitivity_direction_and_constant_param_guard():
    # Plate: vary headers [0, 8], finishing temp [760, 900], and gauge [10, 55] against YS_BN_min
    raw_p = app.defaults("Plate")
    proc_p = {
        "FM_Temp_Before_Rolling": (985.0, 985.0),  # constant -> must produce 0.0 corr/lift without NaN
        "FM_Finish_Temp": (760.0, 900.0),
        "ACC_Laminar_Header": (0, 8),
    }
    geom_p = {
        "Plate_T_mm": (10.0, 55.0),
        "Plate_W_mm": (1889.0, 1889.0),
        "TB_Ratio": (2.0, 3.5),
    }
    probe_p = feas.run_ranged_feasibility("Plate", raw_p, proc_p, geom_p, {"YS_BN_min": 200.0}, samples=200, seed=9)
    ys_med_p = probe_p["summary"]["target_quantiles"]["YS_BN"]["p50"]
    res_p = feas.run_ranged_feasibility(
        "Plate", raw_p, proc_p, geom_p,
        {"YS_BN_min": ys_med_p, "TS_BN_min": None, "TS_BN_max": None, "EL_BN_min": None},
        samples=300, seed=9,
    )
    by_param_p = {d["param"]: d for d in res_p["sensitivity"]}
    assert set(by_param_p) == {
        "FM_Temp_Before_Rolling", "FM_Finish_Temp", "ACC_Laminar_Header",
        "Plate_T_mm", "Plate_W_mm", "TB_Ratio",
    }
    # Constant parameter guard
    assert by_param_p["FM_Temp_Before_Rolling"]["corr_worst_margin"] == 0.0
    assert by_param_p["FM_Temp_Before_Rolling"]["pass_lift_pct"] == 0.0
    # More laminar headers raise yield strength -> positive correlation & lift
    assert by_param_p["ACC_Laminar_Header"]["corr_worst_margin"] > 0.0
    assert by_param_p["ACC_Laminar_Header"]["pass_lift_pct"] > 0.0
    # Hotter finishing or thicker plate softens yield strength -> negative correlation & lift
    assert by_param_p["FM_Finish_Temp"]["corr_worst_margin"] < 0.0
    assert by_param_p["FM_Finish_Temp"]["pass_lift_pct"] < 0.0
    assert by_param_p["Plate_T_mm"]["corr_worst_margin"] < 0.0
    assert by_param_p["Plate_T_mm"]["pass_lift_pct"] < 0.0

    # HRC: hotter coiling softens strength -> negative correlation & negative lift
    raw_h = app.defaults("HRC")
    proc_h = {
        "Temp_Transfer_Bar_C": (937.0, 937.0),
        "Temp_FT_Avg_C": (820.0, 860.0),
        "Temp_CT_Top_C": (480.0, 700.0),
    }
    geom_h = {
        "Coil_T_mm": (3.0, 12.0),
        "Coil_W_mm": (1200.0, 1200.0),
        "TB_Ratio": (2.5, 5.5),
    }
    probe_h = feas.run_ranged_feasibility("HRC", raw_h, proc_h, geom_h, {"TS_BN_min": 300.0}, samples=200, seed=9)
    ts_med_h = probe_h["summary"]["target_quantiles"]["TS_BN"]["p50"]
    res_h = feas.run_ranged_feasibility(
        "HRC", raw_h, proc_h, geom_h,
        {"YS_BN_min": None, "TS_BN_min": ts_med_h, "TS_BN_max": None, "EL_BN_min": None},
        samples=300, seed=9,
    )
    by_param_h = {d["param"]: d for d in res_h["sensitivity"]}
    assert by_param_h["Temp_CT_Top_C"]["corr_worst_margin"] < 0.0
    assert by_param_h["Temp_CT_Top_C"]["pass_lift_pct"] < 0.0
    assert by_param_h["Temp_Transfer_Bar_C"]["corr_worst_margin"] == 0.0


# ------------------------------------------------------------------ 6. Single call, quantiles, determinism
def test_single_score_frame_call_determinism_and_quantiles(monkeypatch):
    calls = []
    real_score = app.score_frame

    def spy_score(mill, frame, engine="Champion GBDT (Recommended)"):
        calls.append((mill, len(frame), engine))
        return real_score(mill, frame, engine=engine)

    monkeypatch.setattr(app, "score_frame", spy_score)

    raw, proc, geom = _default_ranges("HRC", wide=False)
    targets = {"YS_BN_min": 330.0, "TS_BN_min": 450.0, "TS_BN_max": 560.0, "EL_BN_min": 20.0}
    r1 = feas.run_ranged_feasibility("HRC", raw, proc, geom, targets, samples=150, seed=123)
    r2 = feas.run_ranged_feasibility("HRC", raw, proc, geom, targets, samples=150, seed=123)

    assert len(calls) == 2
    assert calls[0] == ("HRC", 150, "Champion GBDT (Recommended)")
    assert r1["samples_df"].equals(r2["samples_df"])
    assert r1["summary"] == r2["summary"]

    for tgt in ("YS_BN", "TS_BN", "EL_BN"):
        q = r1["summary"]["target_quantiles"][tgt]
        assert q["p05"] <= q["p50"] <= q["p95"]
        assert q["std"] >= 0.0


# ------------------------------------------------------------------ 7. Spec target extraction & bare-mode UI render
def test_extract_spec_targets_reads_workbook():
    labels, lookup = sc.spec_grade_options()
    if not labels:
        pytest.skip("specification workbook not available")
    ss400 = next((l for l in labels if "3101" in l and "400" in l), labels[0])
    sp, gr = lookup[ss400]
    tgts, info = feas.extract_spec_targets(sp, gr, 16.0, "Plate")
    assert tgts is not None
    assert info["status"] in ("ok", "out_of_gauge")
    assert tgts["YS_BN_min"] is not None or tgts["TS_BN_min"] is not None


@pytest.mark.parametrize("mill", ["Plate", "HRC"])
def test_view_ranged_feasibility_renders_bare_mode(mill, monkeypatch):
    import streamlit as st

    raw = app.defaults(mill)
    _clear_feas_state()
    real_submit = st.form_submit_button
    try:
        # 1. Idle render (before submitting)
        feas.view_ranged_feasibility(mill, raw)
        assert f"feas_result_{mill}" not in st.session_state

        # 2. Submit path: seed 300 samples for fast test & force submit
        st.session_state[f"feas_samples_{mill}"] = 300
        monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: bool(real_submit(*a, **k)) or True)
        feas.view_ranged_feasibility(mill, raw, engine="Champion GBDT (Recommended)")
        stored = st.session_state.get(f"feas_result_{mill}")
        assert stored is not None and stored["error"] is None
        assert stored["n_samples"] == 300
        assert len(stored["samples_df"]) == 300

        # 3. Replay path (no submit) renders stored result cleanly
        monkeypatch.setattr(st, "form_submit_button", real_submit)
        feas.view_ranged_feasibility(mill, raw, engine="Champion GBDT (Recommended)")
    finally:
        monkeypatch.setattr(st, "form_submit_button", real_submit)
        _clear_feas_state()


def test_view_ranged_feasibility_apptest_prefill_and_submit():
    from streamlit.testing.v1 import AppTest

    _clear_feas_state()
    try:
        script = f"""
import sys
sys.path.insert(0, {APP_DIR!r})
import streamlit_app as app
import feasibility as feas

feas.view_ranged_feasibility("HRC", app.defaults("HRC"), engine="Champion GBDT (Recommended)")
"""
        at = AppTest.from_string(script, default_timeout=30).run()
        assert not at.exception, at.exception

        # Pick a specification if workbook is present to verify pre-fill
        spec_opts = list(at.selectbox(key="feas_spec_pick_HRC").options)
        ss400 = next((o for o in spec_opts if "3101" in o and "400" in o), None)
        if ss400 is not None:
            at.selectbox(key="feas_spec_pick_HRC").select(ss400).run()
            assert not at.exception, at.exception
            assert float(at.number_input(key="feas_ys_min_HRC").value) == pytest.approx(245.0)
            assert float(at.number_input(key="feas_ts_min_HRC").value) == pytest.approx(400.0)
            assert float(at.number_input(key="feas_ts_max_HRC").value) == pytest.approx(510.0)

        at.slider(key="feas_samples_HRC").set_value(300)
        submit_btn = next(b for b in at.button if "Run Feasibility Study" in b.label)
        submit_btn.click().run()
        assert not at.exception, at.exception

        res = at.session_state["feas_result_HRC"]
        assert res is not None and res["error"] is None
        assert res["n_samples"] == 300
        assert len(res["samples_df"]) == 300

        # Switch interactive 2D scatter X/Y axes and replay without error
        at.selectbox(key="feas_x_HRC").select("Temp_CT_Top_C").run()
        assert not at.exception, at.exception
        at.selectbox(key="feas_y_HRC").select("TB_Ratio").run()
        assert not at.exception, at.exception
    finally:
        _clear_feas_state()

