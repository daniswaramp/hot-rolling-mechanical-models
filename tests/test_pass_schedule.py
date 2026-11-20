"""
Pass schedule tab tests (app/pass_schedule.py), following tests/test_contracts.py and
tests/test_atlas_shadow.py conventions: repo root and app/ on sys.path, pure-model
assertions on the calibrated Sellars chain, bare-mode renders of view_pass_schedule for
both mills, and the tab registration inside streamlit_app.main().
"""

import ast
import inspect
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pass_schedule as ps  # noqa: E402
import spec_checker as sc  # noqa: E402
import streamlit_app as app  # noqa: E402

MILLS = ["Plate", "HRC"]


# ------------------------------------------------------------------ 1. calibration sanity
def test_t50_calibration_point_in_sanity_window():
    """t_50 at its anchor (T=1000 C, eps=0.15 roughing-representative, D0=100 um,
    no retardation) lands in [0.5, 20] s. The roughing anchor matters: with the
    old finishing-strain anchor (0.30) the eps^-4 law called every light-draft
    roughing pass NRX, which contradicts plant practice."""
    chem = {"Nb": 0.0, "Ti": 0.0, "C": 0.18, "N": 0.004}
    t50 = ps.t50_seconds(1000.0 + 273.15, 0.15, 100.0, chem["Nb"], chem["Ti"], False)
    assert 0.5 <= t50 <= 20.0, f"t_50 anchor out of window: {t50:.3f} s"


def test_light_roughing_draft_is_prx_not_nrx():
    """A light roughing draft (eps ~ 0.09) at 985 C with 15 s interpass must land
    in PRX territory (partial softening), not the old model's near-zero X_srx."""
    chem = {"Nb": 0.002, "Ti": 0.002, "C": 0.18, "N": 0.004}
    res = ps.compute_schedule(41.0, [3.571], 985.0, 20.0, 15.0, 0.5, 460.0, 100.0, chem)
    first = res["passes"][0]
    assert 0.05 <= first["x_srx"] < 0.95, f"x_srx out of PRX band: {first['x_srx']:.3f}"
    assert first["regime"] == "PRX"


def test_peak_strain_calibration_point_in_sanity_window():
    """eps_p at its anchor (D0=100 um, epsdot=1/s, T=1000 C) lands in [0.10, 0.35]."""
    z = 1.0 * np.exp(300e3 / (8.314 * (1000.0 + 273.15)))
    eps_p = ps.peak_strain(100.0, z)
    assert 0.10 <= eps_p <= 0.35, f"eps_p anchor out of window: {eps_p:.4f}"


def test_recrystallized_grain_size_calibration_point_in_sanity_window():
    """d_rx at its anchor (eps=0.30, D0=100 um) lands in [20, 80] um."""
    d_rx = ps.recrystallized_grain_size(0.30, 100.0)
    assert 20.0 <= d_rx <= 80.0, f"d_rx anchor out of window: {d_rx:.2f} um"


# ------------------------------------------------------------------ 2. regime classification
def _run(tb, drafts, t1, dt, t_int, nb, ti, d0=100.0, dur=0.5, roll=460.0):
    chem = {"Nb": nb, "Ti": ti, "C": 0.18, "N": 0.004}
    return ps.compute_schedule(tb, drafts, t1, dt, t_int, dur, roll, d0, chem)


def test_hot_slow_low_nb_classifies_srx():
    """Hot, slow, low-Nb interpass: t_50 well under the interpass time, so X_srx >= 0.95.

    The pass strains to ~0.155 — just below eps_c at 1150 C, so nothing dynamic happens —
    and the hot interpass then fully recrystallizes within the 15 s.
    """
    res = _run(100.0, [14.36, 14.36], 1150.0, 20.0, 15.0, nb=0.002, ti=0.002)
    first = res["passes"][0]
    assert first["eps"] == pytest.approx(0.155, abs=0.005)
    assert first["eps"] < first["eps_c"], "scenario must sit below the critical strain"
    assert first["x_drx"] == 0.0
    assert first["x_srx"] >= 0.95
    assert first["regime"] == "SRX"


def test_cold_high_nb_is_nrx_or_prx():
    """Cold interpass with high Nb: retardation (and pinning) must keep X_srx below 0.95."""
    res = _run(100.0, [10.0, 10.0, 10.0], 900.0, 20.0, 15.0, nb=0.030, ti=0.002)
    for row in res["passes"]:
        assert row["x_srx"] < 0.95, f"pass {row['pass']} softened too far: {row['x_srx']}"
        assert row["regime"] in ("NRX", "PRX")
    assert res["passes"][0]["regime"] == "NRX"
    # the pinning flag fires: 900 C sits below the Nb(C,N) solution temperature
    assert res["passes"][0]["pinned"] is True


def test_strain_above_critical_flags_drx():
    """A pass strained well past eps_c must classify DRX regardless of the interpass."""
    res = _run(100.0, [30.0, 30.0], 1000.0, 20.0, 1.0, nb=0.0, ti=0.0, dur=0.5)
    first = res["passes"][0]
    assert first["eps"] > first["eps_c"]
    assert first["x_drx"] >= 0.05
    assert first["regime"] == "DRX"


# ------------------------------------------------------------------ 3. retardation ratio
def test_nb_retardation_ratio_at_least_5x():
    """0.030 wt% Nb must retard t_50 at least 5x versus 0.002 wt% at the same state."""
    common = dict(t_k=1000.0 + 273.15, eps=0.30, d0_um=100.0, below_solution=False)
    t50_rich = ps.t50_seconds(nb_wt=0.030, ti_wt=0.0, **common)
    t50_lean = ps.t50_seconds(nb_wt=0.002, ti_wt=0.0, **common)
    assert t50_rich / t50_lean >= 5.0, f"retardation ratio only {t50_rich / t50_lean:.2f}x"


# ------------------------------------------------------------------ 4. penetration
def test_thick_slab_small_draft_small_roll_penetrates_partially():
    res = _run(100.0, [2.0], 1000.0, 20.0, 15.0, nb=0.0, ti=0.0, roll=200.0)
    assert res["passes"][0]["P"] < 0.5
    assert res["summary"]["first_unpenetrated"] == 1


def test_thin_plate_big_draft_fully_penetrates():
    res = _run(10.0, [3.0], 1000.0, 20.0, 15.0, nb=0.0, ti=0.0, roll=460.0)
    assert res["passes"][0]["P"] == pytest.approx(1.0)
    assert res["summary"]["first_unpenetrated"] is None


# ------------------------------------------------------------------ 5. strain accumulation
def test_strain_sum_equals_total_log_ratio():
    drafts = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]
    tb, gauge = 41.0, 16.5
    assert abs(tb - sum(drafts) - gauge) < 1e-9
    res = _run(tb, drafts, 1000.0, 20.0, 15.0, nb=0.002, ti=0.002)
    total = sum(np.log(r["h_in"] / r["h_out"]) for r in res["passes"])
    assert total == pytest.approx(np.log(tb / gauge), abs=1e-9)
    # and the gauge chain is exact
    assert res["passes"][-1]["h_out"] == pytest.approx(gauge, abs=1e-9)


def test_custom_mode_last_pass_absorbs_remainder():
    drafts, err = ps.draft_sequence(41.0, 5, "Custom per-pass (mm)",
                                    0.0, [4.0, 4.0, 4.0, 4.0], 16.0)
    assert err is None
    assert drafts == [4.0, 4.0, 4.0, 4.0, 41.0 - 16.0 - 16.0]
    # validation guards
    _, err2 = ps.draft_sequence(41.0, 5, "Custom per-pass (mm)", 0.0,
                                [10.0, 10.0, 10.0, 10.0], 16.0)
    assert err2 and "exceed" in err2
    _, err3 = ps.draft_sequence(41.0, 5, "Equal drafting (mm)", 8.0, [], 16.0)
    assert err3 and "4 mm" in err3


# ------------------------------------------------------------------ 6. bare-mode render
def _clear_state(prefixes=("ps_",)):
    import streamlit as st
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith(prefixes)]:
        del st.session_state[key]


def test_view_renders_bare_mode_empty_state():
    """Both routes, no stored result: the tab must explain what to press and return."""
    for mill in MILLS:
        _clear_state()
        ps.view_pass_schedule(mill, app.defaults(mill))


def test_view_renders_bare_mode_with_stored_result():
    """A stored analysis replays without recomputing, on both routes."""
    import streamlit as st
    for mill in MILLS:
        raw = app.defaults(mill)
        chem = ps.chem_from_raw(raw)
        tb = float(raw.get("TB_Thickness_mm", raw.get("TB_Thick_mm", 41.0)))
        drafts, err = ps.draft_sequence(tb, 5, "Equal drafting (mm)", 3.0, [], 16.0)
        assert err is None
        _clear_state()
        st.session_state[f"ps_result_{mill}"] = ps.compute_schedule(
            tb, drafts, 1000.0, 20.0, 15.0, 0.5, 460.0, 100.0, chem)
        try:
            ps.view_pass_schedule(mill, raw)
        finally:
            _clear_state()


def test_view_submit_branch_runs_and_stores(monkeypatch):
    """With the form submit monkeypatched on, the view computes and stores the result."""
    import streamlit as st
    mill = "Plate"
    real = st.form_submit_button
    monkeypatch.setattr(st, "form_submit_button",
                        lambda *a, **k: bool(real(*a, **k)) or True)
    _clear_state()
    try:
        ps.view_pass_schedule(mill, app.defaults(mill))
        stored = st.session_state.get(f"ps_result_{mill}")
        assert stored and stored["passes"], "submit branch did not store a result"
        assert stored["summary"]["n_passes"] == 7
    finally:
        _clear_state()
        monkeypatch.setattr(st, "form_submit_button", real)


# ------------------------------------------------------------------ 7. registration & scope
def test_tab_registers_after_shadow_and_about_stays_last():
    source = textwrap.dedent(inspect.getsource(app.main))
    tree = ast.parse(source)
    labels = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "tabs":
            labels = ast.literal_eval(node.args[0])
    assert labels, "main() declares no st.tabs list"
    assert "Pass schedule" in labels
    assert labels.index("Pass schedule") == labels.index("Shadow") + 1, \
        "Pass schedule must sit directly after Shadow"
    assert labels[-1] == "About", f"About must stay the last tab, got {labels[-1]!r}"
    assert "Pass schedule" in source
    assert "pass_schedule.view_pass_schedule(" in source


def test_pass_schedule_never_imports_modal():
    """The tab is pure local numpy: no modal, no remote dispatch anywhere in the module."""
    path = os.path.join(APP_DIR, "pass_schedule.py")
    with open(path, "r", encoding="utf-8") as handle:
        src = handle.read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] in ("modal", "remote_compute",
                                                        "remote_jobs")
                           for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in ("modal", "remote_compute",
                                                             "remote_jobs")
    assert "import modal" not in src and "remote_compute" not in src


# ------------------------------------------------------------------ 8. DBTT Charpy curve
def test_interpolate_charpy_dbtt_exact_at_model_anchors():
    """At 0, -20, -40 C, interpolate_charpy_dbtt returns the direct model prediction dicts."""
    preds = {
        "Impact_0C": {"pred": 140.0, "ci_low": 100.0, "ci_high": 180.0},
        "Impact_m20C": {"pred": 80.0, "ci_low": 40.0, "ci_high": 120.0},
        "Impact_m40C": {"pred": 40.0, "ci_low": 10.0, "ci_high": 70.0},
    }
    assert sc.interpolate_charpy_dbtt(preds, 0) == preds["Impact_0C"]
    assert sc.interpolate_charpy_dbtt(preds, -20) == preds["Impact_m20C"]
    assert sc.interpolate_charpy_dbtt(preds, -40) == preds["Impact_m40C"]
    # Missing or non-finite anchor at 0/-20/-40 returns None rather than synthesizing
    partial = {"Impact_0C": preds["Impact_0C"], "Impact_m20C": preds["Impact_m20C"]}
    assert sc.interpolate_charpy_dbtt(partial, -40) is None
    nan_anchor = dict(preds, Impact_m20C={"pred": float("nan")})
    assert sc.interpolate_charpy_dbtt(nan_anchor, -20) is None


def test_interpolate_charpy_dbtt_monotone_and_bounded():
    """Smooth monotone DBTT transition across [-60, +20], bounded in [5 J, 1.15 * E(0)],
    with 15% widened uncertainty outside [-40, 0]."""
    preds = {
        "Impact_0C": {"pred": 140.0, "ci_low": 110.0, "ci_high": 170.0},
        "Impact_m20C": {"pred": 80.0, "ci_low": 50.0, "ci_high": 110.0},
        "Impact_m40C": {"pred": 40.0, "ci_low": 10.0, "ci_high": 70.0},
    }
    temps = [-60, -50, -40, -30, -29, -20, -15, -10, -5, 0, 10, 20]
    energies = [sc.interpolate_charpy_dbtt(preds, t)["pred"] for t in temps]
    for e_lo, e_hi in zip(energies, energies[1:]):
        assert e_lo <= e_hi, f"DBTT curve is not monotone: {list(zip(temps, energies))}"
    for e in energies:
        assert 5.0 <= e <= 1.15 * 140.0 + 1e-9

    r_m10 = sc.interpolate_charpy_dbtt(preds, -10)
    r_m30 = sc.interpolate_charpy_dbtt(preds, -30)
    r_m50 = sc.interpolate_charpy_dbtt(preds, -50)
    r_p20 = sc.interpolate_charpy_dbtt(preds, 20)
    assert r_m10["key"] == "DBTT_interp" and r_m10["note"] == "DBTT curve interpolation"
    assert r_m30["key"] == "DBTT_interp" and r_m30["note"] == "DBTT curve interpolation"
    assert r_m50["key"] == "DBTT_interp" and r_m50["note"] == "DBTT curve extrapolation"
    assert r_p20["key"] == "DBTT_interp" and r_p20["note"] == "DBTT curve extrapolation"
    # Half-width inside [-40, 0] is 30.0; outside [-40, 0] is widened by 15% -> 34.5
    assert (r_m10["ci_high"] - r_m10["pred"]) == pytest.approx(30.0)
    assert (r_m10["pred"] - r_m10["ci_low"]) == pytest.approx(30.0)
    assert (r_m50["ci_high"] - r_m50["pred"]) == pytest.approx(34.5)
    assert (r_p20["ci_high"] - r_p20["pred"]) == pytest.approx(34.5)


def test_interpolate_charpy_dbtt_enforces_monotonicity_and_two_anchors():
    """Inverted GBDT anchor predictions are isotonized before curve fitting, and 2 anchors suffice."""
    inverted = {
        "Impact_0C": {"pred": 90.0, "ci_low": 60.0, "ci_high": 120.0},
        "Impact_m20C": {"pred": 100.0, "ci_low": 70.0, "ci_high": 130.0},
        "Impact_m40C": {"pred": 120.0, "ci_low": 90.0, "ci_high": 150.0},
    }
    vals = [sc.interpolate_charpy_dbtt(inverted, t)["pred"] for t in (-50, -30, -10, 20)]
    assert all(a <= b for a, b in zip(vals, vals[1:]))
    assert all(5.0 <= v <= 1.15 * max(90.0, vals[-1]) for v in vals)

    two_anchors = {
        "Impact_0C": {"pred": 125.0, "ci_low": 95.0, "ci_high": 155.0},
        "Impact_m40C": {"pred": 35.0, "ci_low": 15.0, "ci_high": 55.0},
    }
    r_m30 = sc.interpolate_charpy_dbtt(two_anchors, -30)
    r_m10 = sc.interpolate_charpy_dbtt(two_anchors, -10)
    assert r_m30 is not None and r_m10 is not None
    assert 35.0 <= r_m30["pred"] <= r_m10["pred"] <= 125.0

    one_anchor = {"Impact_0C": {"pred": 125.0, "ci_low": 95.0, "ci_high": 155.0}}
    assert sc.interpolate_charpy_dbtt(one_anchor, -10) is None


def test_interpolate_charpy_dbtt_empty_predictions_preserves_optimizer_invariant():
    """CRITICAL INVARIANT: empty predictions={} returns None everywhere, and evaluate_mechanical
    keeps key=None and verdict='NOT_VERIFIABLE' at non-{0, -20, -40} temperatures."""
    for t in (20, 10, 0, -5, -10, -15, -20, -29, -30, -40, -50, -60):
        assert sc.interpolate_charpy_dbtt({}, t) is None
        assert sc.interpolate_charpy_dbtt(None, t) is None

    row = pd.Series({
        "SUPPLY CONDITION": "AR",
        "Impact +20 (TR) [J]": "27",
        "Impact +10 (LG) [J]": "27",
        "Impact 0 (TR) [J]": "27",
        "Impact -5 (LG) [J]": "27",
        "Impact -10 (TR) [J]": "27",
        "Impact -15 (LG) [J]": "27",
        "Impact -20 (TR) [J]": "27",
        "Impact -29 (LG) [J]": "27",
        "Impact -30 (TR) [J]": "27",
        "Impact -40 (TR) [J]": "27",
        "Impact -50 (LG) [J]": "27",
    })
    probe = sc.evaluate_mechanical(row, {}, "Plate", "AR")
    by_prop = {c["property"]: c for c in probe["checks"]}
    for label in ("Charpy +20 °C", "Charpy +10 °C", "Charpy -5 °C", "Charpy -10 °C",
                  "Charpy -15 °C", "Charpy -29 °C", "Charpy -30 °C", "Charpy -50 °C"):
        assert by_prop[label]["key"] is None
        assert by_prop[label]["verdict"] == "NOT_VERIFIABLE"
    # Direct model anchors still expose their target keys when probing with {}
    assert by_prop["Charpy 0 °C"]["key"] == "Impact_0C"
    assert by_prop["Charpy -20 °C"]["key"] == "Impact_m20C"
    assert by_prop["Charpy -40 °C"]["key"] == "Impact_m40C"

    # With finite anchor predictions, all 11 temperatures evaluate via direct or DBTT_interp
    preds = {
        "Impact_0C": {"pred": 140.0, "ci_low": 110.0, "ci_high": 170.0},
        "Impact_m20C": {"pred": 90.0, "ci_low": 60.0, "ci_high": 120.0},
        "Impact_m40C": {"pred": 45.0, "ci_low": 15.0, "ci_high": 75.0},
    }
    evaluated = sc.evaluate_mechanical(row, preds, "Plate", "AR")
    assert evaluated["unverifiable_count"] == 0
    assert len(evaluated["checks"]) == 11
    interp_checks = [c for c in evaluated["checks"] if c["key"] == "DBTT_interp"]
    assert len(interp_checks) == 8
    assert all(c["verdict"] in ("COMPLIANT", "MARGINAL_RISK", "NON_COMPLIANT") for c in interp_checks)


# ------------------------------------------------------------------ 9. ferrite microstructure
def test_estimate_ferrite_microstructure_refinement_and_hall_petch():
    """Higher retained strain and higher cooling rate both refine d_alpha and raise ys_micro."""
    base = ps.estimate_ferrite_microstructure(50.0, 0.0, cooling_rate_k_s=5.0, sigma_ss_mpa=80.0)
    more_strain = ps.estimate_ferrite_microstructure(50.0, 0.8, cooling_rate_k_s=5.0, sigma_ss_mpa=80.0)
    faster_cool = ps.estimate_ferrite_microstructure(50.0, 0.0, cooling_rate_k_s=20.0, sigma_ss_mpa=80.0)
    finer_gamma = ps.estimate_ferrite_microstructure(20.0, 0.0, cooling_rate_k_s=5.0, sigma_ss_mpa=80.0)

    assert more_strain["d_alpha_um"] < base["d_alpha_um"]
    assert more_strain["ys_micro_mpa"] > base["ys_micro_mpa"]
    assert faster_cool["d_alpha_um"] < base["d_alpha_um"]
    assert faster_cool["ys_micro_mpa"] > base["ys_micro_mpa"]
    assert finer_gamma["d_alpha_um"] < base["d_alpha_um"]
    assert finer_gamma["ys_micro_mpa"] > base["ys_micro_mpa"]

    # Exact closed-form verification and clip guards
    expected_d = float(np.clip((1.8 + 3.2 * (5.0 ** -0.35)) * (50.0 ** 0.33) * np.exp(0.0), 2.5, 45.0))
    expected_ys = 96.5 + 80.0 + 17.4 * ((expected_d / 1000.0) ** -0.5)
    assert base["d_alpha_um"] == pytest.approx(expected_d)
    assert base["ys_micro_mpa"] == pytest.approx(expected_ys)

    hi_clip = ps.estimate_ferrite_microstructure(5000.0, -2.0, cooling_rate_k_s=0.01)
    lo_clip = ps.estimate_ferrite_microstructure(1.0, 10.0, cooling_rate_k_s=200.0)
    assert hi_clip["d_alpha_um"] == pytest.approx(45.0)
    assert lo_clip["d_alpha_um"] == pytest.approx(2.5)


def test_compute_schedule_populates_ferrite_microstructure_and_view_renders(monkeypatch):
    """compute_schedule returns d_alpha_um and ys_micro_mpa consistent with retained strain."""
    import streamlit as st

    srx_res = _run(100.0, [14.36, 14.36], 1150.0, 20.0, 15.0, nb=0.002, ti=0.002)
    nrx_res = _run(100.0, [10.0, 10.0, 10.0], 900.0, 20.0, 15.0, nb=0.030, ti=0.002)

    for res in (srx_res, nrx_res):
        assert "d_alpha_um" in res and "ys_micro_mpa" in res
        assert res["d_alpha_um"] == pytest.approx(res["summary"]["d_alpha_um"])
        assert res["ys_micro_mpa"] == pytest.approx(res["summary"]["ys_micro_mpa"])
        assert 2.5 <= res["d_alpha_um"] <= 45.0
        assert res["ys_micro_mpa"] > 150.0

    # Pancaked NRX schedule retains strain while SRX resets it to zero
    assert srx_res["summary"]["retained_strain"] == pytest.approx(0.0)
    assert nrx_res["summary"]["retained_strain"] > 0.2

    # view_pass_schedule stores and renders d_alpha_um / ys_micro_mpa in bare mode
    mill = "Plate"
    real = st.form_submit_button
    monkeypatch.setattr(st, "form_submit_button", lambda *a, **k: bool(real(*a, **k)) or True)
    _clear_state()
    try:
        ps.view_pass_schedule(mill, app.defaults(mill))
        stored = st.session_state.get(f"ps_result_{mill}")
        assert stored and "d_alpha_um" in stored["summary"] and "ys_micro_mpa" in stored["summary"]
    finally:
        _clear_state()
        monkeypatch.setattr(st, "form_submit_button", real)
