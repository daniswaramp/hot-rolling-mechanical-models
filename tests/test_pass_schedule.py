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
import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")

for p in [REPO_ROOT, APP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pass_schedule as ps  # noqa: E402
import streamlit_app as app  # noqa: E402

MILLS = ["Plate", "HRC"]


# ------------------------------------------------------------------ 1. calibration sanity
def test_t50_calibration_point_in_sanity_window():
    """t_50 at its anchor (T=1000 C, eps=0.30, D0=100 um, no retardation) lands in [0.5, 20] s."""
    chem = {"Nb": 0.0, "Ti": 0.0, "C": 0.18, "N": 0.004}
    t50 = ps.t50_seconds(1000.0 + 273.15, 0.30, 100.0, chem["Nb"], chem["Ti"], False)
    assert 0.5 <= t50 <= 20.0, f"t_50 anchor out of window: {t50:.3f} s"


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
