"""
Hottest-feasible-first recipe selection tests (redesigned Inverse Optimizer).

Covers the pure selection logic in app/optimizer.py::select_recipe (no Streamlit needed),
the Pareto-frontier and robust CVaR modes, and one end-to-end grid-search smoke run for the
Plate route.
"""

import inspect
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

import optimizer as opt  # noqa: E402


# --------------------------------------------------------------------------- builders
def _req(key="YS_BN", direction="min", required=345.0, half=20.0, label=None, kind="mech"):
    return {"key": key, "kind": kind, "direction": direction, "required": required,
            "half": half, "label": label or key, "unit": "MPa" if kind == "mech" else ""}


def _cand(entry, frt, headers=0.0, ce=0.30, ma=0.02, chem="T1", ct=580.0, ood=False, **preds):
    """One scored Plate candidate row with the columns select_recipe reads."""
    row = {"FM_Temp_Before_Rolling": float(entry), "FM_Finish_Temp": float(frt),
           "ACC_Headers_n": float(headers), "Temp_CT_Top_C": float(ct),
           "ce_iiw": float(ce), "microalloy_total": float(ma),
           "_chemistry": chem, "ood": ood}
    row.update(preds)
    return row


# --------------------------------------------------------------------------- 1
def test_hottest_feasible_wins():
    """A colder corner with a huge (but within-cap) margin must lose to a hotter adequate one."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        # coldest corner: margin 50 = 2.5x interval, inside the 3x overpower cap
        _cand(950, 810, pred_YS_BN=395.0),
        # hotter: margin 24 = 1.2x interval, adequate at k = 1
        _cand(980, 860, pred_YS_BN=369.0),
    ])
    chosen, ladder, notes = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen is not None
    assert chosen["FM_Finish_Temp"] == pytest.approx(860.0)
    assert chosen["FM_Temp_Before_Rolling"] == pytest.approx(980.0)
    assert chosen["margin_YS_BN|min"] == pytest.approx(24.0)
    # the colder candidate is reported as the next feasible alternative
    assert len(ladder) == 1 and ladder[0]["FM_Finish_Temp"] == pytest.approx(810.0)


# --------------------------------------------------------------------------- 2
def test_overpower_deprioritization():
    """A margin beyond the 3x cap deprioritises the candidate behind one inside the band."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        # hottest, but margin 85 = 4.25x interval -> overpower (85-60)/20 = 1.25
        _cand(980, 870, pred_YS_BN=430.0),
        # slightly colder, margin 55 = 2.75x interval -> overpower 0
        _cand(980, 850, pred_YS_BN=400.0),
    ])
    chosen, ladder, notes = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen["FM_Finish_Temp"] == pytest.approx(850.0)
    assert chosen["overpower"] == pytest.approx(0.0)
    # the over-delivering candidate is deprioritised, not hidden
    assert len(ladder) == 1 and ladder[0]["FM_Finish_Temp"] == pytest.approx(870.0)
    assert ladder[0]["overpower"] == pytest.approx(1.25)


# --------------------------------------------------------------------------- 3
def test_step_down_names_failing_property():
    """Every candidate hotter than 840 fails the YS margin; selection steps down to 840."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        _cand(980, 870, pred_YS_BN=350.0),   # margin 5 < 20 -> fail
        _cand(980, 860, pred_YS_BN=350.0),   # fail
        _cand(980, 850, pred_YS_BN=350.0),   # fail
        _cand(980, 840, pred_YS_BN=380.0),   # margin 35 -> feasible
    ])
    chosen, ladder, notes = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen["FM_Finish_Temp"] == pytest.approx(840.0)
    assert ladder == []
    assert notes, "the step-down must be explained"
    assert any("YS" in note for note in notes), notes
    assert any("870" in note for note in notes), notes


# --------------------------------------------------------------------------- 4
def test_lean_chemistry_tiebreak_when_auto():
    """Two chemistries feasible at the same thermal point: lower CE_IIW wins."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        _cand(980, 860, ce=0.42, ma=0.05, chem="RICH", pred_YS_BN=400.0),
        _cand(980, 860, ce=0.30, ma=0.05, chem="LEAN", pred_YS_BN=400.0),
    ])
    chosen, ladder, notes = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen["_chemistry"] == "LEAN"
    assert chosen["ce_iiw"] == pytest.approx(0.30)
    # microalloy total is the final tiebreak when CE_IIW ties
    df2 = pd.DataFrame([
        _cand(980, 860, ce=0.30, ma=0.06, chem="A", pred_YS_BN=400.0),
        _cand(980, 860, ce=0.30, ma=0.02, chem="B", pred_YS_BN=400.0),
    ])
    chosen2, _, _ = opt.select_recipe(df2, reqs, k=1.0, mill="Plate")
    assert chosen2["_chemistry"] == "B"


# --------------------------------------------------------------------------- 5
def test_infeasible_everything_reports_honestly():
    """Nothing clears the margin rule: no recipe is fabricated."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        _cand(980, 870, pred_YS_BN=350.0),
        _cand(960, 850, pred_YS_BN=360.0),
    ])
    chosen, ladder, notes = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen is None
    assert ladder == []
    assert notes, "the infeasibility must be reported with reasons"
    assert any("YS" in note for note in notes)


def test_everything_infeasible_but_ood_preferred_over_extrapolation():
    """OOD candidates are excluded while any in-domain candidate is feasible."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    df = pd.DataFrame([
        # hotter, feasible, but outside the historical envelope
        _cand(980, 880, pred_YS_BN=380.0, ood=True),
        # colder, feasible, inside the envelope
        _cand(960, 840, pred_YS_BN=370.0, ood=False),
    ])
    chosen, _, _ = opt.select_recipe(df, reqs, k=1.0, mill="Plate")
    assert chosen["FM_Finish_Temp"] == pytest.approx(840.0)
    assert chosen["_extrapolated"] is False
    # ... unless nothing in-domain is feasible, then extrapolation is used and flagged
    df2 = pd.DataFrame([_cand(980, 880, pred_YS_BN=380.0, ood=True)])
    chosen2, _, _ = opt.select_recipe(df2, reqs, k=1.0, mill="Plate")
    assert chosen2["FM_Finish_Temp"] == pytest.approx(880.0)
    assert chosen2["_extrapolated"] is True


def test_hrc_selection_ranks_ct_before_chemistry():
    """HRC mirror: same lexicographic keys, coiling temperature ranked, no headers key."""
    reqs = [_req("YS_BN", "min", 245.0, 25.0)]
    rows = [
        {"Temp_Transfer_Bar_C": 950.0, "Temp_FT_Avg_C": 860.0, "Temp_CT_Top_C": 600.0,
         "ce_iiw": 0.30, "microalloy_total": 0.02, "_chemistry": "A", "ood": False,
         "pred_YS_BN": 300.0},
        {"Temp_Transfer_Bar_C": 950.0, "Temp_FT_Avg_C": 860.0, "Temp_CT_Top_C": 560.0,
         "ce_iiw": 0.28, "microalloy_total": 0.02, "_chemistry": "B", "ood": False,
         "pred_YS_BN": 320.0},
    ]
    chosen, _, _ = opt.select_recipe(pd.DataFrame(rows), reqs, k=1.0, mill="HRC")
    # equal overpower (both 0), equal FT and entry -> hottest CT wins over leaner CE
    assert chosen["_chemistry"] == "A"
    assert chosen["Temp_CT_Top_C"] == pytest.approx(600.0)


def test_view_optimizer_call_site_signature_unchanged():
    """streamlit_app calls view_optimizer(mill, raw, engine, spec_rows=SPECS[mill])."""
    params = list(inspect.signature(opt.view_optimizer).parameters)
    assert params[:3] == ["mill", "raw", "engine"]
    assert "spec_rows" in params


# --------------------------------------------------------------------------- 6
def test_plate_grid_search_smoke_a572_gr50_16mm():
    """End-to-end: the real grid search for Plate, ASTM A 572 grade 50 at 16 mm."""
    boosted = os.path.join(REPO_ROOT, "models", "boosted_out")
    if not os.path.isdir(boosted) or not os.listdir(boosted):
        pytest.skip("models/boosted_out artifacts missing")

    import spec_checker as sc

    spec = next((s for s in sc.get_available_specs() if "A 572" in s), None)
    if spec is not None:
        grades = sc.get_available_grades(spec)
        grade = "50" if "50" in grades else grades[0]
        reqs, info = opt.spec_requirements(spec, grade, 16.0, "Plate", "AR")
    else:
        spec, grade, reqs = "ASTM A 572 Gr 50", "-", opt.preset_requirements("Plate", "ASTM A 572 Gr 50")
    if not reqs:
        pytest.skip("specification tier carries no readable limits")

    chemistries = opt.lean_chemistries()
    if not chemistries:
        pytest.skip("models/grade_templates.json missing")

    entry_range, finish_range = (900.0, 1000.0), (780.0, 900.0)
    res = opt.run_recipe_search(
        "Plate", {"Rolling_Type": "AR"}, 16.0, reqs, chemistries,
        entry_range, finish_range, cooling="air", k=1.0)
    assert not res.get("error"), res.get("error")
    chosen = res["chosen"]
    assert chosen is not None, f"no feasible recipe; rejected notes: {res['rejected']}"
    assert res["n_evaluated"] > 0
    entry, frt = float(chosen["FM_Temp_Before_Rolling"]), float(chosen["FM_Finish_Temp"])
    assert entry == entry  # finite
    assert frt == frt
    assert entry_range[0] - 1e-9 <= entry <= entry_range[1] + 1e-9
    assert finish_range[0] - 1e-9 <= frt <= finish_range[1] + 1e-9


# --------------------------------------------------------------------------- slab band caption
def test_slab_band_caption_every_template_never_raises():
    """Regression: the old zip(*gt.band(...)) unpacked dict keys and crashed mid-form.

    gt.band returns two element-keyed dicts with mismatched key sets, so the caption
    must iterate their intersection. Every shipped template must render a string.
    """
    import grade_templates as gt

    names = gt.names()
    assert names, "no grade templates shipped"
    for name in names:
        text = opt._slab_band_text(name)
        assert isinstance(text, str)
        assert "None" not in text
        # a template with any widening band must produce at least one element
        mins, maxs = gt.band(name)
        if any(maxs.get(el, lo) > lo for el, lo in mins.items()
               if isinstance(lo, (int, float))):
            assert text, f"{name}: expected a band caption, got empty string"


def test_slab_band_caption_unknown_and_empty_records():
    assert opt._slab_band_text("no such grade") == ""
    assert opt._slab_band_text("") == ""


def test_view_optimizer_has_no_zip_over_band():
    """The crashing construct must not return."""
    import inspect

    from optimizer import _slab_band_text  # noqa: F401  (exists)
    src = inspect.getsource(opt)
    assert "zip(*gt.band(" not in src


# --------------------------------------------------------------------------- Pareto mode
def test_pareto_rows_are_non_dominated_and_all_feasible():
    """The frontier keeps only margin-feasible, mutually non-dominated candidates.

    Layout (YS_BN min 345, half 20 → k = 1 needs margin ≥ 20, i.e. ≥ 1.00 × interval):
      A  ce 0.30 · air     · margin 55 (2.75×)   feasible, on the frontier
      B  ce 0.28 · air     · margin 35 (1.75×)   feasible, leanest → on the frontier
      C  ce 0.26 · air     · margin 10 (0.50×)   infeasible, must be dropped
      D  ce 0.32 · air     · margin 45 (2.25×)   dominated by A: richer *and* thinner
      E  ce 0.34 · 4 hdrs  · margin 85 (4.25×)   feasible, buys margin with cooling
    """
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    frame = pd.DataFrame([
        _cand(950, 850, headers=0.0, ce=0.30, chem="A", pred_YS_BN=400.0),
        _cand(960, 860, headers=0.0, ce=0.28, chem="B", pred_YS_BN=380.0),
        _cand(960, 870, headers=0.0, ce=0.26, chem="C", pred_YS_BN=355.0),
        _cand(940, 840, headers=0.0, ce=0.32, chem="D", pred_YS_BN=390.0),
        _cand(970, 880, headers=4.0, ce=0.34, chem="E", pred_YS_BN=430.0),
    ])
    res = opt.pareto_rows(frame, reqs, k=1.0, mill="Plate")

    assert res["n_evaluated"] == 5
    assert res["n_feasible"] == 4, "C is the only infeasible candidate"
    frontier = res["frontier"]
    assert [row["chemistry"] for row in frontier] == ["B", "A", "E"], "lean-first order"
    assert "D" not in {row["chemistry"] for row in frontier}, "D is dominated by A"
    assert res["n_frontier"] == 3 and res["n_truncated"] is False

    # every frontier row clears the margin rule select_recipe applies
    for row in frontier:
        assert row["min_margin_interval"] >= 1.0 - 1e-9
        assert row["margins"]["YS_BN|min"] >= 20.0 - 1e-9
        assert row["ood"] is False

    # ... and no frontier row is dominated by another on (CE_IIW, headers, -comfort)
    for i, one in enumerate(frontier):
        for j, other in enumerate(frontier):
            if i == j:
                continue
            dominated = (other["ce_iiw"] <= one["ce_iiw"] + 1e-12
                         and other["headers"] <= one["headers"] + 1e-12
                         and other["min_margin_interval"] >= one["min_margin_interval"] - 1e-12)
            assert not dominated, f"{other['chemistry']} dominates {one['chemistry']}"

    # max_points truncates the head of that ordering and says so
    short = opt.pareto_rows(frame, reqs, k=1.0, mill="Plate", max_points=1)
    assert len(short["frontier"]) == 1 and short["n_truncated"] is True
    assert short["frontier"][0]["chemistry"] == "B"


def test_pareto_rows_reports_an_empty_frontier_honestly():
    """Nothing clears the margin rule: no frontier is fabricated."""
    reqs = [_req("YS_BN", "min", 345.0, 20.0)]
    frame = pd.DataFrame([
        _cand(950, 850, ce=0.30, chem="A", pred_YS_BN=350.0),
        _cand(960, 860, ce=0.28, chem="B", pred_YS_BN=355.0),
    ])
    res = opt.pareto_rows(frame, reqs, k=1.0, mill="Plate")
    assert res["frontier"] == []
    assert res["n_feasible"] == 0
    assert res["n_evaluated"] == 2


def test_nondominated_mask_matches_brute_force():
    """The vectorised sweep must agree with the O(n²) definition it implements."""
    rng = np.random.default_rng(7)

    def brute(f1, f2, f3):
        picked = np.ones(len(f1), dtype=bool)
        for i in range(len(f1)):
            for j in range(len(f1)):
                if i == j:
                    continue
                if (f1[j] <= f1[i] and f2[j] <= f2[i] and f3[j] <= f3[i]
                        and (f1[j] < f1[i] or f2[j] < f2[i] or f3[j] < f3[i])):
                    picked[i] = False
                    break
        return picked

    for _ in range(60):
        n = int(rng.integers(2, 25))
        f1 = rng.integers(0, 5, n).astype(float)          # discrete, like CE_IIW
        f2 = rng.integers(0, 4, n).astype(float)          # discrete, like a header count
        f3 = rng.normal(size=n)                           # continuous, like a margin
        assert np.array_equal(opt.nondominated_mask(f1, f2, f3), brute(f1, f2, f3))

    # an exact duplicate cannot both survive: it is the same schedule twice
    mask = opt.nondominated_mask(np.array([0.30, 0.30, 0.28]),
                                 np.array([0.0, 0.0, 0.0]),
                                 np.array([2.0, 2.0, 1.0]))
    assert mask.sum() == 1 and mask[2]


# --------------------------------------------------------------------------- robust mode
def test_cvar_worst_fraction_mean_matches_hand_computation():
    """CVaR_alpha is the mean of the ceil(alpha × N) worst draws, nothing else."""
    values = np.arange(1.0, 11.0)  # 1 .. 10
    assert opt.cvar_worst_mean(values, alpha=0.20) == pytest.approx(1.5)   # worst 2 of 10
    assert opt.cvar_worst_mean(values, alpha=0.10) == pytest.approx(1.0)   # worst 1 of 10
    assert opt.cvar_worst_mean(values, alpha=0.50) == pytest.approx(3.0)   # worst 5 of 10
    # unsorted input, dropped non-finite draws, and a sample smaller than 1/alpha
    assert opt.cvar_worst_mean([10.0, 1.0, 3.0, 2.0], alpha=0.50) == pytest.approx(1.5)
    assert opt.cvar_worst_mean([5.0, float("nan")], alpha=0.50) == pytest.approx(5.0)
    assert np.isnan(opt.cvar_worst_mean([], alpha=0.10))


def test_robust_metrics_per_candidate_math():
    """Mean / CVaR / P(fail) are computed per candidate and use fail-at-k=0, not margins."""
    margins = [1.0, 2.0, 3.0, 4.0,
               -1.0, 0.5, 2.0, 2.5]
    fails = [False, False, False, False,
             True, False, False, False]
    ids = [0, 0, 0, 0, 1, 1, 1, 1]
    metrics = opt.robust_metrics(margins, fails, ids, alpha=0.25)

    # 25 % of 4 draws = the single worst draw
    assert metrics.loc[0, "cvar_margin"] == pytest.approx(1.0)
    assert metrics.loc[0, "mean_margin"] == pytest.approx(2.5)
    assert metrics.loc[0, "p_fail"] == pytest.approx(0.0)
    assert metrics.loc[1, "cvar_margin"] == pytest.approx(-1.0)
    assert metrics.loc[1, "mean_margin"] == pytest.approx(1.0)
    assert metrics.loc[1, "p_fail"] == pytest.approx(0.25)
    assert int(metrics.loc[1, "n_samples"]) == 4


def _small_grid_inputs(mill="Plate", thickness=16.0):
    """The tiny Plate search both new modes are exercised on (no workbook needed).

    The two *richest* templates from the lean-first list, not the two leanest: the leanest
    chemistries cannot reach A572 Gr 50 strength at 16 mm in this window, which would leave
    the frontier empty and the stress test with no shortlist — that is a valid answer, but a
    test that never exercises the real path proves nothing.
    """
    reqs = opt.preset_requirements(mill, "ASTM A 572 Gr 50")
    chemistries = opt.lean_chemistries()[-2:]
    if not reqs or not chemistries:
        pytest.skip("preset grades or grade_templates.json missing")
    return reqs, chemistries


@pytest.fixture(scope="module")
def boosted_models():
    boosted = os.path.join(REPO_ROOT, "models", "boosted_out")
    if not os.path.isdir(boosted) or not os.listdir(boosted):
        pytest.skip("models/boosted_out artifacts missing")
    return boosted


def test_pareto_frontier_runs_on_a_small_grid(boosted_models):
    """pareto_frontier scores the shared grid and returns a usable, non-dominated frontier."""
    reqs, chemistries = _small_grid_inputs()
    res = opt.pareto_frontier(
        "Plate", {"Rolling_Type": "AR"}, 16.0, reqs, chemistries,
        (900.0, 980.0), (800.0, 880.0), cooling="air", k=1.0, step=20.0)

    assert not res.get("error"), res.get("error")
    assert res["n_evaluated"] == 2 * 5 * 5, "chemistries × entry × finish, air = one header set"
    assert res["n_feasible"] > 0, "this grid must offer at least one feasible schedule"
    assert res["frontier"], "a non-empty feasible pool always has a frontier"
    assert set(res) >= {"frontier", "n_feasible", "n_evaluated", "meta"}
    assert res["meta"]["mill"] == "Plate" and res["meta"]["k"] == pytest.approx(1.0)

    margin_need = {r["key"] + "|" + r["direction"]: 1.0 * float(r["half"])
                   for r in reqs if r.get("kind") == "mech"}
    for row in res["frontier"]:
        assert set(row) >= {"entry", "finish", "headers", "chemistry", "ce_iiw",
                            "microalloy_total", "min_margin_interval", "margins", "ood"}
        assert 900.0 <= row["entry"] <= 980.0
        assert 800.0 <= row["finish"] <= 880.0
        assert row["headers"] == 0.0, "air cooling has no headers"
        # every required property clears k × its interval, on the row itself
        for rid, need in margin_need.items():
            assert row["margins"][rid] >= need - 1e-9, (rid, row["margins"][rid], need)

    # and none of the returned rows is dominated by another one
    for i, one in enumerate(res["frontier"]):
        for j, other in enumerate(res["frontier"]):
            if i == j:
                continue
            assert not (other["ce_iiw"] <= one["ce_iiw"] + 1e-12
                        and other["headers"] <= one["headers"] + 1e-12
                        and other["min_margin_interval"] >= one["min_margin_interval"] - 1e-12)


def test_robust_recipe_runs_on_a_small_grid(boosted_models):
    """robust_recipe stress-tests the shortlist and ranks the chosen row above its runners."""
    reqs, chemistries = _small_grid_inputs()
    res = opt.robust_recipe(
        "Plate", {"Rolling_Type": "AR"}, 16.0, reqs, chemistries,
        (900.0, 980.0), (800.0, 880.0), cooling="air", k=1.0, step=20.0,
        alpha=0.10, n_samples=12, seed=42)

    assert not res.get("error"), res.get("error")
    assert res["n_candidates"] > 0, "this grid must offer a shortlist"
    assert res["n_draws"] == res["n_candidates"] * 12
    chosen = res["chosen"]
    assert chosen is not None, "a non-empty shortlist always has a robust pick"
    assert chosen["n_samples"] == 12
    for field in ("mean_margin", "cvar_margin", "p_fail"):
        assert np.isfinite(float(chosen[field]))
    assert 0.0 <= chosen["p_fail"] <= 1.0
    assert chosen["cvar_margin"] <= chosen["mean_margin"] + 1e-9
    # same scored columns as select_recipe's chosen row, plus the risk fields
    assert "margin_YS_BN|min" in chosen and "pred_YS_BN" in chosen

    cvar = float(chosen["cvar_margin"])
    assert len(res["runners"]) <= 5
    for runner in res["runners"]:
        assert float(runner["cvar_margin"]) <= cvar + 1e-9
    assert res["noise_description"], "the noise model must be described"
    assert "entry ±20" in res["noise_description"]
    assert res["deterministic"] is not None
