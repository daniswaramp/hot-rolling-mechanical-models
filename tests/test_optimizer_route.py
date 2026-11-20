"""
Hottest-feasible-first recipe selection tests (redesigned Inverse Optimizer).

Covers the pure selection logic in app/optimizer.py::select_recipe (no Streamlit needed),
the Pareto-frontier and robust CVaR modes, and one end-to-end grid-search smoke run for the
Plate route.
"""

import inspect
import os
import re
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


# --------------------------------------------------------------------------- merged spec picker
def test_spec_grade_options_labels_and_roundtrip():
    """The merged picker shows one "SPEC GRADE" label per workbook tier and every
    label maps back to the exact (spec, grade) pair. The workbook writes the JIS
    grade as "SS 400", so the label reads "JIS G 3101 SS 400" and norm_key() is
    what lets a picker default match "SS400"."""
    import spec_checker as sc

    labels, lookup = sc.spec_grade_options()
    assert labels, "no spec+grade options built from the shipped workbook"
    assert len(labels) == len(lookup), "duplicate label would silently drop a tier"
    assert "JIS G 3101 SS 400" in labels
    spec, grade = lookup["JIS G 3101 SS 400"]
    assert spec.startswith("JIS G 3101") and grade == "SS 400"
    assert sc.norm_key("JIS G 3101 SS 400") == sc.norm_key("JIS G 3101 SS400")
    # every label's mapped tier really exists in the workbook
    for label, (spec, grade) in lookup.items():
        assert grade in sc.get_available_grades(spec), label


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


# ----------------------------------------------------------- ranged chemistry constraints
def _chem(c=0.06, mn=1.0, si=0.25, nb=0.0, v=0.0, ti=0.0):
    """Synthetic chemistry with the Chem_* keys the helpers read.

    With the other alloying elements at zero: CE_IIW = C + Mn/6 and
    Pcm = C + Si/30 + Mn/20 + V/10.
    """
    return {"Chem_C": c, "Chem_Mn": mn, "Chem_Si": si, "Chem_Nb": nb, "Chem_V": v,
            "Chem_Ti": ti, "Chem_Cr": 0.0, "Chem_Mo": 0.0, "Chem_Ni": 0.0,
            "Chem_Cu": 0.0, "Chem_B": 0.0}


def test_constraint_filter_drops_ce_and_nb_violators():
    """CE above the max and Nb above the cap are dropped; combined limits keep one."""
    chems = [
        ("LEAN", _chem(c=0.06, mn=1.44)),            # CE 0.30, Nb 0
        ("HOTCE", _chem(c=0.18, mn=1.92)),           # CE 0.50 > 0.45 max
        ("RICHNB", _chem(c=0.06, mn=1.44, nb=0.08)), # CE 0.30 but Nb 0.08 > 0.06 cap
        ("OKRICH", _chem(c=0.12, mn=1.20, nb=0.03)), # CE 0.32, Nb 0.03: the intended one
    ]
    filtered, derived, err = opt.apply_chemistry_constraints(
        chems, ce_min=None, ce_max=0.45, pcm_min=None, pcm_max=None,
        nb_max=0.06, v_max=None, ti_max=None)
    assert err is None and derived is None
    assert [name for name, _ in filtered] == ["LEAN", "OKRICH"], "input order preserved"


def test_constraint_derivation_from_leanest_template():
    """No template passes: exactly one derived chemistry, appended last, inside the limits."""
    chems = [("RICH", _chem(c=0.18, mn=1.50, nb=0.08))]   # CE 0.43, Nb 0.08
    filtered, derived, err = opt.apply_chemistry_constraints(
        chems, ce_max=0.40, nb_max=0.03)
    assert err is None and filtered == []
    assert derived is not None
    name, chem = derived
    assert name == "derived (clamped to constraints)"
    assert chem["Chem_Nb"] <= 0.03 + 1e-9, "microalloy clamped to its cap"
    # Mn solved directly onto the CE bound, floored at the app's Chem_Mn minimum
    assert opt.chemistry_ce_iiw(chem) == pytest.approx(0.40, abs=1e-6)
    assert opt.chemistry_ce_iiw(chem) <= 0.40 + 1e-6
    assert chem["Chem_Mn"] >= 0.20 - 1e-9
    # the view composes filtered + [derived]; here filtered is empty, so it is last
    composed = filtered + [derived]
    assert len(composed) == 1 and composed[-1][0] == "derived (clamped to constraints)"


def test_constraint_derivation_walks_mn_for_pcm_then_rechecks_ce():
    """Pcm over its cap walks Mn further down; the resulting CE must re-enter its window."""
    chems = [("P", _chem(c=0.18, mn=1.60, v=0.08))]  # CE 0.463, Pcm 0.276
    filtered, derived, err = opt.apply_chemistry_constraints(
        chems, ce_max=0.45, pcm_max=0.25)
    assert err is None and filtered == []
    name, chem = derived
    assert name == "derived (clamped to constraints)"
    assert opt.chemistry_pcm(chem) <= 0.25 + 1e-6
    assert opt.chemistry_ce_iiw(chem) <= 0.45 + 1e-6
    assert opt.chemistry_ce_iiw(chem) >= 0.0 - 1e-9


def test_constraint_infeasible_reports_error_sentinel():
    """CE max below what the Mn floor can reach: no derivation is fabricated."""
    chems = [("ANY", _chem(c=0.10, mn=1.0))]  # CE floor at Mn 0.20 is 0.1333
    filtered, derived, err = opt.apply_chemistry_constraints(chems, ce_max=0.05)
    assert filtered == [] and derived is None
    assert err and "infeasible" in err, err


def test_constraint_off_returns_input_unchanged():
    """All-None limits: the input list comes back unchanged, same order (checkbox off)."""
    chems = [("B", _chem(c=0.12, mn=1.0)), ("A", _chem(c=0.06, mn=0.8))]
    out, derived, err = opt.apply_chemistry_constraints(chems)
    assert out == chems
    assert derived is None and err is None
    out2, d2, e2 = opt.apply_chemistry_constraints(
        chems, None, None, None, None, None, None, None)
    assert out2 == chems and d2 is None and e2 is None


def test_constraint_filter_keeps_lean_first_order_on_real_templates():
    """On the shipped templates the filter preserves the lean-first CE ordering."""
    chems = opt.lean_chemistries()
    if not chems:
        pytest.skip("models/grade_templates.json missing")
    filtered, derived, err = opt.apply_chemistry_constraints(
        chems, ce_max=0.45, pcm_max=0.25, nb_max=0.06, v_max=0.10, ti_max=0.06)
    assert err is None and derived is None
    assert 0 < len(filtered) <= len(chems)
    ces = [opt.chemistry_ce_iiw(chem) for _, chem in filtered]
    assert ces == sorted(ces), "lean-first order preserved"


def test_view_optimizer_renders_template_count_caption_when_enabled(monkeypatch):
    """Bare-mode render with the gate forced on: no exception, the live count is captioned.

    The checkbox cannot be preseeded in bare mode, so st.checkbox is patched to True
    (view_optimizer has no other checkbox) and st.caption records its calls.
    """
    import streamlit as st

    chems = opt.lean_chemistries()
    if not chems:
        pytest.skip("models/grade_templates.json missing")

    raw = {"Plate_T_mm": 16.0, "Chem_C": 0.10, "Chem_Mn": 1.0, "Chem_Si": 0.20}
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith("opt_")]:
        del st.session_state[key]

    real_caption = st.caption
    seen = []
    # checkbox off (default): the live count must not appear
    monkeypatch.setattr(st, "caption", lambda text, *a, **k: seen.append(str(text)))
    opt.view_optimizer("Plate", raw, "Champion GBDT (Recommended)", spec_rows=None)
    assert not any("templates pass" in s for s in seen)

    # checkbox forced on: the caption appears and names the count
    monkeypatch.setattr(st, "checkbox", lambda *a, **k: True)
    seen.clear()
    opt.view_optimizer("Plate", raw, "Champion GBDT (Recommended)", spec_rows=None)
    assert any("templates pass" in s for s in seen), seen
    assert any(re.search(r"templates pass: \d+ of \d+$", s) for s in seen), seen


# ----------------------------------------------------------- chemistry pre-filter
def test_prefilter_chemistries_drops_ceiling_violators():
    """prefilter_chemistries_for_reqs filters deterministic ce_iiw, pcm, Chem_S, Chem_P ceilings."""
    ok = dict(_chem(c=0.10, mn=1.00, si=0.20), Chem_S=0.005, Chem_P=0.012)      # CE=0.267, Pcm=0.157
    hi_ce = dict(_chem(c=0.22, mn=1.50, si=0.20), Chem_S=0.005, Chem_P=0.012)   # CE=0.470
    hi_pcm = dict(_chem(c=0.16, mn=1.20, si=0.30, v=0.08), Chem_S=0.005, Chem_P=0.012)  # CE=0.376, Pcm=0.238
    hi_s = dict(_chem(c=0.10, mn=1.00, si=0.20), Chem_S=0.030, Chem_P=0.012)
    hi_p = dict(_chem(c=0.10, mn=1.00, si=0.20), Chem_S=0.005, Chem_P=0.035)
    chems = [("OK", ok), ("HI_CE", hi_ce), ("HI_PCM", hi_pcm), ("HI_S", hi_s), ("HI_P", hi_p)]

    reqs = [
        _req("YS_BN", "min", 345.0, 20.0, kind="mech"),
        {"key": "ce_iiw", "kind": "chem", "direction": "max", "required": 0.42, "half": 0.0, "label": "CE IIW"},
        {"key": "pcm", "kind": "chem", "direction": "max", "required": 0.22, "half": 0.0, "label": "Pcm"},
        {"key": "Chem_S", "kind": "chem", "direction": "max", "required": 0.015, "half": 0.0, "label": "Sulfur (S)"},
        {"key": "Chem_P", "kind": "chem", "direction": "max", "required": 0.025, "half": 0.0, "label": "Phosphorus (P)"},
    ]
    kept = opt.prefilter_chemistries_for_reqs(chems, reqs)
    assert [name for name, _ in kept] == ["OK"]

    # Mechanical-only requirements keep all chemistries unchanged.
    assert opt.prefilter_chemistries_for_reqs(chems, [_req("YS_BN", "min", 345.0, 20.0)]) == chems

    # Single chemistry is never filtered out even if it violates a ceiling.
    single = [("HI_CE", hi_ce)]
    assert opt.prefilter_chemistries_for_reqs(single, reqs) == single


def test_prefilter_chemistries_keeps_original_when_all_violate_and_reports_reason(monkeypatch):
    """If every chemistry violates a deterministic ceiling, keep all so _rejected_band_notes names it."""
    chems = [
        ("A", dict(_chem(c=0.20, mn=1.50), Chem_S=0.005, Chem_P=0.015)),  # CE = 0.45
        ("B", dict(_chem(c=0.22, mn=1.50), Chem_S=0.005, Chem_P=0.015)),  # CE = 0.47
    ]
    reqs = [
        _req("YS_BN", "min", 250.0, 20.0, kind="mech"),
        {"key": "ce_iiw", "kind": "chem", "direction": "max", "required": 0.35, "half": 0.0,
         "label": "Carbon Equivalent (CE IIW)"},
    ]
    kept = opt.prefilter_chemistries_for_reqs(chems, reqs)
    assert kept == chems

    # End-to-end with mocked score_targets: rejection notes still report the CE ceiling violation.
    monkeypatch.setattr(opt, "score_targets", lambda mill, X, targets: {
        t: np.full(len(X), 420.0, dtype=float) for t in targets
    })
    res = opt.run_recipe_search(
        "Plate", {"Rolling_Type": "AR"}, 16.0, reqs, chems,
        (940.0, 960.0), (820.0, 840.0), cooling="air", k=1.0, step=20.0)
    assert res["chosen"] is None
    assert res["n_evaluated"] == 2 * 2 * 2
    assert any("Carbon Equivalent" in note and "exceeds" in note for note in res["rejected"]), res["rejected"]


# ----------------------------------------------------------- correlated thermal copula
def test_perturbation_draws_correlated_thermal_copula_preserves_bounds_and_seed():
    """_perturbation_draws couples entry and finish (rho=0.65) with exact uniform [-20, 20] bounds."""
    pool = pd.DataFrame([{
        "FM_Temp_Before_Rolling": 960.0,
        "FM_Finish_Temp": 840.0,
        "ACC_Headers_n": 4.0,
        "ACC_Laminar_Header": 4.0,
        "Plate_T_mm": 16.0,
        "_chemistry": "",
        "Chem_C": 0.16,
        "Chem_Mn": 1.20,
    }])

    frame1, ids1, desc1 = opt._perturbation_draws(
        "Plate", pool, 16.0, n_samples=2000, seed=42, cooling="laminar", headers_range=(4, 8))
    frame2, ids2, desc2 = opt._perturbation_draws(
        "Plate", pool, 16.0, n_samples=2000, seed=42, cooling="laminar", headers_range=(4, 8))
    frame3, _ids3, _desc3 = opt._perturbation_draws(
        "Plate", pool, 16.0, n_samples=2000, seed=99, cooling="laminar", headers_range=(4, 8))

    assert len(frame1) == 2000 and len(ids1) == 2000
    pd.testing.assert_frame_equal(frame1, frame2)
    assert np.array_equal(ids1, ids2)
    assert desc1 == desc2
    assert not np.allclose(frame1["FM_Temp_Before_Rolling"], frame3["FM_Temp_Before_Rolling"])

    d_entry = frame1["FM_Temp_Before_Rolling"].to_numpy(dtype=float) - 960.0
    d_finish = frame1["FM_Finish_Temp"].to_numpy(dtype=float) - 840.0
    assert np.all(d_entry >= -opt.ENTRY_AIM_C - 1e-9)
    assert np.all(d_entry <= opt.ENTRY_AIM_C + 1e-9)
    assert np.all(d_finish >= -opt.FINISH_AIM_C - 1e-9)
    assert np.all(d_finish <= opt.FINISH_AIM_C + 1e-9)
    assert abs(float(np.mean(d_entry))) < 1.5
    assert abs(float(np.mean(d_finish))) < 1.5
    assert float(np.min(d_entry)) < -18.0 and float(np.max(d_entry)) > 18.0

    # Gaussian copula with rho=0.65 yields Pearson correlation ~(6/pi)*arcsin(0.65/2) ≈ 0.632 on uniform marginals.
    corr = float(np.corrcoef(d_entry, d_finish)[0, 1])
    assert 0.50 < corr < 0.78, f"unexpected thermal carry-over correlation: {corr:.4f}"

    assert "entry ±20 °C" in desc1
    assert "correlated thermal copula (ρ=0.65)" in desc1


# ----------------------------------------------------------- ladle-in-hand order triage
def test_triage_specs_for_heat_plate_and_hrc_smoke_and_sorting(boosted_models):
    """Real-model ladle triage ranks certifiable matching specs descending by worst margin."""
    import spec_checker as sc

    if not sc.spec_available():
        pytest.skip("specification workbook missing")

    plate_raw = {
        "Plate_T_mm": 16.0, "Plate_W_mm": 1889.0, "Slab_T_mm": 218.0, "TB_Thickness_mm": 41.0,
        "Chem_C": 0.177, "Chem_Mn": 0.613, "Chem_Si": 0.151, "Chem_P": 0.0121, "Chem_S": 0.005,
        "Chem_Al": 0.03, "Chem_Nb": 0.002, "Chem_V": 0.0039, "Chem_Ti": 0.002, "Chem_Cu": 0.0175,
        "Chem_Ni": 0.02, "Chem_Cr": 0.043, "Chem_Mo": 0.01, "Chem_N": 0.0042,
    }
    res_plate = opt.triage_specs_for_heat("Plate", plate_raw, thickness=16.0, k=1.0, max_specs=10, step=40.0)
    assert res_plate, "expected certifiable Plate specs for default ladle chemistry"
    assert len(res_plate) <= 10
    expected_keys = {"spec", "grade", "supply_condition", "best_entry", "best_finish",
                     "best_cooling", "min_margin_interval", "ce_iiw", "verdict"}
    for row in res_plate:
        assert set(row) >= expected_keys
        assert row["min_margin_interval"] >= 1.0 - 1e-9
        assert row["verdict"] == "CERTIFIABLE"
    margins = [r["min_margin_interval"] for r in res_plate]
    assert margins == sorted(margins, reverse=True)

    hrc_raw = {
        "Coil_T_mm": 6.0, "Coil_W_mm": 1200.0, "TB_Thick_mm": 24.8,
        "Chem_C": 0.176, "Chem_Mn": 0.581, "Chem_Si": 0.127, "Chem_P": 0.015, "Chem_S": 0.005,
        "Chem_Al": 0.036, "Chem_Nb": 0.001, "Chem_V": 0.004, "Chem_Ti": 0.002, "Chem_Cu": 0.02,
        "Chem_Ni": 0.02, "Chem_Cr": 0.03, "Chem_Mo": 0.005, "Chem_N": 0.004, "Chem_B": 0.0,
    }
    res_hrc = opt.triage_specs_for_heat("HRC", hrc_raw, thickness=6.0, k=1.0, max_specs=10, step=40.0)
    assert res_hrc, "expected certifiable HRC specs for default ladle chemistry"
    assert all(r["best_cooling"].startswith("CT ") for r in res_hrc)
    hrc_margins = [r["min_margin_interval"] for r in res_hrc]
    assert hrc_margins == sorted(hrc_margins, reverse=True)


def test_triage_specs_for_heat_filters_thickness_chemistry_and_k(monkeypatch):
    """triage_specs_for_heat enforces thickness bounds, locked chemistry limits, k, and max_specs."""
    import spec_checker as sc

    mock_specs = pd.DataFrame([
        {"SPEC": "SPEC_EASY", "GRADE": "A", "SUPPLY CONDITION": "AR",
         "THICK MIN (mm)": 0.0, "THICK MAX (mm)": 25.0,
         "YS min (TR) [N/mm2]": 250.0, "TS min (TR) [N/mm2]": 400.0,
         "C Max [%]": 0.25, "S Max [%]": 0.03, "CE Max [%]": 0.45},
        {"SPEC": "SPEC_MED", "GRADE": "B", "SUPPLY CONDITION": "TMCP",
         "THICK MIN (mm)": 0.0, "THICK MAX (mm)": 25.0,
         "YS min (TR) [N/mm2]": 330.0, "TS min (TR) [N/mm2]": 430.0,
         "C Max [%]": 0.25, "S Max [%]": 0.03, "CE Max [%]": 0.45},
        {"SPEC": "SPEC_LOW_C", "GRADE": "C", "SUPPLY CONDITION": "AR",
         "THICK MIN (mm)": 0.0, "THICK MAX (mm)": 25.0,
         "YS min (TR) [N/mm2]": 220.0, "TS min (TR) [N/mm2]": 350.0,
         "C Max [%]": 0.10, "S Max [%]": 0.03, "CE Max [%]": 0.45},
        {"SPEC": "SPEC_THICK_ONLY", "GRADE": "D", "SUPPLY CONDITION": "AR",
         "THICK MIN (mm)": 30.0, "THICK MAX (mm)": 60.0,
         "YS min (TR) [N/mm2]": 220.0, "TS min (TR) [N/mm2]": 350.0,
         "C Max [%]": 0.25, "S Max [%]": 0.03, "CE Max [%]": 0.45},
    ])
    monkeypatch.setattr(sc, "load_specifications", lambda: mock_specs)
    monkeypatch.setattr(opt, "interval_half_widths",
                        lambda mill, keys: {key: 20.0 for key in keys})
    monkeypatch.setattr(opt, "score_targets", lambda mill, X, targets: {
        key: np.full(len(X), 360.0 if key.startswith("YS") else (460.0 if key.startswith("TS") else 25.0))
        for key in targets
    })

    raw = dict(_chem(c=0.16, mn=1.20), Chem_S=0.005, Chem_P=0.015, Plate_T_mm=16.0)
    res_k1 = opt.triage_specs_for_heat("Plate", raw, thickness=16.0, k=1.0, max_specs=20, step=40.0)
    assert [r["spec"] for r in res_k1] == ["SPEC_EASY", "SPEC_MED"]
    assert res_k1[0]["min_margin_interval"] == pytest.approx(3.0)
    assert res_k1[1]["min_margin_interval"] == pytest.approx(1.5)

    res_k2 = opt.triage_specs_for_heat("Plate", raw, thickness=16.0, k=2.0, max_specs=20, step=40.0)
    assert [r["spec"] for r in res_k2] == ["SPEC_EASY"]

    res_cap = opt.triage_specs_for_heat("Plate", raw, thickness=16.0, k=1.0, max_specs=1, step=40.0)
    assert len(res_cap) == 1 and res_cap[0]["spec"] == "SPEC_EASY"


def test_view_optimizer_triage_expander_renders_without_checkbox(monkeypatch):
    """The ladle triage expander uses no st.checkbox and does not scan on passive render."""
    import streamlit as st

    raw = {"Plate_T_mm": 16.0, "Chem_C": 0.15, "Chem_Mn": 1.0, "Chem_Si": 0.20}
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith("opt_")]:
        del st.session_state[key]

    checkbox_calls = []
    real_checkbox = st.checkbox
    def track_checkbox(*a, **k):
        checkbox_calls.append(k.get("key") or (a[0] if a else ""))
        return real_checkbox(*a, **k)

    triage_calls = []
    monkeypatch.setattr(st, "checkbox", track_checkbox)
    monkeypatch.setattr(opt, "triage_specs_for_heat", lambda *a, **k: triage_calls.append((a, k)) or [])
    opt.view_optimizer("Plate", raw, "Champion GBDT (Recommended)", spec_rows=None)

    assert len(checkbox_calls) <= 1  # only the auto-search chemistry gate inside the form
    assert not any("triage" in str(c).lower() for c in checkbox_calls)
    assert triage_calls == []

