"""
Hottest-feasible-first recipe selection tests (redesigned Inverse Optimizer).

Covers the pure selection logic in app/optimizer.py::select_recipe (no Streamlit needed)
and one end-to-end grid-search smoke run for the Plate route.
"""

import inspect
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
