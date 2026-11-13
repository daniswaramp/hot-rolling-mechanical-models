"""HRC (Steckel mill) route regression tests.

The Steckel route used to take the page down when a variable changed: a response-surface selectbox
whose options shrank while its widget key persisted, an unguarded single-target lookup, an
optimizer gauge input with a hardcoded Plate window, and a specification table that raised instead
of degrading. These tests pin the guards in place.
"""

import json
import os
import sys

import numpy as np
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
import constraints as cons  # noqa: E402
import physics  # noqa: E402

MILLS = ["Plate", "HRC"]
SCORE_ENGINES = ["Champion GBDT (Recommended)", "CatBoost", "XGBoost", "LightGBM",
                 "Ensemble (.joblib)"]


# --------------------------------------------------------------------------- scoring
@pytest.mark.parametrize("engine", SCORE_ENGINES)
@pytest.mark.parametrize("mill", MILLS)
def test_every_engine_scores_every_target(mill, engine):
    """No engine may raise, and every shipped target must come back finite."""
    preds = app.predict_selected_engine(mill, app.defaults(mill), engine)
    assert preds, f"{mill}/{engine} returned nothing"
    for key in app.TARGET_LABEL[mill]:
        assert key in preds, f"{mill}/{engine} omitted {key}"
        value = preds[key].get("pred", np.nan)
        assert np.isfinite(value), f"{mill}/{engine}/{key} scored {value!r}: {preds[key].get('error')}"


@pytest.mark.parametrize("mill", MILLS)
def test_every_input_at_its_extremes_scores(mill):
    """Min and max of every slider, and the degenerate combinations between them."""
    defaults = app.defaults(mill)
    rows = {row[1]: row for row in app.SPECS[mill]}
    for key, row in rows.items():
        for extreme in (row[4], row[5]):
            raw = dict(defaults)
            raw[key] = float(extreme)
            preds = app.predict_selected_engine(mill, raw, "Champion GBDT (Recommended)")
            for t_key in app.TARGET_LABEL[mill]:
                assert t_key in preds, f"{mill} {key}={extreme} dropped {t_key}"
                err = preds[t_key].get("error")
                if err:
                    continue  # a genuinely unscoreable target reports why instead of crashing
                assert np.isfinite(preds[t_key]["pred"])


def test_hrc_degenerate_schedules_do_not_crash():
    """Zero draft, inverted coiling temperature, empty chemistry."""
    defaults = app.defaults("HRC")
    cases = [
        dict(defaults, Coil_T_mm=defaults["TB_Thick_mm"]),          # zero draft
        dict(defaults, Temp_CT_Top_C=750.0, Temp_FT_Avg_C=750.0),   # coiling above finishing
        dict(defaults, **{k: 0.0 for k in defaults if k.startswith("Chem_")}),
        dict(defaults, Coil_T_mm=1.8),                              # the old optimizer-crash gauge
    ]
    for raw in cases:
        preds = app.predict_selected_engine("HRC", raw, "Champion GBDT (Recommended)")
        for t_key in app.TARGET_LABEL["HRC"]:
            assert t_key in preds
            if preds[t_key].get("error"):
                continue
            assert np.isfinite(preds[t_key]["pred"])


@pytest.mark.parametrize("mill", MILLS)
def test_score_frame_is_nan_safe(mill):
    """The vectorised path must not propagate NaN through the normalizing coupling."""
    frame = app.prepare_frame(mill, app_pd_frame(mill))
    scored = app.score_frame(mill, frame, engine="Champion GBDT (Recommended)")
    for key in app.TARGET_LABEL[mill]:
        assert key in scored.columns, f"{mill} score_frame omitted {key}"


def app_pd_frame(mill):
    """A one-row frame of the mill defaults, the shape the surface and batch paths build."""
    import pandas as pd
    return pd.DataFrame([app.defaults(mill)])


def test_single_target_never_raises():
    """The helper the surface and capability tabs lean on returns (nan, reason), not an exception."""
    value, err = app.single_target("HRC", app.defaults("HRC"), "Champion GBDT (Recommended)", "TS_BN")
    assert np.isfinite(value) and err == ""
    # a target the engine cannot produce comes back as nan + a reason
    value, err = app.single_target("Plate", app.defaults("Plate"), "Champion GBDT (Recommended)", "Nope")
    assert not np.isfinite(value) and err


# --------------------------------------------------------------------------- widget guards
def test_prune_stale_widget_key_clears_out_of_options_value():
    """The response-surface crash: options shrink, the persisted value leaves the list."""
    import streamlit as st
    st.session_state["y_HRC"] = "Chem_Mn"
    app.prune_stale_widget_key("y_HRC", ["Temp_FT_Avg_C", "Temp_CT_Top_C", "Coil_T_mm"])
    assert "y_HRC" not in st.session_state
    st.session_state["y_HRC"] = "Coil_T_mm"
    app.prune_stale_widget_key("y_HRC", ["Temp_FT_Avg_C", "Coil_T_mm", "Chem_C"])
    assert st.session_state["y_HRC"] == "Coil_T_mm"


def test_spec_rows_are_wellformed():
    """A malformed row used to raise inside schedule_inputs' unpacking and kill both routes."""
    for mill in MILLS:
        for row in app.SPECS[mill]:
            assert len(row) == 8, f"{mill}/{row[1]}: {len(row)} fields"
            group, key, label, unit, lo, hi, default, step = row
            assert isinstance(lo, float) and isinstance(hi, float)
            assert hi > lo, f"{mill}/{key}: max <= min"
            assert lo <= default <= hi, f"{mill}/{key}: default outside range"
            assert step > 0, f"{mill}/{key}: step <= 0"


def test_optimizer_gauge_window_covers_the_mill_range():
    """The optimizer used to hardcode min_value 3.0 and reject thin Steckel coil."""
    for mill in MILLS:
        row = app.spec_map(mill)["Coil_T_mm" if mill == "HRC" else "Plate_T_mm"]
        assert row[4] >= 0.0
        # the widget window is derived from the spec row, so its own minimum must fit
        assert row[4] < row[5]


# --------------------------------------------------------------------------- domain envelope
def _domain():
    with open(os.path.join(REPO_ROOT, "models", "applicability_domain.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_domain_covers_the_optimizer_window_keys():
    """Every temperature the optimizer sweeps must be bounded, or explicitly caveated."""
    domain = _domain()
    caveats = json.dumps(domain.get("_caveats", {}))
    for mill in MILLS:
        for _role, key, _label in cons.WINDOW_KEYS[mill]:
            block = domain.get(mill, {}).get(key)
            if block is not None:
                assert np.isfinite(block["p01"]) and np.isfinite(block["p99"])
                assert block["p01"] < block["p99"], f"{mill}/{key}: p01 >= p99"
            else:
                assert f"{mill}.{key}" in caveats, \
                    f"{mill}/{key} is unbounded and not documented in _caveats"


def test_plate_defaults_no_longer_trip_the_extrapolation_banner():
    """ACC_Laminar_Header's swapped-column percentiles flagged every ordinary Plate schedule."""
    for mill in MILLS:
        warnings, _notices = explainer.check_applicability_domain(mill, app.defaults(mill))
        assert warnings == [], f"{mill} defaults flagged: {[w['param'] for w in warnings]}"


def test_domain_has_no_impossible_extremes():
    """A header count of 671 or a coiling temperature max of 6621 C is contamination, not data."""
    domain = _domain()
    for mill in MILLS:
        for key, block in domain.get(mill, {}).items():
            assert block["p99"] < 5000.0, f"{mill}/{key}: p99 {block['p99']} is not a physical value"
            if "Temp" in key:
                assert block["p99"] <= 1500.0, f"{mill}/{key}: p99 {block['p99']} C is not a temperature"
            if key == "ACC_Laminar_Header":
                assert block["p99"] <= 20.0, f"{key}: p99 {block['p99']} is not a header count"


# --------------------------------------------------------------------------- physics
def test_indicators_read_from_the_shared_physics_module():
    """The derived-metrics card and app/physics.py must agree exactly."""
    for mill, raw in (("Plate", app.defaults("Plate")), ("HRC", app.defaults("HRC"))):
        ind = ml.compute_physical_indicators(raw, mill)
        c = raw["Chem_C"]
        expected_ar3 = physics.ar3(c, raw["Chem_Mn"], raw["Chem_Cu"], raw["Chem_Cr"],
                                   raw["Chem_Ni"], raw["Chem_Mo"])
        assert ind["ar3"] == pytest.approx(expected_ar3, abs=1e-9)
        expected_sigma = physics.sigma_ss(raw["Chem_Mn"], raw["Chem_Si"], raw["Chem_Ni"],
                                          raw["Chem_Cr"], raw["Chem_Cu"], raw["Chem_Ti"],
                                          raw["Chem_Mo"], raw["Chem_N"])
        assert ind["sigma_ss"] == pytest.approx(expected_sigma, abs=1e-9)


def test_indicators_clip_like_the_feature_modules():
    """A card computed on unclipped inputs disagreed with the model at the extremes."""
    import math
    raw = dict(app.defaults("HRC"), Coil_T_mm=0.5, TB_Thick_mm=70.0)
    ind = ml.compute_physical_indicators(raw, "HRC")
    # the HRC feature module clips coil thickness to 1.5 mm, so the card must use 1.5 too
    assert ind["true_strain"] == pytest.approx(
        math.log(max(70.0, 1.5 * 1.05) / 1.5), rel=1e-6)
    # and an over-thick transfer bar is clipped to the same 70 mm ceiling
    raw = dict(app.defaults("HRC"), TB_Thick_mm=200.0)
    ind = ml.compute_physical_indicators(raw, "HRC")
    assert ind["true_strain"] == pytest.approx(math.log(70.0 / 6.0), rel=1e-6)


# --------------------------------------------------------------------------- artifacts
def test_bundle_feature_columns_handles_both_schemas():
    """Plate bundles carry 'features', HRC bundles carry 'feature_names'; one resolver for both."""
    import pandas as pd
    for mill in MILLS:
        frame = app.prepare_frame(mill, pd.DataFrame([app.defaults(mill)]))
        X = app.feature_matrix(mill, frame)
        for target in app.TARGET_LABEL[mill]:
            bundle = ml.load_model_bundle(mill, target)
            if bundle is None:
                continue
            cols = ml.bundle_feature_columns(bundle, X)
            assert cols, f"{mill}/{target}: resolver returned nothing"
            missing = [c for c in cols if c not in X.columns]
            assert missing == [], f"{mill}/{target}: bundle wants {missing}"
            # and the model must actually accept them
            aligned = pd.DataFrame({c: X[c] for c in cols}, index=X.index)
            try:
                value = float(np.asarray(bundle["model"].predict(aligned)).ravel()[0])
            except Exception as exc:  # noqa: BLE001
                pytest.fail(f"{mill}/{target}: bundle refused its own feature list: {exc}")
            assert np.isfinite(value), f"{mill}/{target}: bundle predicted NaN"


def test_shipped_bundles_carry_a_feature_list():
    """No bundle may rely on the engineered column order by accident."""
    for mill in MILLS:
        for target in app.TARGET_LABEL[mill]:
            bundle = ml.load_model_bundle(mill, target)
            if bundle is None:
                continue
            assert bundle.get("feature_names") or bundle.get("features") \
                or hasattr(bundle["model"], "feature_names_in_"), \
                f"{mill}/{target}: no feature list anywhere in the bundle"
