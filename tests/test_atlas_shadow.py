"""
Atlas & Shadow contract tests: interpolation exactness, calibration math, the credential-less
fallback path, bare-mode rendering and the tab registration in streamlit_app.main().

Conventions follow tests/test_contracts.py: repo root and app/ go on sys.path, the real
artifacts in models/ are used where they exist, and anything that needs a champion model is
skipped when models/boosted_out is missing.
"""

import ast
import inspect
import io
import os
import sys
import textwrap

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


# --------------------------------------------------------------------------- app_ref
def test_app_ref_resolves_running_main_module(monkeypatch):
    """Regression: `streamlit run` executes the entry as __main__, so a tab module must
    reuse that module instead of re-importing streamlit_app (which re-runs
    st.set_page_config and raises StreamlitSetPageConfigMustBeFirstCommandError)."""
    import types

    import app_ref

    fake_main = types.ModuleType("__main__")
    fake_main.SPECS = {"Plate": []}
    monkeypatch.delitem(sys.modules, "streamlit_app", raising=False)
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    assert app_ref.get_app_module() is fake_main


def test_app_ref_prefers_imported_module_over_bare_main(monkeypatch):
    import types

    import streamlit_app  # noqa: F401

    import app_ref

    monkeypatch.setitem(sys.modules, "__main__", types.ModuleType("__main__"))
    assert app_ref.get_app_module() is streamlit_app


def test_app_ref_never_imports_streamlit_app_when_present(monkeypatch):
    import app_ref

    calls = []

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def spy(name, *args, **kwargs):
        if name == "streamlit_app":
            calls.append(name)
        return real_import(name, *args, **kwargs)

    import streamlit_app  # noqa: F401  (ensure present so no import is needed)

    monkeypatch.setattr("builtins.__import__", spy)
    app_ref.get_app_module()
    assert calls == [], "get_app_module re-imported streamlit_app"

import atlas
import remote_jobs
import shadow
import streamlit_app as app

MILLS = ["Plate", "HRC"]
HAS_BOOSTED = os.path.isdir(os.path.join(REPO_ROOT, "models", "boosted_out"))
NEEDS_MODELS = pytest.mark.skipif(not HAS_BOOSTED, reason="models/boosted_out is not on disk")


@pytest.fixture
def no_credentials(monkeypatch):
    """Guarantee the Modal path cannot be taken, whatever this machine has configured.

    Deleting the token variables is not enough on a developer checkout (``st.secrets`` may
    carry them and ``modal`` may be installed), so ``_secret`` is stubbed out as well: the
    test is about the fallback branch, not about this machine's credentials.
    """
    for key in remote_jobs.TOKEN_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(remote_jobs, "_secret", lambda name: None)
    monkeypatch.setattr(remote_jobs, "_READY", None)
    yield
    monkeypatch.setattr(remote_jobs, "_READY", None)


def defaults(mill):
    return app.defaults(mill)


# ------------------------------------------------------------------ 1. atlas interpolation
def _linear_atlas():
    """A 4D cube over 4 axes holding an affine response, which interpolation must reproduce."""
    axes = ["a", "b", "c", "d"]
    grid = {"a": np.linspace(0.0, 10.0, 5),
            "b": np.linspace(100.0, 200.0, 4),
            "c": np.linspace(-5.0, 5.0, 3),
            "d": np.linspace(1.0, 2.0, 2)}
    coef = {"a": 2.5, "b": -0.3, "c": 4.0, "d": 7.5}
    offset = 11.0
    mesh = np.meshgrid(*[grid[key] for key in axes], indexing="ij")
    values = offset + sum(coef[key] * plane for key, plane in zip(axes, mesh))
    return axes, grid, coef, offset, values


def test_interpolation_reproduces_a_linear_response_exactly():
    """Grid nodes and midpoints both come back to 1e-9: multilinear interpolation is exact
    on an affine response, so anything the slice shows between nodes is the model's own trend."""
    axes, grid, coef, offset, values = _linear_atlas()

    def linear(a, b, c, d):
        return offset + coef["a"] * a + coef["b"] * b + coef["c"] * c + coef["d"] * d

    fixed = {"b": 137.5, "d": 1.25}
    xs, ys, z = atlas.interpolate_slice(values, grid, axes, "a", "c", fixed, n=17)
    assert z.shape == (len(ys), len(xs)), "the slice must be shaped (y, x) for Plotly"
    expected = np.array([[linear(x, fixed["b"], y, fixed["d"]) for x in xs] for y in ys])
    assert np.max(np.abs(z - expected)) < 1e-9

    # the evaluation grid spans the stored window, so its ends are the stored nodes
    assert xs[0] == pytest.approx(grid["a"].min()) and xs[-1] == pytest.approx(grid["a"].max())
    assert ys[0] == pytest.approx(grid["c"].min()) and ys[-1] == pytest.approx(grid["c"].max())

    # exact at the stored nodes as well, including the two the 17-point grid does not land on
    nodes = atlas.interpolate_slice(values, grid, axes, "a", "c", fixed, n=len(grid["a"]))
    for row, y in enumerate(nodes[1]):
        for column, x in enumerate(nodes[0]):
            if x in tuple(grid["a"]) and y in tuple(grid["c"]):
                assert nodes[2][row, column] == pytest.approx(linear(x, fixed["b"], y, fixed["d"]),
                                                             abs=1e-9)

    # a midpoint of the held axes is not in the stored grid: still exact for an affine response
    mid = {key: float(np.mean(node)) for key, node in grid.items()}
    xs_mid, ys_mid, z_mid = atlas.interpolate_slice(values, grid, axes, "b", "d", mid, n=9)
    expected_mid = np.array([[linear(mid["a"], x, mid["c"], y) for x in xs_mid] for y in ys_mid])
    assert np.max(np.abs(z_mid - expected_mid)) < 1e-9

    with pytest.raises(ValueError):
        atlas.interpolate_slice(values, grid, axes, "a", "a", fixed)


def test_parse_atlas_rebuilds_the_cube_from_a_tidy_frame():
    """The stored parquet is a flat product table; the reader must recover the dense cube."""
    axes, grid, _coef, _offset, values = _linear_atlas()
    mesh = np.meshgrid(*[grid[key] for key in axes], indexing="ij")
    frame = pd.DataFrame({key: plane.ravel() for key, plane in zip(axes, mesh)})
    frame["TS_BN"] = values.ravel()
    # shuffle the rows: the reader indexes by node position, not by row order
    frame = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    payload = frame.to_parquet(index=False)

    _frame, read_grid, cube = atlas.parse_atlas(payload, axes)
    for key in axes:
        assert np.allclose(read_grid[key], grid[key])
    assert cube["TS_BN"].shape == tuple(len(grid[key]) for key in axes)
    assert np.max(np.abs(cube["TS_BN"] - values)) < 1e-9


# ------------------------------------------------------------------- 2. shadow calibration
def test_residual_summary_median_is_exact_and_spread_is_p10_p90():
    residuals = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    summary = remote_jobs.residual_summary(residuals)
    assert summary["bias"] == 3.0, "the bias is the median residual, exactly"
    assert summary["n"] == 5
    assert summary["p10"] == pytest.approx(np.percentile(residuals, 10))
    assert summary["p90"] == pytest.approx(np.percentile(residuals, 90))
    # the percentile helpers bracket the body of the sample, not its tail
    assert 1.0 < summary["p10"] < 2.0
    assert 4.0 < summary["p90"] < 10.0
    # rows with no measurement never move the median
    with_nan = remote_jobs.residual_summary(np.array([1.0, 2.0, np.nan, 3.0, 4.0, 10.0]))
    assert with_nan["bias"] == summary["bias"] and with_nan["n"] == 5
    assert remote_jobs.residual_summary(np.array([np.nan, np.nan])) is None


def test_calibration_groups_by_grade_with_per_grade_medians():
    residuals = {"TS_BN": np.array([1.0, 2.0, 3.0, 4.0, 10.0]),
                 "YS_BN": np.array([-2.0, -1.0, 0.0, 1.0, 2.0])}
    grades = ["A", "A", "B", "B", "B"]

    global_only = remote_jobs.calibrate_residuals(residuals)
    assert set(global_only["calibration"]) == {"TS_BN", "YS_BN"}
    assert global_only["by_grade"] == {}, "no grades given, no per-grade block"
    assert global_only["calibration"]["TS_BN"]["bias"] == 3.0
    assert global_only["calibration"]["YS_BN"]["bias"] == 0.0

    grouped = remote_jobs.calibrate_residuals(residuals, grades)
    assert set(grouped["by_grade"]) == {"A", "B"}
    assert grouped["by_grade"]["A"]["TS_BN"]["bias"] == pytest.approx(1.5)
    assert grouped["by_grade"]["B"]["TS_BN"]["bias"] == pytest.approx(4.0)
    assert grouped["by_grade"]["A"]["YS_BN"]["bias"] == pytest.approx(-1.5)
    assert grouped["by_grade"]["A"]["TS_BN"]["n"] == 2
    assert grouped["by_grade"]["B"]["TS_BN"]["n"] == 3
    # a global block is still reported alongside the per-grade one
    assert grouped["calibration"]["TS_BN"]["bias"] == 3.0

    # an empty grade cell is not a grade
    with_blank = remote_jobs.calibrate_residuals(residuals, ["A", "", "B", None, "B"])
    assert set(with_blank["by_grade"]) == {"A", "B"}


def test_residual_summary_ewma_and_cusum_drift_metrics():
    import json

    # Constant sequence: EWMA equals the constant value, bias triggers drift alert.
    const = remote_jobs.residual_summary(np.array([5.0, 5.0, 5.0]))
    assert const["ewma"] == pytest.approx(5.0)
    assert const["drift_alert"] is True
    assert const["drift_status"] == "DRIFT_ALERT"

    # Single zero observation: stable with zero EWMA and zero CUSUM.
    zero = remote_jobs.residual_summary(np.array([0.0]))
    assert zero["ewma"] == pytest.approx(0.0)
    assert zero["cusum_max"] == pytest.approx(0.0)
    assert zero["drift_alert"] is False
    assert zero["drift_status"] == "STABLE"

    # Known sequence [1, 2, 3, 4, 10] (lambda = 0.25) with non-finite filtering.
    clean = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    expected_ewma = float(pd.Series(clean).ewm(alpha=0.25).mean().iloc[-1])
    seq = remote_jobs.residual_summary(np.array([1.0, np.nan, 2.0, np.inf, 3.0, 4.0, 10.0]))
    assert seq["ewma"] == pytest.approx(expected_ewma)
    assert seq["cusum_max"] > 4.0
    assert seq["drift_alert"] is True
    assert seq["drift_status"] == "DRIFT_ALERT"
    json.dumps(seq)  # must remain JSON-serializable for modal_jobs.update_shadow

    # Order sensitivity: recent jump weighs more heavily in EWMA.
    late_jump = remote_jobs.residual_summary(np.array([0.0, 0.0, 0.0, 10.0]))
    early_jump = remote_jobs.residual_summary(np.array([10.0, 0.0, 0.0, 0.0]))
    assert late_jump["ewma"] > early_jump["ewma"]

    # Centered symmetric vector stays STABLE.
    balanced = remote_jobs.residual_summary(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
    assert balanced["bias"] == pytest.approx(0.0)
    assert balanced["cusum_max"] == pytest.approx(1.4)
    assert balanced["drift_alert"] is False
    assert balanced["drift_status"] == "STABLE"

    # Step shift where abs(bias) == 0.5 * (p90 - p10) (bias rule alone is False) -> CUSUM catches it.
    step_shift = remote_jobs.residual_summary(np.array([0.0] * 10 + [1.2] * 10))
    assert abs(step_shift["bias"]) == pytest.approx(0.5 * (step_shift["p90"] - step_shift["p10"]))
    assert step_shift["cusum_max"] == pytest.approx(7.0)
    assert step_shift["drift_alert"] is True
    assert step_shift["drift_status"] == "DRIFT_ALERT"

    # Downward step shift triggers via negative tabular CUSUM excursion.
    down_shift = remote_jobs.residual_summary(np.array([0.0] * 8 + [-2.0] * 8))
    assert down_shift["cusum_max"] > 4.0
    assert down_shift["drift_alert"] is True
    assert down_shift["drift_status"] == "DRIFT_ALERT"


def test_shadow_table_displays_ewma_drift_and_grade_preview_scope(monkeypatch):
    import streamlit as st

    calibration = {
        "TS_BN": remote_jobs.residual_summary(np.array([1.0, 2.0, 3.0, 4.0, 10.0])),
        "YS_BN": remote_jobs.residual_summary(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
    }
    table = shadow._calibration_table("HRC", calibration)
    assert "EWMA" in table.columns
    assert "Drift status" in table.columns
    by_key = table.set_index("Key")
    assert by_key.loc["TS_BN", "EWMA"] == pytest.approx(round(calibration["TS_BN"]["ewma"], 2))
    assert by_key.loc["TS_BN", "Drift status"] == "DRIFT_ALERT"
    assert by_key.loc["YS_BN", "EWMA"] == pytest.approx(round(calibration["YS_BN"]["ewma"], 2))
    assert by_key.loc["YS_BN", "EWMA"] > 0.0
    assert by_key.loc["YS_BN", "Drift status"] == "STABLE"

    # Legacy stored calibration without ewma/drift_status still renders cleanly.
    legacy_table = shadow._calibration_table(
        "HRC", {"TS_BN": {"bias": 1.5, "p10": -1.0, "p90": 4.0, "n": 3}}
    )
    assert legacy_table.iloc[0]["EWMA"] is None
    assert legacy_table.iloc[0]["Drift status"] == "—"

    by_grade = {
        "B": {"TS_BN": remote_jobs.residual_summary(np.array([12.0, 12.0, 12.0]))},
        "A": {"TS_BN": remote_jobs.residual_summary(np.array([-4.0, -4.0, -4.0]))},
    }
    grade_table = shadow._grade_table("HRC", by_grade)
    assert "EWMA" in grade_table.columns
    assert "Drift status" in grade_table.columns

    options = shadow._preview_scope_options(by_grade)
    assert options == ["Global route bias", "A", "B"]

    global_cal, global_lbl = shadow._resolve_preview_calibration(
        calibration, by_grade, "Global route bias"
    )
    assert global_cal is calibration and global_lbl == "global route"

    grade_cal, grade_lbl = shadow._resolve_preview_calibration(calibration, by_grade, "B")
    assert grade_cal is by_grade["B"] and grade_lbl == "grade B"

    scored = pd.DataFrame([{"TS_BN": 500.0, "YS_BN": 350.0, "EL_BN": 22.0}])
    rows = shadow._build_preview_rows(app.TARGET_LABEL["HRC"], scored, grade_cal)
    ts_row = next(r for r in rows if r["Property"] == app.TARGET_LABEL["HRC"]["TS_BN"][0])
    assert ts_row["Bias"] == pytest.approx(12.0)
    assert ts_row["Corrected (display only)"] == pytest.approx(512.0)

    # Render view_shadow with preview enabled and a specific grade chosen.
    monkeypatch.setattr(app, "score_frame", lambda mill, frame, engine=None: scored)
    monkeypatch.setattr(st, "checkbox", lambda *a, **kw: True)
    monkeypatch.setattr(st, "selectbox", lambda label, opts, **kw: "B")
    _clear_state()
    st.session_state["shadow_HRC"] = {
        "result": {
            "mill": "HRC",
            "engine": remote_jobs.DEFAULT_ENGINE,
            "timestamp": "2026-11-20T00:00:00+00:00",
            "n_rows": 6,
            "calibration": calibration,
            "by_grade": by_grade,
        },
        "source": "local",
        "engine": remote_jobs.DEFAULT_ENGINE,
    }
    try:
        shadow.view_shadow("HRC", defaults("HRC"))
    finally:
        _clear_state()


def test_modal_warm_models_is_idempotent_and_called_after_enter_repo(monkeypatch):
    pytest.importorskip("modal")
    import modal_app
    import modal_jobs

    for filename, mod in (("modal_app.py", modal_app), ("modal_jobs.py", modal_jobs)):
        path = os.path.join(REPO_ROOT, filename)
        with open(path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        assert hasattr(mod, "_WARMED")
        assert hasattr(mod, "_warm_models")

        checked_fns = 0
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            stmt_names = [
                stmt.value.func.id
                if isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Name)
                else None
                for stmt in node.body
            ]
            if "_enter_repo" in stmt_names:
                checked_fns += 1
                idx = stmt_names.index("_enter_repo")
                assert idx + 1 < len(stmt_names) and stmt_names[idx + 1] == "_warm_models", (
                    f"{filename}:{node.name} must call _warm_models() right after _enter_repo()"
                )
        assert checked_fns >= 4

        calls = []
        monkeypatch.setattr(app, "boosted_index", lambda mill: calls.append(mill) or {})
        monkeypatch.setattr(mod, "_WARMED", False)

        mod._enter_repo()
        mod._warm_models()
        mod._warm_models()
        assert calls == ["Plate", "HRC"]
        assert mod._WARMED is True

        calls.clear()
        mod._warm_models()
        assert calls == []


# --------------------------------------------------------------------- 3. remote fallback
def test_remote_ready_is_false_without_credentials(no_credentials):
    assert remote_jobs.remote_ready(refresh=True) is False
    assert remote_jobs.remote_ready() is False


@NEEDS_MODELS
def test_atlas_build_falls_back_to_local(no_credentials):
    """A 2x2x2x2 grid: no credentials means local compute, and the payload is still parquet."""
    mill = "Plate"
    payload, source = remote_jobs.build_atlas(mill, defaults(mill), atlas.AXES[mill],
                                              [2, 2, 2, 2])
    assert source == "local"
    assert payload and payload[:4] == b"PAR1", "the atlas must travel as parquet bytes"
    frame = pd.read_parquet(io.BytesIO(payload))
    assert len(frame) == 2 ** 4
    for key in atlas.AXES[mill]:
        assert key in frame.columns, f"axis {key} missing from the atlas"
    assert "TS_BN" in frame.columns

    again, load_source = remote_jobs.load_atlas(mill)
    assert load_source == "local" and again == payload, "the local build is replayable"


@NEEDS_MODELS
def test_shadow_update_falls_back_to_local(no_credentials):
    """A 3-row sheet: no credentials means local compute, and the dict is the calibration."""
    mill = "HRC"
    raw = defaults(mill)
    frame = pd.DataFrame([dict(raw) for _ in range(3)])
    frame["Heat_ID"] = ["H1", "H2", "H3"]
    frame["Grade"] = ["A", "A", "B"]
    frame["Measured_TS_BN"] = [420.0, 430.0, 440.0]
    frame["Measured_YS_BN"] = [300.0, 320.0, 340.0]
    payload = frame.to_csv(index=False).encode("utf-8")

    result, source = remote_jobs.update_shadow(mill, payload)
    assert source == "local"
    assert result is not None
    assert result["mill"] == mill and result["n_rows"] == 3
    assert "timestamp" in result
    assert set(result["calibration"]) == {"TS_BN", "YS_BN"}
    assert result["calibration"]["TS_BN"]["n"] == 3
    assert set(result["by_grade"]) == {"A", "B"}
    assert result["by_grade"]["A"]["TS_BN"]["n"] == 2

    stored, load_source = remote_jobs.load_shadow(mill)
    assert load_source == "local" and stored["n_rows"] == 3


def test_remote_jobs_never_imports_modal_at_module_level():
    """The public demo has to import this module on a machine with no Modal install at all."""
    path = os.path.join(APP_DIR, "remote_jobs.py")
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not any(alias.name.split(".")[0] == "modal" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] != "modal"


# ------------------------------------------------------------------- 4. bare-mode render
def _clear_state(prefixes=("atlas_", "shadow_")):
    import streamlit as st
    for key in [k for k in list(st.session_state.keys()) if str(k).startswith(prefixes)]:
        del st.session_state[key]


def test_views_render_in_bare_mode_with_empty_state():
    """Both tabs, both routes, no session state: they must say what to press and return."""
    for mill in MILLS:
        raw = defaults(mill)
        _clear_state()
        atlas.view_atlas(mill, raw)
        shadow.view_shadow(mill, raw)


@NEEDS_MODELS
def test_atlas_renders_a_stored_payload(no_credentials):
    """The render path (parse, slice, plot) must work off stored bytes, with no rebuild."""
    import streamlit as st
    mill = "HRC"
    payload, _source = remote_jobs.build_atlas(mill, defaults(mill), atlas.AXES[mill],
                                               [2, 2, 2, 2])
    assert payload
    _clear_state()
    st.session_state[f"atlas_{mill}"] = {
        "bytes": payload, "source": "local", "id": "test-atlas",
        "built": "2026-01-01 00:00", "axes": list(atlas.AXES[mill]),
        "grid_sizes": [2, 2, 2, 2], "engine": remote_jobs.DEFAULT_ENGINE, "mill": mill,
    }
    try:
        atlas.view_atlas(mill, defaults(mill))
    finally:
        _clear_state()


@NEEDS_MODELS
def test_shadow_renders_a_stored_calibration(no_credentials):
    """The render path must work off a stored dict, including the per-grade table."""
    import streamlit as st
    mill = "HRC"
    raw = defaults(mill)
    frame = pd.DataFrame([dict(raw) for _ in range(3)])
    frame["Heat_ID"] = ["H1", "H2", "H3"]
    frame["Grade"] = ["A", "A", "B"]
    frame["Measured_TS_BN"] = [420.0, 430.0, 440.0]
    result, _source = remote_jobs.update_shadow(mill, frame.to_csv(index=False).encode("utf-8"))
    assert result

    _clear_state()
    st.session_state[f"shadow_{mill}"] = {"result": result, "source": "local",
                                          "engine": remote_jobs.DEFAULT_ENGINE}
    try:
        shadow.view_shadow(mill, raw)
    finally:
        _clear_state()


def test_templates_carry_the_schedule_and_the_measured_columns():
    for mill in MILLS:
        frame = shadow.template_frame(mill)
        assert "Heat_ID" in frame.columns
        for row in app.SPECS[mill]:
            assert row[1] in frame.columns, f"{mill}: template is missing {row[1]}"
        for target in app.TARGET_LABEL[mill]:
            assert f"Measured_{target}" in frame.columns
        # the measured cells are blank: a pre-filled number would calibrate the model on itself
        assert frame.filter(like="Measured_").isna().all().all()


# ------------------------------------------------------------------ 5. tab registration
def test_tabs_register_atlas_and_shadow_before_about():
    source = textwrap.dedent(inspect.getsource(app.main))
    tree = ast.parse(source)
    labels = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "tabs":
            labels = ast.literal_eval(node.args[0])
    assert labels, "main() declares no st.tabs list"
    assert "Atlas" in labels and "Shadow" in labels
    assert labels[-1] == "About", f"About must stay the last tab, got {labels[-1]!r}"
    assert labels.index("Atlas") + 1 == labels.index("Shadow"), "Atlas and Shadow are adjacent"
    # The Pass schedule tab now sits between Shadow and About; the invariant these tabs
    # care about is that both still register before About.
    assert labels.index("Shadow") < labels.index("About")
    assert labels[labels.index("Shadow") + 1] == "Pass schedule"
    assert "atlas.view_atlas(" in source and "shadow.view_shadow(" in source
