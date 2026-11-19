"""
Pure-helper tests for the response-surface overlays (app/streamlit_app.py, 2D contour branch).

No Streamlit rendering: the spec iso-line value resolver (iso_line_value) and the
out-of-envelope grid mask (surface_ood_mask) are exercised directly, against the real GRP
workbook and the real models/applicability_domain.json envelope.
"""

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

import physics  # noqa: E402
import spec_checker as sc  # noqa: E402
import explainer  # noqa: E402
import streamlit_app as sa  # noqa: E402


def _plate_default_schedule():
    values = {row[1]: row[6] for row in sa.PLATE_SPEC}
    values["Rolling_Type"] = "AR"
    values["ACC_Temp_Post_Laminar"] = physics.post_laminar_temp(
        values["FM_Finish_Temp"], values["ACC_Laminar_Header"], values["Plate_T_mm"])
    return values


def _a572_50_label():
    labels, _lookup = sc.spec_grade_options()
    for wanted in (lambda l: "572" in l and "50" in l and "ASTM" in l,
                   lambda l: "572" in l and "50" in l):
        label = next((l for l in labels if wanted(l)), None)
        if label is not None:
            return label
    return None


def test_iso_line_value_a572_50_tensile():
    """A 572 Gr 50 tier: the TS_BN iso-line value is finite and equals the workbook's 450 MPa."""
    label = _a572_50_label()
    if label is None:
        pytest.skip("no A 572 50 tier in the shipped workbook")
    val = sa.iso_line_value("Plate", label, "TS_BN", 16.0)
    assert val is not None and np.isfinite(val)
    assert val == pytest.approx(450.0), f"{label} TS_BN min came out {val}"


def test_iso_line_value_skips_max_and_missing_requirements():
    """Min-type wins over max-type; targets without a requirement resolve to None."""
    labels, _lookup = sc.spec_grade_options()
    label = _a572_50_label()
    if label is not None:
        # the tier states no Z_Test (through-thickness RA) limit -> no iso-line
        assert sa.iso_line_value("Plate", label, "Z_Test", 16.0) is None
    # JIS G 3101 SS 400 carries BOTH a TS min (400) and a TS max ceiling (510):
    # the min-type floor must win for the iso-line overlay.
    ss400 = next((l for l in labels if "3101" in l and "400" in l), None)
    if ss400 is not None:
        val = sa.iso_line_value("HRC", ss400, "TS_BN", 6.0)
        assert val is not None and val == pytest.approx(400.0), \
            "min-type requirement must win over the max-type ceiling"
    # unknown tier label -> None, never an exception
    assert sa.iso_line_value("Plate", "definitely not a tier", "TS_BN", 16.0) is None


def test_surface_ood_mask_default_false_extreme_true():
    """Default schedule sweeps stay inside the P01-P99 envelope; extreme values do not."""
    raw = _plate_default_schedule()
    domain = explainer.get_applicability_domain().get("Plate", {})
    key = "FM_Finish_Temp"
    if key not in domain or "p01" not in domain[key]:
        pytest.skip("no P01-P99 envelope entry for FM_Finish_Temp")
    p01, p99 = float(domain[key]["p01"]), float(domain[key]["p99"])

    # default schedule + default-ish sweep: everything inside the envelope
    m_in = sa.surface_ood_mask("Plate", np.array([float(raw[key])]),
                               np.array([float(raw["ACC_Temp_Post_Laminar"])]),
                               key, "ACC_Temp_Post_Laminar", raw)
    assert m_in.shape == (1, 1)
    assert not m_in.any(), "default schedule must sit inside the historical envelope"

    # sweep well below P01: every grid point flagged
    m_out = sa.surface_ood_mask("Plate", np.array([p01 - 50.0, p01 - 60.0]),
                                np.array([float(raw["ACC_Temp_Post_Laminar"])]),
                                key, "ACC_Temp_Post_Laminar", raw)
    assert m_out.shape == (2, 1)
    assert m_out.all(), "values far below P01 must be flagged out-of-envelope"

    # a value above P99 flags too
    m_hi = sa.surface_ood_mask("Plate", np.array([p99 + 50.0]),
                               np.array([float(raw["ACC_Temp_Post_Laminar"])]),
                               key, "ACC_Temp_Post_Laminar", raw)
    assert m_hi.all()
