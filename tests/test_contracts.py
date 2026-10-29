"""
Contract Tests: Verifies model artifacts, schema consistency, and registry completeness.
"""

import os
import sys
import json
import pytest
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")
PLATE_DIR = os.path.join(REPO_ROOT, "Plate")
HRC_DIR = os.path.join(REPO_ROOT, "HRC")

for p in [REPO_ROOT, APP_DIR, PLATE_DIR, os.path.join(HRC_DIR, "script")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import model_loader as ml


def test_target_config_contracts():
    """Verifies that TARGET_CONFIG defines all expected targets with units and tolerances."""
    for mill in ["Plate", "HRC"]:
        assert mill in ml.TARGET_CONFIG, f"Missing config for {mill}"
        cfg = ml.TARGET_CONFIG[mill]
        assert len(cfg) >= 3, f"{mill} has fewer than 3 targets"
        for t_key, meta in cfg.items():
            assert "name" in meta, f"Missing name for {mill}|{t_key}"
            assert "unit" in meta, f"Missing unit for {mill}|{t_key}"
            assert "tol" in meta, f"Missing tol for {mill}|{t_key}"
            assert meta["tol"] > 0, f"Non-positive tolerance for {mill}|{t_key}"


def test_boosted_models_exist():
    """Verifies that all native boosted models in boosted_metrics.json physically exist on disk."""
    json_path = os.path.join(REPO_ROOT, "models", "boosted_out", "boosted_metrics.json")
    assert os.path.exists(json_path), "boosted_metrics.json not found"
    
    with open(json_path, "r") as f:
        metrics = json.load(f)

    ext_map = {"xgb": "xgb.ubj", "lgbm": "lgbm.txt", "cat": "cat.cbm"}
    for m_key, info in metrics.items():
        mill, target, fam = m_key.split("|")
        fname = f"{mill}_{target}_{ext_map[fam]}"
        fpath = os.path.join(REPO_ROOT, "models", "boosted_out", fname)
        assert os.path.exists(fpath), f"Champion model file missing: {fname}"
        assert os.path.getsize(fpath) > 1000, f"Model file too small / corrupt: {fname}"


def test_conformal_reports_exist():
    """Verifies that conformal_intervals.csv exists and contains required targets for both mills."""
    for mill in ["Plate", "HRC"]:
        conf_path = os.path.join(REPO_ROOT, mill, "reports", "conformal_intervals.csv")
        assert os.path.exists(conf_path), f"Conformal report missing for {mill}"
        df = pd.read_csv(conf_path)
        assert not df.empty, f"Conformal report empty for {mill}"
        assert "Target" in df.columns
        assert "Conformal_Half_Width_95" in df.columns
        assert "Coverage_95_Pct" in df.columns
        assert "Old_Naive_Coverage_Pct" in df.columns

        targets = df["Target"].tolist()
        assert "YS_BN" in targets
        assert "TS_BN" in targets
        assert "EL_BN" in targets


def test_applicability_domain_json():
    """Verifies that applicability_domain.json exists and defines P01, P99 bounds."""
    p = os.path.join(REPO_ROOT, "models", "applicability_domain.json")
    assert os.path.exists(p), "applicability_domain.json missing"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "Plate" in data
    assert "HRC" in data
    assert len(data["Plate"]) >= 15
    assert len(data["HRC"]) >= 15
    for mill in ["Plate", "HRC"]:
        for param, bounds in data[mill].items():
            assert "p01" in bounds
            assert "p99" in bounds
            assert bounds["p01"] <= bounds["p99"], f"Invalid quantile order in {param}"
