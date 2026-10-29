"""
Zero Data Leakage Validation Tests.
Verifies that Heat_ID group splitting eliminates all sister-plate and sister-coil melt leakage.
"""

import os
import sys
import pytest
import pandas as pd

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
PLATE_DIR = os.path.join(REPO_ROOT, "Plate")
HRC_DIR = os.path.join(REPO_ROOT, "HRC")

for p in [PLATE_DIR, os.path.join(HRC_DIR, "script")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def test_plate_zero_heat_leakage():
    """Verifies that Plate GroupShuffleSplit has exactly zero overlapping heats between train and val."""
    from script.ml.data_prep import prepare_target_dataset, split_grouped_data
    from script.ml.features import compute_metallurgical_features

    plate_csv = os.path.join(PLATE_DIR, "csv", "plate_rolling_2026_cleaned.csv")
    if not os.path.exists(plate_csv):
        plate_csv = os.path.join(REPO_ROOT, "..", "Plate", "csv", "plate_rolling_2026_cleaned.csv")
    if not os.path.exists(plate_csv):
        pytest.skip("Plate cleaned CSV not present in local environment (lightweight repo clone)")
    df = pd.read_csv(plate_csv, nrows=3000, low_memory=False)

    X_feats = compute_metallurgical_features(df)
    X_clean, y_clean, groups_clean = prepare_target_dataset(df, "TS_BN", feature_df=X_feats)
    X_train, X_val, y_train, y_val, g_train, g_val = split_grouped_data(
        X_clean, y_clean, groups_clean, test_size=0.2, random_state=42
    )

    train_heats = set(g_train.unique())
    val_heats = set(g_val.unique())
    overlap = train_heats.intersection(val_heats)

    assert len(overlap) == 0, f"Critical Melt Leakage: {len(overlap)} heats overlap between Plate train and val!"
    assert len(train_heats) > 0
    assert len(val_heats) > 0


def test_hrc_zero_heat_leakage():
    """Verifies that HRC GroupShuffleSplit has exactly zero overlapping heats between train and val."""
    from ml.data_prep import prepare_group_split

    hrc_csv = os.path.join(HRC_DIR, "csv", "steckel_mill_cleaned.csv")
    if not os.path.exists(hrc_csv):
        hrc_csv = os.path.join(REPO_ROOT, "..", "HRC", "csv", "steckel_mill_cleaned.csv")
    if not os.path.exists(hrc_csv):
        pytest.skip("HRC cleaned CSV not present in local environment (lightweight repo clone)")
    df = pd.read_csv(hrc_csv, nrows=3000, low_memory=False)

    train_sub, val_sub = prepare_group_split(df, "TS_BN_MPa", group_col="Heat_ID", test_size=0.2, random_state=42)

    train_heats = set(train_sub["Heat_ID"].dropna().unique())
    val_heats = set(val_sub["Heat_ID"].dropna().unique())
    overlap = train_heats.intersection(val_heats)

    assert len(overlap) == 0, f"Critical Melt Leakage: {len(overlap)} heats overlap between HRC train and val!"
    assert len(train_heats) > 0
    assert len(val_heats) > 0
