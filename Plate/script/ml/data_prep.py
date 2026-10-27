"""
Data Preparation, Cleaning, and Outlier Correction Module.
Handles physical boundary checks, decimal-typo corrections, laminar header parsing,
and GroupKFold/GroupShuffleSplit by Heat_ID to eliminate data leakage.
"""

import re
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from script.ml.config import PHYSICAL_BOUNDS, RAW_INPUT_COLS, GROUP_COL, TEST_SIZE, RANDOM_STATE

logger = logging.getLogger("DataPrep")


def parse_laminar_header(val, temp_post):
    """
    Parses raw laminar header text into an integer count of active headers.
    Detects column swap where temperature was accidentally typed into the header column.
    """
    if pd.isna(val):
        return 0, 0.0, temp_post

    val_str = str(val).strip().upper()
    if val_str in ("", "NONE", "NAN", "-"):
        return 0, 0.0, temp_post

    # Check if a temperature (e.g. 700-900) was erroneously entered in header column
    try:
        num_val = float(val_str)
        if 500.0 <= num_val <= 1050.0:
            # Shifted temperature value
            new_temp = temp_post if pd.notna(temp_post) else num_val
            return 0, 0.0, new_temp
        elif 1.0 <= num_val <= 20.0:
            return 1, num_val, temp_post
    except ValueError:
        pass

    # Pattern like '2H', '1H', 'H2', '3H'
    m = re.search(r"(\d+)\s*H|H\s*(\d+)", val_str)
    if m:
        headers_n = float(m.group(1) or m.group(2))
        return 1, headers_n, temp_post

    return 0, 0.0, temp_post


def fix_decimal_typo(val, min_val, max_val):
    """
    Detects and fixes decimal point omission typos (e.g., 41205 -> 412.05).
    """
    if pd.isna(val):
        return val, False

    # Check as-is
    if min_val <= val <= max_val:
        return val, False

    # Check divided by 100
    if min_val <= val / 100.0 <= max_val:
        return val / 100.0, True

    # Check divided by 10
    if min_val <= val / 10.0 <= max_val:
        return val / 10.0, True

    # Check divided by 1000
    if min_val <= val / 1000.0 <= max_val:
        return val / 1000.0, True

    # Cannot fix safely
    return np.nan, True


def clean_plate_rolling_data(df_raw):
    """
    Applies comprehensive domain cleaning rules:
    - Parses Laminar Header & detects swapped columns
    - Corrects decimal typos in physical properties
    - Enforces physical bounds and checks consistency
    - Generates a change log DataFrame
    """
    df = df_raw.copy()
    change_log = []

    # 1. Parse Laminar Headers and Temperatures
    headers_active = []
    headers_count = []
    updated_temps = []

    for idx, row in df.iterrows():
        raw_hdr = row.get("ACC_Laminar_Header")
        raw_tmp = row.get("ACC_Temp_Post_Laminar")
        act, cnt, tmp = parse_laminar_header(raw_hdr, raw_tmp)

        # If temperature post laminar is missing and laminar is not active,
        # plate is air-cooled; set temp post laminar equal to FM finish temp
        if pd.isna(tmp) and act == 0:
            tmp = row.get("FM_Finish_Temp", np.nan)

        headers_active.append(act)
        headers_count.append(cnt)
        updated_temps.append(tmp)

    df["ACC_Active"] = headers_active
    df["ACC_Headers_n"] = headers_count
    df["ACC_Temp_Post_Laminar"] = updated_temps

    # 2. Check and correct physical numerical columns
    for col, (min_b, max_b) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue

        fixed_vals = []
        for idx, val in df[col].items():
            if pd.isna(val):
                fixed_vals.append(np.nan)
                continue

            try:
                num_v = float(val)
            except (ValueError, TypeError):
                fixed_vals.append(np.nan)
                change_log.append({
                    "row_index": idx,
                    "batch_id": df.at[idx, "Batch_ID"] if "Batch_ID" in df.columns else None,
                    "column": col,
                    "old_value": val,
                    "new_value": np.nan,
                    "reason": "Non-numeric string"
                })
                continue

            fixed_v, changed = fix_decimal_typo(num_v, min_b, max_b)
            if changed:
                change_log.append({
                    "row_index": idx,
                    "batch_id": df.at[idx, "Batch_ID"] if "Batch_ID" in df.columns else None,
                    "column": col,
                    "old_value": val,
                    "new_value": fixed_v,
                    "reason": "Decimal typo correction / Out of bounds"
                })
            fixed_vals.append(fixed_v)

        df[col] = fixed_vals

    # 3. Physical consistency checks
    # Tensile Before N: YS < TS
    mask_bn = df["YS_Before_N_MPa"].notna() & df["TS_Before_N_MPa"].notna()
    invalid_bn = mask_bn & ((df["YS_Before_N_MPa"] >= df["TS_Before_N_MPa"]) | (df["YS_Before_N_MPa"] / df["TS_Before_N_MPa"] < 0.45))
    if invalid_bn.any():
        for idx in df[invalid_bn].index:
            change_log.append({
                "row_index": idx,
                "batch_id": df.at[idx, "Batch_ID"],
                "column": "YS_Before_N_MPa",
                "old_value": df.at[idx, "YS_Before_N_MPa"],
                "new_value": np.nan,
                "reason": f"Inconsistent with TS (YS >= TS or ratio < 0.45)"
            })
            df.at[idx, "YS_Before_N_MPa"] = np.nan

    # Tensile After N: YS < TS
    mask_an = df["YS_After_N_MPa"].notna() & df["TS_After_N_MPa"].notna()
    invalid_an = mask_an & ((df["YS_After_N_MPa"] >= df["TS_After_N_MPa"]) | (df["YS_After_N_MPa"] / df["TS_After_N_MPa"] < 0.45))
    if invalid_an.any():
        for idx in df[invalid_an].index:
            change_log.append({
                "row_index": idx,
                "batch_id": df.at[idx, "Batch_ID"],
                "column": "YS_After_N_MPa",
                "old_value": df.at[idx, "YS_After_N_MPa"],
                "new_value": np.nan,
                "reason": f"Inconsistent with TS (YS >= TS or ratio < 0.45)"
            })
            df.at[idx, "YS_After_N_MPa"] = np.nan

    # Plate thickness < Slab thickness
    thick_check = (df["Plate_T_mm"] >= df["Slab_T_mm"]) & df["Plate_T_mm"].notna() & df["Slab_T_mm"].notna()
    if thick_check.any():
        df.loc[thick_check, "Plate_T_mm"] = np.nan

    df_log = pd.DataFrame(change_log)
    logger.info(f"Cleaned dataset: {len(change_log)} value corrections logged.")
    return df, df_log


def prepare_target_dataset(df_cleaned, target_key, feature_df=None):
    """
    Subsets the dataset for a specific target, removes missing values,
    and returns features (X), target (y), and grouping IDs (groups).
    """
    from script.ml.config import TARGETS

    if target_key not in TARGETS:
        raise ValueError(f"Unknown target key: {target_key}")

    target_cfg = TARGETS[target_key]
    target_col = target_cfg["col"]
    filter_fn = target_cfg["filter"]

    # Filter valid target records
    valid_mask = filter_fn(df_cleaned)
    df_sub = df_cleaned[valid_mask].copy()

    if feature_df is not None:
        X_sub = feature_df.loc[df_sub.index].copy()
    else:
        # Default raw input columns
        avail_cols = [c for c in RAW_INPUT_COLS if c in df_sub.columns]
        X_sub = df_sub[avail_cols].copy()

    y_sub = df_sub[target_col].copy()
    groups_sub = df_sub[GROUP_COL].fillna("UNKNOWN_HEAT").copy()

    # Drop any rows where all features are NaN or y is NaN
    non_null_mask = y_sub.notna() & X_sub.notna().any(axis=1)
    X_clean = X_sub[non_null_mask].copy()
    y_clean = y_sub[non_null_mask].copy()
    groups_clean = groups_sub[non_null_mask].copy()

    return X_clean, y_clean, groups_clean


def split_grouped_data(X, y, groups, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Performs GroupShuffleSplit by Heat_ID so sister plates from the same ladle heat
    never appear in both train and validation sets.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
    groups_train, groups_val = groups.iloc[train_idx], groups.iloc[val_idx]

    # Verify zero group leakage
    train_heats = set(groups_train.unique())
    val_heats = set(groups_val.unique())
    leakage = train_heats.intersection(val_heats)
    assert len(leakage) == 0, f"Critical Data Leakage: {len(leakage)} heats overlap between train and val!"

    logger.info(f"Group split complete: Train={len(X_train)} samples ({len(train_heats)} heats), "
                f"Val={len(X_val)} samples ({len(val_heats)} heats). Zero group leakage.")

    return X_train, X_val, y_train, y_val, groups_train, groups_val
