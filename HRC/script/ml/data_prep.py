"""
Data Preparation and Cleaning Module for Steckel Mill (HRC).
Implements:
1. Automated typo fixing (decimal shifts: e.g. YS 38028 -> 380.28)
2. Physical consistency checks (YS < TS, Coil_T < TB_Thick, CT < FT)
3. Zero-leakage Heat_ID GroupSplit
4. Cleaning audit log export
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger("DataPrep_HRC")


def fix_decimal_shift(val, expected_min, expected_max):
    """Detects and fixes decimal shift typos (e.g. 38028 -> 380.28)."""
    if pd.isna(val) or val <= 0:
        return val

    if expected_min <= val <= expected_max:
        return val

    # Try dividing by 10, 100, 1000
    for factor in [10.0, 100.0, 1000.0]:
        candidate = val / factor
        if expected_min <= candidate <= expected_max:
            return candidate

    return val


def clean_hrc_data(df: pd.DataFrame, physical_bounds: dict, reports_dir: str = None) -> pd.DataFrame:
    """
    Cleans raw HRC dataset, corrects typos, applies physical consistency filters,
    and records an audit log.
    """
    cleaned_df = df.copy()
    cleaning_records = []
    initial_rows = len(cleaned_df)

    # 1. Decimal shift typo correction on Tensile & Dimension properties
    for col in ["YS_BN_MPa", "TS_BN_MPa", "EL_BN_pct", "Coil_T_mm", "TB_Thick_mm"]:
        if col in cleaned_df.columns and col in physical_bounds:
            low, high = physical_bounds[col]
            original = cleaned_df[col].copy()
            cleaned_df[col] = cleaned_df[col].apply(lambda v: fix_decimal_shift(v, low, high))
            corrected_mask = (original != cleaned_df[col]) & original.notna()
            n_corr = corrected_mask.sum()
            if n_corr > 0:
                logger.info(f"Corrected {n_corr} decimal-shift typos in column: {col}")
                for idx in cleaned_df[corrected_mask].index[:10]:
                    cleaning_records.append({
                        "index": idx,
                        "coil_no": cleaned_df.loc[idx, "Coil_No"] if "Coil_No" in cleaned_df.columns else None,
                        "column": col,
                        "original_value": original.loc[idx],
                        "cleaned_value": cleaned_df.loc[idx, col],
                        "rule": "Decimal shift typo correction"
                    })

    # 2. Temperature cleaning (ensure CT < FT)
    for t_col in ["Temp_Transfer_Bar_C", "Temp_FT_Avg_C", "Temp_CT_Top_C", "Temp_CT_Bot_C"]:
        if t_col in cleaned_df.columns and t_col in physical_bounds:
            low, high = physical_bounds[t_col]
            out_of_bounds = (cleaned_df[t_col] < low) | (cleaned_df[t_col] > high)
            cleaned_df.loc[out_of_bounds, t_col] = np.nan

    # Average Coiling Temperature
    if "Temp_CT_Top_C" in cleaned_df.columns and "Temp_CT_Bot_C" in cleaned_df.columns:
        cleaned_df["Temp_CT_Avg_C"] = cleaned_df[["Temp_CT_Top_C", "Temp_CT_Bot_C"]].mean(axis=1)

    # 3. Physical consistency filter: YS < TS
    if "YS_BN_MPa" in cleaned_df.columns and "TS_BN_MPa" in cleaned_df.columns:
        inconsistent_strength = (cleaned_df["YS_BN_MPa"] >= cleaned_df["TS_BN_MPa"]) & \
                                cleaned_df["YS_BN_MPa"].notna() & cleaned_df["TS_BN_MPa"].notna()
        n_inconsistent = inconsistent_strength.sum()
        if n_inconsistent > 0:
            logger.warning(f"Found {n_inconsistent} records with YS >= TS. Nullifying target properties.")
            cleaned_df.loc[inconsistent_strength, ["YS_BN_MPa", "TS_BN_MPa"]] = np.nan

    # 4. Physical consistency: Coil_T_mm < TB_Thick_mm
    if "Coil_T_mm" in cleaned_df.columns and "TB_Thick_mm" in cleaned_df.columns:
        invalid_reduction = (cleaned_df["Coil_T_mm"] >= cleaned_df["TB_Thick_mm"]) & \
                            cleaned_df["Coil_T_mm"].notna() & cleaned_df["TB_Thick_mm"].notna()
        cleaned_df.loc[invalid_reduction, "TB_Thick_mm"] = np.nan

    # Export audit log
    if reports_dir and cleaning_records:
        os.makedirs(reports_dir, exist_ok=True)
        log_path = os.path.join(reports_dir, "hrc_cleaning_log.csv")
        pd.DataFrame(cleaning_records).to_csv(log_path, index=False, encoding='utf-8')
        logger.info(f"Saved cleaning audit log to: {log_path}")

    logger.info(f"Cleaning complete: {len(cleaned_df):,} records processed.")
    return cleaned_df


def prepare_group_split(df: pd.DataFrame, target_col: str, group_col: str = "Heat_ID",
                        test_size: float = 0.20, random_state: int = 42):
    """
    Filters rows where target_col is valid, and performs a zero-leakage GroupShuffleSplit
    so coils from the same melt heat are never shared between train and validation.
    """
    valid_mask = df[target_col].notna() & (df[target_col] > 0)
    subset = df[valid_mask].copy().reset_index(drop=True)

    if group_col not in subset.columns or subset[group_col].isna().all():
        logger.warning(f"Group column '{group_col}' missing or invalid; falling back to Coil_No.")
        group_col = "Coil_No"

    # Fill any missing groups with unique row IDs as Series
    groups = subset[group_col].fillna(pd.Series(subset.index.astype(str), index=subset.index))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(subset, groups=groups))

    train_df = subset.iloc[train_idx].copy().reset_index(drop=True)
    val_df = subset.iloc[val_idx].copy().reset_index(drop=True)

    logger.info(f"Target [{target_col}]: Total={len(subset):,}, Train={len(train_df):,} ({len(train_df[group_col].unique())} heats), "
                f"Val={len(val_df):,} ({len(val_df[group_col].unique())} heats)")

    return train_df, val_df
