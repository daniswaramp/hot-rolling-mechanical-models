"""
Configuration settings for Plate Sequential Rolling ML Pipeline.
Centralizes feature definitions, physical limits, targets, and model hyperparameters.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "csv", "plate_rolling_2026_cleaned.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

# Target specifications: target_key -> (csv_column_name, display_name, unit, subset_filter)
TARGETS = {
    "YS_BN": {
        "col": "YS_Before_N_MPa",
        "name": "Yield Strength (Before Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["YS_Before_N_MPa"].notna() & (df["YS_Before_N_MPa"] > 0)
    },
    "TS_BN": {
        "col": "TS_Before_N_MPa",
        "name": "Tensile Strength (Before Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["TS_Before_N_MPa"].notna() & (df["TS_Before_N_MPa"] > 0)
    },
    "EL_BN": {
        "col": "EL_Before_N_pct",
        "name": "Elongation (Before Normalizing)",
        "unit": "%",
        "filter": lambda df: df["EL_Before_N_pct"].notna() & (df["EL_Before_N_pct"] > 0)
    },
    "YS_AN": {
        "col": "YS_After_N_MPa",
        "name": "Yield Strength (After Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["YS_After_N_MPa"].notna() & (df["YS_After_N_MPa"] > 0)
    },
    "TS_AN": {
        "col": "TS_After_N_MPa",
        "name": "Tensile Strength (After Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["TS_After_N_MPa"].notna() & (df["TS_After_N_MPa"] > 0)
    },
    "EL_AN": {
        "col": "EL_After_N_pct",
        "name": "Elongation (After Normalizing)",
        "unit": "%",
        "filter": lambda df: df["EL_After_N_pct"].notna() & (df["EL_After_N_pct"] > 0)
    },
    "Impact_0C": {
        "col": "Impact_0C_E_Avg_J",
        "name": "Impact Energy at 0 °C",
        "unit": "J",
        "filter": lambda df: df["Impact_0C_E_Avg_J"].notna() & (df["Impact_0C_E_Avg_J"] > 0)
    },
    "Impact_m20C": {
        "col": "Impact_m20C_E_Avg_J",
        "name": "Impact Energy at -20 °C",
        "unit": "J",
        "filter": lambda df: df["Impact_m20C_E_Avg_J"].notna() & (df["Impact_m20C_E_Avg_J"] > 0)
    },
    "Impact_m40C": {
        "col": "Impact_m40C_E_Avg_J",
        "name": "Impact Energy at -40 °C",
        "unit": "J",
        "filter": lambda df: df["Impact_m40C_E_Avg_J"].notna() & (df["Impact_m40C_E_Avg_J"] > 0)
    },
    "Z_Test": {
        "col": "Z_Test_Avg_pct",
        "name": "Through-Thickness Reduction of Area (Z-Test)",
        "unit": "%",
        "filter": lambda df: df["Z_Test_Avg_pct"].notna() & (df["Z_Test_Avg_pct"] > 0)
    }
}

# Raw input features requested by user
RAW_INPUT_COLS = [
    "Slab_T_mm",
    "Chem_C", "Chem_Mn", "Chem_Si", "Chem_Al", "Chem_S", "Chem_P",
    "Chem_Nb", "Chem_V", "Chem_Mo", "Chem_Ti", "Chem_Cu", "Chem_Ni",
    "Chem_Cr", "Chem_N", "Chem_CE",
    "Plate_T_mm", "Plate_W_mm",
    "FM_Temp_Before_Rolling", "FM_Finish_Temp",
    "TB_Ratio",
    "ACC_Laminar_Header", "ACC_Temp_Post_Laminar",
    "FM_Last_Pass_Drafting",
    "Rolling_Type"
]

# Physical bounds for validation and outlier detection
PHYSICAL_BOUNDS = {
    "Slab_T_mm": (100.0, 350.0),
    "Plate_T_mm": (4.0, 160.0),
    "Plate_W_mm": (800.0, 4500.0),
    "FM_Temp_Before_Rolling": (750.0, 1150.0),
    "FM_Finish_Temp": (680.0, 1050.0),
    "ACC_Temp_Post_Laminar": (400.0, 950.0),
    "TB_Ratio": (1.05, 6.0),
    "FM_Last_Pass_Drafting": (0.1, 15.0),
    "Chem_C": (0.01, 0.50),
    "Chem_Mn": (0.10, 2.50),
    "Chem_Si": (0.01, 1.00),
    "Chem_CE": (0.10, 0.70),
    "YS_Before_N_MPa": (180.0, 850.0),
    "TS_Before_N_MPa": (280.0, 950.0),
    "EL_Before_N_pct": (10.0, 55.0),
    "YS_After_N_MPa": (180.0, 800.0),
    "TS_After_N_MPa": (280.0, 900.0),
    "EL_After_N_pct": (10.0, 55.0),
    "Impact_0C_E_Avg_J": (5.0, 450.0),
    "Impact_m20C_E_Avg_J": (5.0, 450.0),
    "Impact_m40C_E_Avg_J": (5.0, 450.0),
    "Z_Test_Avg_pct": (15.0, 95.0)
}

# Grouping identifier to prevent data leakage across splits
GROUP_COL = "Heat_ID"
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
BOOTSTRAP_ITERATIONS = 100
