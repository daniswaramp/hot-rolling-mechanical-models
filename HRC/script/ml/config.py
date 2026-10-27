"""
Configuration settings for Steckel Mill (HRC) ML Pipeline.
Centralizes feature definitions, physical limits, targets, and model hyperparameters.
"""

import os

HRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.join(HRC_DIR, "script")
DEFAULT_CSV_PATH = os.path.join(HRC_DIR, "csv", "steckel_mill_cleaned.csv")
MODELS_DIR = os.path.join(HRC_DIR, "models")
REPORTS_DIR = os.path.join(HRC_DIR, "reports")
NOTEBOOKS_DIR = os.path.join(HRC_DIR, "notebooks")

TARGETS = {
    "YS_BN": {
        "col": "YS_BN_MPa",
        "name": "Yield Strength (Before Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["YS_BN_MPa"].notna() & (df["YS_BN_MPa"] > 0)
    },
    "TS_BN": {
        "col": "TS_BN_MPa",
        "name": "Tensile Strength (Before Normalizing)",
        "unit": "MPa",
        "filter": lambda df: df["TS_BN_MPa"].notna() & (df["TS_BN_MPa"] > 0)
    },
    "EL_BN": {
        "col": "EL_BN_pct",
        "name": "Elongation (Before Normalizing)",
        "unit": "%",
        "filter": lambda df: df["EL_BN_pct"].notna() & (df["EL_BN_pct"] > 0)
    }
}

PHYSICAL_BOUNDS = {
    "YS_BN_MPa": (180.0, 750.0),
    "TS_BN_MPa": (280.0, 950.0),
    "EL_BN_pct": (10.0, 55.0),
    "Coil_T_mm": (2.0, 25.0),
    "Coil_W_mm": (800.0, 2200.0),
    "TB_Thick_mm": (15.0, 60.0),
    "Temp_Transfer_Bar_C": (750.0, 1150.0),
    "Temp_FT_Avg_C": (700.0, 1000.0),
    "Temp_CT_Top_C": (400.0, 750.0),
    "Temp_CT_Bot_C": (400.0, 750.0),
    "Chem_C": (0.01, 0.45),
    "Chem_Mn": (0.10, 2.20),
    "Chem_Si": (0.005, 0.80),
    "Chem_P": (0.001, 0.05),
    "Chem_S": (0.0005, 0.04),
    "Chem_Nb": (0.0, 0.12),
    "Chem_V": (0.0, 0.15),
    "Chem_Ti": (0.0, 0.10),
    "Chem_Al": (0.005, 0.10),
    "Chem_N": (0.001, 0.02)
}

SPLIT_SETTINGS = {
    "test_size": 0.20,
    "group_col": "Heat_ID",
    "random_state": 42
}

N_BOOTSTRAP_ITERATIONS = 100
RANDOM_SEED = 42
