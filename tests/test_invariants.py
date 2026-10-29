"""
Domain Metallurgy Invariant Tests.
Verifies physical thermodynamic softening coupling between Before Normalizing (BN)
and After Normalizing (AN) Heavy Plates across multiple industrial steel grades.
"""

import os
import sys
import pytest
import numpy as np

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)
APP_DIR = os.path.join(REPO_ROOT, "app")
PLATE_DIR = os.path.join(REPO_ROOT, "Plate")

for p in [REPO_ROOT, APP_DIR, PLATE_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import model_loader as ml
import streamlit_app as app


INDUSTRIAL_TEST_CASES = [
    {
        "name": "SS400 / A36 (Structural Mild Steel, 16mm)",
        "inputs": {
            "Plate_T_mm": 16.0, "Plate_W_mm": 2000.0, "Slab_T_mm": 200.0,
            "TB_Thickness_mm": 40.0, "FM_Last_Pass_Thick": 19.0,
            "Chem_C": 0.16, "Chem_Mn": 0.85, "Chem_Si": 0.22, "Chem_P": 0.015, "Chem_S": 0.005,
            "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.005,
            "Chem_Nb": 0.002, "Chem_Ti": 0.002, "Chem_Al": 0.025, "Chem_N": 0.003, "Chem_B": 0.0001,
            "FM_Finish_Temp": 840.0, "ACC_Temp_Post_Laminar": 840.0, "ACC_Laminar_Header": 0.0,
            "FM_Temp_Before_Rolling": 985.0
        }
    },
    {
        "name": "SM490B (Normalized Structural Grade, 25mm)",
        "inputs": {
            "Plate_T_mm": 25.0, "Plate_W_mm": 2200.0, "Slab_T_mm": 220.0,
            "TB_Thickness_mm": 55.0, "FM_Last_Pass_Thick": 30.0,
            "Chem_C": 0.15, "Chem_Mn": 1.35, "Chem_Si": 0.28, "Chem_P": 0.012, "Chem_S": 0.003,
            "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.005,
            "Chem_Nb": 0.018, "Chem_Ti": 0.002, "Chem_Al": 0.030, "Chem_N": 0.003, "Chem_B": 0.0001,
            "FM_Finish_Temp": 820.0, "ACC_Temp_Post_Laminar": 620.0, "ACC_Laminar_Header": 4.0,
            "FM_Temp_Before_Rolling": 980.0
        }
    },
    {
        "name": "TMCP High-Cooling Plate (ACC 8H, 12mm)",
        "inputs": {
            "Plate_T_mm": 12.0, "Plate_W_mm": 1800.0, "Slab_T_mm": 200.0,
            "TB_Thickness_mm": 35.0, "FM_Last_Pass_Thick": 15.0,
            "Chem_C": 0.13, "Chem_Mn": 1.45, "Chem_Si": 0.25, "Chem_P": 0.010, "Chem_S": 0.002,
            "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.020,
            "Chem_Nb": 0.025, "Chem_Ti": 0.015, "Chem_Al": 0.035, "Chem_N": 0.003, "Chem_B": 0.0001,
            "FM_Finish_Temp": 810.0, "ACC_Temp_Post_Laminar": 540.0, "ACC_Laminar_Header": 8.0,
            "FM_Temp_Before_Rolling": 960.0
        }
    },
    {
        "name": "SA 516 Gr 70 (Boiler Pressure Vessel Grade, 32mm)",
        "inputs": {
            "Plate_T_mm": 32.0, "Plate_W_mm": 2500.0, "Slab_T_mm": 240.0,
            "TB_Thickness_mm": 75.0, "FM_Last_Pass_Thick": 38.0,
            "Chem_C": 0.22, "Chem_Mn": 1.15, "Chem_Si": 0.30, "Chem_P": 0.012, "Chem_S": 0.003,
            "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.005,
            "Chem_Nb": 0.005, "Chem_Ti": 0.002, "Chem_Al": 0.025, "Chem_N": 0.003, "Chem_B": 0.0001,
            "FM_Finish_Temp": 850.0, "ACC_Temp_Post_Laminar": 850.0, "ACC_Laminar_Header": 0.0,
            "FM_Temp_Before_Rolling": 990.0
        }
    },
    {
        "name": "S355 Heavy Structural (Air-Cooled, 70mm)",
        "inputs": {
            "Plate_T_mm": 70.0, "Plate_W_mm": 2200.0, "Slab_T_mm": 240.0,
            "TB_Thickness_mm": 130.0, "FM_Last_Pass_Thick": 80.0,
            "Chem_C": 0.17, "Chem_Mn": 1.40, "Chem_Si": 0.35, "Chem_P": 0.015, "Chem_S": 0.004,
            "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.010,
            "Chem_Nb": 0.020, "Chem_Ti": 0.002, "Chem_Al": 0.025, "Chem_N": 0.003, "Chem_B": 0.0001,
            "FM_Finish_Temp": 860.0, "ACC_Temp_Post_Laminar": 860.0, "ACC_Laminar_Header": 0.0,
            "FM_Temp_Before_Rolling": 1000.0
        }
    }
]


@pytest.mark.parametrize("case", INDUSTRIAL_TEST_CASES, ids=lambda c: c["name"])
def test_normalizing_softening_invariants(case):
    """
    Validates inviolable physical metallurgy constraints:
    1. TS_AN < TS_BN (Tensile strength softens after furnace normalizing)
    2. YS_AN < YS_BN (Yield strength softens after furnace normalizing)
    3. EL_AN > EL_BN (Elongation increases due to dislocation annihilation)
    """
    raw = case["inputs"]
    preds = app.predict_selected_engine("Plate", raw, "Champion GBDT (Recommended)")

    ts_bn = preds["TS_BN"]["pred"]
    ts_an = preds["TS_AN"]["pred"]
    ys_bn = preds["YS_BN"]["pred"]
    ys_an = preds["YS_AN"]["pred"]
    el_bn = preds["EL_BN"]["pred"]
    el_an = preds["EL_AN"]["pred"]

    assert np.isfinite(ts_bn) and np.isfinite(ts_an)
    assert np.isfinite(ys_bn) and np.isfinite(ys_an)
    assert np.isfinite(el_bn) and np.isfinite(el_an)

    # Physical softening assertion: TS_AN must be strictly less than TS_BN
    assert ts_an < ts_bn, f"[{case['name']}] TS_AN ({ts_an} MPa) is not lower than TS_BN ({ts_bn} MPa)"
    
    # Physical softening assertion: YS_AN must be strictly less than YS_BN
    assert ys_an < ys_bn, f"[{case['name']}] YS_AN ({ys_an} MPa) is not lower than YS_BN ({ys_bn} MPa)"

    # Ductility gain assertion: EL_AN must be strictly greater than EL_BN
    assert el_an > el_bn, f"[{case['name']}] EL_AN ({el_an} %) is not higher than EL_BN ({el_bn} %)"

    # Softening magnitude assertion: delta TS >= 12 MPa
    delta_ts = ts_bn - ts_an
    assert delta_ts >= 12.0, f"[{case['name']}] Delta TS ({delta_ts:.1f} MPa) is less than minimal 12 MPa"


def test_cooling_rate_increases_softening():
    """
    Verifies that a fast-cooled TMCP plate softens more when normalized than an air-cooled plate,
    because normalizing erases the microstructural grain refinement and dislocation accumulation from ACC.
    """
    base_case = {
        "Plate_T_mm": 16.0, "Plate_W_mm": 2000.0, "Slab_T_mm": 200.0,
        "TB_Thickness_mm": 40.0, "FM_Last_Pass_Thick": 19.0,
        "Chem_C": 0.16, "Chem_Mn": 1.20, "Chem_Si": 0.25, "Chem_P": 0.015, "Chem_S": 0.005,
        "Chem_Cu": 0.02, "Chem_Ni": 0.02, "Chem_Cr": 0.02, "Chem_Mo": 0.005, "Chem_V": 0.005,
        "Chem_Nb": 0.015, "Chem_Ti": 0.002, "Chem_Al": 0.025, "Chem_N": 0.003, "Chem_B": 0.0001,
        "FM_Finish_Temp": 820.0, "FM_Temp_Before_Rolling": 985.0
    }
    
    # Air-cooled: headers = 0
    air = dict(base_case, ACC_Laminar_Header=0.0, ACC_Temp_Post_Laminar=820.0)
    # Accelerated-cooled: headers = 8, T_post = 560 C
    acc = dict(base_case, ACC_Laminar_Header=8.0, ACC_Temp_Post_Laminar=560.0)

    delta_ts_air, delta_ys_air, _ = ml.compute_plate_normalizing_deltas(air)
    delta_ts_acc, delta_ys_acc, _ = ml.compute_plate_normalizing_deltas(acc)

    assert delta_ts_acc > delta_ts_air, "ACC plate should soften more than air-cooled plate on normalizing"
    assert delta_ys_acc > delta_ys_air, "ACC plate should lose more yield strength than air-cooled plate"
