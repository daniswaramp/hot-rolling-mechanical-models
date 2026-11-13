"""Single source for the physics the app displays beside a prediction.

The shipped models were fitted on the exact columns Plate/script/ml/features.py and
HRC/script/ml/features.py produce, so those two modules are frozen: their sigma_ss and Ar3
intentionally differ from the published Pickering/Ouchi forms below (the fitted Plate model uses
354 P and 33 Ni and adds a thickness term to Ar3), and editing them would silently shift every
prediction. This module holds the forms the derived-metrics card documents, applied to the same
clipped inputs the models see, so card and model agree at the extremes.

The NLR constitutive fits are a separate exception: their coefficients were fitted on their own
narrower term definitions (scripts/parse_nlr.py::term_values), which streamlit_app keeps inside
the NLR term builder and labels as NLR terms rather than model physics.
"""

import numpy as np

# Clip bounds, copied from the mill feature modules, so an indicator computed here matches what
# a model actually sees for the same schedule.
PLATE_BOUNDS = {
    "Slab_T_mm": (100.0, 350.0),
    "Plate_T_mm": (4.0, 160.0),
    "Plate_W_mm": (800.0, 4500.0),
    "TB_Ratio": (1.05, 6.0),
    "FM_Temp_Before_Rolling": (750.0, 1150.0),
    "FM_Finish_Temp": (680.0, 1050.0),
    "ACC_Temp_Post_Laminar_low": (400.0, None),
    "Chem_C": (0.01, 0.45),
    "Chem_Mn": (0.10, 2.20),
    "Chem_Si": (0.01, 1.00),
    "Chem_Al": (0.001, 0.10),
    "Chem_S": (0.0001, 0.05),
    "Chem_P": (0.001, 0.05),
    "Chem_Nb": (0.0, 0.10),
    "Chem_V": (0.0, 0.15),
    "Chem_Mo": (0.0, 0.50),
    "Chem_Ti": (0.0, 0.08),
    "Chem_Cu": (0.0, 0.60),
    "Chem_Ni": (0.0, 0.80),
    "Chem_Cr": (0.0, 0.60),
    "Chem_N": (0.0005, 0.02),
}

HRC_BOUNDS = {
    "Coil_T_mm": (1.5, 25.0),
    "Coil_W_mm": (700.0, 2200.0),
    "TB_Thick_mm": (15.0, 70.0),
    "Temp_Transfer_Bar_C": (750.0, 1100.0),
    "Temp_FT_Avg_C": (700.0, 980.0),
    "Temp_CT_Top_C": (400.0, 750.0),
    "Temp_CT_Bot_C": (400.0, 750.0),
}


def clip(value, mill, key):
    lo, hi = PLATE_BOUNDS.get(key, HRC_BOUNDS.get(key, (-np.inf, np.inf))) \
        if mill == "Plate" else HRC_BOUNDS.get(key, (-np.inf, np.inf))
    if value is None:
        return None
    value = float(value)
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def sigma_ss(mn, si, ni, cr, cu, ti, mo, n):
    """Pickering solid solution strengthening, MPa (MODEL_CARD section 3)."""
    return (32.0 * mn + 84.0 * si + 43.0 * ni - 30.0 * cr
            + 38.0 * cu + 80.0 * ti + 11.0 * mo + 354.0 * n)


def ce_iiw(c, mn, cr, mo, v, ni, cu):
    return c + mn / 6.0 + (cr + mo + v) / 5.0 + (ni + cu) / 15.0


def pcm(c, si, mn, cu, cr, ni, mo, v, b=0.0):
    return (c + si / 30.0 + (mn + cu + cr) / 20.0 + ni / 60.0
            + mo / 15.0 + v / 10.0 + 5.0 * b)


def ar3(c, mn, cu, cr, ni, mo):
    """Ouchi Ar3, degrees C."""
    return 910.0 - 310.0 * c - 80.0 * mn - 20.0 * cu - 15.0 * cr - 55.0 * ni - 80.0 * mo


def ar1(mn, ni, si, cr):
    """Andrews Ar1, degrees C."""
    return 723.0 - 10.7 * mn - 16.9 * ni + 29.1 * si + 16.9 * cr


def plate_cooling_rate(frt, t_post, plate_t, headers):
    """Laminar accelerated-cooling rate proxy, MPa-equivalent driving force."""
    drop = max(0.0, frt - t_post)
    return (drop * (headers + 0.5)) / max(plate_t, 1.0)


def hrc_cooling_rate(ft, ct_top, ct_bot, coil_t):
    """Runout-table cooling rate proxy."""
    ct = (ct_top + ct_bot) / 2.0
    return max(0.0, ft - ct) / max(coil_t, 1.0)


def true_strain(thick_in, thick_out):
    return float(np.log(max(thick_in, thick_out * 1.05) / max(thick_out, 1e-9)))
