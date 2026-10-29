"""
Feature Engineering Module for Steckel Mill Hot Rolled Coil (HRC).
Computes domain-specific metallurgical features:
- Pickering solid solution strengthening (sigma_ss)
- Hall-Petch grain size proxy (1/sqrt(t))
- Steckel true strain (ln(TB_thick / Coil_T))
- Phase transformation temperatures (Ar3, Ar1)
- Runout Table (ROT) cooling rate proxy (Delta_T_ROT / t)
- Carbon equivalents (CE_IIW, Pcm)
- Microalloying precipitation terms (Nb, V, Ti)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("Features_HRC")


def engineer_hrc_features(df: pd.DataFrame, is_train: bool = True, spec_map: dict = None) -> tuple:
    """
    Computes all metallurgical and physical features for Steckel Mill HRC modeling.
    Returns: (X DataFrame, spec_map dictionary)
    """
    X = pd.DataFrame(index=df.index)

    # 1. Base Dimensions
    t_s = df["Coil_T_mm"] if "Coil_T_mm" in df.columns else pd.Series(6.0, index=df.index)
    t = pd.to_numeric(t_s, errors='coerce').fillna(6.0).clip(lower=1.5, upper=25.0)
    w_s = df["Coil_W_mm"] if "Coil_W_mm" in df.columns else pd.Series(1250.0, index=df.index)
    w = pd.to_numeric(w_s, errors='coerce').fillna(1250.0).clip(lower=700.0, upper=2200.0)
    tb_s = df["TB_Thick_mm"] if "TB_Thick_mm" in df.columns else pd.Series(30.0, index=df.index)
    tb = pd.to_numeric(tb_s, errors='coerce').fillna(30.0).clip(lower=15.0, upper=70.0)

    X["Coil_T_mm"] = t
    X["Coil_W_mm"] = w
    X["TB_Thick_mm"] = tb

    # 2. Chemistry Elements
    chem_elements = ["C", "Si", "Mn", "P", "S", "Nb", "Cu", "Cr", "Ni", "Mo", "V", "Al", "Ti", "N", "B", "CE"]
    for el in chem_elements:
        col = f"Chem_{el}"
        s = df[col] if col in df.columns else pd.Series(0.0, index=df.index)
        val = pd.to_numeric(s, errors='coerce').fillna(0.0)
        X[col] = val

    c = X["Chem_C"]
    mn = X["Chem_Mn"]
    si = X["Chem_Si"]
    p = X["Chem_P"]
    s = X["Chem_S"]
    nb = X["Chem_Nb"]
    cu = X["Chem_Cu"]
    cr = X["Chem_Cr"]
    ni = X["Chem_Ni"]
    mo = X["Chem_Mo"]
    v = X["Chem_V"]
    al = X["Chem_Al"]
    ti = X["Chem_Ti"]
    n = X["Chem_N"]
    b = X["Chem_B"]

    # 3. Solid Solution Strengthening (Pickering equation)
    # sigma_ss = 32*Mn + 84*Si + 43*Ni - 30*Cr + 38*Cu + 80*Ti + 11*Mo + 354*N (MPa)
    X["sigma_ss_MPa"] = 32.0 * mn + 84.0 * si + 43.0 * ni - 30.0 * cr + 38.0 * cu + 80.0 * ti + 11.0 * mo + 354.0 * n

    # 4. Carbon Equivalents
    # CE_IIW = C + Mn/6 + (Cr+Mo+V)/5 + (Ni+Cu)/15
    X["CE_IIW"] = c + (mn / 6.0) + ((cr + mo + v) / 5.0) + ((ni + cu) / 15.0)
    # Pcm = C + Si/30 + (Mn+Cu+Cr)/20 + Ni/60 + Mo/15 + V/10 + 5*B
    X["P_cm"] = c + (si / 30.0) + ((mn + cu + cr) / 20.0) + (ni / 60.0) + (mo / 15.0) + (v / 10.0) + (5.0 * b)

    # 5. Hall-Petch Grain Size Proxies
    # For HRC strip, strip thickness inversely determines final ferrite grain size d_alpha
    X["Hall_Petch_t_inv_sqrt"] = 1.0 / np.sqrt(t)
    X["Hall_Petch_t_inv_quad"] = 1.0 / np.power(t, 0.25)

    # 6. Steckel Mill Deformation & True Strain
    # True strain = ln(TB / t)
    valid_tb = tb.where(tb > t, t * 1.05)
    X["Steckel_True_Strain"] = np.log(valid_tb / t)
    X["Steckel_Draft_Ratio"] = (valid_tb - t) / valid_tb

    # 7. Thermal History
    ttb_s = df["Temp_Transfer_Bar_C"] if "Temp_Transfer_Bar_C" in df.columns else pd.Series(950.0, index=df.index)
    ttb = pd.to_numeric(ttb_s, errors='coerce').fillna(950.0).clip(lower=750.0, upper=1100.0)
    ft_s = df["Temp_FT_Avg_C"] if "Temp_FT_Avg_C" in df.columns else pd.Series(850.0, index=df.index)
    ft = pd.to_numeric(ft_s, errors='coerce').fillna(850.0).clip(lower=700.0, upper=980.0)
    ct_top_s = df["Temp_CT_Top_C"] if "Temp_CT_Top_C" in df.columns else pd.Series(580.0, index=df.index)
    ct_top = pd.to_numeric(ct_top_s, errors='coerce').fillna(580.0).clip(lower=400.0, upper=750.0)
    ct_bot_s = df["Temp_CT_Bot_C"] if "Temp_CT_Bot_C" in df.columns else pd.Series(580.0, index=df.index)
    ct_bot = pd.to_numeric(ct_bot_s, errors='coerce').fillna(580.0).clip(lower=400.0, upper=750.0)
    ct_avg = pd.concat([ct_top, ct_bot], axis=1).mean(axis=1)

    X["Temp_Transfer_Bar_C"] = ttb
    X["Temp_FT_Avg_C"] = ft
    X["Temp_CT_Top_C"] = ct_top
    X["Temp_CT_Bot_C"] = ct_bot
    X["Temp_CT_Avg_C"] = ct_avg

    # ROT (Runout Table) Temperature Drop
    delta_t_rot = (ft - ct_avg).clip(lower=50.0, upper=450.0)
    X["Delta_T_ROT_C"] = delta_t_rot

    # ROT Cooling Rate Proxy: CR ~ Delta_T_ROT / thickness
    X["ROT_Cooling_Rate_Proxy"] = delta_t_rot / t

    # 8. Phase Transformation Limits
    # Ar3 (Ouchi formula)
    X["Ar3_Temp_C"] = 910.0 - (310.0 * c) - (80.0 * mn) - (20.0 * cu) - (15.0 * cr) - (55.0 * ni) - (80.0 * mo)
    # Ar1 (Andrews formula)
    X["Ar1_Temp_C"] = 723.0 - (10.7 * mn) - (16.9 * ni) + (29.1 * si) + (16.9 * cr)

    # Finish Rolling Temperature relative to Ar3
    X["FT_minus_Ar3_C"] = ft - X["Ar3_Temp_C"]
    # Coiling Temperature relative to Ar1
    X["CT_minus_Ar1_C"] = ct_avg - X["Ar1_Temp_C"]

    # 9. Microalloying Potential
    X["Microalloy_Sum"] = nb + v + ti
    X["Mn_to_C_Ratio"] = mn / (c + 0.001)
    X["Ti_to_N_Ratio"] = ti / (n + 0.0005)

    # Note: Specification label is intentionally omitted as properties are governed
    # fundamentally by physical chemistry and thermomechanical rolling conditions.
    return X, None
