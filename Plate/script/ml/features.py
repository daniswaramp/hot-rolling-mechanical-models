"""
Metallurgical Feature Engineering Module.
Computes domain physics features:
- Hall-Petch grain size proxy and thickness effects
- Pickering solid solution strengthening
- Ar3 transformation and Tnr non-recrystallization temperatures
- True strain components (total, roughing, finishing, last pass drafting)
- Accelerated cooling rate and thermal driving forces
- Carbon equivalents (CE_IIW, Pcm) and microalloying precipitation terms
"""

import numpy as np
import pandas as pd

def _get_series(df, col, default):
    s = df[col] if col in df.columns else pd.Series(default, index=df.index)
    return pd.to_numeric(s, errors='coerce').fillna(default)



def compute_metallurgical_features(df):
    """
    Takes cleaned dataframe and engineers domain-physics features.
    Returns dataframe with base features + metallurgical physical terms.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Base Dimensions & Ratios
    slab_t = _get_series(df, "Slab_T_mm", 220.0).clip(lower=100.0, upper=350.0)
    plate_t = _get_series(df, "Plate_T_mm", 16.0).clip(lower=4.0, upper=160.0)
    plate_w = _get_series(df, "Plate_W_mm", 2000.0).clip(lower=800.0, upper=4500.0)
    tb_ratio = _get_series(df, "TB_Ratio", 2.0).clip(lower=1.05, upper=6.0)
    last_draft = _get_series(df, "FM_Last_Pass_Drafting", 2.0).clip(lower=0.1, upper=15.0)

    X["Slab_T_mm"] = slab_t
    X["Plate_T_mm"] = plate_t
    X["Plate_W_mm"] = plate_w
    X["TB_Ratio"] = tb_ratio
    X["FM_Last_Pass_Drafting"] = last_draft
    X["W_over_T"] = plate_w / plate_t
    X["inv_sqrt_t"] = 1.0 / np.sqrt(plate_t)  # Classical Hall-Petch structural proxy

    # 2. Chemistry Extraction with Safe Fill
    c = _get_series(df, "Chem_C", 0.18).clip(lower=0.01, upper=0.45)
    mn = _get_series(df, "Chem_Mn", 0.90).clip(lower=0.10, upper=2.20)
    si = _get_series(df, "Chem_Si", 0.20).clip(lower=0.01, upper=1.00)
    al = _get_series(df, "Chem_Al", 0.025).clip(lower=0.001, upper=0.10)
    s = _get_series(df, "Chem_S", 0.005).clip(lower=0.0001, upper=0.05)
    p = _get_series(df, "Chem_P", 0.015).clip(lower=0.001, upper=0.05)
    nb = _get_series(df, "Chem_Nb", 0.002).clip(lower=0.0, upper=0.10)
    v = _get_series(df, "Chem_V", 0.005).clip(lower=0.0, upper=0.15)
    mo = _get_series(df, "Chem_Mo", 0.005).clip(lower=0.0, upper=0.50)
    ti = _get_series(df, "Chem_Ti", 0.002).clip(lower=0.0, upper=0.08)
    cu = _get_series(df, "Chem_Cu", 0.02).clip(lower=0.0, upper=0.60)
    ni = _get_series(df, "Chem_Ni", 0.02).clip(lower=0.0, upper=0.80)
    cr = _get_series(df, "Chem_Cr", 0.02).clip(lower=0.0, upper=0.60)
    n = _get_series(df, "Chem_N", 0.003).clip(lower=0.0005, upper=0.02)

    chem_dict = {
        "Chem_C": c, "Chem_Mn": mn, "Chem_Si": si, "Chem_Al": al,
        "Chem_S": s, "Chem_P": p, "Chem_Nb": nb, "Chem_V": v,
        "Chem_Mo": mo, "Chem_Ti": ti, "Chem_Cu": cu, "Chem_Ni": ni,
        "Chem_Cr": cr, "Chem_N": n
    }
    for k, v_series in chem_dict.items():
        X[k] = v_series

    # 3. Carbon Equivalents and Solid Solution Strengthening
    # CE IIW
    X["CE_IIW"] = c + (mn / 6.0) + ((cr + mo + v) / 5.0) + ((ni + cu) / 15.0)

    # Ito-Bessyo Pcm
    X["P_cm"] = c + (si / 30.0) + ((mn + cu + cr) / 20.0) + (ni / 60.0) + (mo / 15.0) + (v / 10.0)

    # Pickering solid solution strengthening model (MPa contribution)
    X["sigma_ss"] = (32.3 * mn) + (83.2 * si) + (354.0 * p) + (38.0 * cu) + (33.0 * ni) + (11.0 * mo) - (30.0 * cr)

    # Microalloying precipitation potentials
    X["Microalloy_Total"] = nb + v + ti
    X["Nb_eff"] = nb * (c + (12.0 / 14.0) * n)
    X["Ti_to_N_ratio"] = ti / np.maximum(n, 0.001)

    # 4. Thermal Parameters
    fm_start_temp = _get_series(df, "FM_Temp_Before_Rolling", 930.0).clip(lower=750.0, upper=1150.0)
    frt = _get_series(df, "FM_Finish_Temp", 820.0).clip(lower=680.0, upper=1050.0)
    acc_active = _get_series(df, "ACC_Active", 0.0)
    acc_headers = _get_series(df, "ACC_Headers_n", 0.0)
    t_post_raw = _get_series(df, "ACC_Temp_Post_Laminar", frt)
    t_post = t_post_raw.clip(lower=400.0, upper=frt)

    X["FM_Temp_Before_Rolling"] = fm_start_temp
    X["FM_Finish_Temp"] = frt
    X["ACC_Active"] = acc_active
    X["ACC_Headers_n"] = acc_headers
    X["ACC_Temp_Post_Laminar"] = t_post

    # Thermal drops
    X["FM_Temp_Drop"] = fm_start_temp - frt
    X["ACC_Temp_Drop"] = np.maximum(0.0, frt - t_post)
    X["Cooling_Rate_Proxy"] = (X["ACC_Temp_Drop"] * (acc_headers + 0.5)) / plate_t

    # 5. Metallurgical Critical Transformation Temperatures
    # Ouchi Ar3 model (°C)
    X["Ar3"] = (910.0 - (310.0 * c) - (80.0 * mn) - (20.0 * cu) - (15.0 * cr)
                - (55.0 * ni) - (80.0 * mo) + (0.35 * (plate_t - 8.0)))

    # Boratto Tnr (non-recrystallization temperature, °C)
    nb_term = (6445.0 * nb) - (644.0 * np.sqrt(np.maximum(0.0, nb)))
    v_term = (732.0 * v) - (230.0 * np.sqrt(np.maximum(0.0, v)))
    X["Tnr"] = 887.0 + (464.0 * c) + nb_term + v_term + (890.0 * ti) + (363.0 * al) - (357.0 * si)

    # Thermal Headroom / Pancaking
    X["FRT_minus_Ar3"] = frt - X["Ar3"]
    X["FRT_minus_Tnr"] = frt - X["Tnr"]
    X["Tpost_minus_Ar3"] = t_post - X["Ar3"]

    # 6. True Strain Terms
    X["eps_total"] = np.log(slab_t / plate_t)
    X["eps_FM"] = np.log(tb_ratio)
    X["eps_last_pass"] = np.log((plate_t + last_draft) / plate_t)

    # 7. Hall-Petch Ferrite Grain Size Estimator
    # Grain refinement increases with heavier total deformation, sub-Tnr pancaking, and faster cooling
    d_alpha_proxy = (plate_t ** 0.4) / (np.maximum(0.5, X["eps_total"]) * np.maximum(1.0, X["Cooling_Rate_Proxy"] ** 0.25))
    X["d_alpha_est"] = np.maximum(1.0, d_alpha_proxy)
    X["HP_term"] = 1.0 / np.sqrt(X["d_alpha_est"])  # Hall-Petch term: d^(-1/2)

    # 8. Process Route One-Hot Encoding
    r_type_s = df["Rolling_Type"] if "Rolling_Type" in df.columns else pd.Series("AR", index=df.index)
    rolling_type = r_type_s.fillna("AR").astype(str).str.upper().str.strip()
    X["is_AR"] = (rolling_type == "AR").astype(float)
    X["is_CR"] = (rolling_type == "CR").astype(float)
    X["is_TMCP"] = (rolling_type == "TMCP").astype(float)

    return X
