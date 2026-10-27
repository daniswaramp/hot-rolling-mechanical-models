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


def compute_metallurgical_features(df):
    """
    Takes cleaned dataframe and engineers domain-physics features.
    Returns dataframe with base features + metallurgical physical terms.
    """
    X = pd.DataFrame(index=df.index)

    # 1. Base Dimensions & Ratios
    slab_t = df["Slab_T_mm"].clip(lower=100.0, upper=350.0)
    plate_t = df["Plate_T_mm"].clip(lower=4.0, upper=160.0)
    plate_w = df["Plate_W_mm"].clip(lower=800.0, upper=4500.0)
    tb_ratio = df["TB_Ratio"].fillna(2.0).clip(lower=1.05, upper=6.0)
    last_draft = df["FM_Last_Pass_Drafting"].fillna(2.0).clip(lower=0.1, upper=15.0)

    X["Slab_T_mm"] = slab_t
    X["Plate_T_mm"] = plate_t
    X["Plate_W_mm"] = plate_w
    X["TB_Ratio"] = tb_ratio
    X["FM_Last_Pass_Drafting"] = last_draft
    X["W_over_T"] = plate_w / plate_t
    X["inv_sqrt_t"] = 1.0 / np.sqrt(plate_t)  # Classical Hall-Petch structural proxy

    # 2. Chemistry Extraction with Safe Fill
    c = df["Chem_C"].fillna(0.18).clip(lower=0.01, upper=0.45)
    mn = df["Chem_Mn"].fillna(0.90).clip(lower=0.10, upper=2.20)
    si = df["Chem_Si"].fillna(0.20).clip(lower=0.01, upper=1.00)
    al = df["Chem_Al"].fillna(0.025).clip(lower=0.001, upper=0.10)
    s = df["Chem_S"].fillna(0.005).clip(lower=0.0001, upper=0.05)
    p = df["Chem_P"].fillna(0.015).clip(lower=0.001, upper=0.05)
    nb = df["Chem_Nb"].fillna(0.002).clip(lower=0.0, upper=0.10)
    v = df["Chem_V"].fillna(0.005).clip(lower=0.0, upper=0.15)
    mo = df["Chem_Mo"].fillna(0.005).clip(lower=0.0, upper=0.50)
    ti = df["Chem_Ti"].fillna(0.002).clip(lower=0.0, upper=0.08)
    cu = df["Chem_Cu"].fillna(0.02).clip(lower=0.0, upper=0.60)
    ni = df["Chem_Ni"].fillna(0.02).clip(lower=0.0, upper=0.80)
    cr = df["Chem_Cr"].fillna(0.02).clip(lower=0.0, upper=0.60)
    n = df["Chem_N"].fillna(0.003).clip(lower=0.0005, upper=0.02)

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
    fm_start_temp = df["FM_Temp_Before_Rolling"].fillna(930.0).clip(lower=750.0, upper=1150.0)
    frt = df["FM_Finish_Temp"].fillna(820.0).clip(lower=680.0, upper=1050.0)
    acc_active = df["ACC_Active"].fillna(0.0) if "ACC_Active" in df.columns else np.zeros(len(df))
    acc_headers = df["ACC_Headers_n"].fillna(0.0) if "ACC_Headers_n" in df.columns else np.zeros(len(df))
    t_post = df["ACC_Temp_Post_Laminar"].fillna(frt).clip(lower=400.0, upper=frt)

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
    rolling_type = df["Rolling_Type"].fillna("AR").str.upper().str.strip()
    X["is_AR"] = (rolling_type == "AR").astype(float)
    X["is_CR"] = (rolling_type == "CR").astype(float)
    X["is_TMCP"] = (rolling_type == "TMCP").astype(float)

    return X
