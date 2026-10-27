import re

def normalize_text(text):
    if text is None:
        return ""
    # Replace newlines and multiple spaces
    text = " ".join(str(text).split()).strip().lower()
    return text

# Define the exact mapping rules
# Each field has:
#   name: output column name
#   group: category group
#   cat_patterns: list of lowercase substring or regex patterns for Row 0
#   header_patterns: list of lowercase regex patterns for Row 2
#   dtype: 'str', 'float', 'int', 'date'

SCHEMA = [
    # 0. Identification & Traceability
    {
        "name": "Batch_ID",
        "group": "Identification",
        "cat_patterns": ["identification"],
        "header_patterns": [r"^batch id$", r"^batch_id$"],
        "dtype": "str"
    },
    {
        "name": "Slab_ID",
        "group": "Identification",
        "cat_patterns": ["identification"],
        "header_patterns": [r"^slab id$", r"^slab_id$"],
        "dtype": "str"
    },
    {
        "name": "Heat_ID",
        "group": "Identification",
        "cat_patterns": ["identification"],
        "header_patterns": [r"^heat id$", r"^heat_id$"],
        "dtype": "str"
    },
    {
        "name": "Roll_Date",
        "group": "Identification",
        "cat_patterns": ["identification"],
        "header_patterns": [r"^roll date$", r"^rolling date$"],
        "dtype": "str"
    },
    {
        "name": "Normalizing_Status",
        "group": "Identification",
        "cat_patterns": ["identification"],
        "header_patterns": [r"^normalizing status$"],
        "dtype": "str"
    },
    {
        "name": "Spec",
        "group": "Identification",
        "cat_patterns": ["order"],
        "header_patterns": [r"^spec$", r"^specification$"],
        "dtype": "str"
    },
    {
        "name": "Rolling_Type",
        "group": "Identification",
        "cat_patterns": ["order"],
        "header_patterns": [r"^rolling type$"],
        "dtype": "str"
    },

    # 1. Slab Dimension
    {
        "name": "Slab_Grade",
        "group": "Slab Dimension",
        "cat_patterns": ["slab dimension", "slab"],
        "header_patterns": [r"^slab grade$"],
        "dtype": "str"
    },
    {
        "name": "Slab_T_mm",
        "group": "Slab Dimension",
        "cat_patterns": ["slab dimension", "slab"],
        "header_patterns": [r"^slab t$", r"^slab thickness$"],
        "dtype": "float"
    },
    {
        "name": "Slab_W_mm",
        "group": "Slab Dimension",
        "cat_patterns": ["slab dimension", "slab"],
        "header_patterns": [r"^slab w$", r"^slab width$"],
        "dtype": "float"
    },
    {
        "name": "Slab_L_mm",
        "group": "Slab Dimension",
        "cat_patterns": ["slab dimension", "slab"],
        "header_patterns": [r"^slab l$", r"^slab length$"],
        "dtype": "float"
    },
    {
        "name": "Slab_Wt_ton",
        "group": "Slab Dimension",
        "cat_patterns": ["slab dimension", "slab"],
        "header_patterns": [r"^slab wt$", r"^slab weight$"],
        "dtype": "float"
    },

    # 2. Plate Dimension
    {
        "name": "Plate_T_mm",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^t plate$", r"^plate t$", r"^thickness$"],
        "dtype": "float"
    },
    {
        "name": "Plate_W_mm",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^w plate$", r"^plate w$", r"^width$"],
        "dtype": "float"
    },
    {
        "name": "Plate_L_m",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^l plate$", r"^plate l$", r"^length$"],
        "dtype": "float"
    },
    {
        "name": "Width_Finish_mm",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^width finish$"],
        "dtype": "float"
    },
    {
        "name": "Length_Finish_mm",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^length finish$"],
        "dtype": "float"
    },
    {
        "name": "Plate_Qty",
        "group": "Plate Dimension",
        "cat_patterns": ["plate dimension", "plate"],
        "header_patterns": [r"^qty$", r"^quantity$"],
        "dtype": "int"
    },

    # 3. Chemistry
    {
        "name": "Chem_C",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^c$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Mn",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^mn$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Si",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^si$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Al",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^al$"],
        "dtype": "float"
    },
    {
        "name": "Chem_S",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^s$"],
        "dtype": "float"
    },
    {
        "name": "Chem_P",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^p$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Nb",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^nb$"],
        "dtype": "float"
    },
    {
        "name": "Chem_V",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^v$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Mo",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^mo$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Ti",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^ti$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Cu",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^cu$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Ni",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^ni$"],
        "dtype": "float"
    },
    {
        "name": "Chem_Cr",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^cr$"],
        "dtype": "float"
    },
    {
        "name": "Chem_N",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^n$"],
        "dtype": "float"
    },
    {
        "name": "Chem_CE",
        "group": "Chemistry",
        "cat_patterns": ["chemistry", "slab quality"],
        "header_patterns": [r"^ce$"],
        "dtype": "float"
    },

    # 4. Transfer Bar Thickness
    {
        "name": "TB_Thickness_mm",
        "group": "Transfer Bar Thickness",
        "cat_patterns": ["roughing mill", "roughing", "tb"],
        "header_patterns": [r"^tb$", r"^tb thickness$"],
        "dtype": "float"
    },
    {
        "name": "TB_Ratio",
        "group": "Transfer Bar Thickness",
        "cat_patterns": ["roughing mill", "roughing", "tb"],
        "header_patterns": [r"^tb ratio$"],
        "dtype": "float"
    },

    # 5. Roughing Mill Parameter
    {
        "name": "RM_Charge_Time",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^charge time$"],
        "dtype": "str"
    },
    {
        "name": "RM_Discharge_Time",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^discharge time$"],
        "dtype": "str"
    },
    {
        "name": "RM_Retention_Time_min",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^retention time \(min\)$", r"^retention time$"],
        "dtype": "float"
    },
    {
        "name": "RM_Start_Soaking_Zone_Temp",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^start / soaking zone$", r"^soaking zone$"],
        "dtype": "float"
    },
    {
        "name": "RM_Temp_Pass2_Rolling_PM",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^temp pass 2 rolling pm$", r"^temp pass 2$"],
        "dtype": "float"
    },
    {
        "name": "RM_Finish_Temp",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^rm finish temp$"],
        "dtype": "float"
    },
    {
        "name": "RM_Broadsizing_Pass",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^broadisizng pass$", r"^broadsizing pass$"],
        "dtype": "int"
    },
    {
        "name": "RM_Roughing_Pass",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^roughing pass$"],
        "dtype": "int"
    },
    {
        "name": "RM_Pass_Turning",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^pass turning$"],
        "dtype": "str"
    },
    {
        "name": "RM_Rolling_Strategy",
        "group": "Roughing Mill Parameter",
        "cat_patterns": ["roughing mill", "roughing"],
        "header_patterns": [r"^rolling strategy$"],
        "dtype": "str"
    },

    # 6. Finishing Mill Parameter
    {
        "name": "FM_Temp_Before_Rolling",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^temp before rolling fm$", r"^temp before rolling$"],
        "dtype": "float"
    },
    {
        "name": "FM_Finish_Temp",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^finish temp fm$", r"^finish temp$"],
        "dtype": "float"
    },
    {
        "name": "FM_Number_of_Passes",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^number of passes fm$", r"^number of passes$"],
        "dtype": "int"
    },
    {
        "name": "FM_Thick_n_minus_1_Last_Pass",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^thickness n-1 last pass$", r"^n-1 last pass$"],
        "dtype": "float"
    },
    {
        "name": "FM_Last_Pass_Thick",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^last pass thick$", r"^last pass thickness$"],
        "dtype": "float"
    },
    {
        "name": "FM_Last_Pass_Drafting",
        "group": "Finishing Mill Parameter",
        "cat_patterns": ["finishing mill", "finishing"],
        "header_patterns": [r"^last pass drafting$"],
        "dtype": "float"
    },

    # 7. Accelerated Cooling Setup
    {
        "name": "ACC_Laminar_Header",
        "group": "Accelerated Cooling Setup",
        "cat_patterns": ["finishing mill", "accelerated cooling", "cooling", "laminar"],
        "header_patterns": [r"^laminar header$"],
        "dtype": "str"
    },
    {
        "name": "ACC_Temp_Post_Laminar",
        "group": "Accelerated Cooling Setup",
        "cat_patterns": ["finishing mill", "accelerated cooling", "cooling", "laminar"],
        "header_patterns": [r"^temperature post laminar$", r"^temp post laminar$"],
        "dtype": "float"
    },

    # Targets: Mechanical Properties - Tensile (YS, TS, EL)
    {
        "name": "YS_Before_N_MPa",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile before n", "before n"],
        "header_patterns": [r"^ys$"],
        "dtype": "float"
    },
    {
        "name": "TS_Before_N_MPa",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile before n", "before n"],
        "header_patterns": [r"^ts$"],
        "dtype": "float"
    },
    {
        "name": "EL_Before_N_pct",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile before n", "before n"],
        "header_patterns": [r"^el$"],
        "dtype": "float"
    },
    {
        "name": "YS_After_N_MPa",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile after n", "after n"],
        "header_patterns": [r"^ys$"],
        "dtype": "float"
    },
    {
        "name": "TS_After_N_MPa",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile after n", "after n"],
        "header_patterns": [r"^ts$"],
        "dtype": "float"
    },
    {
        "name": "EL_After_N_pct",
        "group": "Target Tensile Properties",
        "cat_patterns": ["tensile after n", "after n"],
        "header_patterns": [r"^el$"],
        "dtype": "float"
    },

    # Impact Testing at 0 °C
    {
        "name": "Impact_0C_E1_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e1$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_E2_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e2$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_E3_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e3$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_E_Avg_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e avg$", r"^e_avg$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_E_Min_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e min$", r"^e_min$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_E_Std_J",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^e std$", r"^e_std$"],
        "dtype": "float"
    },
    {
        "name": "Impact_0C_Temp",
        "group": "Impact at 0 °C",
        "cat_patterns": ["impact at 0", "0 °c", "0°c", "0 deg"],
        "header_patterns": [r"^temp$"],
        "dtype": "float"
    },

    # Impact Testing at -20 °C
    {
        "name": "Impact_m20C_E1_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e1$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_E2_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e2$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_E3_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e3$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_E_Avg_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e avg$", r"^e_avg$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_E_Min_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e min$", r"^e_min$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_E_Std_J",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^e std$", r"^e_std$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m20C_Temp",
        "group": "Impact at -20 °C",
        "cat_patterns": ["impact at -20", "-20 °c", "-20°c", "-20 deg"],
        "header_patterns": [r"^temp$"],
        "dtype": "float"
    },

    # Impact Testing at -40 °C
    {
        "name": "Impact_m40C_E1_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e1$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_E2_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e2$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_E3_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e3$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_E_Avg_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e avg$", r"^e_avg$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_E_Min_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e min$", r"^e_min$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_E_Std_J",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^e std$", r"^e_std$"],
        "dtype": "float"
    },
    {
        "name": "Impact_m40C_Temp",
        "group": "Impact at -40 °C",
        "cat_patterns": ["impact at -40", "-40 °c", "-40°c", "-40 deg"],
        "header_patterns": [r"^temp$"],
        "dtype": "float"
    },

    # Z Test (Through-thickness reduction in area)
    {
        "name": "Z_Test_Z1_pct",
        "group": "Z Test",
        "cat_patterns": ["z test"],
        "header_patterns": [r"^z1$"],
        "dtype": "float"
    },
    {
        "name": "Z_Test_Z2_pct",
        "group": "Z Test",
        "cat_patterns": ["z test"],
        "header_patterns": [r"^z2$"],
        "dtype": "float"
    },
    {
        "name": "Z_Test_Z3_pct",
        "group": "Z Test",
        "cat_patterns": ["z test"],
        "header_patterns": [r"^z3$"],
        "dtype": "float"
    },
    {
        "name": "Z_Test_Avg_pct",
        "group": "Z Test",
        "cat_patterns": ["z test"],
        "header_patterns": [r"^z avg$", r"^z_avg$"],
        "dtype": "float"
    }
]
