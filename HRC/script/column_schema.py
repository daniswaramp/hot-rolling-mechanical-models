"""
Column Schema Definition for Steckel Mill (HRC) Data Extraction.
Defines dynamic matching rules based on Category (Row 0) and Header (Row 1)
to handle column shifts and naming variations cleanly.
"""

import re
import unicodedata

def normalize_text(text):
    """Normalize text for resilient header/category matching."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

SCHEMA = [
    # Traceability / Identification
    {
        "name": "Coil_No",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"process.*planning"],
        "header_patterns": [r"^coil\s*no"]
    },
    {
        "name": "Rolling_Date",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"process.*planning"],
        "header_patterns": [r"rolling\s*date"]
    },
    {
        "name": "Heat_ID",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"coil\s*identification"],
        "header_patterns": [r"^heat$"]
    },
    {
        "name": "Batch_ID",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"coil\s*identification"],
        "header_patterns": [r"batch\s*id"]
    },
    {
        "name": "Slab_No",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"coil\s*identification"],
        "header_patterns": [r"slab\s*no"]
    },
    {
        "name": "Slab_Grade",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"coil\s*identification"],
        "header_patterns": [r"slab\s*grade"]
    },
    {
        "name": "Specification",
        "group": "Identification",
        "dtype": "str",
        "cat_patterns": [r"specification"],
        "header_patterns": [r"specification"]
    },

    # Dimensional Parameters
    {
        "name": "Coil_T_mm",
        "group": "Dimension",
        "dtype": "float",
        "cat_patterns": [r"dimension"],
        "header_patterns": [r"^t$"]
    },
    {
        "name": "Coil_W_mm",
        "group": "Dimension",
        "dtype": "float",
        "cat_patterns": [r"dimension"],
        "header_patterns": [r"^w$"]
    },
    {
        "name": "TB_Thick_mm",
        "group": "Dimension",
        "dtype": "float",
        "cat_patterns": [r"tb"],
        "header_patterns": [r"tb\s*thick"]
    },
    {
        "name": "WT_Actual_kg",
        "group": "Dimension",
        "dtype": "float",
        "cat_patterns": [r"weight.*yield"],
        "header_patterns": [r"wt\s*actual"]
    },
    {
        "name": "Yield_Act_pct",
        "group": "Dimension",
        "dtype": "float",
        "cat_patterns": [r"weight.*yield"],
        "header_patterns": [r"yield\s*\(act\)"]
    },

    # Thermal & Mill Processing Parameters
    {
        "name": "Temp_Transfer_Bar_C",
        "group": "Thermal",
        "dtype": "float",
        "cat_patterns": [r"transfer\s*temp"],
        "header_patterns": [r"t\.bar\s*temp"]
    },
    {
        "name": "Temp_FT_Avg_C",
        "group": "Thermal",
        "dtype": "float",
        "cat_patterns": [r"rolling\s*temperature"],
        "header_patterns": [r"ft\s*average"]
    },
    {
        "name": "Temp_CT_Top_C",
        "group": "Thermal",
        "dtype": "float",
        "cat_patterns": [r"rolling\s*temperature"],
        "header_patterns": [r"ct\s*top"]
    },
    {
        "name": "Temp_CT_Bot_C",
        "group": "Thermal",
        "dtype": "float",
        "cat_patterns": [r"rolling\s*temperature"],
        "header_patterns": [r"ct\s*bot"]
    },

    # Target Mechanical Properties (Tensile Before Normalizing)
    {
        "name": "YS_BN_MPa",
        "group": "Mechanical_BN",
        "dtype": "float",
        "cat_patterns": [r"mechanical\s*prop.*tensile\s*bn"],
        "header_patterns": [r"^ys$"]
    },
    {
        "name": "TS_BN_MPa",
        "group": "Mechanical_BN",
        "dtype": "float",
        "cat_patterns": [r"mechanical\s*prop.*tensile\s*bn"],
        "header_patterns": [r"^ts$"]
    },
    {
        "name": "EL_BN_pct",
        "group": "Mechanical_BN",
        "dtype": "float",
        "cat_patterns": [r"mechanical\s*prop.*tensile\s*bn"],
        "header_patterns": [r"^el$"]
    },

    # Chemistry Elements (16 Elements)
    {
        "name": "Chem_C",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^c$"]
    },
    {
        "name": "Chem_Si",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^si$"]
    },
    {
        "name": "Chem_Mn",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^mn$"]
    },
    {
        "name": "Chem_P",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^p$"]
    },
    {
        "name": "Chem_S",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^s$"]
    },
    {
        "name": "Chem_Nb",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^nb$"]
    },
    {
        "name": "Chem_Cu",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^cu$"]
    },
    {
        "name": "Chem_Cr",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^cr$"]
    },
    {
        "name": "Chem_Ni",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^ni$"]
    },
    {
        "name": "Chem_Mo",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^mo$"]
    },
    {
        "name": "Chem_V",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^v$"]
    },
    {
        "name": "Chem_Al",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^al$"]
    },
    {
        "name": "Chem_Ti",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^ti$"]
    },
    {
        "name": "Chem_N",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^n$"]
    },
    {
        "name": "Chem_B",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^b$"]
    },
    {
        "name": "Chem_CE",
        "group": "Chemistry",
        "dtype": "float",
        "cat_patterns": [r"z\s*test"],
        "header_patterns": [r"^ce$"]
    }
]
