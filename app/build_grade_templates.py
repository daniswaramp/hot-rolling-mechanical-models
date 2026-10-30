"""Build the internal grade chemistry templates shipped with the app.

Reads the plant's internal grade database (a MIN/MAX pair per grade) and writes
models/grade_templates.json. The median of each element is the midpoint of its
permitted band, with an unspecified lower limit read as zero. The workbook stays
outside the repository; only the compact JSON is committed.

    python app/build_grade_templates.py [path/to/Internal_Grade_Database.xlsx]
"""

import json
import os
import sys
from datetime import date

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
OUT_PATH = os.path.join(ROOT, "models", "grade_templates.json")
DEFAULT_SOURCE = r"C:\05. CASTING DATA\3. Internal Grade GRP\Internal_Grade_Database.xlsx"

ELEMENTS = ["C", "Mn", "Si", "Al", "S", "P", "Nb", "V", "Mo", "Ti", "Cu", "Ni", "Cr", "N"]


def _num(value):
    """'-' and blanks mean the limit is not stated; every other cell is a number."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text in ("-", "—", "–"):
        return None
    return float(text)


def _clean_name(value):
    return " ".join(str(value).split())


def build(source_path):
    import python_calamine as pc

    book = pc.load_workbook(source_path)
    sheet_name = book.sheet_names[0]
    rows = book.get_sheet_by_name(sheet_name).to_python(skip_empty_area=False)
    header = [str(cell).strip() for cell in rows[0]]
    index = {name: i for i, name in enumerate(header)}
    name_col, par_col = index["Target Grade/ Spec"], index["Par"]

    grades = {}
    for row in rows[1:]:
        if not row or row[name_col] in (None, ""):
            continue
        name = _clean_name(row[name_col])
        parity = str(row[par_col]).strip().upper()
        if parity not in ("MIN", "MAX"):
            continue
        record = grades.setdefault(name, {"min": {}, "max": {}})
        for element in ELEMENTS + ["CE"]:
            cell = row[index[element]] if element in index else None
            record[parity.lower()][element] = _num(cell)

    payload = {}
    for name, record in grades.items():
        low, high = record["min"], record["max"]
        median = {}
        for element in ELEMENTS:
            lo = low.get(element)
            hi = high.get(element)
            if lo is None and hi is None:
                # Not stated on either side: the grade does not require the element.
                median[element] = 0.0
                continue
            lo = 0.0 if lo is None else lo
            hi = lo if hi is None else hi
            if hi < lo:
                lo, hi = hi, lo
            median[element] = round((lo + hi) / 2.0, 5)
        ce_lo, ce_hi = low.get("CE"), high.get("CE")
        payload[name] = {
            "min": {k: v for k, v in low.items() if k != "CE" and v is not None},
            "max": {k: v for k, v in high.items() if k != "CE" and v is not None},
            "median": median,
            "ce_median": None if ce_lo is None or ce_hi is None else round((ce_lo + ce_hi) / 2.0, 4),
        }

    return {
        "generated": date.today().isoformat(),
        "source": os.path.basename(source_path),
        "sheet": sheet_name,
        "elements": ELEMENTS,
        "grades": payload,
    }


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("INTERNAL_GRADE_XLSX", DEFAULT_SOURCE)
    if not os.path.exists(source):
        raise SystemExit(f"Workbook not found: {source}")
    payload = build(source)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"{len(payload['grades'])} grades -> {OUT_PATH}")


if __name__ == "__main__":
    main()
