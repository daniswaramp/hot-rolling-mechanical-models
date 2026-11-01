"""Sanitise the cleaned training tables in place.

The ETL exports what the mill logged, including decimal-shift typos: finishing
temperatures of 7778 or 893897 C, a tensile of 32 MPa. Training already repairs
these via PHYSICAL_BOUNDS, but the CSV itself — which the sidebar statistics and
any refit read — still carries them.

This pass applies the same rule as `Plate/script/ml/data_prep.py` (rescale by
10/100/1000 when the result lands inside the band, otherwise blank the cell),
drops exact duplicate rows, and writes an audit trail next to the reports.

    python app/sanitise_training_tables.py --mill Plate [--dry-run]
    python app/sanitise_training_tables.py --mill Both
"""

import argparse
import importlib.util
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)

ROUTES = {
    "Plate": {
        "csv": os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
        "config": os.path.join(ROOT, "Plate", "script", "ml", "config.py"),
        "log": os.path.join(ROOT, "Plate", "reports", "sanitise_log.csv"),
        "id_cols": ("Batch_ID", "Slab_ID", "Heat_ID"),
    },
    "HRC": {
        "csv": os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"),
        "config": os.path.join(ROOT, "HRC", "script", "ml", "config.py"),
        "log": os.path.join(ROOT, "HRC", "reports", "sanitise_log.csv"),
        "id_cols": ("Coil_ID", "Heat_ID"),
    },
}

TARGET_TOKENS = ("YS_", "TS_", "EL_", "Impact_", "Z_Test")


def load_bounds(config_path):
    spec = importlib.util.spec_from_file_location("route_ml_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.PHYSICAL_BOUNDS)


def rescale(value, lo, hi):
    """The rule training uses: try /100, /10, /1000; blank if none lands in band."""
    if lo <= value <= hi:
        return value, ""
    for divisor, reason in ((100.0, "decimal shift /100"), (10.0, "decimal shift /10"),
                            (1000.0, "decimal shift /1000")):
        if lo <= value / divisor <= hi:
            return value / divisor, reason
    return np.nan, "out of bounds"


def sanitise(mill, dry_run=False):
    spec = ROUTES[mill]
    if not os.path.exists(spec["csv"]):
        print(f"[{mill}] no cleaned CSV at {spec['csv']}")
        return 1
    bounds = load_bounds(spec["config"])

    df = pd.read_csv(spec["csv"], low_memory=False)
    before_rows = len(df)
    ids = next((c for c in spec["id_cols"] if c in df.columns), None)

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log = []

    print(f"\n[{mill}] {before_rows:,} rows -> duplicate rows removed: {duplicates}")
    print(f"{'column':26} {'checked':>8} {'rescaled':>9} {'blanked':>8} {'skipped':>8}")

    target_loss = {}
    for column, (lo, hi) in bounds.items():
        if column not in df.columns:
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        present = series.notna()
        if present.sum() == 0 or numeric[present].notna().mean() < 0.90:
            print(f"{column:26} {'':>8} {'':>9} {'':>8} {'non-numeric':>8}")
            continue

        checked = int(numeric[present].notna().sum())
        rescaled = blanked = 0
        for idx in series.index[present]:
            value = numeric.at[idx]
            if pd.isna(value):
                continue
            fixed, reason = rescale(float(value), lo, hi)
            if not reason:
                continue
            if reason == "out of bounds":
                blanked += 1
            else:
                rescaled += 1
            log.append({
                "row_index": int(idx),
                "record_id": df.at[idx, ids] if ids else None,
                "column": column,
                "old_value": value,
                "new_value": fixed,
                "reason": reason,
            })
            df.at[idx, column] = fixed

        if any(column.startswith(t) for t in TARGET_TOKENS):
            lost = blanked
            target_loss[column] = lost / checked if checked else 0.0
        print(f"{column:26} {checked:8d} {rescaled:9d} {blanked:8d} {0:8d}")

    print(f"[{mill}] rows after dedupe: {len(df):,}   cells touched: {len(log):,}")

    if dry_run:
        print(f"[{mill}] dry run — nothing written")
        return 0

    df.to_csv(spec["csv"], index=False, encoding="utf-8")
    print(f"[{mill}] wrote {spec['csv']}")

    if log:
        os.makedirs(os.path.dirname(spec["log"]), exist_ok=True)
        audit = pd.DataFrame(log)
        audit.insert(0, "timestamp", datetime.now().isoformat(timespec="seconds"))
        audit.to_csv(spec["log"], index=False, encoding="utf-8")
        print(f"[{mill}] audit: {spec['log']} ({len(audit):,} rows)")

    bad = {k: v for k, v in target_loss.items() if v > 0.01}
    if bad:
        print(f"[{mill}] ERROR target columns lost > 1 %: {bad}")
        return 2
    return 0


def main():
    parser = argparse.ArgumentParser(description="Sanitise cleaned training tables")
    parser.add_argument("--mill", default="Both", choices=["Plate", "HRC", "Both"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    mills = ["Plate", "HRC"] if args.mill == "Both" else [args.mill]
    status = 0
    for mill in mills:
        status = max(status, sanitise(mill, args.dry_run))
    sys.exit(status)


if __name__ == "__main__":
    main()
