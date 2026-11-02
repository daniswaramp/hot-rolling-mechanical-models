"""Build models/data_audit.json: the governance numbers the Diagnostics tab shows.

Two sections, kept apart on purpose:

* "as trained" — the split and scores the shipped artifacts were actually fitted
  with, during the 2026-10 campaign. Recorded from that campaign's reports and
  training log; re-deriving them from today's cleaned tables would misstate the
  models, because the tables have since been sanitised.
* "current" — what the cleaned tables contain right now: usable rows, unique
  melts, and the sanitising pass counts read from reports/sanitise_log.csv.

    python app/build_data_audit.py
"""

import json
import os
import sys
from datetime import date

import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, APP_DIR)

import training_stats as ts  # noqa: E402

OUT = os.path.join(ROOT, "models", "data_audit.json")

CSV = {
    "Plate": os.path.join(ROOT, "Plate", "csv", "plate_rolling_2026_cleaned.csv"),
    "HRC": os.path.join(ROOT, "HRC", "csv", "steckel_mill_cleaned.csv"),
}
LOG = {
    "Plate": os.path.join(ROOT, "Plate", "reports", "sanitise_log.csv"),
    "HRC": os.path.join(ROOT, "HRC", "reports", "sanitise_log.csv"),
}
LEADERBOARD = {
    "Plate": os.path.join(ROOT, "Plate", "reports", "tuning_and_ensemble_leaderboard.csv"),
    "HRC": os.path.join(ROOT, "HRC", "reports", "hrc_tuning_leaderboard.csv"),
}

TARGET_COLUMNS = {
    "Plate": {
        "YS_BN": "YS_Before_N_MPa", "TS_BN": "TS_Before_N_MPa", "EL_BN": "EL_Before_N_pct",
        "YS_AN": "YS_After_N_MPa", "TS_AN": "TS_After_N_MPa", "EL_AN": "EL_After_N_pct",
        "Impact_0C": "Impact_0C_E_Avg_J", "Impact_m20C": "Impact_m20C_E_Avg_J",
        "Impact_m40C": "Impact_m40C_E_Avg_J", "Z_Test": "Z_Test_Avg_pct",
    },
    "HRC": {"YS_BN": "YS_BN_MPa", "TS_BN": "TS_BN_MPa", "EL_BN": "EL_BN_pct"},
}

# Recorded at training time (2026-10 campaign). The reports only carry the Samples
# column — total usable rows on Plate, holdout rows on HRC — so the split breakdown
# below is carried verbatim from that campaign's training log rather than re-derived.
RECORDED_AT_TRAINING = {
    "Plate": {
        "heats_total": 1765, "heats_train": 1412, "heats_holdout": 353,
        "targets": {
            "YS_BN": {"train_rows": 6302, "holdout_rows": 1632, "usable_rows": 7934, "holdout_heats": 353},
            "TS_BN": {"train_rows": 6302, "holdout_rows": 1631, "usable_rows": 7933, "holdout_heats": 353},
            "EL_BN": {"train_rows": 6284, "holdout_rows": 1627, "usable_rows": 7911, "holdout_heats": 353},
            "YS_AN": {"train_rows": 578, "holdout_rows": 145, "usable_rows": 723, "holdout_heats": 118},
            "TS_AN": {"train_rows": 577, "holdout_rows": 145, "usable_rows": 722, "holdout_heats": 118},
            "EL_AN": {"train_rows": 577, "holdout_rows": 145, "usable_rows": 722, "holdout_heats": 118},
            "Impact_0C": {"train_rows": 933, "holdout_rows": 234, "usable_rows": 1167, "holdout_heats": 212},
            "Impact_m20C": {"train_rows": 372, "holdout_rows": 94, "usable_rows": 466, "holdout_heats": 86},
            "Impact_m40C": {"train_rows": 156, "holdout_rows": 39, "usable_rows": 195, "holdout_heats": 34},
            "Z_Test": {"train_rows": 314, "holdout_rows": 79, "usable_rows": 393, "holdout_heats": 72},
        },
    },
    "HRC": {
        "heats_total": 4595, "heats_train": 3676, "heats_holdout": 919,
        "targets": {
            "YS_BN": {"train_rows": 8884, "holdout_rows": 2221, "usable_rows": 11105, "holdout_heats": 919},
            "TS_BN": {"train_rows": 8884, "holdout_rows": 2221, "usable_rows": 11105, "holdout_heats": 919},
            "EL_BN": {"train_rows": 8884, "holdout_rows": 2221, "usable_rows": 11105, "holdout_heats": 919},
        },
    },
}

SPLIT_METHOD = "GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42) on Heat_ID"
SAMPLES_SEMANTICS = {
    "Plate": "leaderboard Samples = usable rows for that target",
    "HRC": "leaderboard Samples = holdout rows for that target",
}


def sanitise_counts(mill):
    """Cells repaired vs blanked, from the sanitising audit trail."""
    counts = {"cells_rescaled": 0, "cells_blanked": 0, "log_found": False}
    path = LOG[mill]
    if not os.path.exists(path):
        return counts
    try:
        log = pd.read_csv(path)
    except (OSError, ValueError):
        return counts
    counts["log_found"] = True
    reasons = log.get("reason", pd.Series(dtype=str)).astype(str)
    counts["cells_rescaled"] = int(reasons.str.startswith("decimal shift").sum())
    counts["cells_blanked"] = int((reasons == "out of bounds").sum())
    return counts


def current_targets(table, mill):
    out = {}
    for key, column in TARGET_COLUMNS[mill].items():
        if column not in table.columns:
            continue
        values = ts._numeric(table[column])
        rows = int(values.notna().sum())
        heats = int(table.loc[values.notna(), "Heat_ID"].nunique()) if "Heat_ID" in table.columns else None
        if rows:
            out[key] = {"usable_rows": rows, "unique_heats": heats}
    return out


def leaderboard_scores(mill):
    scores = {}
    try:
        lb = pd.read_csv(LEADERBOARD[mill])
    except (OSError, ValueError):
        return scores
    for _, row in lb.iterrows():
        entry = {}
        for source, target in (("Samples", "report_samples"),
                               ("Final_Val_R2", "val_r2"),
                               ("Final_Val_RMSE", "val_rmse"),
                               ("Val_HitRate_20MPa", "hit_rate"),
                               ("Val_HitRate_Tol", "hit_rate")):
            if source in lb.columns and pd.notna(row.get(source)):
                entry[target] = float(row[source])
        if entry:
            scores[row["Target"]] = entry
    return scores


def build(mill):
    table = ts.load_table(mill)
    if table.empty:
        print(f"  {mill}: no cleaned table, skipped")
        return None

    recorded = RECORDED_AT_TRAINING[mill]
    scores = leaderboard_scores(mill)
    as_trained_targets = {}
    for key, split in recorded["targets"].items():
        entry = dict(split)
        entry.update({k: v for k, v in scores.get(key, {}).items()})
        as_trained_targets[key] = entry

    counts = sanitise_counts(mill)
    heats = int(table["Heat_ID"].nunique()) if "Heat_ID" in table.columns else None

    return {
        "split_method": SPLIT_METHOD,
        "as_trained": {
            "note": "recorded at training time (2026-10 campaign); do not refresh from the "
                    "cleaned tables, which have since been sanitised",
            "heats_total": recorded["heats_total"],
            "heats_train": recorded["heats_train"],
            "heats_holdout": recorded["heats_holdout"],
            "targets": as_trained_targets,
        },
        "current": {
            "ingested_rows": int(len(table)) + counts_ingested_note(mill)[0],
            "duplicate_rows_removed": counts_ingested_note(mill)[0],
            "usable_rows": int(len(table)),
            "unique_heats": heats,
            "cells_rescaled": counts["cells_rescaled"],
            "cells_blanked": counts["cells_blanked"],
            "sanitise_log_found": counts["log_found"],
            "targets": current_targets(table, mill),
        },
        "samples_semantics": SAMPLES_SEMANTICS[mill],
    }


# Duplicate records removed by the 2026-10-30 sanitise pass, recorded here because the
# cleaned CSV no longer contains them. Plate had none; HRC carried 1,182 exact rows.
DUPLICATES_REMOVED = {"Plate": 0, "HRC": 1182}


def counts_ingested_note(mill):
    """(duplicates_removed, ) — kept as its own helper so the manifest stays explicit."""
    return DUPLICATES_REMOVED[mill],


def main():
    bundle = {"generated": date.today().isoformat(),
              "ingested_rows_recorded": {"Plate": 23845, "HRC": 20298}}
    for mill in ("Plate", "HRC"):
        payload = build(mill)
        if payload:
            bundle[mill] = payload
            cur = payload["current"]
            print(f"  {mill}: usable {cur['usable_rows']:,} rows / {cur['unique_heats']:,} melts, "
                  f"{cur['cells_rescaled']} cells rescaled, {cur['cells_blanked']} blanked, "
                  f"{cur['duplicate_rows_removed']} duplicates removed")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    print(f"written: {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
