"""
Evaluation and Metrics Computation Module.
Computes R2, RMSE, MAE, MAPE, and Tolerance Hit-Rate (+-20 MPa or +-10%).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred, unit="MPa"):
    """
    Computes standard regression metrics and industrial tolerance accuracy.
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    # Filter out NaNs if any
    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    if mask.sum() < 2:
        return {
            "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
            "MAPE_pct": np.nan, "HitRate_20MPa_pct": np.nan, "HitRate_10pct_pct": np.nan,
            "Samples": int(mask.sum())
        }

    y_t = y_t[mask]
    y_p = y_p[mask]

    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)

    # MAPE
    non_zero = y_t != 0
    mape = np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0

    # Tolerance hits
    abs_errors = np.abs(y_t - y_p)
    hit_20 = np.mean(abs_errors <= 20.0) * 100.0
    hit_10pct = np.mean(abs_errors <= (0.10 * np.abs(y_t))) * 100.0

    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "MAPE_pct": round(float(mape), 2),
        "HitRate_20MPa_pct": round(float(hit_20), 1),
        "HitRate_10pct_pct": round(float(hit_10pct), 1),
        "Samples": int(len(y_t))
    }
