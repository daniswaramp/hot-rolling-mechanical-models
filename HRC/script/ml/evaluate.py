"""
Evaluation Metrics Module for Steckel Mill (HRC).
Computes R2, RMSE, MAE, MAPE, and industrial tolerance hit rates.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred, unit="MPa"):
    """
    Computes regression evaluation metrics:
    - R2 Score
    - RMSE
    - MAE
    - MAPE (%)
    - Industrial Hit Rate: within +-20 MPa (or +-10% relative for elongation)
    """
    y_t = np.array(y_true, dtype=float)
    y_p = np.array(y_pred, dtype=float)

    mask = ~np.isnan(y_t) & ~np.isnan(y_p)
    y_t = y_t[mask]
    y_p = y_p[mask]

    if len(y_t) == 0:
        return {
            "R2": 0.0,
            "RMSE": 999.0,
            "MAE": 999.0,
            "MAPE": 999.0,
            "Hit_Rate_Tol": 0.0,
            "Samples": 0
        }

    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)

    non_zero = y_t != 0
    mape = np.mean(np.abs((y_t[non_zero] - y_p[non_zero]) / y_t[non_zero])) * 100.0 if np.any(non_zero) else 0.0

    if unit.lower() in ("mpa", "j"):
        # Industrial tolerance: +- 20 MPa or +- 20 J
        hit_rate = np.mean(np.abs(y_t - y_p) <= 20.0) * 100.0
    else:
        # Elongation or percentage: +- 2.5% absolute or +- 10% relative
        hit_rate = np.mean(np.abs(y_t - y_p) <= 2.5) * 100.0

    return {
        "R2": round(float(r2), 4),
        "RMSE": round(float(rmse), 2),
        "MAE": round(float(mae), 2),
        "MAPE": round(float(mape), 2),
        "Hit_Rate_Tol": round(float(hit_rate), 1),
        "Samples": int(len(y_t))
    }
