"""
Machine Learning Models Suite for Plate Rolling Prediction.
Provides unified interface across:
1. Whitebox: Ordinary Least Squares (OLS)
2. Whitebox: Non-Linear Regression (NLR with Metallurgy Constitutive Physics)
3. Whitebox: Regularized Ridge Regression
4. Blackbox: XGBoost Regressor
5. Blackbox: LightGBM Regressor
6. Blackbox: CatBoost Regressor
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from scipy.optimize import curve_fit


class OLSModel(BaseEstimator, RegressorMixin):
    """Whitebox Ordinary Least Squares with Imputation, Standardized Scaling, and explicit coefficients."""
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = LinearRegression()
        self.feature_names_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        self.model.fit(X_scaled, y)
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        return self

    def predict(self, X):
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict(X_scaled)

    def get_formula_summary(self):
        """Returns interpretable coefficient table."""
        df_coef = pd.DataFrame({
            "feature": self.feature_names_,
            "std_coef": self.coef_,
            "abs_impact": np.abs(self.coef_)
        }).sort_values("abs_impact", ascending=False)
        return df_coef


class MetallurgicalNLRModel(BaseEstimator, RegressorMixin):
    """
    Whitebox Non-Linear Regression (NLR) fitting a constitutive metallurgical equation:
    Yield / Tensile = sigma0 + c_HP * (inv_sqrt_t) + c_ss * sigma_ss + c_ce * (CE_IIW^p1)
                      + c_cr * (Cooling_Rate^p2) + c_strain * (eps_total^p3)
    """
    def __init__(self):
        self.popt = None
        self.imputer = SimpleImputer(strategy="median")
        self.feature_cols = ["inv_sqrt_t", "sigma_ss", "CE_IIW", "Cooling_Rate_Proxy", "eps_total"]
        self.default_pred = 400.0

    @staticmethod
    def constitutive_func(X_mat, sigma0, c_hp, c_ss, c_ce, p1, c_cr, p2, c_eps):
        """Nonlinear metallurgical equation."""
        inv_sqrt_t = X_mat[:, 0]
        sigma_ss = X_mat[:, 1]
        ce = np.maximum(0.01, X_mat[:, 2])
        cr = np.maximum(0.0, X_mat[:, 3])
        eps = np.maximum(0.01, X_mat[:, 4])

        y_pred = (sigma0
                  + c_hp * inv_sqrt_t
                  + c_ss * sigma_ss
                  + c_ce * (ce ** np.clip(p1, 0.5, 2.0))
                  + c_cr * (cr ** np.clip(p2, 0.1, 1.0))
                  + c_eps * eps)
        return y_pred

    def fit(self, X, y):
        cols = [c for c in self.feature_cols if c in X.columns]
        if len(cols) < len(self.feature_cols):
            self.popt = None
            self.fallback = RidgeCV().fit(self.imputer.fit_transform(X), y)
            return self

        X_sub = self.imputer.fit_transform(X[self.feature_cols])
        y_val = y.values
        self.default_pred = np.mean(y_val)

        p0 = [150.0, 50.0, 1.0, 300.0, 1.0, 5.0, 0.3, 20.0]
        bounds = (
            [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.05, -50.0],
            [500.0, 300.0, 5.0, 1500.0, 2.0, 50.0, 1.0, 200.0]
        )

        try:
            popt, _ = curve_fit(self.constitutive_func, X_sub, y_val, p0=p0, bounds=bounds, maxfev=10000)
            self.popt = popt
        except Exception:
            self.popt = None
            self.fallback = RidgeCV().fit(self.imputer.fit_transform(X), y)

        return self

    def predict(self, X):
        if self.popt is None:
            return self.fallback.predict(self.imputer.transform(X))

        X_sub = self.imputer.transform(X[self.feature_cols])
        return self.constitutive_func(X_sub, *self.popt)

    def get_formula_summary(self):
        if self.popt is None:
            return "NLR fallback linear model active"
        p = self.popt
        formula = (
            f"Stress = {p[0]:.2f} + {p[1]:.2f}*(t^-0.5) + {p[2]:.3f}*(sigma_ss) + "
            f"{p[3]:.2f}*(CE^{p[4]:.2f}) + {p[5]:.2f}*(CoolingRate^{p[6]:.2f}) + {p[7]:.2f}*(eps_total)"
        )
        return formula


class RidgeModel(BaseEstimator, RegressorMixin):
    """Whitebox Regularized Ridge Regression with Cross-Validation tuning."""
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=np.logspace(-3, 4, 30))
        self.feature_names_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        X_imp = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imp)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_imp = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imp)
        return self.model.predict(X_scaled)


def get_model_pipeline(model_name, random_state=42, **kwargs):
    """
    Factory function instantiating model family instances.
    Supported: 'OLS', 'NLR', 'Ridge', 'XGBoost', 'LightGBM', 'CatBoost'.
    """
    name = model_name.upper()

    if name == "OLS":
        return OLSModel()

    elif name == "NLR":
        return MetallurgicalNLRModel()

    elif name == "RIDGE":
        return RidgeModel()

    elif name == "XGBOOST":
        from xgboost import XGBRegressor
        params = {
            "n_estimators": 350,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "n_jobs": -1
        }
        params.update(kwargs)
        return XGBRegressor(**params)

    elif name == "LIGHTGBM":
        from lightgbm import LGBMRegressor
        params = {
            "n_estimators": 400,
            "learning_rate": 0.04,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": random_state,
            "n_jobs": -1,
            "verbose": -1
        }
        params.update(kwargs)
        return LGBMRegressor(**params)

    elif name == "CATBOOST":
        from catboost import CatBoostRegressor
        params = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "random_seed": random_state,
            "verbose": False
        }
        params.update(kwargs)
        return CatBoostRegressor(**params)

    else:
        raise ValueError(f"Unsupported model family: {model_name}")
