"""
Model Architectures for Steckel Mill (HRC) Mechanical Property Prediction.
Implements:
1. Whitebox Models:
   - OLSModel: Ordinary Least Squares with Median Imputation & Standardization
   - NLRModel: Non-Linear Constitutive Metallurgical Curve Fit (Pickering + Hall-Petch + Strain)
   - RidgeModel: Ridge L2 Regression
2. Gradient Boosted Trees:
   - XGBoostModel
   - LightGBMModel
   - CatBoostModel
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.optimize import curve_fit
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import logging

logger = logging.getLogger("Models_HRC")


class OLSModel(BaseEstimator, RegressorMixin):
    """Ordinary Least Squares Regression with Imputer and Scaler."""
    def __init__(self):
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    @property
    def feature_importances_(self):
        if hasattr(self.pipe.named_steps["reg"], "coef_"):
            return np.abs(self.pipe.named_steps["reg"].coef_)
        return None


class RidgeModel(BaseEstimator, RegressorMixin):
    """Ridge L2 Regularized Linear Model."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=self.alpha, random_state=42))
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)

    @property
    def feature_importances_(self):
        if hasattr(self.pipe.named_steps["reg"], "coef_"):
            return np.abs(self.pipe.named_steps["reg"].coef_)
        return None


class NLRModel(BaseEstimator, RegressorMixin):
    """
    Non-Linear Metallurgical Constitutive Fit (Whitebox).
    y = sigma_0 + k_ss * sigma_ss + k_hp * (1/sqrt(t)) + k_eps * eps^n - k_ct * (CT - 500)
    Grounded in Pickering, Hall-Petch, and dislocation recovery theory.
    """
    def __init__(self):
        self.params = None
        self.feature_names = ["sigma_ss_MPa", "Hall_Petch_t_inv_sqrt", "Steckel_True_Strain", "Temp_CT_Avg_C"]

    @staticmethod
    def _constitutive_func(X_arr, s0, k_ss, k_hp, k_eps, n_eps, k_ct):
        sig_ss = X_arr[:, 0]
        hp = X_arr[:, 1]
        eps = np.maximum(X_arr[:, 2], 1e-4)
        ct = np.maximum(X_arr[:, 3] - 500.0, 0.0)
        return s0 + k_ss * sig_ss + k_hp * hp + k_eps * np.power(eps, n_eps) - k_ct * ct

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        for f in self.feature_names:
            if f not in X_df.columns:
                X_df[f] = 0.0

        sub_X = X_df[self.feature_names].fillna(0.0).values
        y_arr = np.array(y, dtype=float)

        p0 = [150.0, 1.0, 200.0, 50.0, 0.5, 0.2]
        bounds = (
            [-200.0, 0.1, 10.0, 0.0, 0.1, 0.0],
            [600.0, 3.0, 800.0, 300.0, 1.5, 2.0]
        )

        try:
            popt, _ = curve_fit(self._constitutive_func, sub_X, y_arr, p0=p0, bounds=bounds, maxfev=5000)
            self.params = popt
        except Exception as e:
            logger.warning(f"NLR curve_fit did not converge ({e}); using initial parameters.")
            self.params = p0

        return self

    def predict(self, X):
        X_df = pd.DataFrame(X)
        for f in self.feature_names:
            if f not in X_df.columns:
                X_df[f] = 0.0
        sub_X = X_df[self.feature_names].fillna(0.0).values
        return self._constitutive_func(sub_X, *self.params)

    def get_formula_string(self, target_name="Target"):
        if self.params is None:
            return "NLR model not fitted."
        s0, k_ss, k_hp, k_eps, n_eps, k_ct = self.params
        return (f"{target_name} = {s0:.2f} + {k_ss:.3f}*(sigma_ss) + {k_hp:.2f}*(1/sqrt(t)) "
                f"+ {k_eps:.2f}*(eps_STM)^{n_eps:.2f} - {k_ct:.3f}*(CT_Avg - 500)")


class XGBoostModel(BaseEstimator, RegressorMixin):
    """XGBoost Regressor Wrapper."""
    def __init__(self, n_estimators=300, max_depth=6, learning_rate=0.05,
                 subsample=0.85, colsample_bytree=0.85, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.model else None


class LightGBMModel(BaseEstimator, RegressorMixin):
    """LightGBM Regressor Wrapper."""
    def __init__(self, n_estimators=350, max_depth=7, num_leaves=31, learning_rate=0.04,
                 subsample=0.85, colsample_bytree=0.85, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.model else None


class CatBoostModel(BaseEstimator, RegressorMixin):
    """CatBoost Regressor Wrapper."""
    def __init__(self, iterations=400, depth=6, learning_rate=0.05, random_state=42):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            random_seed=self.random_state,
            verbose=0,
            thread_count=-1
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_ if self.model else None


def get_model_pipeline(model_name: str, **kwargs):
    """Factory creating fresh instances of models."""
    name_lower = model_name.lower()
    if name_lower == "ols":
        return OLSModel()
    elif name_lower == "nlr":
        return NLRModel()
    elif name_lower == "ridge":
        return RidgeModel(**kwargs)
    elif name_lower == "xgboost":
        return XGBoostModel(**kwargs)
    elif name_lower == "lightgbm":
        return LightGBMModel(**kwargs)
    elif name_lower == "catboost":
        return CatBoostModel(**kwargs)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
