"""Pickle compatibility for the shipped HRC artifacts. See the Plate module for the full note."""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV

from .models import get_model_pipeline


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking meta-learner blending linear and tree predictions."""

    def __init__(self, base_model_names=None):
        self.base_model_names = base_model_names or ["Ridge", "XGBoost", "LightGBM", "CatBoost"]
        self.meta_model = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.fitted_base_models = []

    def fit(self, X, y):
        self.fitted_base_models = []
        meta_features = []
        for name in self.base_model_names:
            model = get_model_pipeline(name)
            model.fit(X, y)
            self.fitted_base_models.append(model)
            meta_features.append(model.predict(X))
        self.meta_model.fit(np.column_stack(meta_features), y)
        return self

    def predict(self, X):
        return self.meta_model.predict(np.column_stack([m.predict(X) for m in self.fitted_base_models]))
