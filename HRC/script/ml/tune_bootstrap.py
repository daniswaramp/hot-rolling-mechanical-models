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

    def _base_models(self):
        stored = getattr(self, "fitted_base_models", None)
        if isinstance(stored, dict):
            return list(stored.values())
        return list(stored or [])

    def _meta(self):
        return getattr(self, "meta_learner", None) or getattr(self, "meta_model", None)

    def predict(self, X):
        """Both routes pickled a class of this name with different internals: plate stored a dict
        of base models with a `meta_learner`, Steckel stored a list with a `meta_model`."""
        base = self._base_models()
        preds = np.column_stack([m.predict(X) for m in base])
        return self._meta().predict(preds)
