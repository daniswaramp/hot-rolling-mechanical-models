"""Pickle compatibility for the shipped Plate artifacts.

The .joblib files record StackingEnsemble under this module path because the training run defined it
here. The tuning CLI that used to live in this file is gone - the notebooks carry that now - but the
class has to stay importable or every shipped model fails to load.
"""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold

from .config import CV_FOLDS
from .models import get_model_pipeline


class StackingEnsemble:
    """Blends base-model predictions through an L2-regularised meta-learner."""

    def __init__(self, model_names=None):
        self.model_names = model_names if model_names is not None else ["OLS", "Ridge", "XGBoost", "LightGBM"]
        self.meta_learner = RidgeCV(alphas=np.logspace(-3, 3, 20))
        self.fitted_base_models = {}

    def fit(self, X, y, groups):
        n_unique_groups = len(np.unique(groups))
        n_splits = min(CV_FOLDS, max(2, n_unique_groups // 2))
        gkf = GroupKFold(n_splits=n_splits)
        oof_preds = np.zeros((len(X), len(self.model_names)))

        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            X_tr, y_tr, X_te = X.iloc[train_idx], y.iloc[train_idx], X.iloc[test_idx]
            for m_i, m_name in enumerate(self.model_names):
                model = get_model_pipeline(m_name)
                model.fit(X_tr, y_tr)
                oof_preds[test_idx, m_i] = model.predict(X_te)

        self.fitted_base_models = {}
        for m_name in self.model_names:
            model = get_model_pipeline(m_name)
            model.fit(X, y)
            self.fitted_base_models[m_name] = model

        self.meta_learner.fit(oof_preds, y)
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
