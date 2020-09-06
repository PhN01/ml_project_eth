import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


class MeanPredictor(BaseEstimator, TransformerMixin):
    """docstring for MeanPredictor"""

    def fit(self, X, y):
        self.mean = y.mean(axis=0)
        return self

    def predict_proba(self, X):
        check_array(X)
        check_is_fitted(self, ["mean"])
        n_samples, _ = X.shape
        return np.tile(self.mean, (n_samples, 1))


class SVClassifier(SVC, TransformerMixin):
    def __init__(self):
        super().__init__(
            C=1.0, kernel="rbf", gamma="auto", probability=True, shrinking=True
        )

    def fit(self, X, y):
        y = np.argmax(y, axis=1)
        return super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def score(self, X, y):
        y = np.argmax(y, axis=1)
        probs = super().predict_proba(X)
        return log_loss(y, probs)


class GBClassifier(GradientBoostingClassifier, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y):
        return super().fit(X, y)

    def predict_proba(self, X):
        return super().predict_proba(X)

    def predict(self, X):
        return super().predict(X)
