import sklearn as skl
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot as plt
from sklearn.linear_model import RidgeCV


class KernelEstimator(skl.base.BaseEstimator, skl.base.TransformerMixin):
    """docstring"""

    def __init__(self, save_path=None):
        super(KernelEstimator, self).__init__()
        self.save_path = save_path

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.y_mean = np.mean(y)
        y -= self.y_mean
        Xt = np.transpose(X)
        cov = np.dot(X, Xt)
        alpha, _, _, _ = np.linalg.lstsq(cov, y)
        self.coef_ = np.dot(Xt, alpha)

        if self.save_path is not None:
            plt.figure()
            plt.hist(self.coef_[np.where(self.coef_ != 0)], bins=50, normed=True)
            plt.savefig(self.save_path + "KernelEstimatorCoef.png")
            plt.close()

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "y_mean"])
        X = check_array(X)

        prediction = np.dot(X, self.coef_) + self.y_mean

        if self.save_path is not None:
            plt.figure()
            plt.plot(prediction, "o")
            plt.savefig(self.save_path + "KernelEstimatorPrediction.png")
            plt.close()

        return prediction

    def score(self, X, y, sample_weight=None):
        scores = (self.predict(X) - y) ** 2 / len(y)
        score = np.sum(scores)

        if self.save_path is not None:
            plt.figure()
            plt.plot(scores, "o")
            plt.savefig(self.save_path + "KernelEstimatorScore.png")
            plt.close()

            df = pd.DataFrame({"score": scores})
            df.to_csv(self.save_path + "KernelEstimatorScore.csv")

        return score

    def set_save_path(self, save_path):
        self.save_path = save_path


class OptimRidgeCV(RidgeCV, skl.base.TransformerMixin):
    def __init__(self):
        super.__init__()
        self.estimate = None
        self.pred = None

    def fit(self, X, y):
        return super().fit(X, y)

    def predict(self, X):
        self.pred = np.round(super().predict(X)).astype(int)

        for i in range(self.pred.shape[0]):
            if self.pred[i, 1] < 18:
                self.pred[i, 1] = 18
            elif self.pred[i, 1] > 96:
                self.pred[i, 1] = 96

        return self.pred
