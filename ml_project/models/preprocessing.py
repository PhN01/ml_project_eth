from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import nilearn.image
import nibabel as nib
from skimage import feature
from biosppy.signals import ecg


class CropVoxelFrame(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cut = None
        self.X_3d = None
        self.X_new = None

    def fit(self, X, y=None):
        self.X_3d = np.reshape(X, (-1, 176, 208, 176))
        nonzero = np.array(np.nonzero(self.X_3d[0, :, :, :]))
        self.cut = [
            [np.amin(nonzero[0, :]), np.amax(nonzero[0, :])],
            [np.amin(nonzero[1, :]), np.amax(nonzero[1, :])],
            [np.amin(nonzero[2, :]), np.amax(nonzero[2, :])],
        ]

    def transform(self, X, y=None):
        check_is_fitted(self, ["cut"])

        self.X_new = self.X_3d[
            :,
            self.cut[0][0] : self.cut[0][1],
            self.cut[1][0] : self.cut[1][1],
            self.cut[2][0] : self.cut[2][1],
        ]

        return self.X_new


class DownsizeVoxel(BaseEstimator, TransformerMixin):
    def __init__(self, box_dim=[8, 10, 8]):
        self.x_dim, self.y_dim, self.z_dim = box_dim
        self.box = []
        self.new_box = []

    def fit(self, X, y=None):
        n = 0
        i = 0
        j = 0
        k = 0

        while n < X.shape[0]:
            self.box.append([])
            x_bound = self.x_dim
            while i < X.shape[1]:
                y_bound = self.y_dim
                while j < X.shape[2]:
                    z_bound = self.z_dim
                    while k < X.shape[3]:
                        self.new_box = X[n, i:x_bound, j:y_bound, k:z_bound]
                        self.box[n].append(np.sum(self.new_box))
                        k = z_bound
                        z_bound = z_bound + self.z_dim
                    k = 0
                    j = y_bound
                    y_bound = y_bound + self.y_dim
                j = 0
                i = x_bound
                x_bound = x_bound + self.x_dim
            i = 0
            n += 1

    def transform(self, X, y=None):

        X_new = self.box

        return X_new


class FlattenVoxel(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.n_samples = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_samples = X.shape[0]

        X_new = np.reshape(X, (n_samples, -1))

        return X_new


class CutSignal(BaseEstimator, TransformerMixin):
    def __init__(self, start=500, stop=7000):
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_new = X[:, self.start : self.stop]

        return X_new


class ECGCollect(BaseEstimator, TransformerMixin):
    def __init__(self, srate=300):
        self.n_samples = None
        self.srate = srate

    def fit(self, X, y=None):
        self.n_samples = X.shape[0]

        return self

    def transform(self, X, y=None):
        X_new = [
            ecg.ecg(signal=X[i, :], sampling_rate=self.srate, show=False)
            for i in range(self.n_samples)
        ]

        return X_new
