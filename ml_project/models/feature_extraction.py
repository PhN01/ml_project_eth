from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pywt
import statsmodels.tsa.stattools as tsa
import cv2
import skimage


class HistFeatures(BaseEstimator, TransformerMixin):
    """Random Selection of features"""

    def __init__(self, breaks=300, lower=10, upper=256):
        self.breaks = breaks
        self.lower = lower
        self.upper = upper
        self.X_hist

    def fit(self, X, y=None):

        X_norm = np.zeros(np.shape(X))

        k = 0
        n_samples = X.shape[0]

        while k < n_samples:
            X_norm[k, :, :, :] = (X[k, :, :, :] - np.min(X[k, :, :, :])) / (
                np.max(X[k, :, :, :]) - np.min(X[k, :, :, :])
            )

        X_norm = 256 * X_norm

        tmp = cv2.calcHist(
            X_norm[0, :, :, :].astype("uint8"),
            [0],
            None,
            [self.breaks],
            [self.lower, self.upper],
        )
        self.X_hist = np.array([]).reshape((0, tmp.shape[0]))

        k = 0
        while k < n_samples:
            self.X_hist = np.vstack(
                [
                    self.X_hist,
                    cv2.calcHist(
                        X_norm[k, :, :, :].astype("uint8"),
                        [0],
                        None,
                        [self.breaks],
                        [self.lower, self.upper],
                    ).ravel(),
                ]
            )
            k += 1

        return self

    def transform(self, X, y=None):

        X_new = self.X_hist

        return X_new


class CannyEdgeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, sigma=1):
        self.edges = None
        self.sigma = sigma

    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):

        self.edges = np.zeros(np.shape(X))

        for i in range(X.shape[0]):
            for j in range(X.shape[3]):
                self.edges[i, :, :, j] = skimage.feature.canny(
                    X[i, :, :, j],
                    sigma=self.sigma,
                    low_threshold=1,
                    high_threshold=255,
                    mask=None,
                )
        X_new = self.edges

        return X_new


class CubeHistFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, box_dim=[8, 10, 8]):
        self.X_new = None
        self.box_dim = box_dim

    def fit(self):
        self.x_dim, self.y_dim, self.z_dim = self.box_dim
        return self

    def transform(self, X):
        n = 0
        i = 0
        j = 0
        k = 0

        self.X_new = []
        while n < X.shape[0]:
            self.X_new.append([])
            x_bound = self.x_dim
            while i < X.shape[1]:
                y_bound = self.y_dim
                while j < X.shape[2]:
                    z_bound = self.z_dim
                    while k < X.shape[3]:
                        l = np.percentile(X[n, i:x_bound, j:y_bound, k:z_bound], 0.3)
                        u = np.percentile(X[n, i:x_bound, j:y_bound, k:z_bound], 0.99)
                        hist = np.histogram(
                            X[n, i:x_bound, j:y_bound, k:z_bound], bins=10, range=(l, u)
                        )
                        self.X_new[n].extend(hist.tolist())

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

        return self.X_new


class ECGFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, alt_wavelet="coif5"):
        self.n_samples = None
        self.mean_hbeats = None
        self.eukldist_hbeats = None
        self.extr_hbeats = None
        self.features = None
        self.ar_coefs_mean = None
        self.ar_coefs_extr = None
        self.db8_wavedec_mean = None
        self.db8_wavedec_extr = None
        self.alt_wavedec_mean = None
        self.alt_wavedec_extr = None
        self.alt_wavelet = alt_wavelet
        self.class0_mean_hbeat = None

    def fit(self, X, y=None):
        self.n_samples = X.shape[0]

        self.mean_hbeats = [np.mean(X[i][4], axis=0) for i in range(self.n_samples)]

        self.eukldist_hbeats = []
        for i in range(self.n_samples):
            sample = []
            for j in range(np.shape(X[i][4])[0]):
                sample.append(abs(np.linalg.norm(X[i][4][j] - self.mean_hbeats[i])))
            self.eukldist_hbeats.append(np.array(sample))

        self.extr_hbeats = []
        for i in range(self.n_samples):
            self.extr_hbeats.append(
                np.array(X[i][4][np.argmax(self.eukldist_hbeats[i])])
            )

        self.class0_mean_hbeat = [
            -4.6525,
            -4.0632,
            -3.3043,
            -2.372,
            -1.2673,
            0.0017,
            1.4174,
            2.9529,
            4.575,
            6.2479,
            7.9379,
            9.6167,
            11.2633,
            12.8633,
            14.4028,
            15.8586,
            17.1881,
            18.3266,
            19.1961,
            19.7236,
            19.8581,
            19.5736,
            18.8541,
            17.6718,
            15.9793,
            13.7282,
            10.9081,
            7.5774,
            3.8561,
            -0.1234,
            -4.2768,
            -8.5935,
            -13.102,
            -17.7969,
            -22.5907,
            -27.342,
            -31.9562,
            -36.486,
            -41.1447,
            -46.2013,
            -51.8286,
            -58.032,
            -64.7461,
            -72.0556,
            -80.3738,
            -90.3961,
            -102.7666,
            -117.5749,
            -133.8887,
            -149.4667,
            -160.6441,
            -162.3157,
            -148.0821,
            -110.8951,
            -44.6655,
            53.0148,
            177.6542,
            315.6235,
            444.6762,
            538.2731,
            573.0706,
            537.0624,
            434.9066,
            287.6889,
            126.6548,
            -16.7794,
            -120.6239,
            -177.5989,
            -194.316,
            -185.4865,
            -166.4374,
            -147.5035,
            -132.3269,
            -119.7634,
            -107.3508,
            -93.9991,
            -80.6319,
            -69.0593,
            -60.3824,
            -54.1969,
            -49.0228,
            -43.4379,
            -36.9941,
            -30.3066,
            -24.394,
            -19.8587,
            -16.5124,
            -13.6383,
            -10.5932,
            -7.2615,
            -4.0551,
            -1.5286,
            0.0524,
            0.8703,
            1.3682,
            1.9035,
            2.517,
            2.9673,
            2.9646,
            2.4064,
            1.4435,
            0.3578,
            -0.6316,
            -1.4829,
            -2.2994,
            -3.1994,
            -4.2005,
            -5.1995,
            -6.047,
            -6.6449,
            -6.9926,
            -7.155,
            -7.189,
            -7.0889,
            -6.7868,
            -6.2001,
            -5.2846,
            -4.057,
            -2.575,
            -0.8984,
            0.9408,
            2.9422,
            5.1185,
            7.4727,
            9.9869,
            12.6246,
            15.3398,
            18.084,
            20.8058,
            23.4481,
            25.9476,
            28.2387,
            30.2608,
            31.9625,
            33.3016,
            34.2419,
            34.7504,
            34.7986,
            34.3665,
            33.449,
            32.0581,
            30.2223,
            27.9818,
            25.3824,
            22.472,
            19.2994,
            15.9176,
            12.3876,
            8.7828,
            5.1885,
            1.6972,
            -1.6026,
            -4.6397,
            -7.3711,
            -9.7847,
            -11.8923,
            -13.7135,
            -15.2601,
            -16.526,
            -17.4907,
            -18.1333,
            -18.4493,
            -18.462,
            -18.2211,
            -17.789,
            -17.2224,
            -16.5585,
            -15.8123,
            -14.985,
            -14.0774,
            -13.1013,
            -12.0826,
            -11.0561,
            -10.0556,
            -9.1054,
            -8.2163,
            -7.3866,
            -6.6068,
            -5.8639,
            -5.146,
        ]

        return self

    def transform(self, X, y=None):
        self.features = pd.DataFrame(np.array([]).reshape((self.n_samples, 0)))

        # heart rate features
        self.features["hr_mean"] = [np.mean(X[i][6]) for i in range(self.n_samples)]
        self.features["ecg_hr_issue"] = list(np.zeros(self.n_samples))
        self.features["ecg_hr_issue"][
            self.features["hr_mean"].index[self.features["hr_mean"].apply(np.isnan)]
        ] = 1
        self.features["hr_mean"][
            self.features["hr_mean"].index[self.features["hr_mean"].apply(np.isnan)]
        ] = [
            np.mean(300 / np.diff(X[i][2]) * 60)
            for i in self.features["hr_mean"].index[
                self.features["hr_mean"].apply(np.isnan)
            ]
        ]
        self.features["hr_std"] = [np.std(X[i][6]) for i in range(self.n_samples)]
        self.features["hr_std"][
            self.features["hr_std"].index[self.features["hr_std"].apply(np.isnan)]
        ] = [
            np.std(300 / np.diff(X[i][2]) * 60)
            for i in self.features["hr_std"].index[
                self.features["hr_std"].apply(np.isnan)
            ]
        ]
        self.features["hr_0dist"] = self.features["hr_mean"] - 72.7
        # distance of the heart rate from the mean heart rate of class 0

        # features concerning the difference between consecutive R peaks
        self.features["rdif_mean"] = [
            np.mean(np.diff(X[i][2])) for i in range(self.n_samples)
        ]
        self.features["rdif_max"] = [
            np.amax(np.diff(X[i][2])) for i in range(self.n_samples)
        ]
        self.features["rdif_min"] = [
            np.amin(np.diff(X[i][2])) for i in range(self.n_samples)
        ]
        self.features["rdif_std"] = [
            np.std(np.diff(X[i][2])) for i in range(self.n_samples)
        ]
        self.features["rdif_q25"] = [
            np.percentile(np.diff(X[i][2]), 0.25) for i in range(self.n_samples)
        ]
        self.features["rdif_q75"] = [
            np.percentile(np.diff(X[i][2]), 0.75) for i in range(self.n_samples)
        ]

        # features concerning the distance of patient hbeats from
        # mean hbeat of class 0
        self.features["meanhb_0meandist"] = [
            abs(np.linalg.norm(self.mean_hbeats[i] - self.class0_mean_hbeat))
            for i in range(self.n_samples)
        ]
        self.features["extrhb_0meandist"] = [
            abs(np.linalg.norm(self.extr_hbeats[i] - self.class0_mean_hbeat))
            for i in range(self.n_samples)
        ]

        # features concerning the distance of the extremest hbeat from the
        # mean hbeat
        self.features["hbeat_meandist"] = [
            np.mean(self.eukldist_hbeats[i]) for i in range(self.n_samples)
        ]
        self.features["hbeat_maxdist"] = [
            np.amax(self.eukldist_hbeats[i]) for i in range(self.n_samples)
        ]
        self.features["hbeat_mindist"] = [
            np.amin(self.eukldist_hbeats[i]) for i in range(self.n_samples)
        ]
        self.features["hbeat_stddist"] = [
            np.std(self.eukldist_hbeats[i]) for i in range(self.n_samples)
        ]
        self.features["hbeat_q25dist"] = [
            np.percentile(self.eukldist_hbeats[i], 0.25) for i in range(self.n_samples)
        ]
        self.features["hbeat_q75dist"] = [
            np.percentile(self.eukldist_hbeats[i], 0.75) for i in range(self.n_samples)
        ]

        # autoregressive features
        self.ar_coefs_mean = [
            tsa.acf(self.mean_hbeats[i], nlags=4)[1:] for i in range(self.n_samples)
        ]  # ar coefs of mean hbeat
        self.ar_coefs_extr = [
            tsa.acf(self.extr_hbeats[i], nlags=4)[1:] for i in range(self.n_samples)
        ]  # ar coefs extreme hbeat
        self.features["meanhb_ar1"] = [
            self.ar_coefs_mean[i][0] for i in range(self.n_samples)
        ]
        self.features["meanhb_ar2"] = [
            self.ar_coefs_mean[i][1] for i in range(self.n_samples)
        ]
        self.features["meanhb_ar3"] = [
            self.ar_coefs_mean[i][2] for i in range(self.n_samples)
        ]
        self.features["meanhb_ar4"] = [
            self.ar_coefs_mean[i][3] for i in range(self.n_samples)
        ]
        self.features["extrhb_ar1"] = [
            self.ar_coefs_extr[i][0] for i in range(self.n_samples)
        ]
        self.features["extrhb_ar2"] = [
            self.ar_coefs_extr[i][1] for i in range(self.n_samples)
        ]
        self.features["extrhb_ar3"] = [
            self.ar_coefs_extr[i][2] for i in range(self.n_samples)
        ]
        self.features["extrhb_ar4"] = [
            self.ar_coefs_extr[i][3] for i in range(self.n_samples)
        ]

        # wavelet features
        # db8
        pca = PCA(n_components=5)
        self.db8_wavedec_mean = [
            pywt.wavedec(
                np.concatenate((self.mean_hbeats[i], self.mean_hbeats[i])),
                "db8",
                level=4,
            )
            for i in range(self.n_samples)
        ]
        mean_db8_a4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_mean)[i, 0])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_db8_d4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_mean)[i, 1])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_db8_d3 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_mean)[i, 2])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_db8_d2 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_mean)[i, 3])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_db8_d1 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_mean)[i, 4])
                    for i in range(self.n_samples)
                ]
            )
        )

        self.features["meanhb_db8a4pc1"] = mean_db8_a4[:, 0]
        self.features["meanhb_db8a4pc2"] = mean_db8_a4[:, 1]
        self.features["meanhb_db8a4pc3"] = mean_db8_a4[:, 2]
        self.features["meanhb_db8a4pc4"] = mean_db8_a4[:, 3]
        self.features["meanhb_db8a4pc5"] = mean_db8_a4[:, 4]
        self.features["meanhb_db8d4pc1"] = mean_db8_d4[:, 0]
        self.features["meanhb_db8d4pc2"] = mean_db8_d4[:, 1]
        self.features["meanhb_db8d4pc3"] = mean_db8_d4[:, 2]
        self.features["meanhb_db8d4pc4"] = mean_db8_d4[:, 3]
        self.features["meanhb_db8d4pc5"] = mean_db8_d4[:, 4]
        self.features["meanhb_db8d3pc1"] = mean_db8_d3[:, 0]
        self.features["meanhb_db8d3pc2"] = mean_db8_d3[:, 1]
        self.features["meanhb_db8d3pc3"] = mean_db8_d3[:, 2]
        self.features["meanhb_db8d3pc4"] = mean_db8_d3[:, 3]
        self.features["meanhb_db8d3pc5"] = mean_db8_d3[:, 4]
        self.features["meanhb_db8d2pc1"] = mean_db8_d2[:, 0]
        self.features["meanhb_db8d2pc2"] = mean_db8_d2[:, 1]
        self.features["meanhb_db8d2pc3"] = mean_db8_d2[:, 2]
        self.features["meanhb_db8d2pc4"] = mean_db8_d2[:, 3]
        self.features["meanhb_db8d2pc5"] = mean_db8_d2[:, 4]
        self.features["meanhb_db8d1pc1"] = mean_db8_d1[:, 0]
        self.features["meanhb_db8d1pc2"] = mean_db8_d1[:, 1]
        self.features["meanhb_db8d1pc3"] = mean_db8_d1[:, 2]
        self.features["meanhb_db8d1pc4"] = mean_db8_d1[:, 3]
        self.features["meanhb_db8d1pc5"] = mean_db8_d1[:, 4]

        self.db8_wavedec_extr = [
            pywt.wavedec(
                np.concatenate((self.extr_hbeats[i], self.extr_hbeats[i])),
                "db8",
                level=4,
            )
            for i in range(self.n_samples)
        ]
        extr_db8_a4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_extr)[i, 0])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_db8_d4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_extr)[i, 1])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_db8_d3 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_extr)[i, 2])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_db8_d2 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_extr)[i, 3])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_db8_d1 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.db8_wavedec_extr)[i, 4])
                    for i in range(self.n_samples)
                ]
            )
        )

        self.features["extrhb_db8a4pc1"] = extr_db8_a4[:, 0]
        self.features["extrhb_db8a4pc2"] = extr_db8_a4[:, 1]
        self.features["extrhb_db8a4pc3"] = extr_db8_a4[:, 2]
        self.features["extrhb_db8a4pc4"] = extr_db8_a4[:, 3]
        self.features["extrhb_db8a4pc5"] = extr_db8_a4[:, 4]
        self.features["extrhb_db8d4pc1"] = extr_db8_d4[:, 0]
        self.features["extrhb_db8d4pc2"] = extr_db8_d4[:, 1]
        self.features["extrhb_db8d4pc3"] = extr_db8_d4[:, 2]
        self.features["extrhb_db8d4pc4"] = extr_db8_d4[:, 3]
        self.features["extrhb_db8d4pc5"] = extr_db8_d4[:, 4]
        self.features["extrhb_db8d3pc1"] = extr_db8_d3[:, 0]
        self.features["extrhb_db8d3pc2"] = extr_db8_d3[:, 1]
        self.features["extrhb_db8d3pc3"] = extr_db8_d3[:, 2]
        self.features["extrhb_db8d3pc4"] = extr_db8_d3[:, 3]
        self.features["extrhb_db8d3pc5"] = extr_db8_d3[:, 4]
        self.features["extrhb_db8d2pc1"] = extr_db8_d2[:, 0]
        self.features["extrhb_db8d2pc2"] = extr_db8_d2[:, 1]
        self.features["extrhb_db8d2pc3"] = extr_db8_d2[:, 2]
        self.features["extrhb_db8d2pc4"] = extr_db8_d2[:, 3]
        self.features["extrhb_db8d2pc5"] = extr_db8_d2[:, 4]
        self.features["extrhb_db8d1pc1"] = extr_db8_d1[:, 0]
        self.features["extrhb_db8d1pc2"] = extr_db8_d1[:, 1]
        self.features["extrhb_db8d1pc3"] = extr_db8_d1[:, 2]
        self.features["extrhb_db8d1pc4"] = extr_db8_d1[:, 3]
        self.features["extrhb_db8d1pc5"] = extr_db8_d1[:, 4]

        # alternative wavelet
        self.alt_wavedec_mean = [
            pywt.wavedec(
                np.concatenate((self.mean_hbeats[i], self.mean_hbeats[i])),
                self.alt_wavelet,
                level=4,
            )
            for i in range(self.n_samples)
        ]
        mean_alt_a4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_mean)[i, 0])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_alt_d4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_mean)[i, 1])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_alt_d3 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_mean)[i, 2])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_alt_d2 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_mean)[i, 3])
                    for i in range(self.n_samples)
                ]
            )
        )
        mean_alt_d1 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_mean)[i, 4])
                    for i in range(self.n_samples)
                ]
            )
        )

        self.features["meanhb_alta4pc1"] = mean_alt_a4[:, 0]
        self.features["meanhb_alta4pc2"] = mean_alt_a4[:, 1]
        self.features["meanhb_alta4pc3"] = mean_alt_a4[:, 2]
        self.features["meanhb_alta4pc4"] = mean_alt_a4[:, 3]
        self.features["meanhb_alta4pc5"] = mean_alt_a4[:, 4]
        self.features["meanhb_altd4pc1"] = mean_alt_d4[:, 0]
        self.features["meanhb_altd4pc2"] = mean_alt_d4[:, 1]
        self.features["meanhb_altd4pc3"] = mean_alt_d4[:, 2]
        self.features["meanhb_altd4pc4"] = mean_alt_d4[:, 3]
        self.features["meanhb_altd4pc5"] = mean_alt_d4[:, 4]
        self.features["meanhb_altd3pc1"] = mean_alt_d3[:, 0]
        self.features["meanhb_altd3pc2"] = mean_alt_d3[:, 1]
        self.features["meanhb_altd3pc3"] = mean_alt_d3[:, 2]
        self.features["meanhb_altd3pc4"] = mean_alt_d3[:, 3]
        self.features["meanhb_altd3pc5"] = mean_alt_d3[:, 4]
        self.features["meanhb_altd2pc1"] = mean_alt_d2[:, 0]
        self.features["meanhb_altd2pc2"] = mean_alt_d2[:, 1]
        self.features["meanhb_altd2pc3"] = mean_alt_d2[:, 2]
        self.features["meanhb_altd2pc4"] = mean_alt_d2[:, 3]
        self.features["meanhb_altd2pc5"] = mean_alt_d2[:, 4]
        self.features["meanhb_altd1pc1"] = mean_alt_d1[:, 0]
        self.features["meanhb_altd1pc2"] = mean_alt_d1[:, 1]
        self.features["meanhb_altd1pc3"] = mean_alt_d1[:, 2]
        self.features["meanhb_altd1pc4"] = mean_alt_d1[:, 3]
        self.features["meanhb_altd1pc5"] = mean_alt_d1[:, 4]

        self.alt_wavedec_extr = [
            pywt.wavedec(
                np.concatenate((self.extr_hbeats[i], self.extr_hbeats[i])),
                self.alt_wavelet,
                level=4,
            )
            for i in range(self.n_samples)
        ]
        extr_alt_a4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_extr)[i, 0])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_alt_d4 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_extr)[i, 1])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_alt_d3 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_extr)[i, 2])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_alt_d2 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_extr)[i, 3])
                    for i in range(self.n_samples)
                ]
            )
        )
        extr_alt_d1 = pca.fit_transform(
            np.array(
                [
                    np.array(np.array(self.alt_wavedec_extr)[i, 4])
                    for i in range(self.n_samples)
                ]
            )
        )

        self.features["extrhb_alta4pc1"] = extr_alt_a4[:, 0]
        self.features["extrhb_alta4pc2"] = extr_alt_a4[:, 1]
        self.features["extrhb_alta4pc3"] = extr_alt_a4[:, 2]
        self.features["extrhb_alta4pc4"] = extr_alt_a4[:, 3]
        self.features["extrhb_alta4pc5"] = extr_alt_a4[:, 4]
        self.features["extrhb_altd4pc1"] = extr_alt_d4[:, 0]
        self.features["extrhb_altd4pc2"] = extr_alt_d4[:, 1]
        self.features["extrhb_altd4pc3"] = extr_alt_d4[:, 2]
        self.features["extrhb_altd4pc4"] = extr_alt_d4[:, 3]
        self.features["extrhb_altd4pc5"] = extr_alt_d4[:, 4]
        self.features["extrhb_altd3pc1"] = extr_alt_d3[:, 0]
        self.features["extrhb_altd3pc2"] = extr_alt_d3[:, 1]
        self.features["extrhb_altd3pc3"] = extr_alt_d3[:, 2]
        self.features["extrhb_altd3pc4"] = extr_alt_d3[:, 3]
        self.features["extrhb_altd3pc5"] = extr_alt_d3[:, 4]
        self.features["extrhb_altd2pc1"] = extr_alt_d2[:, 0]
        self.features["extrhb_altd2pc2"] = extr_alt_d2[:, 1]
        self.features["extrhb_altd2pc3"] = extr_alt_d2[:, 2]
        self.features["extrhb_altd2pc4"] = extr_alt_d2[:, 3]
        self.features["extrhb_altd2pc5"] = extr_alt_d2[:, 4]
        self.features["extrhb_altd1pc1"] = extr_alt_d1[:, 0]
        self.features["extrhb_altd1pc2"] = extr_alt_d1[:, 1]
        self.features["extrhb_altd1pc3"] = extr_alt_d1[:, 2]
        self.features["extrhb_altd1pc4"] = extr_alt_d1[:, 3]
        self.features["extrhb_altd1pc5"] = extr_alt_d1[:, 4]

        return self.features
