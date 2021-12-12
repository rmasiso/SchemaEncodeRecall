import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt


def regressor_to_TR(E, TR, nTR):
    T = E.shape[0]
    nEvents = E.shape[1]

    # HRF (from AFNI)
    dt = np.arange(0, 15)
    p = 8.6
    q = 0.547
    hrf = np.power(dt / (p * q), p) * np.exp(p - dt / q)

    # Convolve event matrix to get design matrix
    D = np.zeros((T, nEvents))
    for e in range(nEvents):
        D[:, e] = np.convolve(E[:, e], hrf)[:T]

    # Downsample event matrix to TRs
    timepoints = np.linspace(0, (nTR - 1) * TR, nTR)
    D_TR = np.zeros((len(timepoints), nEvents))
    for e in range(nEvents):
        D_TR[:, e] = np.interp(timepoints, np.arange(0, T), D[:, e])
        if np.max(D_TR[:, e]) > 0:
            D_TR[:, e] = D_TR[:, e] / np.max(D_TR[:, e])

    return D_TR


def regress_and_deconv(V, E, TR):
    # V is vox x TR data
    # E is time x event regressors
    # TR is time per TR

    # Returns vox x event betas

    D_TR = regressor_to_TR(E, TR, V.shape[1])
    return deconv(V, D_TR)


def deconv(V, D_TR):
    # Run linear regression to deconvolve
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(D_TR, V.T)
    return regr.coef_

