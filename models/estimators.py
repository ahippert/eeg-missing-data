'''
File: permutations.py
Created Date: Monday November 29th 2021 - 11:17am
Authors: Alexandre Hippert-Ferrer, Ammar Mian
Contact: alexandre.hippert-ferrer@centralesupelec.fr
-----
Last Modified: Wed Dec 01 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

import numpy as np
import copy
import logging

import sys
sys.path.append("./")
import utils.permutations as perm

from sklearn.base import BaseEstimator, TransformerMixin

from robuststats.estimation.base import RealEmpiricalCovariance

def EM_covariance(X, init=None, tol=1e-4, iter_max=50, **kwargs):
    """ A function that estimates the covariance matrix of an incomplete dataset with general missing pattern under a Gaussian distribution
        Inputs:
            * 𝐗 = a matrix of size p*N with each observation along column dimension
            * low_rank = if True, perform low-rank estimation
            * rank = rank in R++ of covariance matrix
            * tol = tolerance for convergence of estimator
            * iter_max = number of maximum iterations
            * init = initial estimate for 𝚺, default is the identity matrix
        Outputs:
            * 𝚺 = covariance matrix estimate
            * tau = textures estimate
            * δ = the final distance between two iterations
            * iteration = number of iterations til convergence """

    # Initialisation of Sigma
    (p, N) = X.shape
    if init is None:
        S = np.eye(p)
    else:
        S = init

    # Get row indices where values are missing
    N_mis_indices = np.where(np.sum(np.isnan(X), axis=0))[0]
    N_obs_indices = np.where(np.sum(np.isnan(X), axis=0)==False)[0]
    N_mis = len(N_mis_indices)
    N_obs = len(N_obs_indices)

    # Get number of observed components where observations are missing
    M = p - np.sum(np.isnan(X[:,N_mis_indices]), axis=0)

    # Permute missing data at the bottom of each vector
    Y = copy.copy(X)
    ind = np.empty((N_mis,p), dtype=int)
    for i, i_mis in enumerate(N_mis_indices):
        Y[:,i_mis], ind[i] = perm.permut_row(Y[:,i_mis])

    # Form permutation matrices
    P = np.array([np.eye(p)[ind[i]].T for i in range(N_mis)])

    # Expectation-Maximization loop
    conv = 0       # Iteration number
    delta = np.inf # Distance error between two iterations
    while (delta>tol) and (conv<iter_max):

        # E step: compute conditional probabilities in matrix B
        B = e_step(Y, S, P, N_mis, N_mis_indices, M)

        # M step: update parameter using conditional probabilities
        S_hat = m_step(Y, B, P, N_obs_indices, N_mis)

        # Compute distance
        delta = np.linalg.norm(S_hat - S, 'fro') / np.linalg.norm(S, 'fro')

        # Replace previous covariance by current one
        S = S_hat

        conv = conv + 1

    return S

def e_step(Y, S, P, N_mis, N_mis_indices, M):

    (p, N) = Y.shape

    # Init matrix B (block matrix which is part of the M-step solution)
    B = np.zeros((N_mis,p,p))

    # Fill matrix B
    for (i, i_mis), j in zip(enumerate(N_mis_indices), M):
        D = P[i]@S@P[i].T # Apply P on each side of covariance
        D_mm = D[j:,j:]
        D_mo = D[j:,:j]
        D_oo = D[:j,:j]
        D_om = D[:j,j:]

        cov = D_mm - D_mo@np.linalg.inv(D_oo)@D_om
        mu_mis = D_mo@np.linalg.inv(D_oo)@np.vstack(Y[:j,i_mis])

        B[i, :j, :j] = Y[:j,i_mis,np.newaxis] @ Y[np.newaxis,:j,i_mis] # Upper left block
        B[i, j:, j:] = cov + mu_mis @ mu_mis.T # Lower right block
        B[i, j:, :j] = mu_mis @ Y[np.newaxis,:j,i_mis]
        B[i, :j, j:] = Y[:j,i_mis,np.newaxis] @ mu_mis.T

    return B

def m_step(Y, B, P, N_obs_indices, N_mis):

    (p, N) = Y.shape

    # Compute covariance and normalize by number of observations
    C = Y[:,N_obs_indices]@Y[:,N_obs_indices].T
    for i in range(N_mis):
        C += P[i].T@B[i]@P[i]
    Sigma_new = C.T/N

    # Low-rank approximation
    #if lowrank: #and rank>=opt_rank:
    #    Sigma_new = low_rank_covariance(Sigma_new, rank)

    return Sigma_new

def low_rank_covariance(S, rank):
    """ Performs Low Rank (LR) reconstruction from Algorithm 5 of Sun and Palomar (2016)
        Input:
            * S => covariance matrix
            * rank => desired rank for LR
        Ouput:
            * R => Low rank covariance matrix """

    # Get shape of covariance
    p = S.shape[0]

    # EVD of covariance matrix
    l, v = np.linalg.eig(S)

    # Get mean of last p-rank eigenvalues
    sig = np.mean(l[rank:])

    # Reconstruct R with Low Rank structure
    R = (l[:rank]-sig) * v[:,:rank]@v[:,:rank].T
    R += sig*np.eye(p) # Low eigenvalues part of the signal
    #R /= np.trace(R) # Normalize by trace

    return R

def empirical_covariance(X, transpose=True, assume_centered=False):
    """Computes the Maximum likelihood covariance estimator.
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.
    Returns
    -------
    covariance : ndarray of shape (n_features, n_features)
        Empirical covariance (Maximum Likelihood Estimator).
    """

    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")

    if transpose:
        X = X.T

    if assume_centered:
        covariance = np.dot(X.T, X) / X.shape[0]
    else:
        covariance = np.cov(X.T, bias=1)

    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return covariance

def empirical_covariance_observed(X, assume_centered=False):
    """ A function that computes the SCM for covariance matrix estimation
            Inputs:
                * X = a np.array of dim (n_features, n_samples)
                with each observation along column dimension
                * assume_centered = bool.
                If False, data are centered with empirical mean.
            Outputs:
                * Sigma = the estimate
    """
    p, N = X.shape

    if assume_centered:
        mean = np.mean(X, axis=1, keepdims=True)
        X = X - mean

    N_tab = np.arange(N)
    N_obs = N_tab[np.sum(np.isnan(X), axis=0)==0] # Fully observed columns

    scm = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            scm[i,j] = np.nansum(X[i,N_obs]*X[j,N_obs])

    return scm/len(N_obs)

class CovariancesEstimation(BaseEstimator, TransformerMixin):
    """ Estimate several covariances. Inspired by Covariances class
    of pyriemann at:
    >https://github.com/pyRiemann/pyRiemann/blob/master/pyriemann/estimation.py
    The difference is that estimator is not a string but an instance
    of a scikit-learn compatible estimator of covariance.
    """
    def __init__(self, estimator, **kwargs):
        self.estimator = estimator

    def fit(self, X, y=None):
        """Fit.
        Do nothing. For compatibility purpose.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials,)
            labels corresponding to each trial, not used.
        Returns
        -------
        self : Covariances instance
            The Covariances instance.
        """
        return self

    def transform(self, X):
        """Estimate covariance matrices.
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        Returns
        -------
        covmats : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance matrices for each trials.
        """
        Nt, Ne, Ns = X.shape
        covmats = np.zeros((Nt, Ne, Ne))
        for i in range(Nt):
            covmats[i, :, :] = self.estimator.fit_transform(X[i, :, :])
        return covmats

class EmpiricalCovarianceMissingData(RealEmpiricalCovariance):
    """EM estimator for incomplete data with a Gaussian distribution
    See:
    >A. Hippert-Ferrer et al.
    >"Robust low-rank estimation of the covariance matrix with a general pattern of missing values."
    >Sig. Proc. (in review), 2022.
    >https://arxiv.org/abs/2107.10505
    Parameters
    ----------
    tol : float, optional
        criterion for error between two iterations, by default 1e-4.
    method : str, optional
        way to compute the solution between
        'fixed-point', 'bcd' or 'gradient', by default 'fixed-point'.
    iter_max : int, optional
        number of maximum iterations of algorithm, by default 100.
    normalisation : str, optional
        type of normalisation between 'trace', 'determinant'
        or 'None', by default 'None'.
    verbosity : bool, optional
        show a progressbar of algorithm, by default False.
    Attributes
    ----------
    covariance_ : ndarray of shape (n_features, n_features)
        Estimated covariance matrix
    err_ : float
        final error between two iterations.
    iteration_ : int
        number of iterations done for estimation.
    """

    def __init__(self, method, tol=1e-4,
                 iter_max=50, normalisation='None',
                 verbosity=False):
        super().__init__()
        self.method = method
        self.verbosity = verbosity
        self.tol = tol
        self.iter_max = iter_max
        self.normalisation = normalisation

        self.set_method()
        self.err_ = np.inf
        self.iteration_ = 0

    def set_method(self):
        if self.method == 'EM-SCM':
            def estimation_function(X, init, **kwargs):
                return EM_covariance(X, init=init, tol=self.tol,
                                     iter_max=self.iter_max, **kwargs)
        elif self.method == 'SCM':
            def estimation_function(X, init, **kwargs):
                return empirical_covariance(X)

        elif self.method == 'SCM-obs':
            def estimation_function(X, init, **kwargs):
                return empirical_covariance_observed(X)
        else:
            logging.error("Estimation method not known.")
            raise NotImplementedError(f"Estimation method {self.method}"
                                      " is not known.")

        self._estimation_function = estimation_function

    def fit(self, X, y=None, init=None, **kwargs):
        """Fits the EM-SCM estimator of covariance matrix with the
        specified method when initialized object.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          Training data, where n_samples is the number of samples and
          n_features is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        init : array-like of shape (n_features, n_features), optional
            Initial point to start the estimation.
        Returns
        -------
        self : object
        """
        if self.iteration_ > 0 and self.verbosity:
            logging.warning("Overwriting previous fit.")
        #X = self._validate_data(X)
        covariance = self._estimation_function(X, init, **kwargs)
        self._set_covariance(covariance)
        #self.err_ = err
        #self.iteration_ = iteration
        return self

    def transform(self, X):
        return self.covariance_
