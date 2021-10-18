'''
File: covariance_tyler.py
Created Date: Monday June 7th 2021 - 09:51am
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Jun 07 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc
'''

from pymanopt.manifolds import SpecialHermitianPositiveDefinite, HermitianPositiveDefinite, StrictlyPositiveVectors
import autograd.numpy as np
import autograd.numpy.linalg as la

from pyCovariance.features.covariance_texture import tyler_estimator_normalized_det

from pyCovariance.features.base import Feature, make_feature_prototype

import covariance_missing_values_dev

import matplotlib.pyplot as plt


@make_feature_prototype
def covariance_tyler(weights=None, p=None, N=None, **kwargs):

    name = 'Covariance with tyler normalised det'
    def _tyler(X):
        sigma, tau, _, _ = tyler_estimator_normalized_det(X)
        return sigma, tau

    M = (SpecialHermitianPositiveDefinite, StrictlyPositiveVectors)

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {'sizes': (p, N),
        'weights': weights}

    return Feature(name, _tyler, M, args_M)

@make_feature_prototype
def covariance_tyler_tau_moyenne(p=None, **kwargs):

    name = 'Covariance with tyler multiplied with tau mean'
    def _tyler(X):
        sigma, tau, _, _ = tyler_estimator_normalized_det(X)
        return np.mean(tau)*sigma

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _tyler, M, args_M)

@make_feature_prototype
def covariance_tyler_md(weights=None, p=None, N=None, **kwargs):

    M = (SpecialHermitianPositiveDefinite, StrictlyPositiveVectors)

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {'sizes': (p, N),
              'weights': weights}
    # M = SpecialHermitianPositiveDefinite
    # args_M = {'sizes':p}

    name = 'Covariance with missing values'
    def _tyler_missing(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian(X)
        # plt.plot(tau)
        # plt.show()
        return sigma, tau

    return Feature(name, _tyler_missing, M, args_M)

@make_feature_prototype
def covariance_tyler_md_fast(weights=None, p=None, N=None, **kwargs):

    M = (SpecialHermitianPositiveDefinite, StrictlyPositiveVectors)
    #M = HermitianPositiveDefinite

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {'sizes': (p, N),
              'weights': weights}

    name = 'Covariance with missing values'
    def _tyler_missing(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian_V2(X)
        return sigma, tau

    return Feature(name, _tyler_missing, M, args_M)

@make_feature_prototype
def covariance_tyler_md_lr(k, weights=None, p=None, N=None, **kwargs):

    M = (SpecialHermitianPositiveDefinite, StrictlyPositiveVectors)
    #M = HermitianPositiveDefinite

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {'sizes': (p, N),
              'weights': weights}

    name = 'Covariance with missing values'
    def _tyler_missing(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian(X, rank=k, lowrank=True)
        return sigma, tau

    return Feature(name, _tyler_missing, M, args_M)

@make_feature_prototype
def covariance_tyler_md_tau_moyenne(p=None, **kwargs):

    #M = (HermitianPositiveDefinite, StrictlyPositiveVectors)
    M = HermitianPositiveDefinite
    args_M = {'sizes': p}
    #if weights is None:
    #    weights = (1/p, 1/N)

    #args_M = {'sizes': (p, N),
    #          'weights': weights}

    name = 'Scaled-Gaussian covariance with missing values'
    def _tyler_missing(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian(X)
        return np.mean(tau)*sigma

    return Feature(name, _tyler_missing, M, args_M)

@make_feature_prototype
def covariance_tyler_md_tau_geo_moyenne(p=None, **kwargs):

    # M = (HermitianPositiveDefinite, StrictlyPositiveVectors)
    M = HermitianPositiveDefinite
    args_M = {'sizes': p}
    #if weights is None:
    #    weights = (1/p, 1/N)

    #args_M = {'sizes': (p, N),
    #          'weights': weights}

    name = 'Scaled-Gaussian covariance with missing values'
    def _tyler_missing(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian(X)
        geomean = tau.prod()**(1.0/len(tau))
        return geomean*sigma

    return Feature(name, _tyler_missing, M, args_M)

@make_feature_prototype
def covariance_tyler_md_tau_moyenne_lr(k, weights=None, p=None, N=None, **kwargs):

    M = (HermitianPositiveDefinite, StrictlyPositiveVectors)
    #M = HermitianPositiveDefinite

    if weights is None:
        weights = (1/p, 1/N)

    args_M = {'sizes': (p, N),
              'weights': weights}

    name = 'Scaled-Gaussian low-rank covariance with missing values'
    def _tyler_missing_lr(X):
        sigma, tau = covariance_missing_values_dev.estim_cmpd_gaussian(X, rank=k, lowrank=True)
        return np.mean(tau)*sigma, tau

    return Feature(name, _tyler_missing_lr, M, args_M)

@make_feature_prototype
def covariance_md(p=None, **kwargs):

    name = 'Gaussian covariance with missing values'
    def _scm_missing(X):
        sigma, _ = covariance_missing_values_dev.estim_gaussian(X, rank=None, lowrank=False)
        return sigma

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm_missing, M, args_M)

@make_feature_prototype
def covariance_md_normdet(p=None, **kwargs):

    name = 'Gaussian covariance with missing values'
    def _scm_missing(X):
        sigma, _ = covariance_missing_values_dev.estim_gaussian(X, normdet=True)
        return sigma

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm_missing, M, args_M)

@make_feature_prototype
def covariance_md_lr(k, p=None, **kwargs):

    name = 'Gaussian low-rank covariance with missing values'
    def _scm_missing(X):
        sigma, _ = covariance_missing_values_dev.estim_gaussian(X, rank=k, lowrank=True)
        return sigma

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm_missing, M, args_M)

@make_feature_prototype
def covariance_md_observed(p=None, **kwargs):

    name = 'SCM based on observed values'
    def _scm_missing(X):
        sigma = covariance_missing_values_dev.compute_scm_obs(X)
        return sigma

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _scm_missing, M, args_M)

@make_feature_prototype
def covariance_multiple_imputation(p=None, **kwargs):

    name = 'Covariance multiple imputation'
    def _tyler_multiple(X):
        """ Performs multiple covariance estimation.
        X is a list of k imputed datasets.
        """
        sigma_i = 0.
        trials = len(X)
        for xi in X:
            sigma, tau, _, _ = tyler_estimator_normalized_det(xi)
            sigma_i += sigma
        return sigma_i/trials

    M = HermitianPositiveDefinite
    args_M = {'sizes': p}

    return Feature(name, _tyler_multiple, M, args_M)


