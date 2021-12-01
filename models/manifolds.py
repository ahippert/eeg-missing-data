'''
File: manifolds.py
Created Date: Monday November 29th 2021 - 11:27am
Author: Antoine Collas
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Nov 29 2021
Modified By: Ammar Mian
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

from robuststats.models.manifolds import WeightedProduct,\
                        SpecialHermitianPositiveDefinite,\
                            StrictlyPositiveVectors


class CovarianceTexture(WeightedProduct):
    """Manifold of (p x n)^k complex Hermitian positive definite matrices with 
    a product of (n)^k textures vectors.


    Attributes
    ----------
    _p : int
        Size of covariance matrix A.
    _n : int
        Size of texture vectors.
    _k : int, optional
        Number of covariance/texture features. Default is 1.
    """

    def __init__(self, p, n, alpha, k=1):
        self._p = p
        self._n = n
        self._k = k
        weights = (1/p, 1/n)
        manifolds = (SpecialHermitianPositiveDefinite(p, k),
                     StrictlyPositiveVectors(n, k))
        super().__init__(manifolds, weights)

