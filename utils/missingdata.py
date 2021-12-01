'''
File: missingdata.py
Created Date: Monday November 29th 2021 - 11:18am
Author: Ammar Mian
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Wed Nov 31 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

import numpy as np
import copy

def generate_mcar(X, shape, missing_rate=.05, rng=42):
    """ Generate Missing Completely At Random data
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, shape)
    X_cp = copy.copy(X)
    np.putmask(X_cp, M, np.nan)

    return X_cp

def general_pattern_mask(shape, miss_k, miss_p, block_size, missing_rate=None):
    """ Generate random blocks of missing data in a mask M
    """
    k, p, n = shape
    M = np.zeros((k, p, n))

    #block_size = (missing_rate*k*p*n)//(100*nb_k*nb_p)
    #block_size = n-4*p
    #if block_size >= n//2:
    #    print("Missing block size is too large")

    slide, mod = 0, 50
    for k_num, k in enumerate(miss_k):
        for j in miss_p[k_num]:
            M[k, j, slide:slide+block_size] = 1
        slide += 10
        slide = slide % mod

    return M

def generate_missing_blocks(X, M):
    """ Mask data with a mask
    """
    X_cp = copy.copy(X)
    np.putmask(X_cp, M, np.nan)
    return X_cp


