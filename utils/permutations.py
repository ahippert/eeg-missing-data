'''
File: permutations.py
Created Date: Monday November 29th 2021 - 11:17am
Author: Alexandre Hippert-Ferrer
Contact: ammar.mian@univ-smb.fr
-----
Last Modified: Mon Nov 29 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

#!/usr/bin/python
import numpy as np
import copy

def permut_col(Y):
    """ Permutes block of missing variables to the right
    of matrix Y
    """
    X = copy.copy(Y)
    M = [not any(np.isnan(X[:,i])) for i in range(len(X[0]))]
    ind = np.arange(len(X[0]), dtype=int)
    M = np.array(M)
    ind[ind[M]] = np.arange(np.sum(M))
    ind[ind[M==False]] = np.arange(np.sum(M),len(X[0]))
    X[:,ind] = X[:,np.arange(len(X[0]))] # permutation
    return X

def permut_row(Y):
    """ Permutes block of missing variables to the bottom
    of matrix Y
    """
    M = binary_mask(Y)
    ind = np.arange(len(Y), dtype=int)
    ind[ind[M]] = np.arange(np.sum(M), dtype=int)
    ind[ind[M==False]] = np.arange(np.sum(M), len(Y), dtype=int)
    Y[ind] = Y[np.arange(len(Y))] # permutation
    return Y, ind

def permut_missing_block(X):
    """ Permutes on rows and columns so that the block of missing
    data is at the lower-right corner of matrix X
    """
    return permut_row(permut_col(X))

def permutation_index(Y):
    """ Permutes block of missing variables to the bottom
    of matrix Y
    """
    M = binary_mask(Y)
    ind = np.arange(len(Y), dtype=int)
    M_true_elem = np.sum(M)
    ind_true = ind[ind[M]]
    ind_false = ind[ind[M==False]]
    ind[:M_true_elem] = ind_true
    ind[M_true_elem:] = ind_false
    return ind

def binary_mask(Y):
    X = copy.copy(Y)
    M = [not np.isnan(X[i]) for i in range(len(X))]
    ind = np.arange(len(X), dtype=int)
    return np.array(M)

#def permut_vector(y):
#     """ Permutes missing values (NaN) to the right of vector
#     """
# X = np.array([1., np.nan, 3., np.nan, 5.])
# M = [not any(np.isnan(X[i])) for i in range(len(X))]
# ind = np.arange(len(X))
# M = np.array(M)
# ind[ind[M]] = np.arange(np.sum(M))
# ind[ind[M==False]] = np.arange(np.sum(M),len(X))
# X[ind] = X[np.arange(len(X))] # permutation

Y = np.random.randn(10,20)
Y[0,0] = np.nan
for i in range(len(Y)):
    Y[i], ids = permut_row(Y[i])

