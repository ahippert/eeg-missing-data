'''
File: permutations.py
Created Date: Monday November 29th 2021 - 11:17am
Author: Alexandre Hippert-Ferrer
Contact: alexandre.hippert-ferrer@centralesupelec.fr
-----
Last Modified: Mon Nov 29 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

def compute_P1(X, y):
    """ Perform P1 computation
    """
    return np.mean(X[y=='Target'], axis=0)

class Transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        self.feature = feature

    def fit(self, X, y):
        """fit."""
        return self

    def P1_transform(self, X, P1):
        """ Transform according to A. Barachant (2014)
        """
        Y = []
        for i in range(X.shape[0]):
            Y.append(np.concatenate((P1, X[i]), axis=0))
        Y = np.array(Y)
        return Y

    def transform(self, X):
        """ Transform X into Y = [P1 X]^T
        """
        P1 = self.feature
        Y = self.P1_transform(X, P1)
        return Y
