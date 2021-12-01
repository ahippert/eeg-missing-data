'''
File: imputers.py
Created Date: Monday November 29th 2021 - 11:17am
Author: Alexandre Hippert-Ferrer
Contact: alexandre.hippert-ferrer@centralesupelec.fr
-----
Last Modified: Wed Dec 01 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

import numpy as np

def impute_data(imputer, X_missing, y_missing=None):
    """ Imputation data with a defined imputer
    """
    if imputer is not None:
        X_imputed = imputer.fit_transform(X_missing)
    return X_imputed

def impute_per_epoch(X, imputer):
    """ Imputation per epoch
    """
    temp = []
    for i in range(X.shape[0]):
        temp.append(impute_data(imputer, X[i]))
    X_imput = np.array(temp)
    return X_imput
