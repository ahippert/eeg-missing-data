'''
File: demo_P300_dataset.py
Created Date: Monday November 29th 2021 - 11:17am
Author: Alexandre Hippert-Ferrer
Contact: alexandre.hippert-ferrer@centralesupelec.fr
-----
Last Modified: Wed Dec 01 2021
Modified By: Alexandre Hippert-Ferrer
-----
Copyright (c) 2021 Université Savoie Mont-Blanc, CentraleSupélec
'''

import sys
sys.path.append("../")
from utils.missingdata import *
from utils.transformers import compute_P1, Transformer
from models.imputers import *
from models.estimators import EmpiricalCovarianceMissingData, CovariancesEstimation

#from robuststats.estimation.base import CovariancesEstimation

from numpy.random import default_rng

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

from pyriemann.classification import MDM
from pyriemann.estimation import Covariances

import moabb
from moabb.paradigms import P300
from moabb.datasets import BNCI2014009

import pandas as pd

moabb.set_log_level("info")
rng = default_rng()

N_SPLITS = 5
SAVE_RESULTS = True

EM_estimator = EmpiricalCovarianceMissingData('EM-SCM')
SCM_estimator = EmpiricalCovarianceMissingData('SCM')
SCM_obs_estimator = EmpiricalCovarianceMissingData('SCM-obs')

methods = {'SCM': CovariancesEstimation(SCM_estimator),
           'EM-SCM': CovariancesEstimation(EM_estimator),
           'k-NN + SCM': CovariancesEstimation(SCM_estimator),
           'SCM-obs': CovariancesEstimation(SCM_obs_estimator)}

imputer = KNNImputer(missing_values=np.nan, add_indicator=False)

x_labels = ['Full SCM',
            'EM SCM',
            'k-NN Imputation',
            'SCM obs'
]

def get_score(X, y, method, method_name, verbose):
    """Get the accuracy score using k-folding

    Parameters
    ----------
    X : ndarray of shape (n_trials, n_features, n_samples)
        P300 data to be classified.
    y : ndarray of shape (n_trials,)
        Labels of each trial ("Target" or "Non-Target").
    method : type CovariancesEstimation
        Type of covariance estimator.
    method_name : string
        Name of the covariance estimator.
    verbose : bool
        To display scores for each folder.
    """

    stk_fold = StratifiedKFold(n_splits=N_SPLITS)
    transformer = Transformer(feature=compute_P1)
    scores = []
    pipe = Pipeline([
        ('P_transform', transformer),
        ('Feature', method),
        ('MDM', MDM(n_jobs=-1))
    ])

    for k, (train, test) in enumerate(stk_fold.split(X, y)):

        # Define training and testing sets
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # Impute each epoch individually for adaptive training
        if method_name != 'SCM':
            X_train_imput = impute_per_epoch(X_train, imputer)
            X_to_compute_P1 = X_train_imput
        else:
            X_to_compute_P1 = X_train

        # Compute P1 (mean of target trials)
        P1 = compute_P1(X_to_compute_P1, y_train)

        #### Training ####
        if method_name == 'k-NN + SCM':
            X_train = X_train_imput

        pipe['P_transform'].feature = P1
        pipe.fit(X_train, y_train)

        #### Testing ####
        if method_name == 'k-NN + SCM':
            X_test_imput = impute_per_epoch(X_test, imputer)
            X_test = X_test_imput

        X_test_sup = transformer.P1_transform(X_test, P1)
        y_pred = pipe.predict(X_test)

        # Compute Overall Accuracy
        scores.append(100*np.sum(y_pred==y_test)/len(y_test))
        if verbose:
            print('[fold {0}] score: {1:.2f}'.format(k, scores[k]))

    return scores#np.mean(scores), np.std(scores)

def main():

    # Upload P300 dataset
    paradigm = P300(resample=128)
    dataset = BNCI2014009()
    X, _, _ = paradigm.get_data(dataset, [1])
    shape = X.shape

    # Experience
    mean_accu, std_accu = np.empty((0,4), dtype=object), []
    missing_epochs = rng.choice(shape[0], shape[0], replace=False)
    missing_electrodes = []
    for i in range(len(missing_epochs)):
        missing_electrodes.append(rng.choice(shape[1], 10, replace=False))

    # Ratio of incomplete epochs
    ratio_tab = [120, 345, 518, 690, 860, 1000]

    # Block size of missing data
    block_size = [5, 15, 25, 35, 45, 55]

    # For each tuple (ratio, block_size), get the classification accuracy
    for i_miss, b_size in zip(ratio_tab, block_size):
        ratio = int(100*(i_miss/shape[0]))
        M = general_pattern_mask(shape,
                                 missing_epochs[:i_miss],
                                 missing_electrodes,
                                 b_size)
        subject_no = 1

        for subject in dataset.subject_list:
            # get the data
            X, y, metadata = paradigm.get_data(
                dataset, [subject]
            )

            X_full = copy.copy(X) # make copy before masking
            X_ = copy.copy(X)
            X_ = generate_missing_blocks(X, M) # Generate missing data blocks

            # Get classification scores
            for method in methods:
                if method=='SCM': data = X_full
                else: data = X_

                scores = get_score(data, y, methods[method], method, verbose=True)

                for score in scores:
                    accuracies = np.vstack((mean_accu, np.asarray([[score, method, subject_no, ratio]], object)))
                    mean_accu = accuracies
            subject_no += 1

    #### Save results in pickle object file ####
    accuracies = pd.DataFrame(accuracies,
                              columns=["Accuracy", "Method", "Subject", "Ratio"])

    if SAVE_RESULTS: accuracies.to_pickle("accuracies.pkl")

if __name__ == "__main__":
    main()
