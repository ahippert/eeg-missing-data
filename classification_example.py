import warnings
import numpy as np
from numpy.random import default_rng
import pandas as pd
import copy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# Imputation librairies
from sklearn.impute import KNNImputer

# Dataset, paradigm and evaluation
import moabb
from moabb.datasets import BNCI2014009
from moabb.paradigms import P300

# Covariance estimation
import covariance_tyler
from pyCovariance.features import covariance, covariance_texture
from pyCovariance.classification import MDM

moabb.set_log_level("info")
rng = default_rng()

N_SPLITS = 5
SAVE_RESULTS = False

classifiers = {
    'SCM': MDM(feature=covariance(), n_jobs=-1, verbose=True),
    'EM-SCM': MDM(feature=covariance_tyler.covariance_md(), n_jobs=-1, verbose=True),
    'k-NN + SCM': MDM(feature=covariance(), n_jobs=-1, verbose=True),
    'SCM observed': MDM(feature=covariance_tyler.covariance_md_observed(), n_jobs=-1, verbose=True)
}

imputer = KNNImputer(missing_values=np.nan, add_indicator=False)

x_labels = ['Full SCM',
            'EM SCM',
            'k-NN Imputation',
            'SCM obs'
]

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

def compute_P1(X, y):
    """ Perform P1 computation
    """
    return np.mean(X[y=='Target'], axis=0)

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

def get_score(X, y, classifier, label, verbose):
    """ Get the accuracy score using k-folding
    """
    stk_fold = StratifiedKFold(n_splits=N_SPLITS)
    transformer = Transformer(feature=compute_P1)
    scores = []
    pipe = Pipeline([
        #('Imputer', imputer),
        ('P_transform', transformer),
        ('MDM', classifier)
    ])

    for k, (train, test) in enumerate(stk_fold.split(X, y)):

        # Define training and testing sets
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

        # Impute each epoch individually for adaptive training
        if label != 'SCM':
            X_train_imput = impute_per_epoch(X_train, imputer)
            X_to_compute_P1 = X_train_imput
        else:
            X_to_compute_P1 = X_train

        # Compute P1 (mean of target trials)
        P1 = compute_P1(X_to_compute_P1, y_train)

        #### Training ####
        if label == 'k-NN + SCM':
            X_train = X_train_imput

        pipe['P_transform'].feature = P1
        pipe.fit(X_train, y_train)

        #### Testing ####
        if label == 'k-NN + SCM':
            X_test_imput = impute_per_epoch(X_test, imputer)
            X_test = X_test_imput

        X_test_sup = transformer.P1_transform(X_test, P1)
        y_pred = pipe[-1].predict(X_test_sup)

        # Compute Overall Accuracy
        scores.append(100*np.sum(y_pred==y_test)/len(y_test))
        if verbose:
            print('[fold {0}] score: {1:.2f}'.format(k, scores[k]))

    return scores#np.mean(scores), np.std(scores)

def impute_data(imputer, X_missing, y_missing=None):
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

def main():

    # Upload P300 dataset
    paradigm = P300(resample=128)
    dataset = BNCI2014009()
    dataset.subject_list = dataset.subject_list[:1]
    X, _, _ = paradigm.get_data(dataset, [1])
    shape = X.shape
    shape = (300, 16, 103)

    # Experience
    mean_accu, std_accu = np.empty((0,4), dtype=object), []
    missing_epochs = rng.choice(shape[0], shape[0], replace=False)
    missing_electrodes = []
    for i in range(len(missing_epochs)):
        missing_electrodes.append(rng.choice(shape[1], 10, replace=False))
    #ratio_tab = [120, 345, 518, 690, 860, 1000]
    ratio_tab = [150]
    #block_size = [5, 15, 25, 35, 45, 55]
    block_size = [25]

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
            X = X[500:800]
            y = y[500:800]

            X_full = copy.copy(X) # make copy before masking
            X_ = copy.copy(X)
            X_ = generate_missing_blocks(X, M) # Generate missing data blocks

            # Get classification scores
            for k in classifiers:
                if k=='SCM': data = X_full
                else: data = X_
                scores = get_score(data, y , classifiers[k], k, verbose=True)

                for score in scores:
                    accuracies = np.vstack((mean_accu, np.asarray([[score, k, subject_no, ratio]], object)))
                    mean_accu = accuracies
            subject_no += 1

    #### Save results in pickle object file ####
    accuracies = pd.DataFrame(accuracies,
                              columns=["Accuracy", "Method", "Subject", "Ratio"])

    if SAVE_RESULTS: accuracies.to_pickle("accuracies.pkl")

if __name__ == "__main__":
    main()
