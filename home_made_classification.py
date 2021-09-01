import warnings
import matplotlib.pyplot as plt
import numpy as np
import copy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Imputation librairies
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer

# Dataset, paradigm and evaluation
import moabb
from moabb.datasets import BNCI2014009
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300

# Covariance estimation
import covariance_tyler
from pyCovariance.features import covariance, covariance_texture
from pyCovariance.classification import MDM

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

moabb.set_log_level("info")

N_SPLITS = 5
classifiers = [MDM(feature=covariance(), n_jobs=-1, verbose=True),
               MDM(feature=covariance_texture(), n_jobs=-1, verbose=True),
               MDM(feature=covariance(), n_jobs=-1, verbose=True),
               MDM(feature=covariance(), n_jobs=-1, verbose=True),
               MDM(feature=covariance(), n_jobs=-1, verbose=True)]

imputers = [None,
            None,
            SimpleImputer(missing_values=np.nan, add_indicator=False,
                          strategy='constant', fill_value=0),
            SimpleImputer(missing_values=np.nan, strategy="mean",
                          add_indicator=False),
            KNNImputer(missing_values=np.nan, add_indicator=False)]
#IterativeImputer(missing_values=np.nan, add_indicator=False,
#random_state=0, n_nearest_features=5,
#sample_posterior=True)]

class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, shape):
        self.shape = shape

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (self.shape[0], self.shape[1]*self.shape[2]))

    def inv_transform(self, X):
        """inverser transform"""
        return np.reshape(X, (self.shape[0], self.shape[1], self.shape[2]))

class Concatenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform according to A. Barachant (2014) """
        P1 = np.mean(X, axis=2)
        X = np.concatenate((P1, X), axis=0)
        return X

def concatenate():
    pass

def generate_mcar(X, shape, missing_rate=.3, rng=42):
    """ Generate Missing Completely At Random data
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, shape)
    np.putmask(X, M, np.nan)

    return X

def get_score(X, y, classifier, verbose):
    # full_scores = cross_val_score(classifier, X_full, y_full,
    #                               scoring='accuracy',
    #                               cv=N_SPLITS)
    n_classes = 2
    k_fold = KFold(n_splits=N_SPLITS)
    scores = []
    for k, (train, test) in enumerate(k_fold.split(X)):

        # Training
        X_train, y_train = X[train], y[train]
        classifier.fit(X_train, y_train)

        # Testing
        X_test, gt = X[test], y[test]
        C = classifier.predict(X_test)

        # Compute Overall Accuracy
        scores.append(100*np.sum(C==gt)/len(gt))
        if verbose:
            print('[fold {0}] score: {1:.2f}'.format(k, scores[k]))

    return np.mean(scores), np.std(scores)

def get_imputer_score(X_missing, y_missing, imputer, classifier):
    """Get classification accuracy after data imputation
    """
    if imputer is not None:
        vec = Vectorizer(X_missing.shape)
        X_missing = vec.transform(X_missing)
        X_imputed = imputer.fit_transform(X_missing, y_missing)
        X_missing = vec.inv_transform(X_imputed)

    # Classification
    mean_impute_scores, std_impute_scores = get_score(X_missing, y_missing, classifier, verbose=True)

    return mean_impute_scores, std_impute_scores

def main():

    # Upload P300 dataset
    print("Uploading dataset...")
    paradigm = P300(resample=128)
    dataset = BNCI2014009()
    dataset.subject_list = dataset.subject_list[:1]

    # Create missing values
    print("Creating missing values...")
    for subject in dataset.subject_list:
        # get the data
        X, y, metadata = paradigm.get_data(
            dataset, [subject]
        )
    # mask random values
    X_full = copy.copy(X) # make copy before masking
    X_ = generate_mcar(X, X.shape)

    # Preprocess data for classification
    print("Preprocessing...")
    concatenate()

    # Imputation and classification
    mean_accu, std_accu = [], []
    for imputer, classifier in zip(imputers, classifiers):
        if imputer is None:
            mean_k, std_k = get_score(X_full, y, classifier, verbose=True)
        else:
            mean_k, std_k = get_imputer_score(X_, y, imputer, classifier)
        mean_accu.append(mean_k)
        std_accu.append(std_k)

    # Plot results
    x_labels = ['Full data Gaussian',
                'Full data Compound Gaussian',
                'Zero imputation',
                'Mean Imputation',
                'KNN Imputation',]
    n_bars = len(mean_accu)
    xval = np.arange(n_bars)

    colors = ['r', 'g', 'b', 'orange', 'black']

    plt.figure(figsize=(8, 6))
    ax = plt.subplot()
    for j in xval:
        ax.barh(j, mean_accu[j], xerr=std_accu[j],
                color=colors[j], alpha=0.6, align='center')

    ax.set_title('Imputation Techniques with P300 Data')
    ax.set_xlim(left=np.min(mean_accu) * 0.75,
                 right=np.max(mean_accu) * 1.25)
    ax.set_yticks(xval)
    ax.set_xlabel('Accuracy')
    ax.invert_yaxis()
    ax.set_yticklabels(x_labels)
    plt.show()
if __name__ == "__main__":
    main()
