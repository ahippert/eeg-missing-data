#!/usr/bin/env python
"""
Classification of P300 dataset with missing values.

Author : Alexandre Hippert-Ferrer, L2S, CentraleSupelec
Created : 26/08/2021
Last update : 26/08/2021
License : undefined
"""
import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pyriemann.estimation import Xdawn, XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb
from moabb.datasets import BNCI2014009
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import P300

from pyCovariance.classification import MDM
from pyCovariance.features.covariance import covariance

##############################################################################
# getting rid of the warnings about the future
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

moabb.set_log_level("info")

class Vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return X#np.reshape(X, X.shape)

def generate_mcar(X, shape, missing_rate=.3, rng=42):
    """ Generate Missing Completely At Random data
    """
    if not isinstance(rng, np.random.RandomState):
        rng = np.random.RandomState(rng)

    M = rng.binomial(1, missing_rate, shape)
    np.putmask(X, M, np.nan)

    return X

def main():

    # Create pipelines
    pipelines = {}
    labels_dict = {"Target": 1, "NonTarget": 0}

    pipelines["RG+LDA"] = make_pipeline(
        XdawnCovariances(nfilter=2, classes=[labels_dict["Target"]], estimator="lwf", xdawn_estimator="scm"),
        TangentSpace(),
        LDA(solver="lsqr", shrinkage="auto"),
    )

    pipelines["Xdw+LDA"] = make_pipeline(
        Xdawn(nfilter=2, estimator="scm"),
        Vectorizer(),
        LDA(solver="lsqr", shrinkage="auto")
    )

    pipelines["RG+MDM"] = make_pipeline(
        Xdawn(nfilter=2, estimator="scm"),
        Vectorizer(),
        MDM(covariance(), verbose=True),
    )

    # Upload P300 dataset
    print("Uploading dataset...")
    paradigm = P300(resample=128)
    dataset = BNCI2014009()
    dataset.subject_list = dataset.subject_list[:2]
    #datasets = [dataset]

    # Create missing values
    print("Creating missing values...")
    for subject in dataset.subject_list:
        # get the data
        X, y, metadata = paradigm.get_data(
            dataset, [subject]
        )
        # mask random values
        X_ = generate_mcar(X, X.shape)

    # Preprocess data for classification
    print("Preprocessing...")

    # Classification and evaluation
    print("Classifying...")
    overwrite = True  # set to True if we want to overwrite cached results
    evaluation = WithinSessionEvaluation(
    paradigm=paradigm, datasets=dataset, suffix="examples", overwrite=overwrite
    )
    results = evaluation.process(pipelines)

    # Plot results
    fig, ax = plt.subplots(facecolor="white", figsize=[8, 4])

    sns.stripplot(
        data=results,
        y="score",
        x="pipeline",
        ax=ax,
        jitter=True,
        alpha=0.5,
        zorder=1,
        palette="Set1",
    )
    sns.pointplot(data=results, y="score", x="pipeline", ax=ax, zorder=1, palette="Set1")

    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.5, 1)

    plt.show()

if __name__ == "__main__":
    main()
