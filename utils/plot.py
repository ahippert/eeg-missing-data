'''
File: plot.py
Created Date: Monday November 29th 2021 - 11:12am
Author: Alexandre Hippert
Contact: hippert27@gmail.com
-----
Last Modified: Mon Nov 29 2021
Modified By: Ammar Mian
-----
Copyright (c) 2021 Université Savoie Mont-Blanc / CentraleSupelec
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
sns.plotting_context("notebook", font_scale=6)
sns.set(rc={'figure.figsize':(11,5)})


def plot_bar(accuracies, ratio_tab=[690], showfig=True, savefig=False):
    """Plot bar results from accuracies computed with varying ratio
    of missing data.

    Parameters
    ----------
    accuracies : [type]
        [description]
    ratio_tab : list, optional
        [description], by default [690]
    showfig : bool, optional
        [description], by default True
    savefig : bool, optional
        [description], by default False
    """
    shape = (1728, 16, 103)
    subject_list = np.arange(1,11)

    accuracies = accuracies[accuracies.Ratio != 57]

    # Draw a nested barplot
    for i in ratio_tab:
        g = sns.catplot(
            x="Subject", y="Accuracy", hue="Method",
            data=accuracies.loc[accuracies["Ratio"] == int(100*(i/shape[0]))],
            kind="bar", ci="sd", palette="mako", alpha=.6, height=5,
            legend=False
        )
        g.despine(left=True)
        g.set_axis_labels("Subject", "Mean accuracy (%)")
        plt.legend(loc='lower left')
        plt.ylim(57,95)
        if savefig: plt.savefig('accuracy_vs_subject_'+str(i)+'missingepochs.pdf')

    f = sns.catplot(
        x="Ratio", y="Accuracy", hue="Method",
        data=accuracies, kind="point",
        ci="sd", palette="YlGnBu_d", alpha=.6, height=4,
        legend=False
    )
    f.despine(left=True)
    f.set_axis_labels("Incomplete epochs (%)", "Mean accuracy (%)")
    plt.legend(loc='lower left')
    if savefig: plt.savefig('accuracy_vs_missingness.pdf')
    if showfig: plt.show()
