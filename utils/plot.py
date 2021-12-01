'''
File: plot.py
Created Date: Monday November 29th 2021 - 11:12am
Author: Alexandre Hippert-Ferrer
Contact: hippert27@gmail.com
-----
Last Modified: Wed Dec 01 2021
Modified By: Alexandre Hippert-Ferrer
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

def plot_bar(accuracies, missing_ratio, showfig=True, savefig=False):
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

    #accuracies = accuracies[accuracies.Ratio != 57]

    # Draw a nested barplot
    for i in missing_ratio:
        g = sns.catplot(
            x="Subject", y="Accuracy", hue="Method",
            data=accuracies.loc[accuracies["Ratio"] == int(i)],
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

def main():

    # Read pkl file containing classification accuracies
    accuracies = pd.read_pickle("../simulations/accuracies.pkl")

    # Accuracies are also plotted for a specific missing data ratio
    ratio_missing_data = [39]

    # Plot accuracies
    plot_bar(accuracies, ratio_missing_data)

if __name__ == "__main__":
    main()
