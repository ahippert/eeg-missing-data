import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
sns.set_theme(style="whitegrid")
sns.plotting_context("notebook", font_scale=6)
sns.set(rc={'figure.figsize':(11,5)})

shape = (1728, 16, 103)
subject_list = np.arange(1,11)

accuracies = pd.read_pickle("accuracies.pkl")
ratio_tab = [690]
accuracies = accuracies[accuracies.Ratio != 57]

# Draw a nested barplot
savefig = False
showfig = True
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
