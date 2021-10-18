import numpy as np
import pandas as pd
from scipy.stats import f_oneway

shape = (1728, 16, 103)
subject_list = np.arange(1, 11)
ratio_tab = [690]
accuracies = pd.read_pickle("accuracies.pkl")
gaussian = accuracies.loc[accuracies["Method"] == 'Gaussian']
knn_gaussian = accuracies.loc[accuracies["Method"] == 'k-NN + Gaussian']
em_gaussian = accuracies.loc[accuracies["Method"] == 'Gaussian EM']
obs_gaussian = accuracies.loc[accuracies["Method"] == 'Gaussian observed']
ratio = int(100*(ratio_tab[0]/shape[0]))
ratio1 = gaussian.loc[gaussian["Ratio"] == ratio]
ratio2 = knn_gaussian.loc[knn_gaussian["Ratio"] == ratio]
ratio3 = em_gaussian.loc[em_gaussian["Ratio"] == ratio]
ratio4 = obs_gaussian.loc[obs_gaussian["Ratio"] == ratio]
for subject in subject_list:
    print(subject)
    acc1 = ratio1.loc[ratio1["Subject"] == subject]["Accuracy"]
    acc2 = ratio2.loc[ratio2["Subject"] == subject]["Accuracy"]
    acc3 = ratio3.loc[ratio3["Subject"] == subject]["Accuracy"]
    acc4 = ratio4.loc[ratio4["Subject"] == subject]["Accuracy"]
    print("one-way ANOVA: all groups")
    print(f_oneway(acc1, acc2, acc3, acc4))
    print("one-way ANOVA: KNN, EM, Gaussian observed")
    print(f_oneway(acc2, acc3, acc4))
    print("one-way ANOVA: Gaussian complete, KNN, EM")
    print(f_oneway(acc1, acc2, acc3))
    print("one-way ANOVA: KNN, EM")
    print(f_oneway(acc2, acc3))
    print("one-way ANOVA: KNN, Gaussian observed")
    print(f_oneway(acc2, acc4))
    print("one-way ANOVA: EM, Gaussian observed")
    print(f_oneway(acc3, acc4))
