import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import pandas as pd
from math import pi
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib import rcParams

used_features = [2, 10, 50, 100, 200, 500, 1000]
clfs = ["GNB", "MLP", "HT"]
methods = ["OB", "SEA", "SIN"]
reductions = ["CV", "FS", "PCA"]

rcParams["font.family"] = "monospace"
colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
ls = ["-", "--", ":"]
lw = [1, 1, 1, 1, 1]

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)

for method in methods:
    for reduction in reductions:
        for n_components in used_features:
            selected_scores = np.load('results/%s-%s_%i.npy' %
                              (method, reduction, n_components))
            print("%s_%s_%i" % (method, reduction, n_components))

            mean_scores = np.mean(selected_scores, axis=1)

            fig = plt.figure(figsize=(4, 2.2))
            ax = plt.axes()
            for z, (value, label, mean) in enumerate(
                zip(np.squeeze(selected_scores), clfs, mean_scores)):
                label += "\n{0:.3f}".format(mean[0])
                val = gaussian_filter1d(value, sigma=2, mode="nearest")
                plt.plot(val, label=label, c=colors[z], ls=ls[z], lw=lw[z])

            ax.legend(
                loc=8,
                bbox_to_anchor=(0.5, -0.05),
                fancybox=False,
                shadow=True,
                ncol=3,
                fontsize=8,
                frameon=False,
            )

            plt.grid(ls=":", c=(0.7, 0.7, 0.7))
            plt.xlim(0, 107)
            axx = plt.gca()
            axx.spines["right"].set_visible(False)
            axx.spines["top"].set_visible(False)

            plt.title(
                "%i features" % (n_components),
                fontfamily="serif",
                y=1.04,
                fontsize=12,
            )
            plt.ylim(0.3, 1.0)
            plt.xticks(fontfamily="serif")
            plt.yticks(fontfamily="serif")
            plt.ylabel("score", fontfamily="serif", fontsize=8)
            plt.xlabel("chunks", fontfamily="serif", fontsize=8)
            plt.tight_layout()
            plt.savefig("figures/%s_%s_%i.png" % (method, reduction, n_components), bbox_inches='tight', dpi=250)
            plt.close()
