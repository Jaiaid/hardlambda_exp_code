import os
import matplotlib.pyplot as plot
import numpy as np
import argparse

COMPARED_SYSTEMS_LABELS = ["Default", "SHADE", "DALI", "Proposed"]

BAR_LABELS = ["Default", "SHADE", "DALI", "Proposed"]

# each item is [resnet50, resnet101, efficientnetb0, ssd]
MEAN_TIME = [142, 651, 132, 27.7]
normalized_val = 142

if __name__=="__main__":
    # plot and axis object
    fig, ax = plot.subplots(figsize=(4, 2.25))
    xtick_labels = []

    for i, barval in enumerate(MEAN_TIME):
        ax.bar(i, normalized_val/barval, label=BAR_LABELS[i])
    
    # print(np.arange(0, len(NETWORK_NAMES)))
    # plot.tick_params(axis='both', which='major', labelsize=12)
    plot.yticks(fontweight='bold', fontsize=8)
    # set the legends and limit in y axis
    #ax.legend(ncol=3, loc="upper center", fontsize=6, frameon=False)
    #ax.get_legend().get_frame().set_alpha(None)
    #ax.get_legend().get_frame().set_facecolor((1, 1, 1, 0.1))
    # ax.set_yscale("log")
    ax.legend()
    ax.set_xticks(np.arange(0, len(BAR_LABELS)))
    ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("Avg. Batch Read Speedup", {'fontsize':8, 'fontweight': "bold"})
    
    fig.savefig("fig_mean_batch_readtime.png", dpi=600, bbox_inches="tight")
    # fig.savefig("fig_infer_computelatency.pdf", format="pdf", bbox_inches="tight")
