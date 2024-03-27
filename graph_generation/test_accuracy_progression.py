import os
import re
import matplotlib.pyplot as plot
import numpy as np
import argparse

EPOCH_PLOT = 30
NETWORK_NAME = "vgg16"
DATASET = "cifar10"
LOGFOLDER = "20240327_datacache_nonshuffled_idx_shuffled_cifar10_30epoch"
COMPARED_SYSTEMS_LABELS = ["Default", "SHADE", "Proposed"]
COMPARED_SYSTEMS_LABELS = ["Default", "SHADE", "Proposed"]
DATA_FILES_DICT = {"Default": "rank0_accuracy_{0}_default.log".format(NETWORK_NAME), "Proposed": "rank0_accuracy_{0}_graddistbg.log".format(NETWORK_NAME), "SHADE": "rank0_accuracy_{0}_shade.log".format(NETWORK_NAME)}
VAL_ACC_LINE_REGEX = r" *   Acc@1 (\d*\.\d+) Acc@5 (\d*\.\d+)" 


if __name__=="__main__":
    # plot and axis object
    fig, ax = plot.subplots(figsize=(4, 2.25))
    xtick_labels = []

    for system in COMPARED_SYSTEMS_LABELS:
        data_file = os.path.join(LOGFOLDER, DATA_FILES_DICT[system])
        if not os.path.exists(data_file):
            print("{0} data file not exist, {1} will not be plotted".format(data_file, system))
            continue

        top1acc = []
        top5acc = []
        with open(data_file) as fin:
            for line in fin.readlines():
                match = re.search(VAL_ACC_LINE_REGEX , line)
                if match is not None:
                    top1acc.append(float(match.groups()[0]))
                    top5acc.append(float(match.groups()[1]))

        ax.plot([i for i in range(min(EPOCH_PLOT, len(top1acc)))], top1acc[:min(EPOCH_PLOT, len(top1acc))], label=system, marker='x')
        # print(np.arange(0, len(NETWORK_NAMES)))
        # plot.tick_params(axis='both', which='major', labelsize=12)
        # set the legends and limit in y axis
        #ax.legend(ncol=3, loc="upper center", fontsize=6, frameon=False)
        #ax.get_legend().get_frame().set_alpha(None)
        #ax.get_legend().get_frame().set_facecolor((1, 1, 1, 0.1))
        # ax.set_yscale("log")
    plot.yticks(fontweight='bold', fontsize=8)
    plot.xticks(fontweight='bold', fontsize=6)
    ax.legend()
    ax.set_xticks(np.arange(0, EPOCH_PLOT))
    ax.set_xlabel("Epoch", {'fontsize':8, 'fontweight': "bold"})
    # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("Top-1 Accuracy", {'fontsize':8, 'fontweight': "bold"})
    
    fig.savefig("fig_accuracy_progression_{0}_{1}.pdf".format(NETWORK_NAME, DATASET), format="pdf", bbox_inches="tight")
    # fig.savefig("fig_infer_computelatency.pdf", format="pdf", bbox_inches="tight")
