import os
import re
import matplotlib.pyplot as plot
import numpy as np
import argparse

EPOCH_PLOT = 15
COMPARED_SYSTEMS_LABELS = ["Default", "SHADE", "Proposed"]
DATA_FILES_DICT = {"Default": "rank0_accuracy_default.log", "Proposed": "rank0_accuracy_graddistbg.log", "SHADE": "rank0_accuracy_shade.log"}
TRAIN_LOSS_LINE_REGEX = r"Epoch: \[\d+\]\[.+\].*Time.*(\d*\.\d+).*Data.*(\d*\.\d+).*Loss.*(\d+\.\d+e[-|\+]\d+).*\((\d+\.\d+e[-|\+]\d+)\).*Acc@1.*Acc@5.*" 
VAL_LOSS_LINE_REGEX = r"Test:.*Time.*Loss.*(\d+\.\d+e[-|\+]\d+).*\((\d+\.\d+e[-|\+]\d+)\).*Acc@1.*Acc@5.*"

if __name__=="__main__":
    # plot and axis object
    fig, ax = plot.subplots(figsize=(4, 2.25))
    xtick_labels = []

    for system in COMPARED_SYSTEMS_LABELS:
        data_file = DATA_FILES_DICT[system]
        if not os.path.exists(data_file):
            print("{0} data file not exist, {1} will not be plotted".format(data_file, system))
            continue

        train_loss = []
        val_loss = []
        val_startiteration = []
        iteration_count = 0
        val_phase = False
        with open(data_file) as fin:
            for line in fin.readlines():
                match = re.search(TRAIN_LOSS_LINE_REGEX , line)
                if match is not None:
                    val_phase = False
                    iteration_count += 1
                    # taking averaged over batch training loss 
                    train_loss.append(float(match.groups()[3]))
                    continue
                match = re.search(VAL_LOSS_LINE_REGEX , line)
                if match is not None and not val_phase:
                    val_startiteration.append(iteration_count)
                    # we are only interested after which iteration val phase starts
                    val_phase = True

        # now scan for the validation loss data
        val_phase = False
        with open(data_file) as fin:
            val_iteration = 0
            val_loss_val = 0
            for line in fin.readlines():
                match = re.search(VAL_LOSS_LINE_REGEX , line)
                if match is not None:
                    val_loss_val += float(match.groups()[1])
                    val_phase = True
                    val_iteration += 1
                elif val_phase:
                    val_loss.append(val_loss_val/val_iteration)
                    val_loss_val = 0
                    val_iteration = 0
                    val_phase = False

        ax.plot([i for i in range(len(train_loss))], train_loss, label=system + " train")
        ax.plot(val_startiteration, val_loss, label=system + " val")
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
    #ax.set_xticks(np.arange(0, EPOCH_PLOT))
    # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("loss", {'fontsize':8, 'fontweight': "bold"})
    ax.set_xlabel("iteration", {'fontsize':8, 'fontweight': "bold"})
    
    fig.savefig("fig_loss_progression.png", dpi=600, bbox_inches="tight")
    # fig.savefig("fig_infer_computelatency.pdf", format="pdf", bbox_inches="tight")
