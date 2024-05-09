import os
import matplotlib.pyplot as plot
import numpy as np
import argparse

COMPARED_SYSTEM_LABEL_DICT = {"": "Default", "dali": "Dali", "shade": "SHADE", "graddistbg": "proposed"}
NETWORK_NAMES_LABEL_DICT = {"mobilenet_v2": "MobileNet-V2", "resnet18": "ResNet18", "resnet50": "ResNet50", "resnet101": "ResNet101"}
# hatchlist = ['\\', '\\', '\\', '\\', '/', '/', '/', '/', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x']
hatchlist = ['o', 'x']
DISABLED_PATH = "ablation_data/benchmark_iteration_step_reverseoptimalsequence_bw500_750.tsv"
ENABLED_PATH = "ablation_data/benchmark_iteration_step_optimalsequence_bw500_750.tsv"

BS = 16
DIM = 128

if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='to get input data file path for scalability draw')
    # parser.add_argument("-i", "--input", type=str, help="path to csv/tsv file", required=True)
    # args = parser.parse_args()
    disableddata_dict = {}
    enableddata_dict = {}
                
    with open(DISABLED_PATH) as fin:
        for i, line in enumerate(fin.readlines()):
            tokens = line.rsplit()[0].replace('\t', ',').split(',')
            network_name = tokens[0]
            bs = int(tokens[1])
            dim = int(tokens[2])
            sampler = tokens[4]

            dl_time = float(tokens[5])
            cu_time = float(tokens[6])
            p_time = float(tokens[7])
            exec_time = float(tokens[8])

            if sampler != "graddistbg":
                continue

            if network_name not in disableddata_dict:
                disableddata_dict[network_name] = {}

            if bs not in disableddata_dict[network_name]:
                disableddata_dict[network_name][bs] = {}
            if dim not in disableddata_dict[network_name][bs]:
                disableddata_dict[network_name][bs][dim] = 0
            try:
                disableddata_dict[network_name][bs][dim]  = cu_time
            except Exception as e:
                pass

    with open(ENABLED_PATH) as fin:
        for i, line in enumerate(fin.readlines()):
            tokens = line.rsplit()[0].replace('\t', ',').split(',')
            network_name = tokens[0]
            bs = int(tokens[1])
            dim = int(tokens[2])
            sampler = tokens[4]

            dl_time = float(tokens[5])
            cu_time = float(tokens[6])
            p_time = float(tokens[7])
            exec_time = float(tokens[8])

            if sampler != "graddistbg":
                continue

            if network_name not in enableddata_dict:
                enableddata_dict[network_name] = {}

            if bs not in enableddata_dict[network_name]:
                enableddata_dict[network_name][bs] = {}
            if dim not in enableddata_dict[network_name][bs]:
                enableddata_dict[network_name][bs][dim] = 0
            try:
                enableddata_dict[network_name][bs][dim]  = cu_time
            except Exception as e:
                pass

    fig, ax = plot.subplots(figsize=(4, 2.25))

    bias = 0.5/len(NETWORK_NAMES_LABEL_DICT) 
    width = 0.25

    ax.bar(np.arange(0, len(NETWORK_NAMES_LABEL_DICT)) - bias, [1, 1, 1, 1],
            label="Full System", width=width, hatch=hatchlist[0])

    datalist = [enableddata_dict[network_name][BS][DIM]/disableddata_dict[network_name][BS][DIM] for network_name in NETWORK_NAMES_LABEL_DICT]
    ax.bar(np.arange(0, len(NETWORK_NAMES_LABEL_DICT)) + bias, datalist,
            label="Optimal Seq. Disabled", width=width, hatch=hatchlist[1])
    # print(np.arange(0, len(NETWORK_NAMES)))
    # set the legends and limit in y axis
    #ax.legend(ncol=3, loc="upper center", fontsize=6, frameon=False)
    #ax.get_legend().get_frame().set_alpha(None)
    #ax.get_legend().get_frame().set_facecolor((1, 1, 1, 0.1))

    plot.tick_params(axis='y', which='major', labelsize=6)
    plot.xticks(fontsize=8)
    plot.xticks(np.arange(0, len(NETWORK_NAMES_LABEL_DICT)), [NETWORK_NAMES_LABEL_DICT[network_name] for network_name in NETWORK_NAMES_LABEL_DICT])

    plot.tick_params(axis='y', which='major', labelsize=8)
    plot.yticks(fontsize=8)
    ax.set_yticks(np.arange(0, 1.25, 0.2), minor=False)
    ax.set_yticks(np.arange(0, 1.25, 0.1), minor=True)
    plot.grid(axis='y', which='major')
    ax.set_ylim([0, 1.25])
    # ax.set_yscale("log")
    # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("Normalized Speedup", {'fontsize':8})
    ax.set_xlabel("", {'fontsize':8})
    ax.legend(ncol=3, fontsize=8, frameon=False, handlelength=2.1, handleheight=1.1)
    
    fig.savefig("fig_ablation_cacheseq_noisy.png", dpi=600, bbox_inches="tight")
    fig.savefig("fig_ablation_cacheseq_noisy.pdf", format="pdf", bbox_inches="tight")
