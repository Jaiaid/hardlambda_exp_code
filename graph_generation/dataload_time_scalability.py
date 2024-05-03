import os
import matplotlib.pyplot as plot
import numpy as np
import argparse

COMPARED_SYSTEMS_LABELS_DICT = {"default": "Default", "shade": "SHADE", "dali": "Dali", "graddistbg": "Proposed"}
NETWORK_NAMES = ["MobileNet-V2", "ResNet18", "ResNet50", "ResNet101"]
# hatchlist = ['\\', '\\', '\\', '\\', '/', '/', '/', '/', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x']
hatchlist = ['\\', '/', 'o', 'x']
DATAFILE_PATH = "benchmark_iteration_step.tsv"
DATASET_SIZE_PER_WORKER = 640
BS = 32
DIM = 256

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='to get input data file path for scalability draw')
    parser.add_argument("-i", "--input", type=str, help="path to csv/tsv file", required=True)
    args = parser.parse_args()

    # plot and axis object
    fig, ax = plot.subplots(figsize=(4, 2.25))
    xtick_labels = []
    data_dict = {}

    with open(args.input) as fin:
        for line in fin.readlines():
            tokens = line.rsplit()[0].replace('\t', ',').split(',')
            network_name = tokens[0]
            bs = int(tokens[1])
            dim = int(tokens[2])
            sampler = tokens[4]

            dl_time = float(tokens[5])
            cu_time = float(tokens[6])
            p_time = float(tokens[7])
            exec_time = float(tokens[8])

            if sampler not in data_dict:
                data_dict[sampler] = {}
            if bs not in data_dict[sampler]:
                data_dict[sampler][bs] = {}
            if dim not in data_dict[sampler][bs]:
                data_dict[sampler][bs][dim] = [0, 0, 0, 0, 0, 0, 0, 0]
            if sampler == "default":
                if network_name == "mobilenet_v2":
                    data_dict[sampler][bs][dim][4] = dl_time
                elif network_name == "resnet18":
                    data_dict[sampler][bs][dim][5] = dl_time
                elif network_name == "resnet50":
                    data_dict[sampler][bs][dim][6] = dl_time
                elif network_name == "resnet101":
                    data_dict[sampler][bs][dim][7] = dl_time

            if network_name == "mobilenet_v2":
                data_dict[sampler][bs][dim][0] = dl_time
            elif network_name == "resnet18":
                data_dict[sampler][bs][dim][1] = dl_time
            elif network_name == "resnet50":
                data_dict[sampler][bs][dim][2] = dl_time
            elif network_name == "resnet101":
                data_dict[sampler][bs][dim][3] = dl_time
    # 2nd pass to normalize the values
    with open(args.input) as fin:
        for line in fin.readlines():
            tokens = line.rsplit()[0].replace('\t', ',').split(',')
            network_name = tokens[0]
            bs = int(tokens[1])
            dim = int(tokens[2])
            sampler = tokens[4]

            dl_time = float(tokens[5])
            cu_time = float(tokens[6])
            p_time = float(tokens[7])
            exec_time = float(tokens[8])

            if network_name == "mobilenet_v2":
                data_dict[sampler][bs][dim][0] = data_dict["default"][bs][dim][4]/dl_time
            elif network_name == "resnet18":
                data_dict[sampler][bs][dim][1] = data_dict["default"][bs][dim][5]/dl_time
            elif network_name == "resnet50":
                data_dict[sampler][bs][dim][2] = data_dict["default"][bs][dim][6]/dl_time
            elif network_name == "resnet101":
                data_dict[sampler][bs][dim][3] = data_dict["default"][bs][dim][7]/dl_time

    bias = 0.6/len(NETWORK_NAMES) 
    width = 0.15
    for i, sampler in enumerate(COMPARED_SYSTEMS_LABELS_DICT):
        offset = (i-2) * bias
        ax.bar(np.arange(0, len(NETWORK_NAMES)) + offset, data_dict[sampler][BS][DIM][0:4],
               label=COMPARED_SYSTEMS_LABELS_DICT[sampler], width=width, hatch=hatchlist[i])
    # print(np.arange(0, len(NETWORK_NAMES)))
    # plot.tick_params(axis='both', which='major', labelsize=12)
    # set the legends and limit in y axis
    #ax.legend(ncol=3, loc="upper center", fontsize=6, frameon=False)
    #ax.get_legend().get_frame().set_alpha(None)
    #ax.get_legend().get_frame().set_facecolor((1, 1, 1, 0.1))
    # ax.set_yscale("log")
    plot.yticks(fontweight='bold', fontsize=8)
    plot.xticks(fontweight='bold', fontsize=6)
    ax.legend(ncol=2)
    plot.xticks(np.arange(0, len(NETWORK_NAMES)), NETWORK_NAMES)
    plot.yticks(np.arange(0, 1.75, 0.1))
    plot.grid(axis='y')
    # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("Normalized Speedup", {'fontsize':8, 'fontweight': "bold"})
    ax.set_xlabel("", {'fontsize':8, 'fontweight': "bold"})
    
    fig.savefig("fig_dl_scalability.png", dpi=600, bbox_inches="tight")
    # fig.savefig("fig_infer_computelatency.pdf", format="pdf", bbox_inches="tight")
