import os
import matplotlib.pyplot as plot
import numpy as np
import argparse

COMPARED_LABELS_DICT = {"4worker": "4 worker", "8worker": "8 worker", "16worker": "16 worker"}
NETWORK_NAMES = ["MobileNet-V2", "ResNet-18", "ResNet-50", "ResNet-101"]
# hatchlist = ['\\', '\\', '\\', '\\', '/', '/', '/', '/', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x']
hatchlist = ['\\', '/', 'o', 'x']
DATAFILE_PATH = "benchmark_iteration_step.tsv"
DATAROOT_DIR = "scalability_data"

DATASET_SIZE_PER_WORKER = 640
BS = 32
DIM = 256

if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='to get input data file path for scalability draw')
    # parser.add_argument("-i", "--input", type=str, help="path to csv/tsv file", required=True)
    # args = parser.parse_args()

    # plot and axis object
    fig, ax = plot.subplots(figsize=(4, 2.25))
    xtick_labels = []
    data_dict = {}

    for subfolder in COMPARED_LABELS_DICT:
        filepath = os.path.join(os.path.join(DATAROOT_DIR, subfolder), DATAFILE_PATH)
        
        worker_count_key = subfolder
        if worker_count_key not in data_dict:
            data_dict[worker_count_key] = {} 

        with open(filepath) as fin:
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

                if sampler not in data_dict[worker_count_key]:
                    data_dict[worker_count_key][sampler] = {}
                if bs not in data_dict[worker_count_key][sampler]:
                    data_dict[worker_count_key][sampler][bs] = {}
                if dim not in data_dict[worker_count_key][sampler][bs]:
                    data_dict[worker_count_key][sampler][bs][dim] = [0, 0, 0, 0, 0, 0, 0, 0]
                
                if network_name == "mobilenet_v2":
                    data_dict[worker_count_key][sampler][bs][dim][4] = exec_time
                elif network_name == "resnet18":
                    data_dict[worker_count_key][sampler][bs][dim][5] = exec_time
                elif network_name == "resnet50":
                    data_dict[worker_count_key][sampler][bs][dim][6] = exec_time
                elif network_name == "resnet101":
                    data_dict[worker_count_key][sampler][bs][dim][7] = exec_time

                if network_name == "mobilenet_v2":
                    data_dict[worker_count_key][sampler][bs][dim][0] = exec_time
                elif network_name == "resnet18":
                    data_dict[worker_count_key][sampler][bs][dim][1] = exec_time
                elif network_name == "resnet50":
                    data_dict[worker_count_key][sampler][bs][dim][2] = exec_time
                elif network_name == "resnet101":
                    data_dict[worker_count_key][sampler][bs][dim][3] = exec_time

        # 2nd pass to normalize value w.r.t 4 worker data
        with open(filepath) as fin:
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

                if sampler == "graddistbg":
                    if network_name == "mobilenet_v2":
                        data_dict[worker_count_key][sampler][bs][dim][0] = data_dict[worker_count_key]["default"][bs][dim][4]/exec_time
                    elif network_name == "resnet18":
                        data_dict[worker_count_key][sampler][bs][dim][1] = data_dict[worker_count_key]["default"][bs][dim][5]/exec_time
                    elif network_name == "resnet50":
                        data_dict[worker_count_key][sampler][bs][dim][2] = data_dict[worker_count_key]["default"][bs][dim][6]/exec_time
                    elif network_name == "resnet101":
                        data_dict[worker_count_key][sampler][bs][dim][3] = data_dict[worker_count_key]["default"][bs][dim][7]/exec_time

    bias = 0.6/len(NETWORK_NAMES) 
    width = 0.15
    for i, worker_count_key in enumerate(COMPARED_LABELS_DICT):
        offset = (i-1) * bias
        ax.bar(np.arange(0, len(NETWORK_NAMES)) + offset, data_dict[worker_count_key]["graddistbg"][BS][DIM][0:4],
            label=COMPARED_LABELS_DICT[worker_count_key], width=width, hatch=hatchlist[i])

    plot.tick_params(axis='y', which='major', labelsize=8)
    plot.xticks(fontsize=6)
    plot.xticks(np.arange(0, len(NETWORK_NAMES)), NETWORK_NAMES)

    plot.tick_params(axis='y', which='major', labelsize=8)
    plot.yticks(fontsize=8)
    ax.set_yticks(np.arange(0, 2, 0.25), minor=False)
    ax.set_yticks(np.arange(0, 2, 0.1), minor=True)
    plot.grid(axis='y', which='major')
    ax.set_ylim([0, 1.5])
    # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
    ax.set_ylabel("Normalized Speedup", {'fontsize':8})
    ax.set_xlabel("", {'fontsize':8})
    ax.legend(ncol=3, fontsize=8, frameon=False, handlelength=2.1, handleheight=1.1)
    
    fig.savefig("fig_exec_scalability.png", dpi=600, bbox_inches="tight")
    # fig.savefig("fig_infer_computelatency.pdf", format="pdf", bbox_inches="tight")
