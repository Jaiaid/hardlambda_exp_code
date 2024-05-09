import os
import matplotlib.pyplot as plot
import numpy as np
import argparse

COMPARED_SYSTEM_LABEL_DICT = {"default": "Default", "dali": "Dali", "shade": "SHADE", "graddistbg": "Proposed"}
NETWORK_NAMES_LABEL_DICT = {"mobilenet_v2": "MobileNet-V2", "resnet18": "ResNet-18", "resnet50": "ResNet-50", "resnet101": "ResNet-101"}
# hatchlist = ['\\', '\\', '\\', '\\', '/', '/', '/', '/', 'o', 'o', 'o', 'o', 'x', 'x', 'x', 'x']
hatchlist = ['\\', '/', 'o', 'x']
DATAFILE_PATH = "3p1000_alldata_old.tsv"

BS_LIST = [8, 16, 32]
DIM_LIST = [128, 64]

# a dictionary containing processing time of A40 in a 2 node 
TRACEDICT_DATAFILE_TEMPLATE = "../exp_background/simdata/benchmark_{0}_nn_step.csv" 
# trace data dict
TRACE_DATA_DICT = {}

if __name__=="__main__":
    # parser = argparse.ArgumentParser(description='to get input data file path for scalability draw')
    # parser.add_argument("-i", "--input", type=str, help="path to csv/tsv file", required=True)
    # args = parser.parse_args()
    data_dict = {}

    for network_name in NETWORK_NAMES_LABEL_DICT:
        TRACE_DATA_DICT[network_name] = {}
        # read trace data
        with open(TRACEDICT_DATAFILE_TEMPLATE.format(network_name)) as fin:
            for i, line in enumerate(fin.readlines()):
                if i == 0:
                    continue
                tokens = line.rsplit()
                bs = int(tokens[1])
                dim = int(tokens[0])

                allreduce_time_per_sample = float(tokens[5])
                process_time_per_sample = float(tokens[3])

                if dim not in TRACE_DATA_DICT[network_name]:
                    TRACE_DATA_DICT[network_name][dim] = {}
                TRACE_DATA_DICT[network_name][dim][bs] = [process_time_per_sample, allreduce_time_per_sample]
                
    with open(DATAFILE_PATH) as fin:
        for i, line in enumerate(fin.readlines()):
            if i == 0:
                continue

            tokens = line.replace(',','\t').rsplit()
            network_name = tokens[0]
            bs = int(tokens[1])
            dim = int(tokens[2])
            sampler = tokens[4]

            dl_time = float(tokens[5])
            cu_time = float(tokens[6])
            p_time = float(tokens[7])
            exec_time = float(tokens[8])

            if network_name not in data_dict:
                data_dict[network_name] = {}

            if sampler not in data_dict[network_name]:
                data_dict[network_name][sampler] = {}
            if bs not in data_dict[network_name][sampler]:
                data_dict[network_name][sampler][bs] = {}
            if dim not in data_dict[network_name][sampler][bs]:
                data_dict[network_name][sampler][bs][dim] = [0, 0, 0, 0]
            try:
                data_dict[network_name][sampler][bs][dim][0] = dl_time
                data_dict[network_name][sampler][bs][dim][2] = exec_time# dl_time + cu_time + (TRACE_DATA_DICT[network_name][dim][bs][0] + TRACE_DATA_DICT[network_name][dim][bs][1]) * bs 
            except Exception as e:
                print(network_name, dim, bs)
                pass

        # 2nd pass to normalize value w.r.t 4 worker data
        with open(DATAFILE_PATH) as fin:
            for i, line in enumerate(fin.readlines()):
                if i == 0:
                    continue

                tokens = line.replace(',','\t').rsplit()
                network_name = tokens[0]
                bs = int(tokens[1])
                dim = int(tokens[2])
                sampler = tokens[4]

                dl_time = float(tokens[5])
                cu_time = float(tokens[6])
                p_time = float(tokens[7])
                exec_time = float(tokens[8])
                try:
                    data_dict[network_name][sampler][bs][dim][1] = data_dict[network_name]["default"][bs][dim][0]/dl_time
                    data_dict[network_name][sampler][bs][dim][3] = data_dict[network_name]["default"][bs][dim][2]/exec_time#(dl_time + cu_time + (TRACE_DATA_DICT[network_name][dim][bs][0] + TRACE_DATA_DICT[network_name][dim][bs][1]) * bs)
                except Exception as e:
                    # print(network_name, dim, bs)
                    pass

    for dim in DIM_LIST:
        for bs in BS_LIST:
            # plot and axis object
            fig, ax = plot.subplots(figsize=(4, 2.25))
            # ax1, ax2 = ax.twinx()
            xtick_labels = []
            bias = 0.6/len(NETWORK_NAMES_LABEL_DICT) 
            width = 0.15

            for i, sampler in enumerate(COMPARED_SYSTEM_LABEL_DICT):
                try:
                    datalist = [data_dict[network_name][sampler][bs][dim][3] if dim in data_dict[network_name][sampler][bs] else 0 for network_name in NETWORK_NAMES_LABEL_DICT]
                    offset = (i-1.5) * bias
                    ax.bar(np.arange(0, len(COMPARED_SYSTEM_LABEL_DICT)) + offset, datalist,
                        label=COMPARED_SYSTEM_LABEL_DICT[sampler], width=width, hatch=hatchlist[i])
                except Exception as e:
                    print(network_name, dim, bs, datalist)
                    pass
            # print(np.arange(0, len(NETWORK_NAMES)))
            # set the legends and limit in y axis
            #ax.legend(ncol=3, loc="upper center", fontsize=6, frameon=False)
            #ax.get_legend().get_frame().set_alpha(None)
            #ax.get_legend().get_frame().set_facecolor((1, 1, 1, 0.1))
            # ax.set_yscale("log")

            plot.tick_params(axis='y', which='major', labelsize=6)
            plot.xticks(fontsize=8)
            plot.xticks(np.arange(0, len(NETWORK_NAMES_LABEL_DICT)), [NETWORK_NAMES_LABEL_DICT[network_name] for  network_name in NETWORK_NAMES_LABEL_DICT])

            plot.tick_params(axis='y', which='major', labelsize=8)
            plot.yticks(fontsize=8)
            ax.set_yticks(np.arange(0, 2, 0.25), minor=False)
            # ax.set_yticks(np.arange(0, 2, 0.1), minor=True)
            plot.grid(axis='y', which='major')
            ax.set_ylim([0, 1.9])
            # ax.set_xticklabels(BAR_LABELS, {'fontsize':8, 'fontweight': "bold"})
            ax.set_ylabel("Normalized Speedup", {'fontsize':8})
            ax.set_xlabel("", {'fontsize':9})
            ax.legend(ncol=2, fontsize=8, frameon=False, handlelength=4, handleheight=1.1)
            
            fig.savefig("figures/exectime_bs{0}_is{1}.png".format(bs, dim), dpi=600, bbox_inches="tight")
            # fig.savefig("exectime_bs{0}_is{1}.png".format(bs, dim), format="pdf", bbox_inches="tight")
