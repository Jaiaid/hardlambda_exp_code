#!/bin/bash

# the path will make sense if observed the scripts position in google drive
# https://drive.google.com/drive/folders/1rZ4atboE6FlE4JawXiUapFBo01N9lxKa?usp=drive_link

# dataloading time speedup bar clusters
python3 fixbsoris_barcluster_dataloadtime_graph.py -i performance_data/configuration1/benchmark_iteration_step_3p1000_1epoch_3runs.tsv
python3 fixbsoris_barcluster_dataloadtime_graph.py -i performance_data/configuration2/benchmark_iteration_step_4a400_cacheconf5_3runs.tsv

# execution time speedup bar clusters
python3 fixbsoris_barcluster_exectime_graph.py -i performance_data/configuration1/benchmark_iteration_step_3p1000_3epoch_3runs.tsv
python3 fixbsoris_barcluster_exectime_graph.py -i performance_data/configuration2/4a4000_alldata.tsv

# datasize vs dltime
# it expects performance_data/configuration2/4a4000_alldata.tsv at same folder
python3 datasize_vs_dltime.py

# datasize vs exectime
# it expects performance_data/configuration2/4a4000_alldata.tsv at same folder
python3 datasize_vs_exectime.py

# lossaccuracy curves
# cifar 10
python3 lossaccuracy_curve_tsv_input.py -n resnet18 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs.tsv -dsname cifar10
python3 lossaccuracy_curve_tsv_input.py -n resnet50 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs.tsv -dsname cifar10
python3 lossaccuracy_curve_tsv_input.py -n resnet101 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs.tsv -dsname cifar10
python3 lossaccuracy_curve_tsv_input.py -n mobilenet_v2 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs.tsv -dsname cifar10
# imagenet
python3 lossaccuracy_curve_tsv_input.py -n resnet18 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs_imagenet.tsv -dsname imagenet
python3 lossaccuracy_curve_tsv_input.py -n resnet50 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs_imagenet.tsv -dsname imagenet
python3 lossaccuracy_curve_tsv_input.py -n resnet101 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs_imagenet.tsv -dsname imagenet
python3 lossaccuracy_curve_tsv_input.py -n mobilenet_v2 -cs 10000 -i lossaccuracy_curves/lossaccuracy_epochs_imagenet.tsv -dsname imagenet

# sensitivity study
python3 ablation_leastlatency_read.py
python3 ablation_bgmove.py
python3 ablation_cacheoptimalseq_stable.py
python3 ablation_cacheoptimalseq_noisy.py

# scalability study
python3 dataload_time_scalability.py
python3 cacheupdatetime_scalability.py

