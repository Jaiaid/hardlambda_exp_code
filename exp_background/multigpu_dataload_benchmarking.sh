#!/bin/bash

#!/bin/bash
ROOT_DIR=..
BS=$1
EPOCH=$2

for NETARCH in mobilenet_v2 resnet18 resnet50 efficientnet_b1 vgg16;do
    for SAMPLER in default dali shade graddistbg;do
        sed "s/NETARCH/"$NETARCH"/" system_2a4000_1p1000.txt > system_2a4000_1p1000_tmp.txt
        sed -i "s/SAMPLER/"$SAMPLER"/" system_2a4000_1p1000_tmp.txt
        sed -i "s/BS/"$BS"/" system_2a4000_1p1000_tmp.txt
        sed -i "s/EPOCH/"$EPOCH"/" system_2a4000_1p1000_tmp.txt
        bash $ROOT_DIR/job_distributor/master.sh system_2a4000_1p1000_tmp.txt
        rm system_2a4000_1p1000_tmp.txt
    done
done