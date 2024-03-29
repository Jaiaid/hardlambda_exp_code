#!/bin/bash

CPUNODE=$1
MEMNODE=$2
NETARCH=$3
SAMPLER=$4
BS=$5
EPOCH=$6
WORLD=$7
DISTURL=$8
IFACE=$9
RANK=${10}
CACHEDESC=${11}
IMGSIZE=${12}

# to activate virtual environment
source ../../venv/bin/activate

if [[ $RANK -eq 0 ]]
then
    mkdir -p imagedim${IMGSIZE} 
    time numactl --cpunodebind=$CPUNODE --membind=$MEMNODE python3 ../benchmarking_scripts/model_accuracy_measure.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -cachedesc $CACHEDESC > /dev/null
    mv benchmark_iteration_step.csv imagedim${IMGSIZE}/benchmark_iteration_step_${SAMPLER}_${NETARCH}_${BS}.csv 
else
    time numactl --cpunodebind=$CPUNODE --membind=$MEMNODE python3 ../benchmarking_scripts/model_accuracy_measure.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -cachedesc $CACHEDESC  > /dev/null
fi