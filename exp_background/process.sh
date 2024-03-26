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
RANK=$10
DMOVERIP=$11
DMOVERPORT=$12

# to activate virtual environment
source ../../venv/bin/activate

time numactl --cpunodebind=$CPUNODE --membind=$MEMNODE python3 ../benchmarking_scripts/dataload_benchmarking.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -ipm $DMOVERIP -portm $DMOVERPORT