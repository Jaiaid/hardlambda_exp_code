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

time numactl --cpunodebind=0 --membind=0 python3 benchmarking_scripts/dataload_benchmarking.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -ipm $DMOVERIP -portm $DMOVERPORT