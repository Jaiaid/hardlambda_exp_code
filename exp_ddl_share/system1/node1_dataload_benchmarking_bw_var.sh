#!/bin/bash
ROOT_DIR=..
SAMPLER=graddistbg
ARCH=resnet18
EPOCH=4
BWMBPS=100
BS=16

sudo tc qdisc add dev eno2 root netem rate ${BWMBPS}mbit
# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/dataload_benchmarking.py -b $BS --epochs $EPOCH --world-size 4 --dist-url tcp://10.21.12.239:44144 -if eno2 --rank 0 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50524 > rank0_bwvar_$SAMPLER_$BS_$BWMBPS.log  &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/dataload_benchmarking.py -b $BS --epochs $EPOCH --world-size 4 --dist-url tcp://10.21.12.239:44144 -if eno2 --rank 1 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50525 > rank1_bwvar_$SAMPLER_$BS_$BWMBPS.log
