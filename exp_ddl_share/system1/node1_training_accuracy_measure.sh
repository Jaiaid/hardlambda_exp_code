#!/bin/bash
ROOT_DIR=..
SAMPLER=default
ARCH=resnet18
EPOCH=10

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/model_accuracy_measure.py --arch $ARCH -b 16 --epochs $EPOCH --world-size 2 --evaluate --dist-url tcp://10.21.12.239:44144 -if eno2 --rank 0 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50524 > rank0_accuracy_$ARCH.log  &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/model_accuracy_measure.py --arch $ARCH -b 16 --epochs $EPOCH --world-size 2 --evaluate --dist-url tcp://10.21.12.239:44144  -if eno2 --rank 1 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50525 > rank1_accuracy_$ARCH.log  &
