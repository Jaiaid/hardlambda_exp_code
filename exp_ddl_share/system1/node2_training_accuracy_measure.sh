#!/bin/bash
ROOT_DIR=..
SAMPLER=default
ARCH=resnet18
EPOCH=10

PROF_TIME=$((MIN_PER_EPOCH_TIME_SEC*EPOCH))

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno4
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/model_accuracy_measure.py --arch $ARCH -b 16 --epochs $EPOCH --world-size 4 --evaluate --dist-url tcp://10.21.12.239:44144 -if eno4 --rank 2 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50524 > rank0_accuracy_$ARCH.log  &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/model_accuracy_measure.py --arch $ARCH -b 16 --epochs $EPOCH --world-size 4 --evaluate --dist-url tcp://10.21.12.239:44144 -if eno4 --rank 3 --sampler $SAMPLER -ipm 127.0.0.1 -portm 50525 > rank1_accuracy_$ARCH.log
