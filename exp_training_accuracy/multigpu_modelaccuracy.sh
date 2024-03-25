#!/bin/bash

#!/bin/bash
ROOT_DIR=..
BS=$1
EPOCH=$2

for NETARCH in mobilenet_v2 resnet18 resnet50 efficientnet_b1 vgg16;do
    for SAMPLER in default dali shade graddistbg;do
        # the interface should be changed according to which interface we will be using
        # here the private network 10.21.12.0/24 is on eno2
        # numa node 0
        time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/benchmarking_scripts/dataload_benchmarking.py -dset cifar10 -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size 2 --gpu 0 --dist-url tcp://10.21.12.241:44144 -if eno2 --rank 0 -ipm 127.0.0.1 -portm 50524 > rank0_accuracy_${NETARCH}_${SAMPLER}.log &
        # numa node 1
        time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/benchmarking_scripts/dataload_benchmarking.py -dset cifar10 -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size 2 --gpu 1 --dist-url tcp://10.21.12.241:44144 -if eno2 --rank 1 -ipm 127.0.0.1 -portm 50525 > /dev/null
    done
done