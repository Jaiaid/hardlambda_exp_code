#!/bin/bash
ROOT_DIR=..
EPOCH=10

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/dataload_benchmarking.py -b 16 --epochs $EPOCH --world-size 4 --dist-url tcp://10.21.12.239:44144 -if eno2 --rank 0 -ipm 127.0.0.1 -portm 50524  &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/dataload_benchmarking.py -b 16 --epochs $EPOCH --world-size 4 --dist-url tcp://10.21.12.239:44144 -if eno2 --rank 1 -ipm 127.0.0.1 -portm 50525
