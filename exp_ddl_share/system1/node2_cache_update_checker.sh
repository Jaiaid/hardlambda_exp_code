#!/bin/bash
ROOT_DIR=..

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/cache_update_checker.py -b 16 --dist-url tcp://10.21.12.239:44144 --world-size 4 --rank 2 -ipm 127.0.0.1 -portm 50524 > /dev/null  &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/cache_update_checker.py -b 16 --dist-url tcp://10.21.12.239:44144 --world-size 4 --rank 3 -ipm 127.0.0.1 -portm 50525 > /dev/null
