#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..
SAMPLER=graddistbg
EPOCH=10
MIN_PER_EPOCH_TIME_SEC=300

PROF_TIME=$((MIN_PER_EPOCH_TIME_SEC*EPOCH))

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e $EPOCH -nc 1000 -n 2 -ddl -ip 10.21.12.239 --port 44144 -if eno2 -id 0 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler $SAMPLER -ipm 127.0.0.1 -portm 50524 &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/main.py -bs 16 -e $EPOCH -nc 1000 -n 2 -ddl -ip 10.21.12.239 --port 44144 -if eno2 -id 1 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler $SAMPLER -ipm 127.0.0.1 -portm 50525 &

# collect memory data at 2s interval for 10 min
sudo timeout $PROF_TIME $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data_${SAMPLER}_node2.csv &
# collect numa data at 2s interval for 10 min
sudo timeout $PROF_TIME $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data_${SAMPLER}_node2.csv &
# collect network data at 2s interval for 10 min
sudo timeout $PROF_TIME sar -n DEV 2 > net_data_${SAMPLER}_node2.csv