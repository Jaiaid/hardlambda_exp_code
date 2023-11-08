#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..

# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -ddl -ip 129.21.22.239 --port 44144 -if eno2 -id 0 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler default &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -ddl -ip 129.21.22.239 --port 44144 -if eno2 -id 1 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler default &

# collect memory data at 2s interval
sudo $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data.csv &
# collect numa data at 2s interval
sudo $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data.csv &
# collect network data at 2s interval
sudo sar -n DEV 2 > net_node1.csv