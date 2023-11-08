#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -gpu -ddl -ip 129.21.22.239 --port 44144 -if eno3 -id 2 -ws 4 -m resnet18 -d imagenet -s distributed_random -sampler graddist &
# numa node 1
time numactl --cpunodebind=1 --membind=1  python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -gpu -ddl -ip 129.21.22.239 --port 44144 -if eno3 -id 3 -ws 4 -m resnet18 -d imagenet -s distributed_random -sampler graddist &

# collect memory data at 2s interval
sudo $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data_node2.csv &
# collect numa data at 2s interval
sudo $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data_node2.csv &
# collect network data at 2s interval
sudo sar -n DEV 2 > net_node2.csv