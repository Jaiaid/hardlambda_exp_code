#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..

# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e 5 -nc 10 -n 2 -ddl -ip 129.21.22.239 -if eno2 -id 0 -ws 4 -m toy -d cifar10 -s local_random &
# numa node 1
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/main.py -bs 16 -e 5 -nc 10 -n 2 -ddl -ip 129.21.22.239 -if eno2 -id 1 -ws 4 -m toy -d cifar10 -s local_random &

# collect memory data at 2s interval
sudo $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data.csv &
# collect numa data at 2s interval
sudo $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data.csv