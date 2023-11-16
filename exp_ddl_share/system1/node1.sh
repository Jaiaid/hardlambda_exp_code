#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno2
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 DataMovementService.py --seqno 0 -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p 50524
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -ddl -ip 10.21.12.239 --port 44144 -if eno2 -id 0 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler graddistbg -ipm 127.0.0.1 -portm 50524 &
# numa node 1
time numactl --cpunodebind=0 --membind=0 python3 DataMovementService.py --seqno 1 -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p 50525
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -ddl -ip 10.21.12.239 --port 44144 -if eno2 -id 1 -ws 4 -m resnet18 -d imagenet -s distributed_random -gpu -sampler graddistbg -ipm 127.0.0.1 -portm 50525 &

# wait for 5 min before starting collecting metric
sleep 300

# collect memory data at 2s interval for 10min
sudo timeout 600 $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data_node1.csv &
# collect numa data at 2s interval for 10 min
sudo timeout 600 $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data_node1.csv &
# collect network data at 2s interval for 10 min
sudo timeout 600 sar -n DEV 2 > net_node1.csv