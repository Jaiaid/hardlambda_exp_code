#!/bin/bash
PROF_TOOL_DIR=~/application/proftools/pcm/build/bin/
ROOT_DIR=..

# the interface should be changed according to which interface we will be using
# here the private network 10.21.12.0/24 is on eno4
# numa node 0
time numactl --cpunodebind=0 --membind=0 python3 DataMovementService.py --seqno 2 -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p 50524
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -gpu -ddl -ip 129.21.22.239 --port 44144 -if eno4 -id 2 -ws 4 -m resnet18 -d imagenet -s distributed_random -sampler graddist  -ipm 127.0.0.1 -portm 50524 &
# numa node 1
time numactl --cpunodebind=0 --membind=0 python3 DataMovementService.py --seqno 2 -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p 50525
time numactl --cpunodebind=1 --membind=1  python3 $ROOT_DIR/main.py -bs 16 -e 10 -nc 1000 -n 2 -gpu -ddl -ip 129.21.22.239 --port 44144 -if eno4 -id 3 -ws 4 -m resnet18 -d imagenet -s distributed_random -sampler graddist -ipm 127.0.0.1 -portm 50525  &

# wait for 5 min before starting collecting metric
sleep 300

# collect memory data at 2s interval for 10 min
sudo timeout 600 $PROF_TOOL_DIR/pcm-memory 2 -csv=memory_data_node2.csv &
# collect numa data at 2s interval for 10 min
sudo timeout 600 $PROF_TOOL_DIR/pcm-numa 2 -csv=numa_data_node2.csv &
# collect network data at 2s interval for 10 min
sudo timeout 600 sar -n DEV 2 > net_node2.csv