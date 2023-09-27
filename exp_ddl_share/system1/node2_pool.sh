#!/bin/bash

ROOT_DIR=..

# numa node 0 pool
time numactl --cpunodebind=0 --membind=0 python3 $ROOT_DIR/data_provider_redis.py -conf $ROOT_DIR/system1/redis2.conf -p 26379 &
# numa node 1 pool
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/data_provider_redis.py -conf $ROOT_DIR/system1/redis3.conf -p 26380 &