#!/bin/bash

ROOT_DIR=..

# numa node 1 pool
time numactl --cpunodebind=1 --membind=1 python3 $ROOT_DIR/data_provider_redis.py -conf $ROOT_DIR/system1/redis1.conf &