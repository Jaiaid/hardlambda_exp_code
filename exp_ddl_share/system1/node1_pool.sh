#!/bin/bash

ROOT_DIR=..
CACHE_SIZE=4000

# numa node 0 pool
time numactl --cpunodebind=0 --membind=0  redis-server $ROOT_DIR/system1/redis2.conf &
python3 $ROOT_DIR/data_provider_redis.py -data imagenet -root /sandbox1/data/imagenet/2017/ -size $CACHE_SIZE -offset 0 -conf $ROOT_DIR/system1/redis2.conf -p 26379 &
# numa node 1 pool
time numactl --cpunodebind=0 --membind=0 redis-server $ROOT_DIR/system1/redis3.conf &
python3 $ROOT_DIR/data_provider_redis.py -data imagenet -root /sandbox1/data/imagenet/2017/ -size 0 -offset $CACHE_SIZE -conf $ROOT_DIR/system1/redis3.conf -p 26380 &