#!/bin/bash

ROOT_DIR=..
CACHE_SIZE=15000

# kill the existing redis-server
for pid in `lsof -i -P -n | grep "redis-ser" | awk -F' ' '{ print $2 }' | tail -n +2`;do
    kill $pid
done

# numa node 1 pool
time numactl --cpunodebind=1 --membind=1 redis-server $ROOT_DIR/system1/redis1.conf &
python3 $ROOT_DIR/data_provider_redis.py -data cifar10 -root /sandbox1/data/imagenet/2017/ -size $CACHE_SIZE -offset $((CACHE_SIZE+CACHE_SIZE)) -conf $ROOT_DIR/system1/redis1.conf -p 26379 &