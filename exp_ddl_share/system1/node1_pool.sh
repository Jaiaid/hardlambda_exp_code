#!/bin/bash

ROOT_DIR=../..
CACHE_SIZE=15000

# kill the existing redis-server
for pid in `lsof -i -P -n | grep "redis-ser" | awk -F' ' '{ print $2 }' | tail -n +2`;do
    kill $pid
done

# numa node 0 pool
time numactl --cpunodebind=0 --membind=0  redis-server $ROOT_DIR/exp_ddl_share/system1/redis2.conf &
python3 $ROOT_DIR/cache_scripts/data_provider_redis.py -data cifar10 -root /sandbox1/data/imagenet/2017/ -size $CACHE_SIZE -offset 0 -conf $ROOT_DIR/exp_ddl_share/system1/redis2.conf -p 26379 &
# numa node 1 pool
time numactl --cpunodebind=1 --membind=1 redis-server $ROOT_DIR/exp_ddl_share/system1/redis3.conf &
python3 $ROOT_DIR/cache_scripts/data_provider_redis.py -data cifar10 -root /sandbox1/data/imagenet/2017/ -size $CACHE_SIZE -offset $CACHE_SIZE -conf $ROOT_DIR/exp_ddl_share/system1/redis3.conf -p 26380 &
