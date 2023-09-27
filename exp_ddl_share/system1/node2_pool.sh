#!/bin/bash

# numa node 0 pool
time numactl --cpunodebind=0 --membind=0 python3 data_provider_redis.py -conf system1/redis2.conf
# numa node 1 pool
time numactl --cpunodebind=1 --membind=1 python3 data_provider_redis.py -conf system1/redis3.conf