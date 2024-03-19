#!/bin/bash

ROOT_DIR=..
CACHE_SIZE=$1
CPU_NODE=$2
MEM_NODE=$3
REDIS_PORT=$4
DATASET_TYPE=$5
DATASET=$6
OFFSET=$7
SEQNO=$8
BATCH_SIZE=$9
CACHEDESC_FILE=$10
$DMOVE_PORT$11


# create configuration
sed "92s/port 6379/port "$REDIS_PORT"/" redis.conf > redis_out.conf
sed -i "88s/protected-mode yes/protected-mode no/" redis_out.conf
# numa node pool
time numactl --cpunodebind=$CPU_NODE --membind=$MEM_NODE  redis-server redis_out.conf &
python3 $ROOT_DIR/cache_scripts/data_provider_redis.py -data $DATASET_TYPE -root $DATASET -size $CACHE_SIZE -offset $OFFSET -conf redis_out.conf -p $REDIS_PORT &
# create the movement service
python3 $ROOT_DIR/cache_scripts/DataMovementService.py --seqno $SEQNO -bs $BATCH_SIZE -cdesc $CACHEDESC_FILE -pn 10.21.12.239 10.21.12.222 -p $DMOVE_PORT