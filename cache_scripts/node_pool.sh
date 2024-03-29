#!/bin/bash

ROOT_DIR=..
CPU_NODE=$1
MEM_NODE=$2
DATASET_TYPE=$3
DATASETROOT=$4
CACHE_SIZE=$5
OFFSET=$6
IMAGEDIM=$7
REDIS_PORT=$8
SEQNO=$9
CACHEDESC_FILE=${10}

# set -x

# kill the existing redis-server
for pid in `lsof -i -P -n | grep "redis-ser" | awk -F' ' '{ print $2 }' | tail -n +2`;do
    kill $pid
done
# kill if previous associated python service is still holding the port
# kill the existing redis-server
for pid in `lsof -i -P -n | grep "python3" | awk -F' ' '{ print $2 }'`;do
    kill $pid
done

sleep 5

# to activate virtual environment
source ../../venv/bin/activate
# create configuration
sed "92s/port 6379/port "$REDIS_PORT"/" redis.conf > redis_out$REDIS_PORT.conf
sed -i "88s/protected-mode yes/protected-mode no/" redis_out$REDIS_PORT.conf
# numa node pool
echo "starting redis server"
time numactl --cpunodebind=$CPU_NODE --membind=$MEM_NODE  redis-server redis_out$REDIS_PORT.conf > /dev/null &
sleep 5
echo "starting redis cache populating"
python3 $ROOT_DIR/cache_scripts/data_provider_redis.py -data $DATASET_TYPE -root $DATASETROOT -dim $IMAGEDIM -size $CACHE_SIZE -offset $OFFSET -conf redis_out.conf -p $REDIS_PORT
sleep 10
# create the movement service
echo "creating data movement service"
python3 $ROOT_DIR/cache_scripts/DataMovementService.py --seqno $SEQNO -cdesc $CACHEDESC_FILE > cache$SEQNO.txt 2>&1 &
echo "created data movement service"
# set +x