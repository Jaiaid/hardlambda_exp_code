#!/bin/bash

CPUNODE=$1
MEMNODE=$2
NETARCH=$3
SAMPLER=$4
BS=$5
EPOCH=$6
WORLD=$7
DISTURL=$8
IFACE=$9
RANK=${10}
CACHEDESC=${11}
IMGSIZE=${12}

# kill previous job
for pid in `lsof -i -P -n | grep ":44144" | awk -F' ' '{ print $2 }'`;do
    kill $pid
done
sleep 5

# to activate virtual environment
source ../../venv/bin/activate

set -x

if [[ $RANK -eq 0 ]]
then
    mkdir -p imagedim${IMGSIZE} 
    time numactl --cpunodebind=$CPUNODE --membind=$MEMNODE python3 ../benchmarking_scripts/dataload_benchmarking.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -cachedesc $CACHEDESC > t.log
    touch imagedim${IMGSIZE}/benchmark_iteration_step_${SAMPLER}_${NETARCH}_${BS}.tsv 
    tail -1 benchmark_iteration_step.tsv >> imagedim${IMGSIZE}/benchmark_iteration_step_${SAMPLER}_${NETARCH}_${BS}.tsv 
else
    time numactl --cpunodebind=$CPUNODE --membind=$MEMNODE python3 ../benchmarking_scripts/dataload_benchmarking.py -a $NETARCH -sampler $SAMPLER -b $BS --epochs $EPOCH --world-size $WORLD --dist-url $DISTURL -if $IFACE --rank $RANK -cachedesc $CACHEDESC > t_$RANK.log
fi

redis-cli -h 127.0.0.1 -p 6379 FLUSHDB
set +x