#!/bin/bash

JOBLISTFILE=$1

jobcount=0
# read the hostfile
while read -r line; do
    jobargs=()
    for arg in $line; do
        jobargs+=($arg)
    done
    jobcount=$((jobcount+1))

    # create the cmd
    cmd="bash"
    for arg in ${jobargs[@]:3}; do
        cmd=$cmd" "$arg
    done

    # create the script
    echo $cmd>>worker_${jobcount}.sh
    # transfer the script
    ip=${jobargs[0]}
    port=${jobargs[1]}
    workdir=${jobargs[2]}
    script=${jobargs[3]}
    scp -P $port worker_${jobcount}.sh $script $ip:$workdir
    rm worker_${jobcount}.sh
done < $JOBLISTFILE

jobcount=0
while read -r line; do
    jobargs=()
    for arg in $line; do
        jobargs+=($arg)
    done
    jobcount=$((jobcount+1))

    # transfer the script
    ip=${jobargs[0]}
    port=${jobargs[1]}
    workdir=${jobargs[2]}
    cmdstr='cd '${workdir}';bash 'worker_${jobcount}.sh
    echo $cmdstr
    ssh -p 1440 $ip $cmdstr &
    PID=$!
done < $JOBLISTFILE

# assumption is all job will complete near same time
# therefore checking only one's pid is good enough
while ps -p ${PID} > /dev/null
do
	sleep 1
done
# 30s sleep for extra safety
sleep 30
