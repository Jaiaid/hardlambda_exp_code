#!/bin/bash

#!/bin/bash
ROOT_DIR=..
EPOCH=2
CACHEJOB_SCRIPT=$1
TRAINJOB_SCRIPT=$2

# set -x

for IMGDIM in 32 64 128 256 512 1024;do
    pushd $ROOT_DIR/cache_scripts
    sed "s/IMGDIM/"$IMGDIM"/" $CACHEJOB_SCRIPT > cachetmp.txt
    # following is not very suitable because it is targeted to exit when the last job finishes
    # but we are creating a service which never ends therefore the ssh invocation keeps working
    # that's why the & at the end
    bash $ROOT_DIR/job_distributor/master.sh cachetmp.txt &
    # give some time to create the data cache
    # we are experimenting with small so 1minute should suffice
    echo "cache creation done, waitign 60s to be sure"
    sleep 60
    echo "cache creation done"
    popd
    for BS in 4 8 16 32 64;do
        for NETARCH in mobilenet_v2 resnet18 resnet50 resnet101;do
            for SAMPLER in graddistbg shade default dali;do
                sed "s/NETARCH/"$NETARCH"/" $TRAINJOB_SCRIPT > jobtmp.txt
                sed -i "s/SAMPLER/"$SAMPLER"/" jobtmp.txt
                sed -i "s/BS/"$BS"/" jobtmp.txt
                sed -i "s/EPOCH/"$EPOCH"/" jobtmp.txt
                sed -i "s/IMGDIM/"$IMGDIM"/" jobtmp.txt
                bash $ROOT_DIR/job_distributor/master.sh jobtmp.txt
                rm jobtmp.txt
            done
        done
    done
done 

# set +x