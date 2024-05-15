#!/bin/bash

#!/bin/bash
ROOT_DIR=..
EPOCH=1
CACHEJOB_SCRIPT=$1

# set -x

for IMGDIM in 32 64 128 256 512 1024;do
    sed "s/IMGDIM/"$IMGDIM"/" $CACHEJOB_SCRIPT > $ROOT_DIR/cache_scripts/cachetmp.txt
    mv batch_latency_benchmark_cacheconf.yaml $ROOT_DIR/cache_scripts/tmp.yaml
    pushd $ROOT_DIR/cache_scripts
    # following is not very suitable because it is targeted to exit when the last job finishes
    # but we are creating a service which never ends therefore the ssh invocation keeps working
    # that's why the & at the end
    bash $ROOT_DIR/job_distributor/master.sh cachetmp.txt &
    # give some time to create the data cache
    # we are experimenting with small so 1minute should suffice
    echo "cache creation done, waitign 60s to be sure"
    sleep 240
    echo "cache creation done"
    mv cache0.txt $ROOT_DIR/exp_motivation/cache0$IMGDIM.txt
    rm tmp.yaml
    popd
done 

# set +x