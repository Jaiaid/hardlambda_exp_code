#!/bin/bash

#!/bin/bash
ROOT_DIR=..
CACHEJOB_SCRIPT=$1
TRAINJOB_SCRIPT=$2
BS=64
EPOCH=30

# set -x

pushd $ROOT_DIR/cache_scripts
sed "s/IMGDIM/"$IMGDIM"/" $CACHEJOB_SCRIPT > cachetmp_acc.txt
# following is not very suitable because it is targeted to exit when the last job finishes
# but we are creating a service which never ends therefore the ssh invocation keeps working
# that's why the & at the end
bash $ROOT_DIR/job_distributor/master.sh cachetmp_acc.txt &
# give some time to create the data cache
# we are experimenting with small so 1minute should suffice
echo "cache creation done, waitign $((12*(5000/640)))s to be sure"
sleep $((12*(5000/640)))
echo "cache creation done"
popd

for NETARCH in resnet18;do
    for SAMPLER in shade;do
        sed "s/NETARCH/"$NETARCH"/" $TRAINJOB_SCRIPT > jobtmp_acc.txt
        sed -i "s/SAMPLER/"$SAMPLER"/" jobtmp_acc.txt
        sed -i "s/BS/"$BS"/" jobtmp_acc.txt
        sed -i "s/EPOCH/"$EPOCH"/" jobtmp_acc.txt
        sed -i "s/IMGDIM/"$IMGDIM"/" jobtmp_acc.txt
        bash $ROOT_DIR/job_distributor/master.sh jobtmp_acc.txt
        rm jobtmp_acc.txt
    done
done

# set +x
