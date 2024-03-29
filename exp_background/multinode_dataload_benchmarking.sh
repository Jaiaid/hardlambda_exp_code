#!/bin/bash

#!/bin/bash
ROOT_DIR=..
EPOCH=2

for IMGDIM in 32 64 128 256 512 1024;do
    pushd $ROOT_DIR/cache_scripts
    bash $ROOT_DIR/job_distributor/master.sh cachecreation_joblist_conf2.txt
    # give some time to create the data cache
    # we are experimenting with small so 1minute should suffice
    sleep 60
    popd
    for BS in 4 8 16 32 64;do
        for NETARCH in mobilenet_v2 resnet18 resnet50 resnet101;do
            for SAMPLER in default dali shade graddistbg;do
                sed "s/NETARCH/"$NETARCH"/" system_2a4000_1p1000.txt > system_2a4000_1p1000_tmp.txt
                sed -i "s/SAMPLER/"$SAMPLER"/" system_2a4000_1p1000_tmp.txt
                sed -i "s/BS/"$BS"/" system_2a4000_1p1000_tmp.txt
                sed -i "s/EPOCH/"$EPOCH"/" system_2a4000_1p1000_tmp.txtIMGDIM
                sed -i "s/IMGDIM/"$IMGDIM"/" system_2a4000_1p1000_tmp.txt
                bash $ROOT_DIR/job_distributor/master.sh system_2a4000_1p1000_tmp.txt
                rm system_2a4000_1p1000_tmp.txt
            done
        done
    done
done 