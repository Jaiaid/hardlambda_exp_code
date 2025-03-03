#!/bin/bash

source ../../venv/bin/activate

for network in mobilenet_v2 resnet18 resnet50 resnet101;do
    python3 nn_distributed_step_benchmarking.py --rank 0 --dist-url $1 --arch $network  --iface $2

    # to kill everything of python
    for pid in `lsof -i -P -n | grep "python3" | awk -F' ' '{ print $2 }' | tail -n +2`;do kill $pid;done
done
