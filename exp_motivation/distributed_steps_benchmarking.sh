#!/bin/bash

source ../../venv/bin/activate

python3 nn_distributed_step_benchmarking.py --rank 0 --dist-url $1 &
python3 nn_distributed_step_benchmarking.py --rank 1 --dist-url $1

# to kill everything of python
for pid in `lsof -i -P -n | grep "python3" | awk -F' ' '{ print $2 }' | tail -n +2`;do kill $pid;done
