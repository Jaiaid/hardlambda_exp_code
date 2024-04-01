#!/bin/bash

source ../../venv/bin/activate

python3 nn_distributed_step_benchmarking.py --rank 0 --dist-url $1 &
python3 nn_distributed_step_benchmarking.py --rank 1 --dist-url $1
