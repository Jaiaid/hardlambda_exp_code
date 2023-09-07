import sys
import os
import torch.multiprocessing

from trainer_process import get_training_process

if __name__ == "__main__":
    # number of process 
    # will be used to create unique id from 0-3
    num_rank = int(sys.argv[1])
    strategy = sys.argv[2]

    torch.multiprocessing.spawn(get_training_process(strategy),
             args=(16, 2, "cifar10", "model"),
             nprocs=num_rank,
             join=True)