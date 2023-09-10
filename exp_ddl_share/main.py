import sys
import argparse
import os
import torch.multiprocessing

from trainer_process import get_training_process

if __name__ == "__main__":
    # add number of process, batch size and dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size per iteration, default 16"
        , default=16)
    parser.add_argument("-e", "--epoch_count", type=int, help="how many epoch to train per node"
        , default=2)
    parser.add_argument("-nc", "--num_class", type=int, help="how many class in dataset", required=True)
    parser.add_argument("-n", "--process_count", type=int
        , help="number of process in multi process training, default 1", default=1)
    parser.add_argument("-m", "--model", choices=["toy"], help="which model to train, select"
        , default="toy", required=True)
    parser.add_argument("-d", "--dataset", choices=["cifar10", "cifar100"]
        , help="on which dataset to train, select from 'cifar10', 'cifar100'", required=True)
    parser.add_argument("-s", "--store_strategy", choices=["baseline", "sharedlocal", "disaggregated"]
        , help="on which dataset to train, select from 'baseline', 'shared' or 'sharedpool'", required=True)
    # get arguments
    args = parser.parse_args()

    # start training
    torch.multiprocessing.spawn(get_training_process(args.store_strategy),
        args=(args.batch_size, args.epoch_count, args.num_class, args.dataset, args.model),
        nprocs=args.process_count,
        join=True
    )
