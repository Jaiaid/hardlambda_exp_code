import sys
import argparse
import os
import torch.multiprocessing

from trainer_process import get_training_process

if __name__ == "__main__":
    # add number of process, batch size and dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("-bs", "--batch_size", type=int, help="batch size per iteration, default 16",
                        default=16)
    parser.add_argument("-e", "--epoch_count", type=int, help="how many epoch to train per node",
                        default=2)
    parser.add_argument("-nc", "--num_class", type=int, help="how many class in dataset", required=True)
    parser.add_argument("-n", "--process_count", type=int,
                        help="number of process in multi process training, default 1", default=1)
    parser.add_argument("-m", "--model", choices=["toy", "resnet18"], help="which model to train, select",
                        default="toy", required=True)
    parser.add_argument("-d", "--dataset", choices=["cifar10", "cifar100", "imagenet"],
                        help="on which dataset to train, select from 'cifar10', 'cifar100'", required=True)
    parser.add_argument("-s", "--store_strategy", choices=["baseline", "sharedlocal", "disaggregated", "local_random", "distributed_random"]
        , help="on which dataset to train, select from 'baseline', 'shared' or 'sharedpool'", required=True)
    parser.add_argument("-ddl", "--distributed", default=False, action="store_true", help="if it will be distributed")
    parser.add_argument("-gpu", "--gpu", default=False, action="store_true", help="if it will be distributed")
    parser.add_argument("-sampler", "--sampler", choices=["default", "distaware", "shade", "graddist", "graddistbg"],
                        help="what sampler will be used")

    # ddl over network iface related parameters
    parser.add_argument("-id", "--rank", type=int, help="process rank", default=0, required=False)
    parser.add_argument("-ws", "--world_size", type=int, help="world size", default=2, required=False)
    parser.add_argument("-ip", "--ip", type=str, help="master ip", default="127.0.0.1", required=False)
    parser.add_argument("-p", "--port", type=int, help="master port", default=26379, required=False)
    parser.add_argument("-if", "--iface", type=str, help="network device name", default="lo", required=False)
    parser.add_argument("-ipm", "--ip_mover", type=str, help="data move service ip", default="lo", required=False)
    parser.add_argument("-portm", "--port_mover", type=str, help="data move service port", default="lo", required=False)
    
    # get arguments
    args = parser.parse_args()

    rank = args.rank
    os.environ["RANK"] = str(rank)
    world_size = args.world_size
    os.environ["WORLD_SIZE"] = str(world_size)
    ip = args.ip
    os.environ["MASTER_ADDR"] = str(ip)
    port = args.port
    os.environ["MASTER_PORT"] = str(port)
    iface = args.iface
    os.environ["GLOO_SOCKET_IFNAME"] = str(iface)

    # for iamgenet stop-gap measure
    if args.dataset == "imagenet":
        args.num_class = 1000

    # start training
    if args.distributed:
        get_training_process(args.store_strategy
            rank=args.rank, batch_size=args.batch_size, epoch_count=args.epoch_count,
            num_classes=args.num_class, dataset_name=args.dataset, model_name=args.model,
            num_replicas=args.world_size, ddl=args.distributed, gpu=args.gpu,
            sampler=args.sampler, args=args
        )
    else:
        torch.multiprocessing.spawn(get_training_process(args.store_strategy),
            args=(args.batch_size, args.epoch_count, args.num_class, args.dataset, args.model, args.process_count),
            nprocs=args.process_count,
            join=True
        )
