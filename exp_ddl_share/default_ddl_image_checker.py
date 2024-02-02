# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py

import argparse
import os
import time
import subprocess
from enum import Enum
import PIL
import numpy as np

# import torch
# import torch.backends.cudnn as cudnn
import torch.distributed as dist
# import torch.multiprocessing as mp
# import torch.nn as nn
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torchvision.transforms as transforms
# from torch.optim.lr_scheduler import StepLR
# from torch.utils.data import Subset

# custom dataset, dataloader related packages
from dataset import SharedDistRedisPool, DatasetPipeline
from DistribSampler import DefaultDistributedSampler

# proposed method related variable
data_mover = None


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://10.21.12.239:44144', type=str,
                    help='url used to set up distributed training')
# ddl interface related
parser.add_argument("-if", "--iface", type=str, help="network device name", default="lo", required=False)
# for data movement service
parser.add_argument("-ipm", "--ip_mover", type=str, help="data move service ip", default="lo", required=False)
parser.add_argument("-portm", "--port_mover", type=str, help="data move service port", default="lo", required=False)

best_acc1 = 0


def main():
    args = parser.parse_args()

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    global best_acc1, data_mover
    args.gpu = gpu

    iface = args.iface
    os.environ["GLOO_SOCKET_IFNAME"] = str(iface)

    dist.init_process_group(backend="gloo", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # creating the custom data loading mechanism
    print("creating data pipeline")
    dataset = SharedDistRedisPool()

    data_sampler = DefaultDistributedSampler(
            dataset=dataset, num_replicas=args.world_size)
    # edited for custom data loading
    train_loader = DatasetPipeline(dataset=dataset, batch_size=args.batch_size,
                                       sampler=data_sampler, num_replicas=args.world_size)

    # extract the image for 3 epoch
    # because there are 3 cache, 
    # if cache update is correct all data should be seen in three epoch
    img_count = 0
    for epoch in range(3):
        # custom dataloader
        train_loader.set_epoch(epoch)

        for i, (images, target) in enumerate(train_loader):
            # data is read cache can be updated
            data_mover.updatecache(i)

            # only one should do the saving
            if args.rank == 0:
                # save images to cache_data_folder
                for j in range(args.batch_size):
                    im = PIL.Image.fromarray((images[j,:]*255).numpy().transpose(1, 2, 0).astype(np.uint8))
                    im.save(os.path.join("cache_data_default", "{0}.jpg".format(img_count)))
                    img_count  += 1

def train(train_loader, model, criterion, optimizer, epoch, device, args):
    global data_mover


if __name__ == '__main__':
    main()
