# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py

import argparse
import os
import random
import math
import yaml
import time
import subprocess
import psutil
import warnings
import redis
import datetime
import numpy as np
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

# to use package in a script and the pacakege in different folder
import sys
sys.path.append("..")

# custom dataset, dataloader related packages
from datatrain.dataset import SharedDistRedisPool, DatasetPipeline, PyTorchDaliPipeline, GraddistBGPipeline
from datatrain.DistribSampler import DefaultDistributedSampler, GradualDistAwareDistributedSamplerBG
from datatrain.shade_modified import ShadeDataset, ShadeSampler
from datatrain.DataMovementService import DataMoverServiceInterfaceClient

# local cluster environment specific
IMAGENET_DATA_DIR = "/sandbox1/data/imagenet/2017"
# what size of image used
image_size = None
# benchmark data dict
benchmark_data_dict = {}
# proposed method related variable
data_mover = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Benchmarking')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://10.21.12.239:44144', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
# ddl interface related
parser.add_argument("-if", "--iface", type=str, help="network device name", default="lo", required=False)
# sampler related
parser.add_argument("-sampler", "--sampler", default="default", required=False, choices=["default", "shade", "graddistbg", "dali"],
                        help="what sampler will be used")
# for proposed method specialized dataset initiation
parser.add_argument("-cachedesc", "--cache-descriptor", type=str, help="yaml file describing caches", default="cache.yaml", required=False)
# parameter synchornization modification related
parser.add_argument("-esync", "--epoch-sync", action='store_true', help="use to sync gradient at epoch boundary")


def main():
    args = parser.parse_args()

    network_arch = args.arch
    sampler = args.sampler
    iface = args.iface
    os.environ["GLOO_SOCKET_IFNAME"] = str(iface)
    # for deterministic training
    args.seed = 3400
    benchmark_data_dict[network_arch] = {}
    print("benchmarking for network arch: {0}, sampler: {1}".format(network_arch, sampler))
        
    args.sampler = sampler
    benchmark_data_dict[network_arch][sampler] = [math.nan, math.nan, math.nan]
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        # it seems it is not easy to make it deterministic
        # some layer may not have deterministic implementation available
        # Hence use deterministic is commented
        # also see https://stackoverflow.com/questions/70178014/something-about-the-reproducibility-of-pytorch-on-multi-gpu
        # torch.use_deterministic_algorithms(True)
        cudnn.deterministic = False
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')


    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.dist_backend = "gloo"
    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args, network_arch)    

    # dump the data
    # only rank 0 process will do that
    if args.rank == 0:
        with open("lossaccuracy_epochs.tsv", "a") as fout:
            for entry in benchmark_data_dict[network_arch][sampler]:
            # fout.write("Network Arch\tBatch Size\tImage Size\tEpoch\tSampler\tdataload time\tdata process time\tcache update time\texec time\trss(MiB)\tvms(MiB)\tmax rss(MiB)\tmax vms(MiB)\n")
                epoch = entry[0]
                loss = entry[1]
                acc1 = entry[2]
                acc5 = entry[3]
                cachesize = entry[4] 
                fout.write(
                    "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(
                        network_arch, args.batch_size, image_size[2], args.epochs, sampler+"_ereduce" if args.epoch_sync else sampler, epoch, loss, acc1, acc5, cachesize)
                )

def main_worker(gpu, ngpus_per_node, args, arch):
    global data_mover, benchmark_data_dict

    # added following
    # https://github.com/pytorch/torchrec/issues/328
    torch.cuda.set_device(args.rank%ngpus_per_node)
    try:
        # following https://github.com/lkeab/BCNet/issues/53
        dist.init_process_group(backend=args.dist_backend, timeout=datetime.timedelta(seconds=100000), init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    except Exception as e:
        print(e, args.rank, args.world_size, ngpus_per_node)
        sys.exit(0)
    # create model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    model.cuda()
    # DistributedDataParallel will divide and allocate batch_size to all
    # available GPUs if device_ids are not set
    model = torch.nn.parallel.DistributedDataParallel(model)
    device = torch.device("cuda")
    # define loss function (criterion), optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Data loading code
    # creating the custom data loading mechanism
    print("creating data pipeline")
    # Define the transformations for data preprocessing
    if args.sampler != "shade":
        dataset = SharedDistRedisPool(cachedesc_filepath=args.cache_descriptor)
    else:
        # IMPORTANT:
        # WE ASSUME IN A HOST AT MOST TWO DIFFERENT WORKER CAN BE
        # THAT'S WHY THIS MODULUS, TO MAKE IT GENERAL WE HAVE TO BOTH SCRIPTIFY REDIS START AT PARTICULAR NODE 
        # AND PROVIDE WHICH REDIS TO CONNECT AS PARAMETER
        dataset = ShadeDataset(cachedesc_filepath=args.cache_descriptor, port_num=6379+(args.rank%2))

    with open(args.cache_descriptor) as fin:
        cachedatadict = yaml.safe_load(fin)

    cachedatadict = cachedatadict["cachedict"]
    cache_nodes_dict = {}
    rank_id_dict = {}
    for i, key in enumerate(cachedatadict):
        rank = key
        rank_id_dict[i] = rank

        ip = cachedatadict[key][0].split(":")[0]
        serviceport = cachedatadict[key][1]["serviceport"]
        cachesize = cachedatadict[key][1]["length"]
        cache_nodes_dict[i] = [ip, serviceport]

    # create the sampler
    if args.sampler == "shade":
        data_sampler = ShadeSampler(
            dataset=dataset, num_replicas=args.world_size, batch_size=args.batch_size, host_ip="0.0.0.0", port_num=6379+(args.rank%2))
    elif args.sampler == "graddistbg":
        data_sampler = GradualDistAwareDistributedSamplerBG(
            dataset=dataset, num_caches=len(cache_nodes_dict), batch_size=args.batch_size)
        data_sampler.set_rank(rank=args.rank)
    else:
        data_sampler = DefaultDistributedSampler(
            dataset=dataset, num_replicas=args.world_size)

    # edited for custom data loading
    train_loader = DatasetPipeline(dataset=dataset, batch_size=args.batch_size,
                                    sampler=data_sampler, num_replicas=args.world_size)

    if args.sampler == "graddistbg":
        # try 10 times to connect
        connection_refused_count = 0
        while connection_refused_count < 10: 
            try:
                data_mover = DataMoverServiceInterfaceClient(ip=cache_nodes_dict[args.rank][0], port=cache_nodes_dict[args.rank][1])
                # connection established send the batch size
                data_mover.set_batchsize(batch_size=args.batch_size)
                break
            except ConnectionError as e:
                connection_refused_count += 1
                print("connection establish attempt {0} failed".format(connection_refused_count))
                # sleep for a second
                time.sleep(1)

    try:
        loss = AverageMeter()
        acc1 = AverageMeter()
        acc5 = AverageMeter()
        benchmark_data_dict[arch][args.sampler] = []

        for epoch in range(args.epochs):
            loss.reset()
            acc1.reset()
            acc5.reset()    

            if args.distributed:
                # custom dataloader
                train_loader.set_epoch(epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, device, args, loss_store=loss, acc1_store=acc1,
                    acc5_store=acc5)
            
            scheduler.step()

            loss.all_reduce()
            acc1.all_reduce()
            acc5.all_reduce()
            dist.barrier()

            # we are doing string otherwise the object reference will be there in the list 
            # which will cause every loss entry to become the last lose entry
            # same for acc 
            benchmark_data_dict[arch][args.sampler].append([epoch + 1, str(loss), str(acc1), str(acc5), cachesize])
        
        if args.sampler == "graddistbg":
            data_mover.close()
        if args.sampler == "shade":
            # clean the cached data for next run
            redis_host = 'localhost'  # Change this to your Redis server's host
            redis_port = 6379  # Change this to your Redis server's port
            redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
            redis_client.flushdb()

    except Exception as e:
        print(e)
    finally:
        dist.destroy_process_group()

def train(train_loader, model, criterion, optimizer, epoch, device, args, loss_store, acc1_store, acc5_store):
    global data_mover, image_size
    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):
        # to ensure data is actually read
        images = images.to("cpu")
        target = target.to("cpu")
        if args.sampler == "graddistbg":
            data_mover.updatecache(i)

        # move data to the same device as model
        image_size = images.shape
        images = images.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_store.update(loss.item())
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1_store.update(acc1)
        acc5_store.update(acc5)

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, summary_type=Summary.AVERAGE):
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        return str(self.avg)


class MaxMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, summary_type=Summary.AVERAGE):
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.max = 0
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.max = max(self.max, val)

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        max_tensor = torch.tensor([self.max], dtype=torch.float32, device=device)
        dist.all_reduce(max_tensor, dist.ReduceOp.MAX, async_op=False)
        self.max = max_tensor.tolist()[0]

    def __str__(self):
        return str(self.max)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
