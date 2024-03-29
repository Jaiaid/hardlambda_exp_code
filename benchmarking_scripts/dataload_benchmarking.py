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

    benchmark_data_dict[network_arch] = {}
    print("benchmarking for network arch: {0}, sampler: {1}".format(network_arch, sampler))
        
    args.sampler = sampler
    benchmark_data_dict[network_arch][sampler] = [math.nan, math.nan, math.nan]
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    if torch.cuda.is_available():
        ngpus_per_node = torch.cuda.device_count()
        # nccl does not work with single GPU
        # https://discuss.pytorch.org/t/ncclinvalidusage-of-torch-nn-parallel-distributeddataparallel/133183
        # https://discuss.ray.io/t/ray-train-parallelize-on-single-gpu/11483/2
        if ngpus_per_node == 1:
            args.dist_backend = "gloo"
    else:
        ngpus_per_node = 0
        assert args.dist_backend != "nccl",\
        "nccl backend does not work without GPU, see https://pytorch.org/docs/stable/distributed.html"
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, network_arch))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, network_arch)
    
    # print("Benchmarking for network: {0} and sampler: {1} could not be done".format(network_arch, sampler))
        

    # dump the data
    # only rank 0 process will do that
    if args.rank == 0:
        with open("benchmark_iteration_step.tsv", "a") as fout:
            # fout.write("Network Arch\tBatch Size\tImage Size\tEpoch\tSampler\tdataload time\tdata process time\tcache update time\texec time\trss(MiB)\tvms(MiB)\tmax rss(MiB)\tmax vms(MiB)\n")
            datatime = benchmark_data_dict[network_arch][sampler][0]
            cache_time = benchmark_data_dict[network_arch][sampler][1]
            processtime = benchmark_data_dict[network_arch][sampler][2]
            exec_time = benchmark_data_dict[network_arch][sampler][3]
            rss = benchmark_data_dict[network_arch][sampler][4]
            vms = benchmark_data_dict[network_arch][sampler][5]
            rss_peak = benchmark_data_dict[network_arch][sampler][6]
            vms_peak = benchmark_data_dict[network_arch][sampler][7]
            fout.write(
                "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}\t{12}\n".format(
                    network_arch, args.batch_size, image_size, args.epochs, sampler+"_ereduce" if args.epoch_sync else sampler, datatime, cache_time,
                    processtime, exec_time, rss, vms, rss_peak, vms_peak
                )
            )

def main_worker(gpu, ngpus_per_node, args, arch):
    global data_mover, benchmark_data_dict

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        # added following
        # https://github.com/pytorch/torchrec/issues/328
        torch.cuda.set_device(args.rank%ngpus_per_node)
        # let rank 0 to catch up in mulit node setup
        if args.rank != 0:
            time.sleep(10)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if torch.cuda.is_available():
            if args.gpu is not None:
                # torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs of the current node.
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
        dataset = ShadeDataset(cachedesc_filepath=args.cache_descriptor)

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
        cache_nodes_dict[i] = [ip, serviceport]

    # create the sampler
    if args.sampler == "shade":
        data_sampler = ShadeSampler(
            dataset=dataset, num_replicas=args.world_size, batch_size=args.batch_size, host_ip="0.0.0.0")
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

    process_time = AverageMeter()
    data_time = AverageMeter()
    cacheupdate_time = MaxMeter()
    exec_time = MaxMeter()
    rss_amount = AverageMeter()
    rss_peak = MaxMeter()
    vms_amount = AverageMeter()
    vms_peak = MaxMeter()

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            # custom dataloader
            train_loader.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args, process_time=process_time, data_time=data_time,
                cacheupdate_time=cacheupdate_time, memory_rss=rss_amount, memory_vms=vms_amount, rss_peak=rss_peak, vms_peak=vms_peak)

        scheduler.step()
    
    if args.sampler == "graddistbg":
        data_mover.close()
    if args.sampler == "shade":
        # clean the cached data for next run
        redis_host = 'localhost'  # Change this to your Redis server's host
        redis_port = 6379  # Change this to your Redis server's port
        redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
        redis_client.flushdb()

    exec_time.update(time.time() - start_time)
    print("network {0} took {1}s".format(arch, str(exec_time)))

    process_time.all_reduce()
    data_time.all_reduce()
    cacheupdate_time.all_reduce()
    exec_time.all_reduce()
    rss_amount.all_reduce()
    vms_amount.all_reduce()
    rss_peak.all_reduce()
    vms_peak.all_reduce()
    benchmark_data_dict[arch][args.sampler] = [data_time, cacheupdate_time, process_time, exec_time, rss_amount, vms_amount, rss_peak, vms_peak]

    dist.barrier()
    dist.destroy_process_group()

def train(train_loader, model, criterion, optimizer, epoch, device, args, process_time, data_time,
          cacheupdate_time, memory_rss, memory_vms, rss_peak, vms_peak):
    global data_mover, image_size
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.sampler == "graddistbg":
            # data is read cache can be updated
            cacheupdate_cmd_send_time = time.time()
            data_mover.updatecache(i)
            cacheupdate_time.update(time.time() - cacheupdate_cmd_send_time)

        process_start_time = time.time()
        # move data to the same device as model
        image_size = images.shape
        images = images.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)

        if args.epoch_sync and i < len(train_loader)/args.batch_size-1:
            with model.no_sync():
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # measure elapsed time
        end = time.time()
        process_time.update(time.time() - process_start_time)
        memory_rss.update(psutil.Process().memory_info().rss>>20)
        memory_vms.update(psutil.Process().memory_info().vms>>20)
        rss_peak.update(psutil.Process().memory_info().rss>>20)
        vms_peak.update(psutil.Process().memory_info().vms>>20)

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
