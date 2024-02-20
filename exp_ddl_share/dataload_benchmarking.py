# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py

import argparse
import os
import random
import shutil
import time
import subprocess
import warnings
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

# custom dataset, dataloader related packages
from dataset import SharedDistRedisPool, DatasetPipeline, PyTorchDaliPipeline
from DistribSampler import DefaultDistributedSampler, GradualDistAwareDistributedSamplerBG
from shade_modified import ShadeDataset, ShadeSampler
from DataMovementService import DataMoverServiceInterfaceClient

# local cluster environment specific
IMAGENET_DATA_DIR = "/sandbox1/data/imagenet/2017"
# networks to benchmark, all are taken from torchvision
NETWORK_LIST = ["efficientnet_b1", "mobilenet_v2", "resnet18", "resnet50", "resnet101", "vgg16"]
# sampler list to benchmark
SAMPLER_LIST = ["default", "dali", "shade", "graddistbg"]
# benchmark data dict
benchmark_data_dict = {}
# proposed method related variable
data_mover = None

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Benchmarking')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
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
parser.add_argument("-sampler", "--sampler", default="default", required=False, choices=["default", "distaware", "shade", "graddist", "graddistbg", "dali"],
                        help="what sampler will be used")
# for data movement service
parser.add_argument("-ipm", "--ip_mover", type=str, help="data move service ip", default="lo", required=False)
parser.add_argument("-portm", "--port_mover", type=str, help="data move service port", default="lo", required=False)


def main():
    args = parser.parse_args()

    iface = args.iface
    os.environ["GLOO_SOCKET_IFNAME"] = str(iface)

    for network_arch in NETWORK_LIST:
        benchmark_data_dict[network_arch] = {}
        for sampler in SAMPLER_LIST:
            args.sampler = sampler
            benchmark_data_dict[network_arch][sampler] = []
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
    # dump the data
    # only rank 0 process will do that
    if args.rank == 0:
        with open("benchmark_iteration_step.tsv", "w") as fout:
            fout.write("Network Arch\tSampler\tdataload time\tdata process time\n")
            for network_arch in NETWORK_LIST:
                for sampler in SAMPLER_LIST:
                    datatime = benchmark_data_dict[network_arch][sampler][0]
                    processtime = benchmark_data_dict[network_arch][sampler][1]
                    fout.write("{0}\t{1}\t{2}\t{3}\n".format(network_arch, sampler, datatime, processtime))

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
        # torch.cuda.set_device(args.rank)
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
                torch.cuda.set_device(args.gpu)
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
        dataset = SharedDistRedisPool()
    else:
        dataset = ShadeDataset()

    # create the sampler
    if args.sampler == "shade":
        data_sampler = ShadeSampler(
            dataset=dataset, num_replicas=args.world_size, batch_size=args.batch_size, host_ip="0.0.0.0")
    elif args.sampler == "graddistbg":
        data_sampler = GradualDistAwareDistributedSamplerBG(
            dataset=dataset, num_replicas=args.world_size, batch_size=args.batch_size)
        data_sampler.set_rank(rank=args.rank)
        # starting the background data mover service
        data_mover_service = subprocess.Popen(
            """python3 {2}/DataMovementService.py --seqno {0}
            -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p {1}""".format(
                args.rank if args.rank < 3 else 2, args.port_mover, os.path.dirname(os.path.abspath(__file__))).split()
        )
        # check if running
        if data_mover_service.poll() is None:
            print("data mover service is running")
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
                data_mover = DataMoverServiceInterfaceClient(args.ip_mover, args.port_mover)
                break
            except ConnectionError as e:
                connection_refused_count += 1
                print("connection establish attempt {0} failed".format(connection_refused_count))
                # sleep for a second
                time.sleep(1)

    process_time = AverageMeter()
    data_time = AverageMeter()

    start_time = time.time()
    for epoch in range(args.epochs):
        if args.distributed:
            # custom dataloader
            train_loader.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args, process_time=process_time, data_time=data_time)

        scheduler.step()
    
    if args.sampler == "graddistbg":
        data_mover.close()
        data_mover_service.kill()

    print("network {0} took {1}s".format(arch, time.time() - start_time))

    process_time.all_reduce()
    data_time.all_reduce()
    benchmark_data_dict[arch][args.sampler] = [data_time, process_time]

    dist.destroy_process_group()

def train(train_loader, model, criterion, optimizer, epoch, device, args, process_time, data_time):
    global data_mover
    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.sampler == "graddistbg":
            # data is read cache can be updated
            data_mover.updatecache(i)

        process_start_time = time.time()
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

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
        return self.avg

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
