import copy
import random
import time
import math
import argparse

import torch
import torchvision
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

from tqdm import tqdm
from nvitop import Device, ResourceMetricCollector


NETWORKS =  ["resnet50"]
BATCH_SIZES = [[2048, 20], [1024, 20], [512,20], [256,20], [128,20], [64,20], [32,20], [16,20], [8,20], [4,20], [2,20], [1,20]]
IMGDIMS = [1024, 512, 256, 128, 64, 32]
LEARNING_RATE = 0.001
NUMBER_OF_CLASSES = 1000

gpu_utilization_dict = {}
gpu_utilization_list = []

# resource status collection
collector = ResourceMetricCollector(Device.cuda.all())


# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Benchmarking')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://10.21.12.239:44144', type=str,
                    help='url used to set up distributed training')

# will do full update through standard pytorch optim module
def training_all_param_update(args, nn_model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, batch_size: int, maxiter: int):
    global gpu_utilization_list
    # emptying the list before starting
    gpu_utilization_list = []
    # empty the cuda cache done by torch
    torch.cuda.empty_cache()
    # for collection sanity test
    data_collection_count = 0

    # start the resource collector with tag
    collector.start(tag="test")
    # NOT DOING DAEMONIZING
    # daemonize it, this returns a threading.thread object
    # will GIL cause issue? not sure yet
    # IT IS AN ISSUE MOST PROBABLY, daemon thread not getting invoked for given interval
    # daemon = collector.daemonize(on_collect, interval=0.1, on_stop=None)

    # move to GPU
    torch.cuda.set_device(args.rank%torch.cuda.device_count())
    
    # following https://github.com/lkeab/BCNet/issues/53
    dist.init_process_group(backend="gloo", init_method=args.dist_url,
                        world_size=2, rank=args.rank)

    try:
        nn_model.cuda()
        nn_model = torch.nn.parallel.DistributedDataParallel(nn_model)

        # init loss and optimizer
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), lr=LEARNING_RATE)

        # this is for loss calculated forward backward
        nn_model.train()
        iter_count = 0
        fw_time_count = 0
        bw_time_count = 0
        all_reduce_fw_time_count = 0
        all_reduce_bw_time_count = 0
        train_data_load_time = 0
        data_load_start_time = time.time()
        for input_data, label in tqdm(dataloader, leave=False):
            # zero the optimizer grad
            optimizer.zero_grad()

            if iter_count % 2 == 0:
                with nn_model.no_sync():
                    # calculate loss of model
                    try:
                        one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
                        input_data = input_data.cuda()
                        train_data_load_time += time.time() - data_load_start_time 
                        t = time.time()
                        fw = nn_model(input_data)
                        fw_time_count += time.time() - t
                        # collect metrics
                        metrics = collector.collect()
                        gpu_utilization_list.append(
                            [
                                metrics["test/timestamp"],
                                metrics["test/host/cpu_percent (%)/mean"],
                                metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                                metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                            ]
                        )
                        data_collection_count += 1
                    except torch.cuda.OutOfMemoryError as e:
                        print(
                            "batch size {0} for cifar10 with given size is not suitable for forward pass and loss calc with GPU memory {1}GB".format(
                                batch_size, torch.cuda.mem_get_info()[1]>>30
                            )
                        )
                        fw_time_count = math.nan
                        bw_time_count = math.nan
                        iter_count = 1
                        break

                    # backprop
                    try:
                        
                        t = time.time()
                        loss = loss_func(fw, one_hot)
                        loss.backward()
                        # optimization step
                        optimizer.step()
                        # append the loss
                        loss_val = loss.data.cpu().item()
                        # per iteration loss record
                        bw_time_count += time.time() - t
                        # collect metrics
                        metrics = collector.collect()
                        gpu_utilization_list.append(
                            [
                                metrics["test/timestamp"],
                                metrics["test/host/cpu_percent (%)/mean"],
                                metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                                metrics["test/host/cpu_percent (%)/mean"],
                                metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                            ]
                        )
                        data_collection_count += 1
                    except torch.cuda.OutOfMemoryError as e:
                        print(
                            "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                                batch_size, torch.cuda.mem_get_info()[1]>>30
                            )
                        )
                        bw_time_count = math.nan
                        iter_count = 1
                        break
            else:
                # calculate loss of model
                try:
                    one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
                    input_data = input_data.cuda()
                    train_data_load_time += time.time() - data_load_start_time 
                    t = time.time()
                    fw = nn_model(input_data)
                    all_reduce_fw_time_count += time.time() - t
                    # collect metrics
                    metrics = collector.collect()
                    gpu_utilization_list.append(
                        [
                            metrics["test/timestamp"],
                            metrics["test/host/cpu_percent (%)/mean"],
                            metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                            metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                        ]
                    )
                    data_collection_count += 1
                except torch.cuda.OutOfMemoryError as e:
                    print(
                        "batch size {0} for cifar10 with given size is not suitable for forward pass and loss calc with GPU memory {1}GB".format(
                            batch_size, torch.cuda.mem_get_info()[1]>>30
                        )
                    )
                    all_reduce_fw_time_count = math.nan
                    all_reduce_bw_time_count = math.nan
                    iter_count = 1
                    break

                # backprop
                try:
                    
                    t = time.time()
                    loss = loss_func(fw, one_hot)
                    loss.backward()
                    # optimization step
                    optimizer.step()
                    # append the loss
                    loss_val = loss.data.cpu().item()
                    # per iteration loss record
                    all_reduce_bw_time_count += time.time() - t
                    # collect metrics
                    metrics = collector.collect()
                    gpu_utilization_list.append(
                        [
                            metrics["test/timestamp"],
                            metrics["test/host/cpu_percent (%)/mean"],
                            metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                            metrics["test/host/cpu_percent (%)/mean"],
                            metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                        ]
                    )
                    data_collection_count += 1
                except torch.cuda.OutOfMemoryError as e:
                    print(
                        "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                            batch_size, torch.cuda.mem_get_info()[1]>>30
                        )
                    )
                    all_reduce_bw_time_count = math.nan
                    iter_count = 1
                    break

            iter_count += 1
            if iter_count == maxiter:
                break
            data_load_start_time = time.time()
            # per epoch loss record
            # print("average forward time {0}s with batch size {1}".format(fw_time_count/iter_count, batch_size))
            # print("average loss calc time {0}s with batch size {1}".format(loss_time_count/iter_count, batch_size))
            # print("average backprop time {0}s with batch size {1}".format(bw_time_count/iter_count, batch_size))
            # print("average optimizer step time {0}s with batch size {1}".format(optstep_time_count/iter_count, batch_size))
    except Exception as e:
        print(e, args.rank)
    finally:
        dist.destroy_process_group()
        
        # clear collector status
        collector.clear()
        collector.stop(tag="test")

        del nn_model

        allreduce_time = (all_reduce_fw_time_count + all_reduce_bw_time_count) - (fw_time_count + bw_time_count)
        
        # print("\n\n", data_collection_count, "\n\n")
        return (fw_time_count + bw_time_count)/iter_count, (fw_time_count + bw_time_count)/(iter_count*batch_size), allreduce_time, allreduce_time/(iter_count * batch_size), train_data_load_time/iter_count, train_data_load_time/(iter_count*batch_size)


if __name__ == "__main__":
    args = parser.parse_args()

    for network in NETWORKS:
        benchmark_dict = {}

        first_time_write = True
        for img_siz in IMGDIMS:
            for batch_data in BATCH_SIZES:
                batch_size = batch_data[0]
                iteration = batch_data[1]
                random.seed(3400)
                data_dict_per_iteration = {}
                data_dict_per_epoch = {}
                # load dataset and data loader for cifar10
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                    torchvision.transforms.Resize([img_siz,img_siz]),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
                ])
                dataset = torchvision.datasets.CIFAR10(root=".", train=True, transform=transform, download=True)
                dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
                # model
                nn_model = torchvision.models.__dict__[network]()
                # nn_model.fc = torch.nn.Linear(512, NUMBER_OF_CLASSES)
                
                ptime, ptime_per_sample, allreduce, allreduce_per_sample, traindata_load_time, traindata_load_per_sample  = training_all_param_update(
                    args, nn_model=copy.deepcopy(nn_model), dataloader=dataloader, batch_size=batch_size, maxiter=iteration
                )

                benchmark_dict[batch_size] = [ptime, ptime_per_sample, allreduce, allreduce_per_sample, traindata_load_time, traindata_load_per_sample]
                gpu_utilization_dict[batch_size] = copy.deepcopy(gpu_utilization_list)

            if first_time_write and args.rank==0:
                with open("benchmark_{0}_nn_step.csv".format(network), "w") as fout:
                    fout.write("IMGDIM\tBatch Size\tprocess time\tprocess time per sample\tallreduce time\tall reduce per sample\tTrain data load\ttrain data load per sample\n")
                    for batch_data in BATCH_SIZES:
                        batch_size = batch_data[0]
                        data_list = benchmark_dict[batch_size]
                        fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                                img_siz, batch_size, data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5]
                            )
                        )
                first_time_write = False
            elif args.rank == 0:
                with open("benchmark_{0}_nn_step.csv".format(network), "a") as fout:
                    for batch_data in BATCH_SIZES:
                        batch_size = batch_data[0]
                        data_list = benchmark_dict[batch_size]
                        fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                                img_siz, batch_size, data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5]
                            )
                        )

        # with open("benchmark_{0}_gpu_utilization.csv".format(network), "w") as fout:
        #     for batch_size in BATCH_SIZES:
        #         fout.write("{0}\t{0}\t{0}\t".format(batch_size))
        #     fout.write("\n")
        #     for batch_size in BATCH_SIZES:
        #         fout.write("cpu usage(%)\tgpu usage(%)\tgpu memory usage (MiB)\t".format(batch_size))
        #     fout.write("\n")
            
        #     maxlen = max([len(gpu_utilization_dict[batch_size]) for batch_size in gpu_utilization_dict])
        #     for i in range(maxlen):
        #         for batch_size in BATCH_SIZES:
        #             if batch_size in gpu_utilization_dict:
        #                 if i < len(gpu_utilization_dict[batch_size]):
        #                     data_list = gpu_utilization_dict[batch_size][i]
        #                     fout.write("{0}\t{1}\t{2}\t".format(data_list[1], data_list[2], data_list[3]))
        #                 else:
        #                     fout.write("{0}\t{1}\t{2}\t".format(math.nan, math.nan, math.nan))
        #         fout.write("\n")
