import os
import psutil
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models import resnet18
from dataset import SharedRedisPool, SharedDistRedisPool, DatasetPipeline
from DistribSampler import DistAwareDistributedSampler, DefaultDistributedSampler, GradualDistAwareDistributedSampler, GradualDistAwareDistributedSamplerBG
from shade_modified import ShadeDataset, ShadeSampler
from DataMovementService import DataMoverServiceInterfaceClient

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ModelPipeline():
    def __init__(self, model:torch.nn.Module) -> None:
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001)

    def make_gpu(self, device_idx):
        self.model = self.model.cuda(torch.device('cuda', device_idx))

    def make_distributed(self):
        self.model = DDP(self.model)

    def run_train_step(self, inputs:torch.Tensor, labels:torch.Tensor):
        # zero the parameter gradients
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        try:
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
        except Exception as e:
            print(e)
            exit(0)

def get_training_process(strategy_name):
    if strategy_name == "baseline":
        return train_process_local_pool
    elif strategy_name == "distributed_random":
        return train_process_pool_distrib_shuffle
    return None

def get_model(name:str, num_classes:int) -> torch.nn.Module:
    if name=="resnet18":
        model = resnet18()
        model.fc = nn.Linear(512, num_classes) 
        return model
    return ToyModel()


def train_process_pool_distrib_shuffle(rank, batch_size, epoch_count, num_classes,
                                       dataset_name, model_name, num_replicas=None, ddl=None, gpu=False,
                                       sampler=None, args=None):
    print("creating model pipeline")
    # create the model training pipeline
    training_pipeline = ModelPipeline(model=get_model(model_name, num_classes=num_classes))
    # to select which gpu to use if multiple gpu
    device_idx = rank if torch.cuda.device_count()>rank else 0

    if gpu:
        training_pipeline.make_gpu(device_idx)
    if ddl:
        print("creating process group")
        dist.init_process_group("gloo", rank=rank, world_size=num_replicas)
        print("making model distributed")
        training_pipeline.make_distributed()
    else:
        dist.init_process_group("gloo", rank=rank, world_size=num_replicas)

    print("creating data pipeline")
    # Define the transformations for data preprocessing
    if sampler != "shade":
        dataset = SharedDistRedisPool()
    else:
        dataset = ShadeDataset()

    # create the sampler
    if sampler == "distaware":
        data_sampler = DistAwareDistributedSampler(
            dataset=dataset, num_replicas=num_replicas)
    elif sampler == "shade":
        data_sampler = ShadeSampler(
            dataset=dataset, num_replicas=num_replicas, batch_size=batch_size, host_ip="0.0.0.0")
    elif sampler == "graddist":
        data_sampler = GradualDistAwareDistributedSampler(
            dataset=dataset, num_replicas=num_replicas, batch_size=batch_size)
    elif sampler == "graddistbg":
        data_sampler = GradualDistAwareDistributedSamplerBG(
            dataset=dataset, num_replicas=num_replicas, batch_size=batch_size)
    else:
        data_sampler = DefaultDistributedSampler(
            dataset=dataset, num_replicas=num_replicas)

    # create the pipeline from sampler
    dataset_pipeline = DatasetPipeline(dataset=dataset, batch_size=batch_size,
                                       sampler=data_sampler, num_replicas=num_replicas)
    # to keep track of which data batch reading takes what time
    batch_read_time = [0] * int(dataset.nb_samples/batch_size + 1)

    import time
    total_time = 0
    count = 0
    # to clean existing file
    fout = open("latency_data_rank_{0}.csv".format(rank), "w")
    fout.write("epoch,batch,batch size,read time (s)\n")
    fout.close()

    if sampler == "graddistbg":
        print("starting data movement service")
        data_mover = DataMoverServiceInterfaceClient(args.ip_mover, args.port_mover)
        print("data movement service is working in background")

    for epoch in range(epoch_count):
        print("starting training epoch {0}...".format(epoch))
        dataset_pipeline.set_epoch(epoch)
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        try:
            epoch_start_time = time.time()
            for i, data in enumerate(dataset_pipeline):
                t = time.time()
                inputs, labels = data
                # got the data now issue cache update cmd
                # if data movement in background is running
                # which is the case for "graddistbg" sampler
                if sampler == "graddistbg":
                    data_mover.updatecache(i)
                # generate label and move to GPU
                one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                if gpu:
                    inputs = inputs.cuda(torch.device('cuda', device_idx))
                    one_hot = one_hot.cuda(torch.device('cuda', device_idx))
                # record the batch_read time
                # we are adding with the assumption that all rank's train equal epoch count
                # therefore, summation of time is indicative of delay
                batch_read_time[i] += time.time() - t
                # set the time
                data_sampler.set_batch_time(i, batch_read_time[i])
                # train one iteration
                training_pipeline.run_train_step(inputs=inputs, labels=one_hot)
                total_time += time.time() - t
                count += 1
            print("epoch {0} took: {1}s".format(epoch, time.time() - epoch_start_time))
        except KeyboardInterrupt as e:
            print(total_time/count)
            exit(0)

        print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        
        with open("latency_data_rank_{0}.csv".format(rank), "a") as fout:
            for batch_no, latency in enumerate(batch_read_time):
                fout.write(str(epoch) + "," + str(batch_no) + "," + str(batch_size) + "," + str(latency) + "\n")
                batch_read_time[batch_no] = 0
    
        
    if sampler == "graddistbg":
        data_mover.close()

    # dump data read freq, only rank 0 will init that
    dataset_pipeline.dump_data_read_freq("freq_data_rank_{0}.csv".format(rank))

