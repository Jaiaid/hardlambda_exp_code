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

from dataset import LocalPool, SharedPool, SharedRedisPool, DatasetPipeline


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
    elif strategy_name == "sharedlocal":
        return train_process_shared_pool_local
    elif strategy_name == "disaggregated":
        return train_process_shared_pool_far
    elif strategy_name == "local_random":
        return train_process_local_pool_distrib_shuffle
    return None

def get_model(name:str) -> torch.nn.Module:
    return ToyModel()

def train_process_local_pool(rank, batch_size, epoch_count, num_classes, dataset_name, model_name):
    # create the model training pipeline
    training_pipeline = ModelPipeline(model=get_model(model_name))
    # Define the transformations for data preprocessing
    dataset_pipeline = DatasetPipeline(LocalPool(dataset_name=dataset_name), batch_size=batch_size)
    
    for epoch in range(epoch_count):
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        for i, data in enumerate(dataset_pipeline):
            inputs, labels = data
            # generate label and move to GPU
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            # train one iteration
            training_pipeline.run_train_step(inputs=inputs, labels=one_hot)

        print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")

def train_process_local_pool_distrib_shuffle(rank, batch_size, epoch_count, num_classes, dataset_name, model_name, num_replicas=None, ddl=None, gpu=False):
    print("creating model pipeline")
    # create the model training pipeline
    training_pipeline = ModelPipeline(model=get_model(model_name))
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
    dataset_pipeline = DatasetPipeline(LocalPool(dataset_name=dataset_name), batch_size=batch_size, sampler="dist", num_replicas=num_replicas)
    import time
    total_time = 0
    count = 0
    for epoch in range(epoch_count):
        print("starting training epoch {0}...".format(epoch))
        dataset_pipeline.set_epoch(epoch)
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        try:
            for i, data in enumerate(dataset_pipeline):
                inputs, labels = data
                # generate label and move to GPU
                if not gpu:
                    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                else:
                    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).cuda(torch.device('cuda', device_idx))
                # train one iteration
                t = time.time()
                training_pipeline.run_train_step(inputs=inputs, labels=one_hot)
                total_time += time.time() - t
                count += 1
        except KeyboardInterrupt as e:
            print(total_time/count)
            exit(0)

        print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")


def train_process_shared_pool_local(rank, batch_size, epoch_count, num_classes, dataset_name, model_name):
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epoch_count):
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            input = input.cuda()
            # generate label and move to GPU
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            one_hot = one_hot.cuda()
            
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")


def train_process_shared_pool_far(rank, batch_size, epoch_count, num_classes, dataset_name, model_name):
    # create the model training pipeline
    training_pipeline = ModelPipeline(model=get_model(model_name))
    # Define the transformations for data preprocessing
    dataset_pipeline = DatasetPipeline(SharedRedisPool(), batch_size=batch_size)

    for epoch in range(epoch_count):
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        for i, data in enumerate(dataset_pipeline, 0):
            inputs, labels = data
            # inputs = inputs.cuda()
            # generate label and move to GPU
            one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
            # one_hot = one_hot.cuda()
            
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
