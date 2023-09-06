import psutil
import sys
import ctypes
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms

import redis


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

class SharedPool(Dataset):
    def __init__(self):
        self.nb_samples = 50000
        c = 3
        h = w = 32
        shared_array_base = mp.Array(ctypes.c_float, self.nb_samples*c*h*w)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(self.nb_samples, c, h, w)
        self.shared_array = torch.from_numpy(shared_array)
    
    def __getitem__(self, index):
        x = self.shared_array[index]
        return x, 0
    
    def __len__(self):
        return self.nb_samples

class SharedRedisPool(Dataset):
    def __init__(self):
        # prepare redis client
        redis_host = 'localhost'  # Change this to your Redis server's host
        redis_port = 6379  # Change this to your Redis server's port
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
        self.nb_samples = int.from_bytes(self.redis_client.get("length"), 'little')
    
    def __getitem__(self, index):
        deser_x = self.redis_client.get("data" + str(index))
        deser_y = self.redis_client.get("label" + str(index))
        
        x = torch.from_numpy(np.frombuffer(deser_x, dtype=np.float32).reshape(3,32,32))
        y = int.from_bytes(deser_y, 'little')

        return x, y
    
    def __len__(self):
        return self.nb_samples


def train_process(rank, world_size, train_strategy, train_dataset):
    model = ToyModel()
    # model.to("cuda")
    # Define the transformations for data preprocessing
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
    ])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    if train_strategy is None or train_strategy == "baseline":
        # Download and load the CIFAR-10 training dataset
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

        # Create data loaders for training and test datasets
        # each process will create its own dataloader to create their own copy
        batch_size = 16
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        data_list = []
        label_list = []
        for i, data in enumerate(train_loader, 0):
            data_list.append(data[0])
            label_list.append(data[1])

        for epoch in range(2):
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
            for i in range(0, len(data_list), batch_size):
                inputs = data_list[i]
                labels = label_list[i]
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
                try:
                    loss = loss_fn(outputs, one_hot)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(outputs, one_hot)
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")

    elif train_strategy == "shared":
        batch_size = 16
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(2):
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                one_hot = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=10).float()
                try:
                    loss = loss_fn(outputs, one_hot)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(e)
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")

    elif train_strategy == "sharedpool":
        batch_size = 16
        train_dataset = SharedRedisPool()
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(2):
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float()
                try:
                    loss = loss_fn(outputs, one_hot)
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(e)
            print("Memory rss footprint of process ", rank, " at epoch", epoch, " end ", (psutil.Process().memory_info().rss)>>20, "MiB")
            print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")


    return


if __name__ == "__main__":
    # number of process 
    # will be used to create unique id from 0-3
    num_rank = int(sys.argv[1])
    strategy = sys.argv[2]
    # first argument to train_process is its rank
    # which will be controlled by spawner
    if strategy == "shared":
        # Define the transformations for data preprocessing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
        ])
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    else:
        train_dataset = None

    mp.spawn(train_process,
             args=(1, strategy, train_dataset),
             nprocs=num_rank,
             join=True)
