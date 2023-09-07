import numpy as np
import redis
import ctypes
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


def get_dataset(dataset_name:str):
    if dataset_name == "cifar10":
        # Define the transformations for data preprocessing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
        ])
        return torchvision.datasets.CIFAR10(root="./data", download=True, transform=transform)
    elif dataset_name == "cifar100":
        # Define the transformations for data preprocessing
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
        ])

        return torchvision.datasets.CIFAR100(root="./data", download=True, transform=transform)


class LocalPool(Dataset):
    def __init__(self, dataset_name):
        self.dataset = get_dataset(dataset_name=dataset_name)
        self.nb_samples = len(self.dataset)
        # cache it in memory by creating list
        self.data_list = []
        self.label_list = []
        for i, data in enumerate(self.dataset, 0):
            self.data_list.append(data[0])
            self.label_list.append(data[1])

    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]
    
    def __len__(self):
        return self.nb_samples

class SharedPool(Dataset):
    def __init__(self, dataset_name):
        self.nb_samples = 50000
        c = 3
        h = w = 32
        shared_array_base = torch.multiprocessing.Array(ctypes.c_float, self.nb_samples*c*h*w)
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
    

class DatasetPipeline():
    def __init__(self, dataset:Dataset, batch_size:int) -> None:
        self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        return self.dataloader._get_iterator()