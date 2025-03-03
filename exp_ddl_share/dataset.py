import math
import psutil
import numpy as np
import redis
import ctypes
import torch
import torchvision
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def, Pipeline



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
        x = np.frombuffer(deser_x, dtype=np.float32)
        dim_x = int(math.ceil(math.sqrt(x.shape[0]/3)))
        x = torch.from_numpy(x.reshape(3,dim_x,dim_x))
        y = int.from_bytes(deser_y, 'little')

        return x, y
    
    def __len__(self):
        return self.nb_samples

class SharedDistRedisPool(Dataset):
    def __init__(self):
        # prepare redis client
        redis_host1 = '10.21.12.222'  # Change this to your Redis server's host
        redis_port1 = 26379  # Change this to your Redis server's port
        self.redis_client1 = redis.StrictRedis(host=redis_host1, port=redis_port1, db=0)
        self.nb_samples = int.from_bytes(self.redis_client1.get("length"), 'little')

        redis_host2 = '10.21.12.239'  # Change this to your Redis server's host
        redis_port2 = 26379  # Change this to your Redis server's port
        self.redis_client2 = redis.StrictRedis(host=redis_host2, port=redis_port2, db=0)
        self.nb_samples += int.from_bytes(self.redis_client2.get("length"), 'little')

        redis_host3 = '10.21.12.239'  # Change this to your Redis server's host
        redis_port3 = 26380  # Change this to your Redis server's port
        self.redis_client3 = redis.StrictRedis(host=redis_host3, port=redis_port3, db=0)
        self.nb_samples += int.from_bytes(self.redis_client3.get("length"), 'little')

        self.redis_query_stat = { '10.21.12.222:26379': 0, '10.21.12.239:26379': 0, '10.21.12.239:26380': 0}
        self.index_queried_stat = {}
    
    def __getitem__(self, index):
        if index not in self.index_queried_stat:
            self.index_queried_stat[index] = 0
        self.index_queried_stat[index] += 1

        if index >= 2*self.nb_samples/3:
            index -= int(2*self.nb_samples/3)
            select_redis_client = self.redis_client1
            self.redis_query_stat['10.21.12.222:26379'] += 1
        elif index >= self.nb_samples/3:
            index -= int(self.nb_samples/3)
            select_redis_client = self.redis_client2
            self.redis_query_stat['10.21.12.239:26379'] += 1
        else:
            select_redis_client = self.redis_client3
            self.redis_query_stat['10.21.12.239:26380'] += 1
        
        deser_x = select_redis_client.get("data" + str(index))
        deser_y = select_redis_client.get("label" + str(index))
        x = np.frombuffer(deser_x, dtype=np.float32)
        dim_x = int(math.ceil(math.sqrt(x.shape[0]/3)))
        x = torch.from_numpy(x.reshape(3,dim_x,dim_x))
        y = int.from_bytes(deser_y, 'little')

        return x, y
    
    def __len__(self):
        return self.nb_samples

    def get_query_stat(self):
        return self.redis_query_stat
    
    def get_index_query_stat(self):
        return self.index_queried_stat

class PyTorchDaliPipeline(Pipeline):
    def __init__(self, pytorch_dataset, batch_size, num_threads, device_id):
        super(PyTorchDaliPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.PythonFunction(function=self.load_data, output_layout=[("data", types.FLOAT), ("label", types.INT32)])
        self.pytorch_dataset = pytorch_dataset
        self.batch_size = batch_size

    def load_data(self):
        # Use the PyTorch dataset to load data
        for _ in range(self.batch_size):
            sample = self.pytorch_dataset[np.random.randint(len(self.pytorch_dataset))]
            yield (sample['data'], sample['label'])

    def define_graph(self):
        return self.input()

class DatasetPipeline():
    def __init__(self, dataset:Dataset, batch_size:int, sampler:DistributedSampler=None, num_replicas:int=None) -> None:
        self.pipeline = None
        self.sampler = sampler
        if sampler is not None:
            self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        elif sampler == "dali":
            num_threads = int(psutil.cpu_count(logical=False)//2)
            device_id = 0
            self.dataset = dataset
            self.pipeline = PyTorchDaliPipeline(
                pytorch_dataset=dataset, batch_size=batch_size,
                num_threads=num_threads, device_id=device_id)
            dali_pipeline.build()
        else:
            self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        if self.sampler == "dali":
            return DALIGenericIterator(
                pipelines=[self.pipeline], output_map=["data", "label"],
                size=len(self.dataset)
            )
        return self.dataloader._get_iterator()
    
    def __len__(self):
        if self.sampler == "dali":
            return len(self.dataset)
        return len(self.dataloader)
    
    def dump_data_read_freq(self, output_file_path:str):
        self.sampler.dump_data_read_freq(output_file_path=output_file_path)
