import math
import psutil
import numpy as np
import redis
import yaml
import ctypes
import torch
import torchvision
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import pipeline_def, Pipeline, ops



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
    def __init__(self, cachedesc_filepath:str):
        with open(cachedesc_filepath) as fin:
            datadict = yaml.safe_load(fin)

        datadict = datadict["cachedict"]
        cache_nodes_dict = {}
        rank_id_dict = {}
        for i, key in enumerate(datadict):
            rank = key
            rank_id_dict[i] = rank

            ip = datadict[key][0].split(":")[0]
            port = datadict[key][0].split(":")[1]
            offset = datadict[key][1]["offset"]
            length = datadict[key][1]["length"]
            cache_nodes_dict[i] = [ip, port, offset, length]
        # prepare redis client
        self.cache_connection_list = []
        self.nb_samples = 0
        self.cache_query_stat = {}
        # create all the connection object by traversing parsed cache nodes description file
        for i, cache_node_rank in enumerate(cache_nodes_dict):
            redis_host = cache_nodes_dict[cache_node_rank][0]
            redis_port = cache_nodes_dict[cache_node_rank][1]
            # each entry is [connection object, offset, length of dataset]
            self.cache_connection_list.append(
                [redis.StrictRedis(host=redis_host, port=redis_port, db=0),
                 cache_nodes_dict[cache_node_rank][2],
                 cache_nodes_dict[cache_node_rank][3],
                 redis_host+":"+redis_port]
            )
            self.nb_samples += int.from_bytes(self.cache_connection_list[-1][0].get("length"), 'little')

            self.cache_query_stat[redis_host+":"+redis_port] = 0
        # sort according to offset, for later ease
        self.cache_connection_list.sort(key=lambda x:x[1])

    def __getitem__(self, index):
        # determine which one contains the data
        select_redis_client = self.cache_connection_list[-1][0]
        select_offset = self.cache_connection_list[-1][1]
        select_querycache_key = self.cache_connection_list[-1][3]
        for i in range(len(self.cache_connection_list)-1):
            if self.cache_connection_list[i][1] <= index and index < self.cache_connection_list[i+1][1]:
                select_redis_client = self.cache_connection_list[i][0]
                select_offset = self.cache_connection_list[i][1]
                select_querycache_key = self.cache_connection_list[i][3]
        # for stat purpose
        self.cache_query_stat[select_querycache_key] += 1

        # subtract the offset
        index -= select_offset
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
        return self.cache_query_stat


class PyTorchDaliPipeline(Pipeline):
    def __init__(self, pytorch_dataset, batch_size, num_threads, device_id):
        super(PyTorchDaliPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = fn.python_function(function=self.load_data) #, output_layouts=[("data", types.FLOAT), ("label", types.INT32)])
        self.pytorch_dataset = pytorch_dataset
        self.batch_size = batch_size

    def load_data(self):
        # Use the PyTorch dataset to load data
        for _ in range(self.batch_size):
            sample = self.pytorch_dataset[np.random.randint(len(self.pytorch_dataset))]
            yield (sample['data'], sample['label'])

    def define_graph(self):
        # Use the PyTorch dataset to load data
        for _ in range(self.batch_size):
            sample = self.pytorch_dataset[np.random.randint(len(self.pytorch_dataset))]
            yield (sample['data'], sample['label'])

        # return self.input()


class DatasetPipeline():
    def __init__(self, dataset:Dataset, batch_size:int, sampler:DistributedSampler=None,
                 num_replicas:int=None, num_threads:int=0) -> None:
        self.pipeline = None
        self.sampler = sampler
        if sampler == "dali":
            # num_threads = int(psutil.cpu_count(logical=False)//2)
            device_id = 0
            self.dataset = dataset
            self.pipeline = PyTorchDaliPipeline(
                pytorch_dataset=dataset, batch_size=batch_size,
                num_threads=num_threads, device_id=device_id)
            self.pipeline.build()
        elif sampler is not None:
            self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        else:
            self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)

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


class GraddistBGPipeline():
    def __init__(self, dataset:Dataset, batch_size:int, sampler:DistributedSampler=None, num_replicas:int=None) -> None:
        print("dataset length: {0}".format(len(dataset)))
        self.sampler = sampler
        if sampler is not None:
            self.dataloader: DataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
        self.count = 0
        self.batch_size = batch_size

    def set_epoch(self, epoch: int):
        self.sampler.set_epoch(epoch)

    def __iter__(self):
        # x_list = []
        # y_list = []
        # for i in range(self.batch_size):
        #     idx = next(self.sampler.__iter__())
        #     x, y = self.dataloader.dataset[idx]
        #     x_list.append(x)
        #     y_list.append(y)
        
        # self.count += 1
        # # if self.count % self.sampler.batch_size == 0 and idx>0:
        # #     self.sampler.data_mover.updatecache((idx-1)//self.sampler.batch_size)
        # yield torch.stack(x_list, dim=0), torch.tensor(y_list, dtype=torch.long).reshape(self.batch_size)
        return self.dataloader._get_iterator()

    def dump_data_read_freq(self, output_file_path:str):
        self.sampler.dump_data_read_freq(output_file_path=output_file_path)
