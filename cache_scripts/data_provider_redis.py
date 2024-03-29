import time
import psutil
import redis
import subprocess
import os
import random
import torch
import torchvision
import argparse
# from torch.utils.data import Dataset, DataLoader

SEED = 3400

class SharedDataRedis():
    def __init__(self, port, cachesize, dataset, dataroot, dataoffset, imgsize):
        # the normalization mean and std for cifar10,100 got from
        # https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb
        if dataset == "cifar10":
            if imgsize == 32:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],)  # Normalize to mean 0 and standard deviation 1
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(imgsize),
                    torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],)  # Normalize to mean 0 and standard deviation 1
                ])

            self.train_dataset = torchvision.datasets.CIFAR10(root=".", train=True, transform=transform, download=True)
        elif dataset == "cifar100":
            if imgsize == 32:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],)  # Normalize to mean 0 and standard deviation 1
                ])
            else:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(imgsize),
                    torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],)  # Normalize to mean 0 and standard deviation 1
                ])

            self.train_dataset = torchvision.datasets.CIFAR100(root=".", train=True, transform=transform, download=True)
        elif dataset == "imagenet":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(imgsize + int(imgsize*0.125)),
                torchvision.transforms.CenterCrop(imgsize),
                torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.train_dataset = torchvision.datasets.ImageNet(root=dataroot, transform=transform)

        # prepare redis client
        redis_host = 'localhost'  # Change this to your Redis server's host
        redis_port = port  # Change this to your Redis server's port
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)

        stored_count = 0
        for i, data in enumerate(self.train_dataset):
            if i < dataoffset:
                continue
            if stored_count == cachesize:
                break
            # Serialize the tensor to binary
            input, label= data
            serialized_input_tensor = input.numpy().tobytes()
            redis_key = str(stored_count) 
            self.redis_client.set("data" + redis_key, serialized_input_tensor)
            self.redis_client.set("label" + redis_key, label.to_bytes(4, 'little'))
            stored_count += 1
        # store length
        self.redis_client.set("length", min(int(cachesize), len(self.train_dataset)).to_bytes(4, 'little'))


if __name__=='__main__':
    redis_server_cmd = "redis-server"  # Use the actual path if not in PATH
    redis_server_options = "redis.conf"  # Replace with your Redis configuration file if needed

    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf_file", type=str, help="configuration file path", default=redis_server_options)
    parser.add_argument("-p", "--port", type=str, help="server port", required=True)
    parser.add_argument("-data", "--dataset", choices=["cifar10", "cifar100", "imagenet"], help="which dataset to use", default="cifar10")
    parser.add_argument("-root", "--dataset-root", type=str, help="where dataset is", default="./data")
    parser.add_argument("-offset", "--store-offset", type=int, help="from which offset image will be stored", required=True)
    parser.add_argument("-size", "--cache-size", type=int, help="number of data samples", required=True)
    parser.add_argument("-dim", "--image-dimension", type=int, help="dimension of image", required=False, default=224)
    args = parser.parse_args()


    time.sleep(2)
    data_pool = SharedDataRedis(port=args.port, dataset=args.dataset, dataroot=args.dataset_root, cachesize=args.cache_size, dataoffset=args.store_offset, imgsize=args.image_dimension)
