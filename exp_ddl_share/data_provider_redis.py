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
    def __init__(self, port, cachesize, dataset, dataroot, dataoffset):
        if dataset == "cifar10":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
            ])

            self.train_dataset = torchvision.datasets.CIFAR10(root=dataroot, train=True, transform=transform, download=True)
        elif dataset == "cifar100":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
            ])

            self.train_dataset = torchvision.datasets.CIFAR100(root=dataroot, train=True, transform=transform, download=True)
        elif dataset == "imagenet":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
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
            if dataset == "imagenet" and stored_count == cachesize:
                break
            # Serialize the tensor to binary
            input, label= data
            if dataset == "imagenet":
                shape = [1] + list(input.shape)
                input = torch.nn.functional.interpolate(
                    torch.reshape(input, tuple(shape)), size=(224, 224), mode='bilinear', align_corners=False
                )
            serialized_input_tensor = input.numpy().tobytes()
            redis_key = str(stored_count) 
            self.redis_client.set("data" + redis_key, serialized_input_tensor)
            self.redis_client.set("label" + redis_key, label.to_bytes(4, 'little'))
            stored_count += 1
        # store length
        if dataset == "imagenet":
            self.redis_client.set("length", int(cachesize).to_bytes(4, 'little'))
        else:
            self.redis_client.set("length", len(self.train_dataset).to_bytes(4, 'little'))


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
    args = parser.parse_args()

    try:
        # Start the Redis server as a subprocess
        # redis_server_process = subprocess.Popen([redis_server_cmd, args.conf_file]
        # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

        # Wait for Redis to start (adjust the sleep duration as needed)
        time.sleep(2)
        data_pool = SharedDataRedis(port=args.port, dataset=args.dataset, dataroot=args.dataset_root, cachesize=args.cache_size, dataoffset=args.store_offset)
        # sleep idefinitely until keyboard exception
        # print("Memory rss footprint of redis process ", (psutil.Process(redis_server_process.pid).memory_info().rss)>>20, "MiB")
        # print("Memory shared footprint of redis starter process ", (psutil.Process().memory_info().shared)>>20, "MiB")

        print("Press Ctl+C to exit")
        while True:
            # out, err = redis_server_process.communicate()
            # print(out)
            # exit_code = redis_server_process.wait()
            # print("redis server exited with exit code {0}".format(exit_code))
            break
    except Exception as e:
        # kill the process
        # redis_server_process.kill()
        print(e)
        print("exiting")

    