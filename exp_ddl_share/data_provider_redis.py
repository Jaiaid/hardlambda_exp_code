import time
import psutil
import redis
import subprocess
import os
import torch
import torchvision
import argparse
# from torch.utils.data import Dataset, DataLoader

class SharedDataRedis():
    def __init__(self, port, dataset, dataroot):
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
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
            ])

            self.train_dataset = torchvision.datasets.ImageNet(root=dataroot, train=True, transform=transform, download=True)

        # prepare redis client
        redis_host = 'localhost'  # Change this to your Redis server's host
        redis_port = port  # Change this to your Redis server's port
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0)
        
        for i, data in enumerate(self.train_dataset):
            if dataset == "imagenet" and i > 50000:
                break
            # Serialize the tensor to binary
            input, label= data
            serialized_input_tensor = input.numpy().tobytes()
            redis_key = str(i) 
            self.redis_client.set("data" + redis_key, serialized_input_tensor)
            self.redis_client.set("label" + redis_key, label.to_bytes(4, 'little'))
        # store length
        if dataset == "imagenet":
            self.redis_client.set("length", int(50000).to_bytes(4, 'little'))
        else:
            self.redis_client.set("length", len(self.train_dataset).to_bytes(4, 'little'))


if __name__=='__main__':
    redis_server_cmd = "redis-server"  # Use the actual path if not in PATH
    redis_server_options = "redis.conf"  # Replace with your Redis configuration file if needed

    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf_file", type=str, help="configuration file path", default=redis_server_options)
    parser.add_argument("-p", "--port", type=str, help="server port", required=True)
    parser.add_argument("-data", "--dataset", , choices=["cifar10", "cifar100", "imagenet"], help="which dataset to use", default="cifar10")
    parser.add_argument("-root", "--dataset-root", type=str, help="where dataset is", default="./data")
    args = parser.parse_args()

    try:
        # Start the Redis server as a subprocess
        # redis_server_process = subprocess.Popen([redis_server_cmd, args.conf_file]
        # , stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)

        # Wait for Redis to start (adjust the sleep duration as needed)
        time.sleep(2)
        data_pool = SharedDataRedis(port=args.port, dataset=args.dataset, dataroot=args.dataset_root)
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

    