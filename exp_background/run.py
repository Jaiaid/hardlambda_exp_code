import os
import json
import time
import sys
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torchvision import datasets, transforms
# from time import time
import os
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from datetime import timedelta
import logging

#possible improvement: add args for momentum, lr, batch_size, dataset?
PYTORCH_DIR = './'
log_interval = 100  # multiple of the batch size (e.g. will print an update every batch_size * log_interval)
batch_size = 100
momentum = 0.25
learning_rate = 0.00001
cuda = None


def preprocess_cifar(image_size: 32, world_size, rank):

    # transform original images to image_size^2 size
    if image_size == 32:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])
    else:
        transform = transforms.Compose(
            [transforms.Resize(image_size),
             transforms.ToTensor(),
             transforms.Normalize((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    # download the dataset if needed
    if not os.path.exists("./cifar_data"):
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                               download=True, transform=transform)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./cifar_data', train=True,
                                                download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./cifar_data', train=False,
                                               download=False, transform=transform)

    # the test set will be fed to a simple data loader
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)

    # for the train set, we create a sampler for each worker with a non-overlapping split of the whole dataset
    # we then pass the worker specific sampler to the trainloader
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=False, sampler=sampler, pin_memory=False, num_workers=0)

    return trainloader, testloader

def save_checkpoint(log_dir, model, optimizer, epoch):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  state = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
  }
  torch.save(state, filepath)


def load_checkpoint(log_dir, epoch=1):
  filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
  return torch.load(filepath)


def launchTraining(modelName, trainloader, PATH, num_epochs, testloader, rank, world_size, single_node_log_dir, learning_rate, ip, port, verbose=False):
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        model = getattr(torchvision.models, modelName)(pretrained=True)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model = getattr(torchvision.models, modelName)(pretrained=True)

    dist.init_process_group("gloo")
    ddp_model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    if rank == 0:
        print("Training Started")
        toc = time.time()

    for epoch in range(1, num_epochs + 1):

        running_loss, total, correct = 0.0, 1.0, 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(trainloader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            # per-batch training accuracy calculation
            running_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total += target.size(0)
            correct += torch.sum(predicted == target).item()
            accuracy = 100. * correct / total
            if batch_idx % log_interval == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(trainloader) * len(data),
                           100. * batch_idx / len(trainloader), loss.item(), accuracy))

        # saves a checkpoint in the rank 0 node
        if rank == 0:
            save_checkpoint(single_node_log_dir, model, optimizer, num_epochs)
    try:
        dist.destroy_process_group()
    except:
        None

def launchTrainings(allModels, trainloader, PATHS, epochs, testloader):
    for PATH, modelName in zip(PATHS, allModels):
        launchTraining(modelName, trainloader, PATH, epochs, testloader)

def create_log_dir():
    log_dir = os.path.join(PYTORCH_DIR, str(time.time()), 'log')
    os.makedirs(log_dir)
    return log_dir

def main():
    # cuda = torch.device('cuda')

    # creates experiment log directory
    single_node_log_dir = "./"

    all_models = ["resnet18", "alexnet", "vgg16"]

    # parses arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default="squeezenet1_0", required=False, choices=all_models)
    parser.add_argument("--epochs", type=int, help="number of epochs", default=1, required=False)
    parser.add_argument("--length", type=int, help="image length (image size = length^2)", default=32, required=False)
    parser.add_argument("--rank", type=int, help="process rank", default=0, required=False)
    # parser.add_argument("--size", type=int, help="world size", default=2, required=False)
    # parser.add_argument("--ip", type=str, help="master ip", default="10.188.75.1", required=False)
    # parser.add_argument("--port", type=int, help="master port", default=29500, required=False)
    # parser.add_argument("--iface", type=str, help="network device name", default="ens3", required=False)
    args = parser.parse_args()
    rank = args.rank
    os.environ["RANK"] = str(rank)
    # world_size = args.size
    world_size = 1
    os.environ["WORLD_SIZE"] = 1 #str(world_size)
    # ip = args.ip
    ip = "127.0.0.1"
    os.environ["MASTER_ADDR"] = str(ip)
    # port = args.port
    port = 12355
    os.environ["MASTER_PORT"] = str(port)
    iface = args.iface
    os.environ["GLOO_SOCKET_IFNAME"] = str(iface)
    model = args.model
    epochs = args.epochs
    length = args.length

    trainloader, testloader = preprocess_cifar(length, world_size, rank)

    if model == "all":
        PATHS = ["./checkpoints/" + model + "_cifar_checkpoint.pth" for model in all_models]
        launchTrainings(all_models, trainloader, PATHS, epochs, testloader)
    else:
        PATH = "./checkpoints/" + model + "_cifar_checkpoint.pth"
        launchTraining(model, trainloader, PATH, epochs, testloader, rank, world_size, single_node_log_dir, learning_rate, ip, port)

if __name__ == '__main__':
    main()