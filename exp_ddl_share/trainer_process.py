import os
import psutil
import subprocess
import time
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
    t_beg = time.time()
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
        data_sampler.set_rank(rank=rank)
        # starting the background data mover service
        data_mover_service = subprocess.Popen(
            """python3 {2}/DataMovementService.py --seqno {0}
            -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p {1}""".format(
                rank if rank < 3 else 2, args.port_mover, os.path.dirname(os.path.abspath(__file__))).split()
        )
        # check if running
        if data_mover_service.poll() is None:
            print("data mover service is running")
    else:
        data_sampler = DefaultDistributedSampler(
            dataset=dataset, num_replicas=num_replicas)

    # create the pipeline from sampler
    dataset_pipeline = DatasetPipeline(dataset=dataset, batch_size=batch_size,
                                       sampler=data_sampler, num_replicas=num_replicas)
    # to keep track of which data batch reading takes what time
    batch_read_time = [0] * int(dataset.nb_samples/batch_size + 1)

    total_time = 0
    count = 0
    # to clean existing file
    fout = open("latency_data_rank_{0}.csv".format(rank), "w")
    fout.write("epoch,batch,batch size,read time (s)\n")
    fout.close()

    if sampler == "graddistbg":
        # try 10 times to connect
        connection_refused_count = 0
        while connection_refused_count < 10: 
            try:
                data_mover = DataMoverServiceInterfaceClient(args.ip_mover, args.port_mover)
                break
            except ConnectionError as e:
                connection_refused_count += 1
                print("connection establish attempt {0} failed".format(connection_refused_count))
                # sleep for a second
                time.sleep(1)

        if connection_refused_count == 10:
            print("connection failed, exiting...")
            data_mover_service.kill()
            exit(1)
        else:
            print("connection successful after {0} attempt".format(connection_refused_count))
            print("data movement service client interface is opened")

    # if epoch profiling only run one epoch
    if args.epoch_prof:
        batch_read_avg_time = 0
        cache_update_avg_time = 0
        backprop_step_avg_time = 0
        processed_batch_count = 0

    print("initialization took {0}s".format(time.time() - t_beg))
    t_train = time.time()
    for epoch in range(epoch_count):
        print("starting training epoch {0}...".format(epoch))
        dataset_pipeline.set_epoch(epoch)
        print("Memory rss footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().rss)>>20, "MiB")
        print("Memory shared footprint of process ", rank, " at epoch", epoch, " start", (psutil.Process().memory_info().shared)>>20, "MiB")
        try:
            epoch_start_time = time.time()
            for i in range(len(dataset)/(4*batch_size)):
                t = time.time()
                data = next(iter(dataset_pipeline))
                inputs, labels = data
                if args.epoch_prof:
                    batch_read_avg_time += time.time() - t
                # record the batch_read time
                # we are adding with the assumption that all rank's train equal epoch count
                # therefore, summation of time is indicative of delay
                batch_read_time[i] += time.time() - t
                # got the data now issue cache update cmd
                # if data movement in background is running
                # which is the case for "graddistbg" sampler
                if sampler == "graddistbg":    
                    if args.epoch_prof:
                        t2 = time.time()
                    data_mover.updatecache(i)
                    if args.epoch_prof:
                        cache_update_avg_time += time.time() - t2
                # generate label and move to GPU
                if args.epoch_prof:
                    t3 = time.time()
                one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
                if gpu:
                    inputs = inputs.cuda(torch.device('cuda', device_idx))
                    one_hot = one_hot.cuda(torch.device('cuda', device_idx))
                # set the time
                if sampler != "shade":
                    data_sampler.set_batch_time(i, batch_read_time[i])
                # train one iteration
                training_pipeline.run_train_step(inputs=inputs, labels=one_hot)
                if args.epoch_prof:
                    backprop_step_avg_time = time.time() - t3
                    processed_batch_count += 1

                # it seems del does not cause garbage collection 
                # but it reduces reference which should cause garbage collection
                # to my understanding at end of each iteration memory should be reclaimed
                # https://stackoverflow.com/questions/14969739/python-del-statement
                # this is to ensure that we are not keeping data cached
                del inputs
                del labels
                
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
    
    print("training took {0}s".format(time.time() - t_train))
    print("per iteration average time {0}s with iteration count {1}".format(total_time / count, count))

    t_dump = time.time()
    # dump prof data in rank<rank>_<sampler>.timedata
    if args.epoch_prof:
        with open("rank{0}_{1}.timedata".format(rank, sampler), "w") as fin:
            fin.write("batch read avg time: {0}s\n".format(batch_read_avg_time/processed_batch_count))
            fin.write("cache update avg time: {0}s\n".format(cache_update_avg_time/processed_batch_count))
            fin.write("backprop step avg time: {0}s\n".format(backprop_step_avg_time/processed_batch_count))
            fin.write("processed batch count: {0}\n".format(processed_batch_count))
            if sampler != "shade":
                fin.write("redis query count: {0}\n".format(dataset.get_query_stat()))

    if sampler == "graddistbg":
        data_mover.close()
        data_mover_service.kill()

    if sampler != "shade":
        # dump data read freq, only rank 0 will init that
        dataset_pipeline.dump_data_read_freq("freq_data_rank_{0}.csv".format(rank))

    print("data dump took {0}s".format(time.time() - t_dump))