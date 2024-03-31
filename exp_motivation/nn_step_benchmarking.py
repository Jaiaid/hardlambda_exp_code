import copy
import random
import torch
import torchvision
import time
import math

from tqdm import tqdm
from nvitop import Device, ResourceMetricCollector


NETWORKS =  ["resnet50"]
BATCH_SIZES = [2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
LEARNING_RATE = 0.001
NUMBER_OF_CLASSES = 1000
IMAGECOUNT_PER_RUN = 2048
ITERATION_COUNT_PER_RUN = 10000000

gpu_utilization_dict = {}
gpu_utilization_list = []

# resource status collection
collector = ResourceMetricCollector(Device.cuda.all())


# copied from https://raw.githubusercontent.com/pytorch/examples/main/imagenet/main.py
model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))

# will do full update through standard pytorch optim module
def training_all_param_update(nn_model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, batch_size: int):
    global gpu_utilization_list
    # emptying the list before starting
    gpu_utilization_list = []
    # empty the cuda cache done by torch
    torch.cuda.empty_cache()
    # for collection sanity test
    data_collection_count = 0

    # start the resource collector with tag
    # We will collect one sample per inference iteration
    # two sample per 
    collector.start(tag="test")
    # NOT DOING DAEMONIZING
    # daemonize it, this returns a threading.thread object
    # will GIL cause issue? not sure yet
    # IT IS AN ISSUE MOST PROBABLY, daemon thread not getting invoked for given interval
    # daemon = collector.daemonize(on_collect, interval=0.1, on_stop=None)

    # move to GPU
    nn_model.cuda()
    # we are doing iteration steps benchmarking so no need to train more than one epoch

    # this is for only inference forward
    nn_model.eval()
    evalfw_time_count = 0
    eval_iter_count = 0 
    infer_data_load_time = 0
    data_load_start_time = time.time()
    for input_data, label in tqdm(dataloader, leave=False):
        try:
            one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
            input_data = input_data.cuda()
            infer_data_load_time += time.time() - data_load_start_time
            t = time.time()
            fw = nn_model(input_data)
            evalfw_time_count += time.time() - t
            t = time.time()
            # collect metrics
            # collecting after measuring the time to not affect time measurement due to metric collection overhead
            # As we are collecting mean, we should still see the effect from last sampling 
            metrics = collector.collect()
            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for inference with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            evalfw_time_count = math.nan
            eval_iter_count = 1
            break
        eval_iter_count += 1
        if eval_iter_count > IMAGECOUNT_PER_RUN/batch_size or eval_iter_count > ITERATION_COUNT_PER_RUN:
            break

        data_load_start_time = time.time()

    # to reinit every thing before training benchmarking
    # clear collector status
    collector.clear()
    collector.stop(tag="test")
    # move to CPU
    nn_model.cpu()
    # empty the cuda cache done by torch
    torch.cuda.empty_cache()
    # start collector
    collector.start(tag="test")
    # move to GPU
    nn_model.cuda()
    # init loss and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(nn_model.parameters(), lr=LEARNING_RATE)

    # this is for loss calculated forward backward
    nn_model.train()
    iter_count = 0
    fw_time_count = 0
    bw_time_count = 0
    train_data_load_time = 0
    data_load_start_time = time.time()
    for input_data, label in tqdm(dataloader, leave=False):
        # zero the optimizer grad
        optimizer.zero_grad()
        # calculate loss of model
        try:
            one_hot = torch.nn.functional.one_hot(label, num_classes=1000).float().cuda()
            input_data = input_data.cuda()
            train_data_load_time += time.time() - data_load_start_time 
            t = time.time()
            fw = nn_model(input_data)
            fw_time_count += time.time() - t
            # collect metrics
            metrics = collector.collect()
            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for forward pass and loss calc with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            fw_time_count = math.nan
            bw_time_count = math.nan
            iter_count = 1
            break

        # backprop
        try:
            t = time.time()
            loss = loss_func(fw, one_hot)
            loss.backward()
            # optimization step
            optimizer.step()
            # append the loss
            loss_val = loss.data.cpu().item()
            # per iteration loss record
            bw_time_count += time.time() - t
            # collect metrics
            metrics = collector.collect()
            gpu_utilization_list.append(
                [
                    metrics["test/timestamp"],
                    metrics["test/host/cpu_percent (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/gpu_utilization (%)/mean"],
                    metrics["test/host/cpu_percent (%)/mean"],
                    metrics["test/cuda:0 (gpu:0)/memory_used (MiB)/mean"]
                ]
            )
            data_collection_count += 1
        except torch.cuda.OutOfMemoryError as e:
            print(
                "batch size {0} for imagenet 3x224x224 is not suitable for backward pass and update with GPU memory {1}GB".format(
                    batch_size, torch.cuda.mem_get_info()[1]>>30
                )
            )
            bw_time_count = math.nan
            iter_count = 1
            break

        iter_count += 1
        if iter_count > IMAGECOUNT_PER_RUN/batch_size or eval_iter_count > ITERATION_COUNT_PER_RUN:
            break
        data_load_start_time = time.time()
        # per epoch loss record
        # print("average forward time {0}s with batch size {1}".format(fw_time_count/iter_count, batch_size))
        # print("average loss calc time {0}s with batch size {1}".format(loss_time_count/iter_count, batch_size))
        # print("average backprop time {0}s with batch size {1}".format(bw_time_count/iter_count, batch_size))
        # print("average optimizer step time {0}s with batch size {1}".format(optstep_time_count/iter_count, batch_size))

    # clear collector status
    collector.clear()
    collector.stop(tag="test")

    del nn_model
    
    # print("\n\n", data_collection_count, "\n\n")
    return evalfw_time_count/eval_iter_count, evalfw_time_count/(eval_iter_count*batch_size), fw_time_count/iter_count, fw_time_count/(iter_count*batch_size), bw_time_count/iter_count, bw_time_count/(iter_count*batch_size), infer_data_load_time/(iter_count*batch_size), train_data_load_time/(iter_count*batch_size)


if __name__ == "__main__":
    for network in NETWORKS:
        benchmark_dict = {}

        for batch_size in BATCH_SIZES:
            random.seed(3400)
            data_dict_per_iteration = {}
            data_dict_per_epoch = {}
            # load dataset and data loader for cifar10
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),               # Convert images to PyTorch tensors
                torchvision.transforms.Resize([224,224]),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to mean 0 and standard deviation 1
            ])
            dataset = torchvision.datasets.CIFAR10(root=".", train=True, transform=transform, download=True)
            dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)
            # model
            nn_model = torchvision.models.__dict__[network]()
            # nn_model.fc = torch.nn.Linear(512, NUMBER_OF_CLASSES)
            
            evalfw, evalfw_per_samp, fw, fw_per_samp, bw, bw_per_samp, inferdataload_per_sample, traindata_load_per_sample  = training_all_param_update(
                nn_model=copy.deepcopy(nn_model), dataloader=dataloader, batch_size=batch_size
            )

            benchmark_dict[batch_size] = [evalfw, evalfw_per_samp, fw, fw_per_samp, bw, bw_per_samp, inferdataload_per_sample, traindata_load_per_sample]
            gpu_utilization_dict[batch_size] = copy.deepcopy(gpu_utilization_list)

        with open("benchmark_{0}_nn_step.csv".format(network), "w") as fout:
            fout.write("Batch Size\tInference\tInference Per Sample\tForward\tForward Per Sample\tBackward\tBackward Per Sample\tInfer Data Load\tTrain data load\n")
            for batch_size in BATCH_SIZES:
                data_list = benchmark_dict[batch_size]
                fout.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\n".format(
                        batch_size, data_list[0], data_list[1], data_list[2], data_list[3], data_list[4], data_list[5], data_list[6], data_list[7]
                    )
                )

        # with open("benchmark_{0}_gpu_utilization.csv".format(network), "w") as fout:
        #     for batch_size in BATCH_SIZES:
        #         fout.write("{0}\t{0}\t{0}\t".format(batch_size))
        #     fout.write("\n")
        #     for batch_size in BATCH_SIZES:
        #         fout.write("cpu usage(%)\tgpu usage(%)\tgpu memory usage (MiB)\t".format(batch_size))
        #     fout.write("\n")
            
        #     maxlen = max([len(gpu_utilization_dict[batch_size]) for batch_size in gpu_utilization_dict])
        #     for i in range(maxlen):
        #         for batch_size in BATCH_SIZES:
        #             if batch_size in gpu_utilization_dict:
        #                 if i < len(gpu_utilization_dict[batch_size]):
        #                     data_list = gpu_utilization_dict[batch_size][i]
        #                     fout.write("{0}\t{1}\t{2}\t".format(data_list[1], data_list[2], data_list[3]))
        #                 else:
        #                     fout.write("{0}\t{1}\t{2}\t".format(math.nan, math.nan, math.nan))
        #         fout.write("\n")
