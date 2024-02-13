from typing import TypeVar, Optional, Iterator
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data import Dataset
import time
import math

T_co = TypeVar('T_co', covariant=True)

from torch.utils.data.distributed import DistributedSampler

class DefaultDistributedSampler(DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 16) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.data_batch_read_latency = [0] * int(self.num_samples*num_replicas)
        self.data_batch_read_freq = [0] * int(self.num_samples*num_replicas)

    def __iter__(self) -> Iterator[T_co]:
        iterator = super().__iter__()
        iter_copy = copy.deepcopy(iterator)
        for index in iterator:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1
        return iter_copy

    def set_batch_time(self, batch_no:int, read_time:float):
        pass

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")


class DistAwareDistributedSampler(DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 16) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.data_batch_read_latency = [0] * int(self.num_samples*num_replicas)
        self.data_batch_read_freq = [0] * int(self.num_samples*num_replicas)
        # self.rank_cache = redis.StrictRedis(host=metadata_cache_ip, port=metadata_cache_port, db=0)

        # read the whole dataset and make a ranking
        benchmarking_time_start = time.time()
        time_list = []
        total_read_time = 0
        for i in range(0, len(dataset), batch_size):
            t = time.time()
            # we are only checking how much time to read from memory
            for j in range(i, i+batch_size):
                input = dataset[i]
                total_read_time += time.time() - t
            time_list.append(total_read_time)
            total_read_time = 0
        
        self.batch_dist_ranking_list = list(numpy.argsort(time_list))
        benchmarking_time_end = time.time()
        print("benchmarking took {0}s".format(benchmarking_time_end - benchmarking_time_start))

    def __iter__(self) -> Iterator[T_co]:
        total_batch = len(self.batch_dist_ranking_list)
        num_batch_per_replica = int(self.total_size / (self.num_replicas * self.batch_size))
        start_idx = (self.epoch * num_batch_per_replica) % total_batch
        # we are not modding this for batch generation logic simplification
        end_idx = (start_idx + num_batch_per_replica)

        # create indices array
        indices = []
        for i in range(start_idx, end_idx):
            idx = i % total_batch
            indices += list(
                range(
                    self.batch_dist_ranking_list[idx] * self.batch_size,
                    self.batch_dist_ranking_list[idx] * self.batch_size + self.batch_size
                )
            )

        # record the data's access frequency
        # still not read but we are passing the indices so it should be read
        for index in indices:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1

        return iter(indices)

    def set_batch_time(self, batch_no:int, read_time:float):
        idx = self.rank*self.num_replicas + batch_no

    def get_batch_time(self, batch_no:int) -> float:
        key = self.rank*self.num_replicas + batch_no
        return 0

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")


class GradualDistAwareDistributedSampler(DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 16) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        batch_count = 0
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
        self.data_batch_read_latency = [np.nan] * int(batch_count)
        self.data_batch_read_freq = [0] * int(batch_count)
        self.batch_dist_ranking_list = list(np.argsort(self.data_batch_read_latency))
        
    def set_epoch(self, epoch: int) -> None:
        # sort at the beginning of epoch after first epoch
        if epoch > 0:
            self.batch_dist_ranking_list = list(np.argsort(self.data_batch_read_latency))
        return super().set_epoch(epoch)

    def __iter__(self) -> Iterator[T_co]:
        total_batch = len(self.batch_dist_ranking_list)
        num_batch_per_replica = int(self.total_size / (self.num_replicas * self.batch_size))
        start_idx = (self.epoch * num_batch_per_replica) % total_batch
        # we are not modding this for batch generation logic simplification
        end_idx = (start_idx + num_batch_per_replica)

        if self.epoch >= 1:
            indices = []
            for i in range(start_idx, end_idx):
                idx = i % total_batch
                indices += list(
                    range(
                        self.batch_dist_ranking_list[idx] * self.batch_size,
                        self.batch_dist_ranking_list[idx] * self.batch_size + self.batch_size
                    )
                )
        else:
            # for first batch work like default
            # from the 2nd batch we will have some information on data access latency
            iterator = super().__iter__()
            iter_copy = copy.deepcopy(iterator)
            for index in iterator:
                batch_no = int(index/self.batch_size)
                self.data_batch_read_freq[batch_no] += 1
            return iter_copy


        # record the data's access frequency
        # still not read but we are passing the indices so it should be read
        for index in indices:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1

        return iter(indices)

    def set_batch_time(self, batch_no:int, read_time:float):
        idx = batch_no
        self.data_batch_read_latency[idx] = read_time

    def get_batch_time(self, batch_no:int) -> float:
        idx = batch_no
        return self.data_batch_read_latency[idx]

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")


class GradualDistAwareDistributedSamplerBG():
    r"""Sampler that restricts data loading to a subset of the dataset.

    
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 16) -> None:
        #super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.total_batch = math.ceil(len(dataset)/batch_size)
        batch_count = 0
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
        self.data_batch_read_latency = [np.nan] * int(batch_count)
        self.data_batch_read_freq = [0] * int(batch_count)
        # needed to provide index from appropriate offset
        self.rank = 0
        self.total_size = len(dataset)
        self.num_replicas = 2

    def set_epoch(self, epoch: int) -> None:
        return super().set_epoch(epoch)
    
    def set_rank(self, rank: int) -> None:
        self.rank = rank

    def __iter__(self) -> Iterator[T_co]:
        num_batch_per_replica = int(self.total_size / (self.num_replicas * self.batch_size))
        # we are not modding this for batch generation logic simplification
        end_idx = num_batch_per_replica

        indices = list(range(self.rank * num_batch_per_replica * self.batch_size,
                             self.rank * num_batch_per_replica * self.batch_size + end_idx * self.batch_size))

        for index in indices:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1

        return iter(indices)

    def set_batch_time(self, batch_no:int, read_time:float):
        idx = batch_no
        self.data_batch_read_latency[idx] = read_time

    def get_batch_time(self, batch_no:int) -> float:
        idx = batch_no
        return self.data_batch_read_latency[idx]

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")
