import copy
import numpy as np
import torch.distributed as dist
import time
import math
import subprocess
import os

from typing import TypeVar, Optional, Iterator
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from DataMovementService import DataMoverServiceInterfaceClient


T_co = TypeVar('T_co', covariant=True)

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


class GradualDistAwareDistributedSamplerBG():
    r"""Sampler that restricts data loading to a subset of the dataset.

    
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, batch_size: int = 16,
                 ip_mover=None, port_mover=None) -> None:
        self.num_replicas = num_replicas
        self.batch_size = batch_size
        self.dataset = dataset
        self.total_size = len(dataset)
        self.total_batch = math.ceil(len(dataset)/batch_size)
        batch_count = 0
        for i in range(0, len(dataset), batch_size):
            batch_count += 1
        self.data_batch_read_latency = [np.nan] * int(batch_count)
        self.data_batch_read_freq = [0] * int(batch_count)
        # needed to provide index from appropriate offset
        self.rank = rank
        self.epoch = 0

        # starting the background data mover service
        self.data_mover_service = subprocess.Popen(
            """python3 {2}/DataMovementService.py --seqno {0}
            -bs 16 -cn 10.21.12.239 26379 10.21.12.239 26380 10.21.12.222 26379 -pn 10.21.12.239 10.21.12.222 -p {1}""".format(
                rank if rank < 3 else 2, port_mover, os.path.dirname(os.path.abspath(__file__))).split()
        )
        # check if running
        if self.data_mover_service.poll() is None:
            print("data mover service is running")

        # try 10 times to connect
        connection_refused_count = 0
        while connection_refused_count < 10: 
            try:
                self.data_mover = DataMoverServiceInterfaceClient(ip_mover, port_mover)
                break
            except ConnectionError as e:
                connection_refused_count += 1
                print("connection establish attempt {0} failed".format(connection_refused_count))
                # sleep for a second
                time.sleep(1)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
 
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
#            self.data_batch_read_freq[batch_no] += 1
        
        return iter(indices)

    def set_batch_time(self, batch_no:int, read_time:float):
        idx = batch_no
#        self.data_batch_read_latency[idx] = read_time

    def get_batch_time(self, batch_no:int) -> float:
        idx = batch_no
#        return self.data_batch_read_latency[idx]

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")

    def __exit__(self):
        # cleanup
        try:
            self.data_mover.close()
            self.data_mover_service.kill()
        except Exception as e:
            print(e)

    def __del__(self):
        # cleanup
        try:
            self.data_mover.close()
            self.data_mover_service.kill()
        except Exception as e:
            print(e)
