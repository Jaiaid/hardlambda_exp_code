import math
from typing import TypeVar, Optional, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)

from torch.utils.data.distributed import DistributedSampler

class CustomDistributedSampler(DistributedSampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size: int = 16) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.data_batch_read_latency = [0] * int(self.num_samples*num_replicas/self.batch_size + 1)
        self.data_batch_read_freq = [0] * int(self.num_samples*num_replicas/self.batch_size + 1)

    def __iter__(self) -> Iterator[T_co]:
        print(self.num_samples)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # record the data's access frequency
        # still not read but we are passing the indices so it should be read
        for index in indices:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1

        return iter(indices)

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

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # record the data's access frequency
        # still not read but we are passing the indices so it should be read
        for index in indices:
            batch_no = int(index/self.batch_size)
            self.data_batch_read_freq[batch_no] += 1

        return iter(indices)

    def dump_data_read_freq(self, output_file_path):
        r"""dump the data access freuqncy in text to given output file path
        
        Args:
            output_file_path: output file where the data will be dumped
        """
        with open(output_file_path, "w") as fout:
            fout.write("batch,frequency\n")
            for batch_no, read_freq in enumerate(self.data_batch_read_freq):
                fout.write(str(batch_no) + "," + str(read_freq) + "\n")
