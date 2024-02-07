import bisect
import random
import warnings

from torch._utils import _accumulate
from torch import randperm
# No 'default_generator' in torch/__init__.pyi
from torch import default_generator  # type: ignore
from typing import TypeVar, Generic, Iterable, Iterator, Sequence, List, Optional, Tuple
from torch import Tensor, Generator

import math

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

import os
import time
from datetime import datetime
import argparse
#import torchvision
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import random
#import pandas
import PIL.Image as Image
import numpy as np
import redis
import io
from io import BytesIO
import numpy as np
from torch._utils import ExceptionWrapper
import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

#class ShadeDataset(torch.utils.data.Dataset):
class ShadeDataset(Dataset):

    def __init__(self, transform=None, target_transform=None, cache_data = True,
        wss = 0.1, host_ip = '0.0.0.0', port_num = '6379'):
        self.samples = []
        self.classes = []
        self.transform = transform
        self.target_transform = target_transform
        self.loader = None
        self.cache_data = cache_data
        self.wss = wss
        # we will read from redis memory pool
        # prepare redis client
        redis_host1 = '10.21.12.222'  # Change this to your Redis server's host
        redis_port1 = 26379  # Change this to your Redis server's port
        self.redis_client1 = redis.StrictRedis(host=redis_host1, port=redis_port1, db=0)
        self.nb_samples = int.from_bytes(self.redis_client1.get("length"), 'little')

        redis_host2 = '10.21.12.239'  # Change this to your Redis server's host
        redis_port2 = 26379  # Change this to your Redis server's port
        self.redis_client2 = redis.StrictRedis(host=redis_host2, port=redis_port2, db=0)
        self.nb_samples += int.from_bytes(self.redis_client2.get("length"), 'little')

        redis_host3 = '10.21.12.239'  # Change this to your Redis server's host
        redis_port3 = 26380  # Change this to your Redis server's port
        self.redis_client3 = redis.StrictRedis(host=redis_host3, port=redis_port3, db=0)
        self.nb_samples += int.from_bytes(self.redis_client3.get("length"), 'little')

        self.cache_portion = self.wss * self.nb_samples
        self.cache_portion = int(self.cache_portion // 1)

        if host_ip == '0.0.0.0':
            self.key_id_map = redis.Redis()
        else:
            self.startup_nodes = [{"host": host_ip, "port": port_num}]
            self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

        self.PQ = heapdict.heapdict()
        self.ghost_cache = heapdict.heapdict()
        self.key_counter = 0
        # to keep track who is queried what amount
        self.index_queried_stat = {}
    
    # to get stat on which indexed queried what amount
    def get_index_query_stat(self):
        return self.index_queried_stat
    
    def random_func(self):
        return 0.6858089651836363

    def set_num_local_samples(self,n):
        self.key_counter = n

    def set_PQ(self,curr_PQ):
        self.PQ = curr_PQ

    def set_ghost_cache(self,curr_ghost_cache):
        self.ghost_cache = curr_ghost_cache

    def get_PQ(self):
        return self.PQ

    def get_ghost_cache(self):
        return self.ghost_cache

    def cache_and_evict(self, index):
        if self.cache_data and self.key_id_map.exists(index):
            deser_x = self.key_id_map.get(index)
            x = np.frombuffer(deser_x, dtype=np.float32)
            dim_x = int(math.ceil(math.sqrt(x.shape[0]/3)))
            sample = torch.from_numpy(x.reshape(3,dim_x,dim_x))
        else:
            transformed_index = index
            if index >= 2*self.nb_samples/3:
                transformed_index -= int(2*self.nb_samples/3)
                select_redis_client = self.redis_client3
            elif index >= self.nb_samples/3:
                transformed_index -= int(self.nb_samples/3)
                select_redis_client = self.redis_client2
            else:
                select_redis_client = self.redis_client1

            deser_x = select_redis_client.get("data" + str(transformed_index))
            x = np.frombuffer(deser_x, dtype=np.float32)
            dim_x = int(math.ceil(math.sqrt(x.shape[0]/3)))
            sample = torch.from_numpy(x.reshape(3,dim_x,dim_x))

            if index in self.ghost_cache:
                print('miss %d' %(index))
            keys_cnt = self.key_counter + 50

            if(keys_cnt >= self.cache_portion):
                try:
                    peek_item = self.PQ.peekitem()
                    if self.ghost_cache[index] > peek_item[1]: 
                        evicted_item = self.PQ.popitem()
                        print("Evicting index: %d Weight: %.4f Frequency: %d" %(evicted_item[0], evicted_item[1][0], evicted_item[1][1]))

                        if self.key_id_map.exists(evicted_item[0]):
                            self.key_id_map.delete(evicted_item[0])
                        keys_cnt-=1
                except:
                    print("Could not evict item or PQ was empty.")
                    pass

            if self.cache_data and keys_cnt < self.cache_portion:
                self.key_id_map.set(index, x.tobytes())
                # print("Index: ", index)
            # sample = image.convert('RGB')
        return sample

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """

        if index not in self.index_queried_stat:
            self.index_queried_stat[index] = 0
        self.index_queried_stat[index] += 1

        if index >= 2*self.nb_samples/3:
            index -= int(2*self.nb_samples/3)
            select_redis_client = self.redis_client3
        elif index >= self.nb_samples/3:
            index -= int(self.nb_samples/3)
            select_redis_client = self.redis_client2
        else:
            select_redis_client = self.redis_client1
        
        deser_y = select_redis_client.get("label" + str(index))
        target = int.from_bytes(deser_y, 'little')

        # insertion_time = datetime.now()
        # insertion_time = insertion_time.strftime("%H:%M:%S")
        # print("train_search_index: %d time: %s" %(index, insertion_time))

        sample = self.cache_and_evict(index)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(y)

        return sample, target

    def __len__(self) -> int:
        return self.nb_samples

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

import numpy as np 
from collections import defaultdict
import random
import redis
import heapdict
import PIL
# from rediscluster import RedisCluster
from collections import OrderedDict


T_co = TypeVar('T_co', covariant=True)


class ShadeSampler(Sampler[T_co]):
    r"""ShadeSampler that uses fine-grained rank-based importance and 
    PADS policy to sample data.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.ShadeSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset with repetitive samples.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        PADS (bool): If ``True`` (default), sampler will use PADS policy for selecting indices.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        batch_size (int): The number of sample in a batch. Used for ranking samples.
        replacement (bool): Replacement allows SHADE to have repetitive samples.
        host_ip (str): Redis master node IP address
        port_num (str): Port at which Redis instances are listening
        rep_factor (int/float): factor by which the samples in cache are multiplied to train more 
        on hard-to-learn samples.
        init_fac (int): initialization factor used to set the weights of each sample
        ls_init_fac : importance sampling factor used after each epoch

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, PADS: bool = True, batch_size: int = 64,
                 seed: int = 0, drop_last: bool = False, replacement = True, host_ip = None, port_num = None, rep_factor = 1, init_fac = 1, ls_init_fac = 1e-2 ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:                      
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        if host_ip is None:
            raise RuntimeError("Requires Redis Host Node IP.")
        # if port_num is None:
        # 	raise RuntimeError("Requires Redis Port Number.")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.batch_wts = []

        self.num_samples = len(dataset)
        
        #initialize importance sampling factors
        self.init_fac, self.ls_init_fac = init_fac, ls_init_fac
        #initialize weights of all of the indices
        self.weights = torch.ones(self.num_samples)*init_fac
        #sampling with replacement
        self.replacement = replacement

        #variable for understanding the portion of processed indices.
        self.item_curr_pos = 0
        self.batch_size = batch_size

        #fix the weights of indices in the log scale for rank-based importance.
        for j in range(batch_size):
          self.batch_wts.append(math.log(j+10))
        self.batch_wts = torch.tensor(self.batch_wts)
        self.ls_param = 0

        #starting the cache.
        if host_ip == '0.0.0.0':
            self.key_id_map = redis.Redis()
        else:
            self.startup_nodes = [{"host": host_ip, "port": port_num}]
            self.key_id_map = RedisCluster(startup_nodes=self.startup_nodes)

        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.PADS = PADS
        self.seed = seed

        #initializing parameter to decide between aggressive and relaxed sampling.
        self.curr_val_score = 100
        self.rep_factor = rep_factor

    def __iter__(self) -> Iterator[T_co]:
        if self.PADS:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
            self.idxes = torch.multinomial(self.weights.add(self.ls_param), len(self.dataset), self.replacement)
            self.indices = self.idxes.tolist()
            if self.epoch == 0:
                random.shuffle(self.indices)
            if self.epoch > 0 and self.curr_val_score > 0:
                self.indices = self.pads_sort(self.indices)
        else:
            self.idxes = torch.multinomial(self.weights.add(self.ls_param), len(self.dataset), self.replacement)
            self.indices = self.idxes.tolist()
        
        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(self.indices)
            if padding_size <= len(self.indices):
                self.indices += self.indices[:padding_size]
            else:
                self.indices += (self.indices * math.ceil(padding_size / len(self.indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            self.indices = self.indices[:self.total_size]
        assert len(self.indices) == self.total_size

        # take repetitive samples based on loss/val_accuracy.
        self.indices = self.score_based_rep()

        #increase hit rates through PADS
        cache_hit_list, cache_miss_list, num_miss_samps = self.prepare_hits(self.rep_factor)
        
        #create the indices list for processing in the next epoch
        self.indices = cache_hit_list + cache_miss_list[:num_miss_samps]

        print(f'hit_list_len: {len(cache_hit_list)}')
        print(f'miss_list_len: {len(cache_miss_list[:num_miss_samps])}')
        print(len(self.indices))

        #sanity check
        self.indices = self.indices[:self.num_samples]

        assert len(self.indices) == self.num_samples
        self.indices_for_process = self.indices
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

    def on_epoch_end(self, metrics):
        if not hasattr(self, 'prev_loss'):
            self.prev_loss = metrics
            self.ls_param = self.prev_loss * self.ls_init_fac
        else:
            cur_loss = metrics
            # assume normal learning curve
            ls_fac = np.exp((cur_loss - self.prev_loss) / self.prev_loss)
            self.ls_param = self.ls_param * ls_fac
            self.prev_loss = cur_loss
        self.item_curr_pos = 0

    def set_importance_per_batch(self, wts):
        end = min(self.item_curr_pos + self.batch_size, self.num_samples)
        # self.idxes is the entire set of indices of the dataset that will be processed in the epoch. For example : [3,2,1,5,4,7,6,9,8,12,11,10]
        # Getting the portion of those indices that was processed in the current batch.
        # if batch size = 3 and second batch is getting processed, then indices [3,4,5] will be processed. self.i: end = 3:6
        # Finding the actual indices in the dataset between 3:6 which is [5,4,7]
        new_wts = torch.argsort(wts)
        items = len(wts)
        self.updated_idx_list = self.idxes[self.item_curr_pos+new_wts]
        self.weights[self.updated_idx_list] = self.batch_wts[0:items]
        self.item_curr_pos += self.batch_size
    
    def pass_batch_important_scores(self, raw_score):
        self.set_importance_per_batch(raw_score)
    def get_weights(self):
        return self.weights.add(self.ls_param)
    def get_idxes(self):
        return self.idxes
    def get_indices_for_process(self):
        return self.indices_for_process
    def get_sorted_index_list(self):
        return self.updated_idx_list

    def pass_curr_val_change(self,val_change_avg):
        self.curr_val_score = val_change_avg


    def pads_sort(self,l):
        # l -> list to be sorted
        n = len(l)

        # d is a hashmap
        d = {}
        #d = defaultdict(lambda: 0)
        for i in range(n):
            #d[l[i]] += 1
            d[l[i]] = 1 + d.get(l[i],0)

        # Sorting the list 'l' where key
        # is the function based on which
        # the array is sorted
        # While sorting we want to give
        # first priority to Frequency
        # Then to value of item
        l.sort(key=lambda x: (-d[x], x))

        return l

    def score_based_rep(self):
        if self.epoch > 0 and self.curr_val_score > 0:
            cvs = self.curr_val_score
            ce = self.epoch
            print(f'epoch: {ce}, curr_val_score: {cvs}, so doing aggressive sampling.')
            self.indices = self.indices[:self.num_samples]
            random.shuffle(self.indices)
        else:
            self.indices = self.indices[self.rank:self.total_size:self.num_replicas]

        return self.indices

    def prepare_hits(self,r):
        hit_list = []
        miss_list = []
        for ind in self.indices:
            if self.key_id_map.exists(ind):
                hit_list.append(ind)
            else:
                miss_list.append(ind)

        # print(f'hit_list_len: {len(hit_list)}')
        # print(f'miss_list_len: {len(miss_list)}')

        # if rep_factor is a multiple of 0.5
        if r % 1 != 0:
            r = r - 0.5
            r = int(r)
            hit_samps = len(hit_list) * r + len(hit_list)//2
            miss_samps = len(self.indices) - hit_samps

            print(f'hit_samps: {hit_samps}')
            print(f'miss_samps: {miss_samps}')

            art_hit_list = hit_list*r + hit_list[:len(hit_list)//2]
            art_miss_list = miss_list

            random.shuffle(art_hit_list)
            random.shuffle(art_miss_list)
        else:
            r = int(r)
            hit_samps = len(hit_list) * r
            miss_samps = len(self.indices) - hit_samps

            print(f'hit_samps: {hit_samps}')
            print(f'miss_samps: {miss_samps}')

            art_hit_list = hit_list*r 
            art_miss_list = miss_list

            random.shuffle(art_hit_list)
            random.shuffle(art_miss_list)

        return art_hit_list,art_miss_list,miss_samps
