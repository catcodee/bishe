
from collections import defaultdict
import torch
import numpy as np
from torch.utils.data import Sampler

class RandomIdentitySampler(Sampler):
    """Modified from https://github.com/Cysu/open-reid/blob/master/reid/utils/data/sampler.py"""
    def __init__(self, data_source, k=1):
        self.data_source = data_source
        self.k = k
        self.index_dic = defaultdict(list)
        for index in range(len(data_source)):
            self.index_dic[self.data_source.loc[index]['pid']].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_pids = len(self.pids)

    def __len__(self):
        return self.num_pids * self.k

    def __iter__(self):
        indices = torch.randperm(self.num_pids)
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.k:
                t = np.random.choice(t, size=self.k, replace=False)
            else:
                t = np.random.choice(t, size=self.k, replace=True)
            ret.extend(t)
        return iter(ret)

class RandomIdentityBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, glob_var, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.glob_var = glob_var
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.glob_var.src_batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.glob_var.src_batch_size
        else:
            return (len(self.sampler) + self.glob_var.src_batch_size - 1) // self.glob_var.src_batch_size
        
class RandomTrackletBatchSampler(Sampler):
    def __init__(self,dataset,glob_var):

        self.dataset = dataset 
        self.glob_var = glob_var
        self.index_dict = defaultdict(list)
        self.tracklets = list(set(self.dataset['tracklet_idx']))
        self.num_tracklets = len(self.tracklets)

    def __len__(self):
        return self.num_tracklets // self.glob_var.src_batch_size
        

    def __iter__(self):
        self.indices = torch.randperm(len(self.tracklets)).numpy().tolist()
        self.index = 0
        return self

    def __next__(self):
        
        k = self.index + self.glob_var.tgt_batch_size
        if k >= self.num_tracklets:
            raise StopIteration
        ret = self.indices[self.index:k]
        self.index = k
        return ret
        
