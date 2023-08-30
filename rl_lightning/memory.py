
from collections import OrderedDict, deque
import random
from typing import Iterator, List, Tuple, Callable, Any, Dict

import numpy as np
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data.dataset import IterableDataset


class Memory(IterableDataset):
    
    insert_fields = ()
    derived_fields = ()

    def __init__(self, capacity: int, epoch_size: int):
        self.fields = self.insert_fields + self.derived_fields
        super().__init__()
        self._capacity = capacity
        self._epoch_size = epoch_size
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: deque(maxlen=self._capacity) for field in self.fields}

    def clear(self):
        self._size = 0
        self._not_computed = 0
        self._buffer = {field: deque(maxlen=self._capacity) for field in self.fields}        

    def __getitem__(self, key):
        if key in self.fields:
            return self._buffer[key]
        else:
            raise KeyError("Given key is not recorded!")

    def step(self, *data):
        for key, value in zip(self.insert_fields, data):
            self._buffer[key].append(value)
        self._not_computed += 1
        if self._size < self._capacity:
            self._size += 1 

    def reset(self):
        pass

    def _sample_by_indices(self, indices):
        output = []
        for field in self.fields:
            sampled = [self._buffer[field][i] for i in indices]
            stacked = np.stack(sampled, axis=0)
            output.append(stacked)
        return tuple(output)

    def _get_last_n(self, deq, n):
        idxs = range(self._size - n, self._size)
        elements = [deq[i] for i in idxs]
        return np.stack(elements, axis=0)
        
    def compute(self, agent):
        if self._not_computed == 0:
            return None
        episode_args = (self._get_last_n(self._buffer[key], self._not_computed) for key in self.insert_fields)
        episode_results = self.compute_function(agent, *episode_args)
        for key, value in zip(self.derived_fields, episode_results):
            self._buffer[key].extend(value)
        self._not_computed = 0

    def __iter__(self):
        if self._not_computed != 0:
            raise AssertionError("Please call compute function to compute remaining features!")
        if self._size > self._epoch_size:
            indices = random.sample(range(self._size), self._epoch_size)
        else:
            multiple, remainder = self._epoch_size // self._size, self._epoch_size % self._size
            indices = random.sample(range(self._size), remainder) + multiple*list(range(self._size))
            random.shuffle(indices)
        for data in zip(*self._sample_by_indices(indices)):
            yield data

    def compute_function(self, **data):
        raise NotImplementedError
        
    def rollout(self, agent, num_steps: int, **kwargs):
        raise NotImplementedError

