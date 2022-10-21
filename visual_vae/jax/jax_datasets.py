import jax
import flax
import numpy as np
import torch
from ..dataset_utils import InfiniteDataLoader

def load_and_shard_numpy_batch(xs, global_batch_size):
    local_device_count = jax.local_device_count()
    def _prepare(x):
        if len(x.shape) == 4:
            x = np.transpose(x, (0, 2, 3, 1))
        return x.reshape((local_device_count, global_batch_size // local_device_count) + x.shape[1:])
    
    out = jax.tree_map(_prepare, xs)
    return out

def numpy_loader_to_jax_dataset(dataset, batch_size):
    dataset = map(lambda x: load_and_shard_numpy_batch(x, batch_size), dataset)
    dataset = flax.jax_utils.prefetch_to_device(dataset, 1) #one is probably okay? info here: https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html says so
    return dataset
    
def cifar_loader_wrapper(loader):
    for x, y in loader:
        yield {"x": x, "y": y}

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], torch.Tensor):
        return np.stack([np.array(x) for x in batch])
    elif isinstance(batch[0], (tuple,list)):
       transposed = zip(*batch)
       return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

class NumpyLoader(InfiniteDataLoader):
    def __init__(self, dataset, batch_size=1,
                    shuffle=False, sampler=None,
                    batch_sampler=None, num_workers=0,
                    pin_memory=False, drop_last=False,
                    timeout=0, worker_init_fn=None):
        super(self.__class__, self).__init__(dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn)
