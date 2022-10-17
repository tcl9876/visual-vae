import math
import random

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

import os
from tensorflow.io import gfile
# Code credit for data loading: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py,
# also https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html for the NumpyLoader class
# we don't use mpi, instead we use huggingface accelerate. we also remove the dependency on blobfile module, instead use tf

import torch
import jax
import flax
def load_and_shard_numpy_batch(xs, global_batch_size):
    local_device_count = jax.local_device_count()
    def _prepare(x):
        return x.reshape((local_device_count, global_batch_size // local_device_count) + x.shape[1:])
    
    out = jax.tree_map(_prepare, xs)
    return out

def numpy_loader_to_jax_dataset(dataset, batch_size):
    dataset = map(lambda x: load_and_shard_numpy_batch(x, batch_size), dataset)
    dataset = flax.jax_utils.prefetch_to_device(dataset, 1) #one is probably okay? info here: https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html says 
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

class NumpyLoader(DataLoader):
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

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(gfile.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif gfile.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]