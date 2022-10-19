from PIL import Image
import numpy as np
import random
import os
import torch
from tensorflow.io import gfile
from torch.utils.data import DataLoader, Dataset
from .dataset_utils import cifar_loader_wrapper, _list_image_files_recursively, random_crop_arr, center_crop_arr, NumpyLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

#see https://github.com/pytorch/pytorch/issues/42402 for PIL slowness.
class NoPIL_CIFAR10(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

#source: https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

def create_dataset(dargs, repeating=True):
    is_cifar = "cifar" in dargs.dataset_name.lower()

    if is_cifar:
        def map_fn(x):
            x = x/127.5 - 1.
            return torch.tensor(x).float().permute(2, 0, 1)
                
        dataset = NoPIL_CIFAR10(dargs.data_dir, download=True, 
            transform=transforms.Lambda(map_fn)
        )
    else:
        dataset = ImageDataset(dargs)

    if dargs.framework.lower() in ['jax', 'flax']:
        print("building numpy loader...")
        loader = NumpyLoader(
            dataset, batch_size=dargs.batch_size, num_workers=1, drop_last=True
        )
    else:
        loader = InfiniteDataLoader(
            dataset, batch_size=dargs.batch_size, num_workers=1, drop_last=True
        )
    
    #maybe bring this back for JAX version? accelerate doesnt like it since it converts to a generator
    #if is_cifar:
    #    loader = cifar_loader_wrapper(loader)
    
    if dargs.framework.lower() in ['jax', 'flax']:
        from .dataset_utils import numpy_loader_to_jax_dataset #lazy import so we don't depend on JAX
        loader = numpy_loader_to_jax_dataset(loader, dargs.batch_size)
    
    return loader

#Code credit for this class: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py
class ImageDataset(Dataset):
    def __init__(self, dargs):
        super().__init__()
        self.dargs = dargs

        if not dargs.data_dir:
            raise ValueError("unspecified data directory")
        self.local_images = _list_image_files_recursively(dargs.data_dir)
        self.local_classes = None
        if dargs.class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [os.path.basename(path).split("_")[0] for path in self.local_images]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            self.local_classes = [sorted_classes[x] for x in class_names]
        
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        dargs = self.dargs

        path = self.local_images[idx]
        with gfile.GFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        if self.random_crop:
            img = random_crop_arr(pil_image, dargs.resolution)
        else:
            img = center_crop_arr(pil_image, dargs.resolution)

        if dargs.random_flip and random.random() < 0.5:
            img = img[:, ::-1]

        img = img.astype(np.float32) / 127.5 - 1
        
        if dargs.framework.lower() not in ['jax', 'flax']:
            img = np.transpose(img, [2, 0, 1])

        out_dict = {
            "x": img
        }
        if dargs.num_classes is not None:
            out_dict["y"] = np.array(dargs.num_classes[idx], dtype=np.int64)
        
        if img_lr is not None:
            img_lr = np.array(
                Image.fromarray(img).resize(dargs.lower_res)
            )
            img_lr += np.random.normal(size=img_lr.shape) * dargs.sigma_aug

            if dargs.framework.lower() not in ['jax', 'flax']:
                img_lr = np.transpose(img_lr, [2, 0, 1])
            
            out_dict["x_lr"] = img_lr

        return out_dict