from PIL import Image
import numpy as np
import random
import os
import torch
from tensorflow.io import gfile
from torch.utils.data import  Dataset
from .dataset_utils import InfiniteDataLoader, _list_image_files_recursively, random_crop_arr, center_crop_arr
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

def create_dataset(dargs):
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
        from .jax.jax_datasets import NumpyLoader, numpy_loader_to_jax_dataset #lazy import so we don't depend on JAX
        print("building numpy loader...")
        loader = NumpyLoader(
            dataset, batch_size=dargs.batch_size, num_workers=1, drop_last=True
        )
        loader = numpy_loader_to_jax_dataset(loader, dargs.batch_size)
    else:
        loader = InfiniteDataLoader(
            dataset, batch_size=dargs.batch_size, num_workers=1, drop_last=True
        )
    
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
        if dargs.is_labeled:
            # Assume classes are the first part of the filename,
            # before an underscore.
            self.local_classes = [os.path.basename(path).split("_")[0] for path in self.local_images]
        
    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        dargs = self.dargs

        path = self.local_images[idx]
        with gfile.GFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        img = center_crop_arr(pil_image, dargs.native_res)

        if dargs.flip and random.random() < 0.5:
            img = img[:, ::-1]

        img = img.astype(np.float32) / 127.5 - 1
        
        out_dict = {
            "x": np.transpose(img, [2, 0, 1])
        }
        
        if dargs.is_labeled:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            
        if dargs.lower_res is not None:
            img_lr = np.array(
                Image.fromarray(img).resize(dargs.lower_res)
            )
            img_lr += np.random.normal(size=img_lr.shape) * dargs.sigma_aug
            
            out_dict["x_lr"] = np.transpose(img_lr, [2, 0, 1])

        return out_dict