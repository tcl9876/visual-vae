import numpy as np
import torch
from tensorflow.io import gfile
import os
from accelerate import Accelerator

from absl import app, flags
from ml_collections.config_flags import config_flags

from visual_vae.torch.torch_utils import restore_checkpoint
from visual_vae.general_utils import denormalize, save_images
from visual_vae.torch.model import VAE

args = flags.FLAGS
config_flags.DEFINE_config_file("config", "./config/cifar10.py", "the location of the config path for the model. e.g. ./config/cifar10.py.")
config_flags.DEFINE_config_file("superres_config", None, "the location of the config path for a super-resolution model. e.g. ./config/imagenet64.py. ")
flags.DEFINE_string("save_dir", None, "the global directory you will save your results into.")
flags.DEFINE_string("checkpoint_path", None, "the path you will restore the model checkpoint from.")
flags.DEFINE_string("superres_checkpoint_path", None, "the path(s) you will restore the super-resolution model checkpoint from.")
flags.DEFINE_integer("n_samples", 36, "the number of samples you want to create.")
flags.DEFINE_integer("nrow", 6, "if you are making a grid, the number of columns in the grid. By default, we use 6 columns.")
flags.DEFINE_integer("max_batch_size", 64, "the maximum allowable batch size for sampling.")
flags.DEFINE_float("mean_scale", 2.0, "the guidance weight for classifier-free guidance of the mean.")
flags.DEFINE_float("var_scale", 4.0, "the guidance weight for classifier-free guidance of the variance.")
flags.DEFINE_integer("label", -1, "If the model is class conditional, generates images from a certain class. Set to -1 for randomly chosen classes. Set to the number of classes in your dataset (e.g. 1000 for imagenet) for unconditional sampling.")
flags.DEFINE_string("save_format", "grid", "either 'grid' or 'npz'. determines whether to save results as a grid of images (default, best for <= 100 images), or as an .npz file (for evaluation).")
flags.mark_flags_as_required(["config", "checkpoint_path", "save_dir"])


def main(_):

    if not gfile.isdir(args.save_dir):
        gfile.makedirs(args.save_dir)

    configs = [args.config]
    if args.superres_config:
        assert args.superres_checkpoint_path is not None, "You must provide --superres_checkpoint_path if you want to perform super-resolution."
        configs += [args.superres_config]
        
    ckpt_paths = [args.checkpoint_path]
    if args.superres_checkpoint_path:
        assert args.superres_checkpoint_path is not None, "You must provide --superres_config if you are trying to load the super-resolution model."
        ckpt_paths += [args.superres_checkpoint_path]

    accelerator = Accelerator()
    models = []
    for config, ckpt_path in zip(configs, ckpt_paths):
        margs = config.model
        model = VAE(**margs)
        restore_checkpoint(model, ckpt_path=ckpt_path)
        model = accelerator.prepare(model)
        models.append(model)

    res = configs[-1].model.resolution
    samples = np.zeros([0, res, res, 3]).astype('uint8')
    
    
    for n in range(0, args.n_samples, args.max_batch_size):
        batch_size = min(args.max_batch_size, args.n_samples - n)
        
        device = next(model.parameters()).device
        if not models[0].num_classes:
            batch_label = None
        elif args.label == -1:
            batch_label = torch.randint(size=[batch_size], high=configs[0].model.num_classes, device=device)
        else:
            batch_label = torch.tensor([args.label] * batch_size, device=device).int()
        
        if args.label == model.num_classes:
            #if we're using unconditional sampling, so no need to do guidance.
            args.mean_scale, args.var_scale = 0, 0

        current_images = models[0].p_sample(num=batch_size, label=batch_label, img_lr=None, mweight=args.mean_scale, vweight=args.var_scale)
        
        if len(models) == 2:
            current_images = models[1].p_sample(num=batch_size, label=batch_label, img_lr=current_images)
        
        current_images = accelerator.gather(current_images).cpu()
        current_images = denormalize(current_images.permute(0, 2, 3, 1).numpy())
        samples = np.concatenate((samples, current_images), axis=0)
    
    ext = "png" if args.save_format.lower() == "grid" else "npz"
    if model.num_classes == 0 or args.label == model.num_classes:
        label_string = "uncond"
    elif args.label == -1:
        label_string = f"random_classes_m{args.mean_scale}_v{args.var_scale}"
    else:
        label_string = f"class_{str(args.label)}_m{args.mean_scale}_v{args.var_scale}"

    samples_identifier = f"{len(gfile.glob(f'{args.save_dir}/*.{ext}'))}_{label_string}"
    samples_path = os.path.join(args.save_dir, f"samples_{samples_identifier}.{ext}")
    
    if args.save_format.lower() == "grid":
        save_images(samples, samples_path, nrow=args.nrow)
    else:
        np.savez(samples_path, arr0=samples)

    print(f"Saved {len(samples)} samples to {samples_path}")

if __name__ == '__main__':
    app.run(main)