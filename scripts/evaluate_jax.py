import jax
import jax.random as random
import jax.numpy as jnp
from flax.training.train_state import TrainState
from optax import adam
from tensorflow.io import gfile
import numpy as np
import os

from absl import app, flags
from ml_collections.config_flags import config_flags

from visual_vae.jax.jax_utils import restore_checkpoint
from visual_vae.jax.model import VAE
from visual_vae.general_utils import denormalize, save_images

args = flags.FLAGS
config_flags.DEFINE_config_file("config", "./config/cifar10.py", "the location of the config path for the model. e.g. ./config/cifar10.py.")
config_flags.DEFINE_config_file("superres_config", None, "the location of the config path for a super-resolution model. e.g. ./config/imagenet64.py. ")
flags.DEFINE_string("save_dir", None, "the global directory you will save your results into.")
flags.DEFINE_string("checkpoint_path", None, "the path you will restore the model checkpoint from.")
flags.DEFINE_string("superres_checkpoint_path", None, "the path(s) you will restore the super-resolution model checkpoint from.")
flags.DEFINE_integer("n_samples", 36, "the number of samples you want to create.")
flags.DEFINE_integer("nrow", 6, "if you are making a grid, the number of columns in the grid. By default, we use 6 columns.")
flags.DEFINE_integer("max_batch_size", 64, "the maximum allowable batch size for sampling.")
flags.DEFINE_float("mean_scale", 1.0, "the guidance weight for classifier-free guidance of the mean.")
flags.DEFINE_float("var_scale", 1.0, "the guidance weight for classifier-free guidance of the variance.")
flags.DEFINE_integer("label", -1, "If the model is class conditional, generates images from a certain class. Set to -1 for randomly chosen classes. Set to the number of classes in your dataset (e.g. 1000 for imagenet) for unconditional sampling.")
flags.DEFINE_string("save_format", "grid", "either 'grid' or 'npz'. determines whether to save results as a grid of images (default, best for <= 100 images), or as an .npz file (for evaluation).")
flags.mark_flags_as_required(["config", "checkpoint_path", "save_dir"])


def build_jax_state(model):
    
    rng = random.PRNGKey(0)
    rng, sample_key, init_key = random.split(rng, num=3) 
    dummy_input = random.normal(sample_key, [1, model.resolution, model.resolution, 3])

    if model.num_classes:
        dummy_label = jnp.zeros([1,], dtype=jnp.int32)
    else:
        dummy_label = None

    if model.is_superres:
        lower_res = model.resolution//(2**(len(model.nlayers) - 1))
        dummy_lowres = random.normal(sample_key, [1, lower_res, lower_res, 3])
    else:
        dummy_lowres = None
    
    init_out = model.init(init_key, sample_key, dummy_input, dummy_label, dummy_lowres)
    init_params = init_out['params']
    tx = adam(learning_rate=0.) #dummy optimizer because flax TrainState needs you to pass one in
    state = TrainState.create(apply_fn=model.p_sample, params=init_params, tx=tx)

    return state

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

    models = []
    states = []
    for config, ckpt_path in zip(configs, ckpt_paths):
        config.unlock()
        margs = config.model
        model = VAE(**margs)
        state = build_jax_state(model)
        state = restore_checkpoint(state, ckpt_path=ckpt_path)
        models.append(model)
        states.append(state)

    res = configs[-1].model.resolution
    samples = np.zeros([0, res, res, 3]).astype('uint8')
    rng = random.PRNGKey(0)

    max_batch_size = min(args.max_batch_size, args.n_samples)
    def lr_gen_func():
        return models[0].apply({'params': states[0].params}, rng=rng, num=max_batch_size, 
            label=batch_label, img_lr=None, method=models[0].p_sample, mutable=['singular_vectors'], mweight=args.mean_scale, vweight=args.var_scale
        )
    
    lr_gen_func = jax.jit(lr_gen_func)

    if len(models) == 2:
        def sr_gen_func(img_lr):
            return models[1].apply({'params': states[1].params}, rng=rng, num=max_batch_size, 
                label=batch_label, img_lr=img_lr, method=models[1].p_sample, mutable=['singular_vectors'], mweight=args.mean_scale, vweight=args.var_scale
            )
        
        sr_gen_func = jax.jit(sr_gen_func)


    for n in range(0, args.n_samples, args.max_batch_size):
        batch_size = min(args.max_batch_size, args.n_samples - n)
        
        if not models[0].num_classes:
            batch_label = None
        elif args.label == -1:
            rng, label_gen_key = random.split(rng)
            batch_label = random.randint(label_gen_key, [batch_size], minval=0, maxval=configs[0].model.num_classes)
        else:
            batch_label = jnp.array([args.label] * batch_size, dtype=jnp.int32)

        current_images, _ = lr_gen_func()
        if len(models) == 2:
            current_images, _ = sr_gen_func(current_images)

        current_images = current_images[:batch_size]
        current_images = denormalize(np.array(current_images))
        samples = np.concatenate((samples, current_images), axis=0)
    
    ext = "png" if args.save_format.lower() == "grid" else "npz"
    if model.num_classes == 0 or args.label == model.num_classes:
        label_string = "uncond"
    elif args.label == -1:
        label_string = "random_classes_m{args.mean_scale}_v{args.var_scale}"
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
