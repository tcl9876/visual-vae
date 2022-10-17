import jax
import jax.numpy as jnp
import numpy as np
import flax
from jax import random
import time
import os
from functools import partial
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile, write_file


from absl import app, flags
from ml_collections.config_flags import config_flags

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)
jnp.set_printoptions(precision=4)

from visual_vae.jax.training_utils import CosineDecay, EMATrainState, EMATrainStateAccum, Adam, Adamax, training_losses_fn, train_step_fn
from visual_vae.jax.jax_utils import unreplicate, copy_pytree, count_params, generation_step_fn, save_checkpoint, restore_checkpoint
from visual_vae.general_utils import get_rate_schedule, print_and_log, denormalize, save_images, extract_train_inputs_fn, Metrics
from visual_vae.image_datasets import create_dataset
from visual_vae.jax.model import VAE

args = flags.FLAGS
config_flags.DEFINE_config_file("config", None, "the location of the config path you will use to train the model. e.g. ./config/cifar10.py")
flags.DEFINE_string("global_dir", None, "the global directory you will save all training stuff into.")
flags.mark_flags_as_required(["config", "global_dir"])

def main(_):
    #setup basic config stuff
    config, global_dir = args.config, args.global_dir
    config.unlock()

    margs = config.model
    dargs = config.dataset
    targs = config.training
    oargs = config.optimizer

    if not gfile.isdir(global_dir):
        gfile.makedirs(global_dir)
    
    dargs.data_dir = dargs.data_dir.format(global_dir)
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    dargs.framework = "JAX"
        
    try:
        dataset = create_dataset(dargs)
    except BaseException as e:
        print(e)
        print("Dataset creation failed. Perhaps you forgot to specify which dataset you're using via the --dataset argument?")
        return 

    #create logfile
    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path):
        write_file(logfile_path, "")
    printl = partial(print_and_log, logfile_path=logfile_path)


    #create rate schedule
    rate_schedule = get_rate_schedule(targs.rate_schedule, margs.nlayers)
    print("The rate schedule, on a per-layer information level, is: ", rate_schedule)

    
    #create relevant train functions from arguments
    extract_train_inputs = partial(extract_train_inputs_fn, is_labeled=dargs.is_labeled, lower_res=dargs.lower_res)
    n_accums = int(np.ceil(targs.total_batch_size/dargs.batch_size))
    if targs.total_batch_size%dargs.batch_size:
        real_batch_size = dargs.batch_size*n_accums
        print(f"Warning: The total batch size of {real_batch_size} is not a multiple of the microbatch size {dargs.batch_size}")
        print(f"Will use a real batch size of {dargs.batch_size} x {n_accums} = {real_batch_size}")
    else:
        real_batch_size = targs.total_batch_size

    training_losses = partial(training_losses_fn, global_sr_weight=targs.global_sr_weight, sigma_q=targs.sigma_q, rate_schedule=rate_schedule)
    train_step = partial(train_step_fn, training_losses=training_losses, resolution=dargs.native_res, skip_threshold=targs.skip_threshold, n_accums=n_accums)
    generation_step = partial(generation_step_fn, extract_train_inputs=extract_train_inputs, num=5)

    #set devices, init rng, and spectral regularization vectors
    devices = jax.devices()
    print("Devices:", devices)

    rng = random.PRNGKey(123)
    rng, sample_key, init_key = random.split(rng, num=3)    
    model = VAE(**margs)

    dummy_train_inputs = unreplicate(next(dataset))
    dummy_data, dummy_label, dummy_lowres = extract_train_inputs(dummy_train_inputs)

    init_out = model.init(init_key, sample_key, dummy_data, dummy_label, dummy_lowres)
    init_params = init_out['params']
    init_sn = init_out['singular_vectors']


    #make the optimizer
    learn_rate = CosineDecay(oargs.startlr, oargs.maxlr, oargs.minlr, oargs.warmup_steps, oargs.decay_steps)
    if oargs.opt_type=='adam':
        optimizer = Adam(learn_rate, b1=oargs.beta1, b2=oargs.beta2).make()
    else:
        optimizer = Adamax(learn_rate, b1=oargs.beta1, b2=oargs.beta2).make()
    print('Using optimizer with the following arguments:\n', oargs)

    #create the train state. Only create an instance of EMATrainStateAccum if we need to, otherwise create a regular EMATrainState that does not store extra grads.
    if n_accums > 1:
        print(f"Using gradient accumulation with microbatch size of {dargs.batch_size}, real batch size of {real_batch_size}, and {n_accums} gradient accumulations per update.")
        zero_grad = copy_pytree(init_params)
        zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), zero_grad)
        state = EMATrainStateAccum.create(
            apply_fn=model.apply,
            params=init_params,
            ema_params=copy_pytree(init_params),
            tx=optimizer,
            ema_decay=oargs.ema_decay,
            singular_vectors=init_sn,
            current_grads=zero_grad
        )
    else:
        print(f"Using a batch size of {real_batch_size} without gradient accumulation.")
        state = EMATrainState.create(
            apply_fn=model.apply,
            params=init_params,
            ema_params=copy_pytree(init_params),
            tx=optimizer,
            ema_decay=oargs.ema_decay,
            singular_vectors=init_sn
        )


    #give some information about our model and dataset, then restore checkpoint
    print('Total Parameters', count_params(state.params))
    print("trying to restore checkpoint...")
    state = restore_checkpoint(state, targs.checkpoint_dirs[0])
    print("global step after restore:", int(state.step))
    state = flax.jax_utils.replicate(state, devices=devices)

    x, _, x_lr = extract_train_inputs(next(dataset))
    h, w, c = list(x.shape[-3:])
    x = np.reshape(jax.device_get(x), (-1, h, w, c))
    x = denormalize(x)
    save_images(x[:16], os.path.join(targs.log_dir, "dataset_examples.png"))

    if x_lr is not None:
        h, w, c = list(x_lr.shape[-3:])
        x_lr = np.reshape(jax.device_get(x_lr), (-1, h, w, c))
        x_lr = denormalize(x_lr)
        save_images(x_lr[:16], os.path.join(targs.log_dir, "dataset_examples_lowres.png"))
    

    #make distributed train/sample fns, training rng, and metrics
    p_train_step = jax.pmap(
        fun=jax.jit(train_step),
        axis_name='shards',
    )
    
    p_generation_step = jax.pmap(
        jax.jit(generation_step),
        axis_name='batch'
    )

    rng = jax.random.PRNGKey(seed=0) #note, every time when restarting from preemption, this will use the same rng numbers. is this undesirable or does it not matter?
    metrics = Metrics(['loss', 'distortion term', 'kl term', 'sr loss'])
    skip_counter = flax.jax_utils.replicate(jnp.int32(0), devices=devices) #note: the skip counter is resetting each time, fix sometime.

    #train
    printl(f"starting/resuming training from step {int(unreplicate(state.step))}")
    s=time.time()
    for global_step, (train_inputs) in zip(range(int(unreplicate(state.step)), targs.iterations), dataset):
        # Train step
        rng, *train_step_rng = random.split(rng, num=jax.local_device_count() + 1)
        train_step_rng = jax.device_put_sharded(train_step_rng, devices)

        train_inputs = extract_train_inputs(train_inputs)
        state, new_metrics, global_norm, skip_counter = p_train_step(
            train_step_rng,
            state,
            train_inputs,
            skip_counter
        )
        
        if global_step%20==0:
            new_metrics = unreplicate(new_metrics)
            new_metrics = jax.tree_map(lambda x: float(x.mean()), new_metrics)
            gnorm = unreplicate(global_norm)
            gnorm = jax.tree_map(lambda x: float(x.mean()), gnorm)

            new_metrics['global_norm'] = gnorm
            metrics.update(new_metrics)

        if global_step % targs.log_freq==0: 
            skipc = unreplicate(skip_counter)
            printl(f'Real Step: {unreplicate(state.step)}, Batches passed this session: {global_step},  Metrics: {metrics}, Gnorm: {gnorm}, Time {round(time.time()-s)}s, Skips: {skipc}')

            metrics.reset_states()
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0:
                save_checkpoint(state, checkpoint_dir, unreplicate=True, keep=num_checkpoints)
        
        try:
            if global_step%min(targs.save_freq)==0:
                vae = VAE(**dict(margs))
                ims = p_generation_step(model, vae, state.params, train_step_rng, next(dataset))
                h, w, c = list(ims.shape[-3:])
                ims = np.reshape(jax.device_get(ims), (-1, h, w, c))
                save_images(denormalize(ims), os.path.join(targs.log_dir, f'images{global_step}.png'))

        except:
            printl("generation failed for some reason. moving on.")

if __name__ == '__main__':
    app.run(main)