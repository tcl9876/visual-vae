import time
import os
import numpy as np
from functools import partial
from copy import deepcopy
import torch
from accelerate import Accelerator
from tensorflow.config import set_visible_devices as tf_set_visible_devices
from tensorflow.io import gfile, write_file

from absl import app, flags
from ml_collections.config_flags import config_flags

tf_set_visible_devices([], device_type="GPU")
np.set_printoptions(precision=4)

from visual_vae.torch.training_utils import TrainState, CosineDecayScheduler, training_losses_fn, make_optimizer
from visual_vae.torch.torch_utils import count_params, save_checkpoint, restore_checkpoint
from visual_vae.general_utils import Metrics, get_rate_schedule, print_and_log, denormalize, save_images, extract_train_inputs_fn
from visual_vae.image_datasets import create_dataset
from visual_vae.torch.model import VAE


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
    
    dargs.data_dir = dargs.data_dir.format(global_dir)
    targs.checkpoint_dirs = [subdir.format(global_dir) for subdir in targs.checkpoint_dirs]
    targs.log_dir = targs.log_dir.format(global_dir)
    dargs.framework = "torch"

    gfile.makedirs(global_dir)
    [gfile.makedirs(ckpt_dir) for ckpt_dir in targs.checkpoint_dirs]
    gfile.makedirs(targs.log_dir)

    n_accums = int(np.ceil(targs.total_batch_size/dargs.batch_size))    
    accelerator = Accelerator(gradient_accumulation_steps=n_accums)
    ismain = accelerator.is_main_process
    
    #create logfile
    logfile_path = os.path.join(targs.log_dir, 'logfile.txt')
    if not gfile.exists(logfile_path):
        write_file(logfile_path, "")

    #shortcut to print (and optionally log) only when on the main process.
    def printm(*args, log=False):
        if ismain:
            if log:
                print_and_log(*args, logfile_path=logfile_path)
            else:
                print(*args)

    if targs.total_batch_size%dargs.batch_size:
        real_batch_size = dargs.batch_size*n_accums
        printm(f"Warning: The total batch size of {real_batch_size} is not a multiple of the microbatch size {dargs.batch_size}")
        printm(f"Will use a real batch size of {dargs.batch_size} x {n_accums} = {real_batch_size}")
    else:
        real_batch_size = targs.total_batch_size

    if n_accums > 1:
        printm(f"Using gradient accumulation with microbatch size of {dargs.batch_size}, real batch size of {real_batch_size}, and {n_accums} gradient accumulations per update.")
    else:
        printm(f"Using a batch size of {real_batch_size} without gradient accumulation.")
    
    try:
        dataset = create_dataset(dargs)
    except BaseException as e:
        if ismain:
            print(e)
            print("Dataset creation failed. Perhaps you forgot to specify which dataset you're using via the --dataset argument?")
            return 

    model = VAE(**margs)
    optimizer = make_optimizer(model.parameters(), oargs.maxlr, oargs.opt_type, oargs.ema_decay, betas=(oargs.beta1, oargs.beta2))
    if oargs.opt_type=='adam':
        optimizer = torch.optim.Adam(model.parameters(), oargs.maxlr, betas=(oargs.beta1, oargs.beta2))
    else:
        optimizer = torch.optim.Adamax(model.parameters(), oargs.maxlr, betas=(oargs.beta1, oargs.beta2))

    scheduler = CosineDecayScheduler(oargs.startlr, oargs.maxlr, oargs.minlr, oargs.warmup_steps, oargs.decay_steps)
    
    #give some information about our model and dataset, then restore checkpoint
    printm('Total Parameters', count_params(model))
    printm("trying to restore checkpoint...")
    state = restore_checkpoint(state, targs.checkpoint_dirs[0])
    printm("global step after restore:", state.get_step())

    x, _, x_lr = extract_train_inputs(next(dataset))
    x = x.cpu().permute(0, 3, 1, 2).numpy()
    x = denormalize(x)
    if ismain:
        save_images(x[:16], os.path.join(targs.log_dir, "dataset_examples.png"), nrow=4)

    if x_lr is not None:
        x_lr = x_lr.cpu().permute(0, 3, 1, 2).numpy()
        x_lr = denormalize(x_lr)
        if ismain:
            save_images(x_lr[:16], os.path.join(targs.log_dir, "dataset_examples_lowres.png"), nrow=4)
    
    #distribute objects and make things needed for training - don't send scheduler through prepare
    model, optimizer, dataset = accelerator.prepare(
        model, optimizer, dataset
    )
        
    state = TrainState(model, optimizer, scheduler)
    device_count = 1 #TODO: make it based on how many devices we actually have
    
    rate_schedule = get_rate_schedule(targs.rate_schedule, margs.nlayers)
    printm("The rate schedule, on a per-layer information level, is: ", rate_schedule)
    
    extract_train_inputs = partial(extract_train_inputs_fn, is_labeled=dargs.is_labeled, lower_res=dargs.lower_res)
    training_losses = partial(training_losses_fn, model=model, device_count=device_count, global_sr_weight=targs.global_sr_weight, sigma_q=targs.sigma_q, rate_schedule=rate_schedule)

    printm(f"starting/resuming training from step {int(state.get_step())}")
    s=time.time()

    metrics = Metrics(['loss', 'distortion term', 'kl term', 'sr loss'])

    for global_step, (train_inputs) in zip(range(state.get_step(), targs.iterations), dataset):
        # Train step
        with accelerator.accumulate(model):
            
            #TODO: gradient skipping, tracking global norm and # of skips. # of skips perhaps as a superclass of the optimizer.
            train_inputs = extract_train_inputs(train_inputs)
            loss, current_metrics = training_losses(train_inputs)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            metrics.update(current_metrics)
            """
            new_metrics = {}
            for key, value in metrics:
                new_metrics[key] = accelerator.gather(value).mean().item()
            metrics.update(new_metrics)
            """


        if global_step % targs.log_freq==0 and ismain: 
            accelerator.wait_for_everyone()
            printm(f'Real Step: {state.get_step()}, Batches passed this session: {global_step},  Metrics: {metrics}, Time {round(time.time()-s)}s', log=True)

            metrics.reset_states()
        
        
        for checkpoint_dir, num_checkpoints, save_freq in zip(targs.checkpoint_dirs, targs.num_checkpoints, targs.save_freq):
            if global_step%save_freq==0 and ismain:
                accelerator.wait_for_everyone()
                save_checkpoint(state, checkpoint_dir, accelerator=accelerator, keep=num_checkpoints)
                
        if global_step%min(targs.save_freq)==0:
            accelerator.wait_for_everyone()
            _, label, img_lr = train_inputs
            num = 16//device_count
            if label is not None: label = label[:num]
            if img_lr is not None: img_lr = img_lr[:num]

            samples = model.p_sample(num=num, label=label, img_lr=img_lr)
            samples = accelerator.gather(samples)
            samples = denormalize(samples.cpu().permute(0, 2, 3, 1).numpy())
            save_images(samples, os.path.join(targs.log_dir, f'images{global_step}.png'), nrow=4)

if __name__ == '__main__':
    app.run(main)