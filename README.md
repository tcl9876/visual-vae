# Visual VAE

Code for the paper "[Optimizing Hierarchical Image VAEs for Sample Quality](https://arxiv.org/abs/2210.10205)". Hierarchical VAEs are an extension of regular VAEs which uses a sequence of learned normal distributions for the prior and posterior. Notable examples include [NVAE](https://arxiv.org/abs/2007.03898) and [Very Deep VAE](https://arxiv.org/abs/2011.10650). We propose changes to these Hierarchical VAEs that help them generate better-looking samples, namely: 
* controlling how much information is added in each latent variable layer
* Using a continuous Gaussian KL loss instead of a discrete (mixture of logistic distributions) loss.
* using a guided sampling strategy similar to  [classifier-free guidance](https://openreview.net/forum?id=qw8AKxfYbI) in diffusion models

This release includes models for CIFAR-10 and ImageNet $64^2$ $-$ the ImageNet $64^2$ model consists of a base model that generates $32^2$ images, followed by a $2 \times$ super-resolution model. On these two datasets, we achieve FID scores of 20.82 and 17.5 respectively; these results are considerably better than previous state-of-the-art VAEs.

# Instructions for usage
First, clone our repository and change directory into it.  Install the requirements via:
```pip install -r requirements.txt```

Then do :
```pip install -e .```

## Sampling

To sample from our pretrained models, you should first download them using the links from the Pretrained Models section below. In these examples, we assume you've downloaded the relevant model files into the directory "./models". 

To create an $8 \times 8$ grid of CIFAR-10 samples:

```python scripts/evaluate.py --config "config/cifar10.py" --save_dir "./results" --checkpoint_path "./models/cifar10_ema_weights.pt" --n_samples 64 --nrow 8 ```

To create a grid of ImageNet $64^2$ samples with a mean guidance strength of 1.5, and a variance guidance strength of 5.0:

```python scripts/evaluate.py --config "config/imagenet32.py" --save_dir "./results" --checkpoint_path "./models/i32_ema_weights.pt" --superres_config "config/imagenet64.py" --superres_checkpoint_path "./models/i64_ema_weights.pt" --n_samples 36 --nrow 6 --mean_scale 1.5 --var_scale 5.0  ```

To perform unguided sampling (ImageNet only), set ``--mean_scale 0.0`` and ``--var_scale 0.0``. If not specified, the default guidance values are 2.0 and 4.0 respectively. For unconditional sampling, set ```--label 1000```. Alternatively, to generate images from a specific class, set ``label $LABEL_NUM`` (see [this website](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/) for the list of ImageNet class numbers.

 To generate $32^2$ ImageNet samples (no super-resolution model in the pipeline), simply remove the ``supperres_config`` and ``--superres_checkpoint_path`` arguments.

To create a .npz file of instead of a grid, e.g. for FID evaluation, add the argument ```--save_format "npz"```.

If you trained your own model with a different config, remember to set the correct model config via  ``--config "config/my_new_config.py"`` 

#### Sampling in JAX 

The instructions above are for sampling with PyTorch. Sampling with the JAX models is essentially the same, except:
* use ``scripts/evaluate_jax.py`` instead of ``scripts/evaluate.py`` 
* use the JAX checkpoints instead of the PyTorch ones (e.g. ``cifar10_ema_weights_jax.p`` instead of ``cifar10_ema_weights.pt``)

<b>Note:</b> you will need to install JAX and flax, e.g. via ``pip install jax>=0.3.0 flax`` 

## Training

The training configuration files are located within the  ``config`` folder - the config is divided into 4 parts: model architecture hyperparameters, dataset information, training details, and optimizer settings. We encourage you to look at the existing config files for more information, or if you want to change certain hyperparameters.

Training in PyTorch (CIFAR-10 dataset):

``python scripts/train.py --config config/cifar10.py --global_dir "./training_results"``

This will save the training checkpoints and logs to the folder ./training_results. 

<b>Note:</b> Training code is still a work in progress but should be finalized within the next couple of days. 


# Pretrained Models

We include model checkpoints for our CIFAR-10, ImageNet $32^2$ and ImageNet $32^2 \rightarrow 64^2$ models, which you can download them from Google Drive:

PyTorch checkpoints:
 * CIFAR-10 : [cifar10_ema_weights.pt](https://docs.google.com/uc?export=download&id=1OWcVyWyyKlyj2aIAE7tQfsmdmEuePz8q)
 * ImageNet $32^2$: [i32_ema_weights.pt](https://docs.google.com/uc?export=download&id=17Gsehu-4o0rDfGN0tf-aswecnWfbtbCC)
 * ImageNet $32^2 \rightarrow 64^2$: [i64_ema_weights.pt](https://docs.google.com/uc?export=download&id=1Z6Yehkp0DnxYjD5qAzjuBAE-4RMoagjx)

JAX checkpoints:
 * CIFAR-10 : [cifar10_ema_weights_jax.p](https://docs.google.com/uc?export=download&id=12zH4p7vo8Z3vkhQsLW9YJSSVRhDmtP04)
 * ImageNet $32^2$: [i32_ema_weights_jax.p](https://docs.google.com/uc?export=download&id=1d3spI5F8ue8sWXD6thvesGBLWTT_ex2x)
 * ImageNet $32^2 \rightarrow 64^2$: [i64_ema_weights_jax.p](https://docs.google.com/uc?export=download&id=1Y2U5H_R02_u7Y_xvFl6d1RGMhZAP1v6i)


# Acknowledgements:

* This research was supported by Google's TPU Research Cloud (TRC) Program, which provided Google Cloud TPUs for training the models. 

* Portions of our codebase were adapted from the [Efficient-VDVAE](https://github.com/Rayhane-mamah/Efficient-VDVAE), [Progressive Distillation](https://github.com/google-research/google-research/tree/master/diffusion_distillation), and [Guided Diffusion](https://github.com/openai/guided-diffusion) repositories - thanks for open-sourcing!

# Citation:

If you found this repository useful to your research, please consider citing our paper:
```
@misc{luhman2022optimizing,
      title={Optimizing Hierarchical Image VAEs for Sample Quality}, 
      author={Eric Luhman and Troy Luhman},
      year={2022},
      eprint={2210.10205},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
