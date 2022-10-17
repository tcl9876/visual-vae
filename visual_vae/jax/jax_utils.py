import jax.numpy as jnp
import jax
import functools
import flax
from flax.training import checkpoints
from tensorflow.io import gfile
import pickle

#checkpoint related

#a modified version of flax's save checkpoint function to support saving to a specific path ckpt_path.
#this is so we can save just the model parameters, which takes up less space, rather than saving the entire TrainState w/ flax's regular saving/loading utils
def save_checkpoint(state, ckpt_dir, ckpt_path=None, unreplicate=False, keep=99):
    if unreplicate: state = jax.device_get(flax.jax_utils.unreplicate(state))

    if isinstance(ckpt_path, str) and (ckpt_path.endswith(".p") or ckpt_path.endswith(".pkl")):
        with gfile.GFile(ckpt_path, 'wb') as f:
            pickle.dump(state, f)
    else:
        checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, target=state, step=state.step, keep=keep)

#a modified version of flax's restore checkpoint function to support loading from a specific path ckpt_path.
def restore_checkpoint(empty_state, ckpt_dir=None, ckpt_path=None, step=None):
    if isinstance(ckpt_path, str) and (ckpt_path.endswith(".p") or ckpt_path.endswith(".pkl")):
        with gfile.GFile(ckpt_path, 'rb') as f:
            state_dict = pickle.load(f)
        return empty_state.replace(params=state_dict)
    else:
        return checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=empty_state, step=step)

def calc_weight_for_kl(x, a, b, c):
    def to_float(x):
        return x.astype(jnp.float32)

    mask1 = to_float(jnp.less(x, a))
    mask2 = to_float(jnp.logical_and(jnp.greater_equal(x, a), jnp.less(x, b)))
    mask3 = to_float(jnp.logical_and(jnp.greater_equal(x, b), jnp.less(x, c)))
    mask4 = to_float(jnp.greater_equal(x, c))

    piece1 = x/a
    piece2 = 1.
    piece3 = (x-b) / (c-b) + 1.
    piece4 = 2.
    weighting_function_output = piece1*mask1 + piece2*mask2 + piece3*mask3 + piece4*mask4
    return jnp.maximum(weighting_function_output, 0.1) #the weight should never be zero.

def weighted_kl(unweighted_kl, sum_of_kl, lower_percentage, upper_percentage):
    a = lower_percentage * sum_of_kl * 0.01  #the 0.01 is because its a percentage
    b = upper_percentage * sum_of_kl * 0.01 
    w = jax.lax.stop_gradient(calc_weight_for_kl(unweighted_kl, a, b, a+b))
    adjusted_kl = unweighted_kl * w
    return jnp.sum(adjusted_kl)

#VAE related utils
def get_smoothed_variance(var_unconstrained):
    return (1/0.693) * jnp.log(1 + jnp.exp(0.693 * var_unconstrained))

def sample_diag_mvn(rng, mean, var, temp=1.):
    eps = jax.random.normal(next(rng), shape=mean.shape, dtype=mean.dtype)   
    return mean + jnp.sqrt(var) * eps * temp

def sample_mvn_deterministic(eps, mean, var, temp=1.):
    return mean + jnp.sqrt(var) * eps * temp


def transposed_matmul(a, b, perm):
    b = jnp.transpose(b, axes=perm)
    return jnp.matmul(a, b)

def compute_mvn_kl(mu1, var1, mu2, var2, raxis=(-1, -2, -3)):
    logvar1 = jnp.log(var1)
    logvar2 = jnp.log(var2)
    C = 0.5 * (logvar2 - logvar1 - 1)

    kl = 0.5 * (var1 + (mu1 - mu2) **2) / var2 + C
    kl = jnp.sum(kl, axis=raxis)
    return kl

def count_params(pytree):
    return sum([x.size for x in jax.tree_leaves(pytree)])
        
# JAX related utils
def compute_global_norm(grads):
    norms, _ = jax.flatten_util.ravel_pytree(jax.tree_map(jnp.linalg.norm, grads))
    return jnp.linalg.norm(norms)

def copy_pytree(pytree):
  return jax.tree_map(jnp.array, pytree)

@functools.partial(jax.jit, static_argnums=(2,))
def _foldin_and_split(rng, foldin_data, num):
    return jax.random.split(jax.random.fold_in(rng, foldin_data), num)

def unreplicate(x):
    return jax.device_get(flax.jax_utils.unreplicate(x))

class RngGen(object):
    """Random number generator state utility for Jax."""
    def __init__(self, init_rng):
        self._base_rng = init_rng
        self._counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.advance(1)

    def advance(self, count):
        self._counter += count
        return jax.random.fold_in(self._base_rng, self._counter)

    def split(self, num):
        self._counter += 1
        return _foldin_and_split(self._base_rng, self._counter, num)


#Sampling function
def generation_step_fn(model, vae, params, rng, train_inputs, extract_train_inputs, num):
    _, label, img_lr = extract_train_inputs(train_inputs)

    if not vae.is_superres:
        img_lr = None
    if vae.num_classes == 0:
        label = None

    outputs, _ = vae.apply({'params': params}, rng=rng, num=num, 
        label=label, img_lr=img_lr,
        method=model.p_sample, mutable=['singular_vectors']
    )
    return outputs
