import optax
import jax
import flax
import jax.numpy as jnp

from typing import Any, Callable, Optional, Union
from optax._src.base import Params, identity
from optax._src import combine
from optax._src.alias import _scale_by_learning_rate, ScalarOrSchedule
from optax._src import transform
from optax._src import base
from optax._src import numerics
from optax._src.transform import update_moment, bias_correction, ScaleByAdamState

from flax.training.train_state import TrainState

from .jax_utils import copy_pytree, compute_mvn_kl, weighted_kl, compute_global_norm


def training_losses_fn(params, rng, state, train_inputs, global_sr_weight, sigma_q, rate_schedule): 
    img, label, img_lr = train_inputs

    outputs, singular_vectors = state.apply_fn({'params': params}, rng=rng, img=img, label=label, img_lr=img_lr, mutable=['singular_vectors'])
    model_out, unweighted_kls, sr_loss = outputs
    sr_loss *= global_sr_weight

    mean_output, var_output = jnp.split(model_out, 2, axis=-1)
    var_output = jnp.exp(var_output)
    qvar = (sigma_q ** 2)
    neg_logpx_z = compute_mvn_kl(img, qvar, mean_output, var_output, raxis=None) #shape is ()

    total_kl_per_image = jnp.sum(jnp.stack(unweighted_kls, axis=-1), axis=-1)  #(B, )
    KL_Loss = jnp.float32(0.)
    for i, k in enumerate(unweighted_kls):
        w = rate_schedule[i]
        k_weighted = weighted_kl(k, total_kl_per_image, w, 2.*w)
        KL_Loss += k_weighted    

    per_replica_bs = img.shape[0] / jax.local_device_count()
    neg_logpx_z /= per_replica_bs
    KL_Loss /= per_replica_bs
    total_loss = neg_logpx_z + KL_Loss + sr_loss
    metrics = {'loss': total_loss, 'distortion term': neg_logpx_z, 'kl term': KL_Loss, 'sr loss': sr_loss}
    return total_loss, (metrics, singular_vectors)

def safe_update(state, grads, vectors, global_norm, clip_value):
    def update(_):
        return state.apply_gradients(grads_and_vectors=(grads,vectors)) 

    def do_nothing(_):
        return state

    state = jax.lax.cond(global_norm < clip_value, update, do_nothing, operand=None)
    skip_bool = jnp.logical_or(global_norm >= clip_value, jnp.isnan(global_norm))  
    return state, jnp.int32(skip_bool)

def train_step_fn(rng, state, train_inputs, skip_counter, training_losses, resolution, skip_threshold, n_accums):

    grad_fn = jax.value_and_grad(training_losses, has_aux=True, argnums=0)
    (_, metrics_and_vectors), grads = grad_fn(state.params, rng, state, train_inputs)
    metrics, vectors = metrics_and_vectors

    grads = jax.lax.pmean(grads, axis_name='shards')
    global_norm = compute_global_norm(grads) / (3 * resolution**2)

    def update_func(_):
        if n_accums > 1:
            return state.update_gradients(grads_and_vectors=(grads, vectors))
        else:
            return state.apply_gradients(grads_and_vectors=(grads, vectors))

    def do_nothing(_):
        return state

    state = jax.lax.cond(global_norm < skip_threshold, update_func, do_nothing, operand=None)
    skip_bool = jnp.logical_or(global_norm >= skip_threshold, jnp.isnan(global_norm))  

    state = accum_apply(state, n_accums)

    skip_counter += jnp.int32(skip_bool)
    return state, metrics, global_norm, skip_counter

def accum_apply(state, n_accums):
    def update(_):
        return state.apply_gradients() 

    def do_nothing(_):
        return state

    state = jax.lax.cond(state.accum_step>=n_accums, update, do_nothing, operand=None)
    return state



#optimizer related stuff

def update_infinite_moment(updates, moments, decay, eps):
    """Compute the exponential moving average of the infinite moment."""
    return jax.tree_map(
        lambda g, t: jnp.maximum(decay * t, jnp.abs(g) + eps), updates, moments)  # max(β2 · ut−1, |gt|)


def scale_by_adamax(
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8
) -> base.GradientTransformation:
    """Rescale updates according to the Adamax algorithm.
    References:
      [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
    Args:
      b1: decay rate for the exponentially weighted average of grads.
      b2: decay rate for the exponentially weighted average of squared grads.
      eps: term added to the denominator to improve numerical stability.
    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        mu = jax.tree_map(jnp.zeros_like, params)  # First moment
        nu = jax.tree_map(jnp.zeros_like, params)  # Infinite moment
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = update_moment(updates, state.mu, b1, 1)
        nu = update_infinite_moment(updates, state.nu, b2, eps)  # No bias correction for infinite moment
        count_inc = numerics.safe_int32_increment(state.count)
        mu_hat = bias_correction(mu, b1, count_inc)
        updates = jax.tree_map(
            lambda m, v: m / v, mu_hat, nu)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)

class BaseWeightDecayOptimizer:
    def __init__(self, learning_rate, use_weight_decay, l2_weight, l2_mask):
        self.learning_rate = learning_rate
        self.use_weight_decay = use_weight_decay
        self.l2_weight = l2_weight
        self.l2_mask = l2_mask

    def _add_weight_decay_and_lr_transformations(self, transforms):
        if self.use_weight_decay:
            assert self.l2_weight != 0.
            transforms.append(transform.add_decayed_weights(weight_decay=self.l2_weight, mask=self.l2_mask))

        transforms.append(_scale_by_learning_rate(self.learning_rate))
        return transforms

    def create_transforms(self):
        """Method that returns a list of optax transformations to apply during optimization"""
        raise NotImplementedError('Do not use BaseWeightDecayOptimizer, create your own optimizer on top!')

    def make(self):
        return combine.chain(*self.create_transforms())

class Adam(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 eps_root: float = 0.,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.eps_root = eps_root
        super(Adam, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                   l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [transform.scale_by_adam(b1=self.b1, b2=self.b2, eps=self.eps, eps_root=self.eps_root)]
        return self._add_weight_decay_and_lr_transformations(transforms)
        
class Adamax(BaseWeightDecayOptimizer):
    def __init__(self, learning_rate: ScalarOrSchedule,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 eps: float = 1e-8,
                 use_weight_decay: bool = False,
                 l2_weight: float = 0.,
                 l2_mask: Optional[Union[Any, Callable[[Params], Any]]] = None):
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        super(Adamax, self).__init__(learning_rate=learning_rate, use_weight_decay=use_weight_decay, l2_weight=l2_weight,
                                     l2_mask=l2_mask)

    def create_transforms(self):
        transforms = [scale_by_adamax(b1=self.b1, b2=self.b2, eps=self.eps)]
        return self._add_weight_decay_and_lr_transformations(transforms)
        
class EMATrainState(TrainState):
    ema_decay: float
    ema_params: flax.core.FrozenDict[str, Any]
    singular_vectors: flax.core.FrozenDict[str, Any]

    def update_gradients(self, *, grads_and_vectors, **kwargs):
        raise RuntimeError("The 'update_gradients' method only works for a TrainState object that does gradient accumulation.")

    def apply_gradients(self, *, grads_and_vectors, **kwargs):
        grads, vectors = grads_and_vectors
        vectors = vectors['singular_vectors']

        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            singular_vectors=vectors
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )

class EMATrainStateAccum(TrainState):
    ema_decay: float
    ema_params: flax.core.FrozenDict[str, Any]
    singular_vectors: flax.core.FrozenDict[str, Any]
    current_grads: flax.core.FrozenDict[str, Any]
    accum_step: int=0
    
    def update_gradients(self, *, grads_and_vectors, **kwargs):
        grads, vectors = grads_and_vectors
        vectors = vectors['singular_vectors']
        
        new_grads = jax.tree_util.tree_map(lambda x,y: x+y, self.current_grads, grads)
        return self.replace(
            step=self.step,
            params=self.params,
            ema_params=self.ema_params,
            opt_state=self.opt_state,
            singular_vectors=vectors,
            current_grads=new_grads,
            accum_step=self.accum_step+1
        )
        

    def apply_gradients(self):
        updates, new_opt_state = self.tx.update(
            self.current_grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        new_ema_params = jax.tree_util.tree_map(lambda ema, p: ema * self.ema_decay + (1 - self.ema_decay) * p,
                                      self.ema_params, new_params)
        
        zero_grad = copy_pytree(self.params)
        zero_grad = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), zero_grad)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            ema_params=new_ema_params,
            opt_state=new_opt_state,
            singular_vectors=self.singular_vectors,
            current_grads=zero_grad,
            accum_step=self.accum_step * 0,
        )

    @classmethod
    def create(cls, *, apply_fn, params, ema_params, tx, ema_decay, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            ema_params=ema_params,
            tx=tx,
            opt_state=opt_state,
            ema_decay=ema_decay,
            **kwargs,
        )

class CosineDecay:
    def __init__(self, startlr, maxlr, minlr, warmup_steps, decay_steps):
        self.startlr = startlr
        self.maxlr = maxlr
        self.minlr = minlr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        
    def __call__(self, step):
        step = jnp.minimum(step, self.decay_steps)
        startlr, maxlr, minlr = self.startlr, self.maxlr, self.minlr
        warmup = startlr + step/self.warmup_steps * (maxlr - startlr)

        decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * step/self.decay_steps))
        decay_factor = (1 - minlr/maxlr) * decay_factor + minlr/maxlr
        lr = maxlr * decay_factor
        return jnp.minimum(warmup, lr)
