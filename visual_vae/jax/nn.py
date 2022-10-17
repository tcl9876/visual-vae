import jax
import jax.numpy as jnp
import numpy as np

from typing import Any
from flax import linen as nn
from flax.linen import initializers 

from .jax_utils import transposed_matmul

def scaled_init(scale, dtype=jnp.float_):
    glorot_init = initializers.glorot_uniform()

    def init(key, shape, dtype=dtype):
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        initial_values = glorot_init(key, shape, dtype=dtype)
        return initial_values * scale
    return init

def Conv2D(c, kernel_size, w_scale=1., strides=1, use_bias=True):
    conv = nn.Conv(
        features=c,
        kernel_size=(kernel_size, kernel_size),
        strides=(strides, strides),
        use_bias=use_bias,
        padding='SAME',
        kernel_init=scaled_init(scale=w_scale)
    )
    return conv

class ConvSR(nn.Module):
    c: int
    kernel_size: int
    w_scale: float = 1.
    sr_lam: float = 1.  

    @nn.compact
    def __call__(self, x):
        conv = Conv2D(self.c, self.kernel_size, w_scale=self.w_scale)
        out = conv(x)
        w = conv.variables['params']['kernel']
        
        fn = lambda s: jnp.asarray(np.random.normal(size=s))
        u = self.variable('singular_vectors', 'sn_u', fn, (1, self.c))
        w = jnp.reshape(w, [-1, self.c]) #[k, k, C, C] -> [k** 2 * C, C]

        v_unnormalized = transposed_matmul(u.value, w, perm=[1, 0])
        v = v_unnormalized / jnp.linalg.norm(v_unnormalized, ord=2)
        u_unnormalized = jnp.matmul(v, w)
        u.value = u_unnormalized / jnp.linalg.norm(u_unnormalized, ord=2)

        sigma = transposed_matmul(jnp.matmul(v, w), u.value, perm=[1, 0])
        return out, jnp.squeeze(sigma * self.sr_lam)

class Downsample(nn.Module):
    c: Any
    strides: int
    
    @nn.compact
    def __call__(self, x, label=None):
        B, H, W, C = x.shape
        strides = self.strides
        x = nn.avg_pool(x, (strides, strides), (strides, strides))

        if self.c is not None:
            x = Conv2D(self.c, 3)(x)
        return x

class Upsample(nn.Module):
    c: Any
    strides: int
    
    @nn.compact
    def __call__(self, x, label=None):
        B, H, W, C = x.shape
        x = jax.image.resize(x, shape=[B, H * self.strides, W * self.strides, C], method='nearest')

        if self.c is not None:
            x = Conv2D(self.c, 3)(x)
        return x
        
class ResBlock(nn.Module):
    c: int
    c_out: int
    w_scale: float=1.
    sr_lam: Any=None
    num_classes: int=0
    is_conv: bool=True

    @nn.compact
    def __call__(self, inputs, label=None):
        x = inputs

        if self.num_classes > 0: 
            label = nn.Embed(self.num_classes+1, self.c//4)(label)[:, None, None, :] #always use +1 for cf-guidance
            label = nn.Dense(x.shape[-1])(label)
            x += label

        if self.is_conv: ksize = 3
        else: ksize = 1

        x = Conv2D(self.c, ksize)(nn.swish(x))
        if self.sr_lam is not None:
            x, sr_loss = ConvSR(self.c_out, ksize, w_scale=self.w_scale, sr_lam=self.sr_lam)(nn.swish(x))            
            return x, sr_loss
        else:
            x = Conv2D(self.c_out, ksize, w_scale=self.w_scale)(nn.swish(x))            
            if self.c==self.c_out: 
                x += inputs
            return x

class AttentionLayer(nn.Module):
    c: int
    num_heads: int=1

    @nn.compact
    def __call__(self, x, label=None):
        c = self.c
        num_heads = self.num_heads
        dhead = c // num_heads
        B, H, W, C = x.shape
        x = jnp.reshape(x, [B, H*W, C])

        # with 1 head, we only use a q-projection, because we fuse Q and K projections together, and the V and output projections are fused together too
        # note that this doesn't limit the expressivity, b/c when there's one head:
        # (x @ Wq) @ (x @ Wk).T = (x @ Wq) @ (Wk.T @ x.T) = x @ (Wq @ Wk.T) @ x.T -> notice how only one weight matrix is needed
        if num_heads==1:
            q = nn.Dense(c)(x)
            qk = transposed_matmul(q, x, perm=[0, 2, 1]) / jnp.sqrt(c)
            attention_weights = jax.nn.softmax(qk, axis=-1) 
            attn_out = jnp.matmul(attention_weights, x)   
        else:
            qkv = nn.Dense(c*3)(x)
            q, k, v = jnp.split(qkv, 3, axis=-1)

            q = jnp.reshape(q, [B, H*W, num_heads, dhead]) 
            k = jnp.reshape(k, [B, H*W, num_heads, dhead]) 
            v = jnp.reshape(v, [B, H*W, num_heads, dhead]) 
            qk = jnp.einsum('bqhd,bkhd->bhqk', q, k) / jnp.sqrt(dhead)

            attention_weights = jax.nn.softmax(qk, axis=-1)
            attn_out = jnp.einsum('bhqk,bkhd->bqhd', attention_weights, v)

        out = nn.Dense(c, kernel_init=initializers.zeros)(attn_out)
        x = x + out
        x = jnp.reshape(x, [B, H, W, C])
        return x