import jax
import jax.numpy as jnp
import numpy as np

from typing import Iterable, Any
from flax import linen as nn
from flax.linen import initializers 

from .jax_utils import get_smoothed_variance, sample_mvn_deterministic, sample_diag_mvn, RngGen, compute_mvn_kl
from ..general_utils import get_resolutions, get_evenly_spaced_indices
from .nn import Conv2D, Downsample, Upsample, ResBlock, AttentionLayer

class StochasticConvLayer(nn.Module):
    c: int
    zdim: int
    resolution: int
    w_scale: float=1.0
    num_classes: int=0

    def setup(self):
        c = self.c
        zdim = self.zdim
        w_scale = self.w_scale
        sr_lam = float(self.resolution)
        is_conv = self.resolution>=4

        self.prior_block = ResBlock(c, c+zdim*2, w_scale=1e-10, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv)
        self.posterior_block = ResBlock(c, zdim*2, w_scale=1.0, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv)
        self.zproj = Conv2D(self.c, kernel_size=1, w_scale=w_scale, use_bias=False)
        self.shared_block = ResBlock(c, c, w_scale=w_scale, num_classes=self.num_classes, is_conv=is_conv)
        
    def p_sample(self, rng, x_c, x_u=None, label=None, mweight=1.0, vweight=1.0, T=1.):
        zdim = self.zdim
        is_guided = (x_u is not None)

        p_out_c, _ = self.prior_block(x_c, label)
        pmean_c, pv_unconstrained_c, h_c = p_out_c[..., :zdim], p_out_c[..., zdim:zdim*2], p_out_c[..., zdim*2:]
        pvar_c = get_smoothed_variance(pv_unconstrained_c)
        
        if is_guided:
            uncond_label = jnp.ones_like(label) * self.num_classes
            
            p_out_u, _ = self.prior_block(x_u, uncond_label)
            pmean_u, pv_unconstrained_u, h_u = p_out_u[..., :zdim], p_out_u[..., zdim:zdim*2], p_out_u[..., zdim*2:]
            pvar_u = get_smoothed_variance(pv_unconstrained_u)

            #compute distribution means and variances according to the guided sampling formula.
            pmean = (pmean_u + mweight * (pmean_c - pmean_u))
            logvar_c, logvar_u = jnp.log(pvar_c) , jnp.log(pvar_u)
            pvar = pvar_u * jnp.exp(vweight * (logvar_c - logvar_u))
        else:
            #use the conditional mean and variance as usual.
            pmean, pvar = pmean_c, pvar_c

        z = sample_diag_mvn(rng, pmean, pvar, temp=T)
        z = self.zproj(z)
        
        x_c += (z + h_c)
        x_c = self.shared_block(x_c, label)
        
        if is_guided:
            x_u += (z + h_u)
            x_u = self.shared_block(x_u, uncond_label)

        return x_c, x_u

    def __call__(self, eps, x, acts, label=None):
        if label is not None: label, cf_guidance_label = label #q is always conditional, p is not
        else: cf_guidance_label = None
        zdim = self.zdim
        p_out, sr_loss_p = self.prior_block(x, cf_guidance_label)
        pmean, pv_unconstrained, h = p_out[..., :zdim], p_out[..., zdim:zdim*2], p_out[..., zdim*2:]

        concatted = jnp.concatenate((x, acts), axis=-1)
        q_out, sr_loss_q = self.posterior_block(concatted, label)
        qmean, qv_unconstrained = jnp.split(q_out, 2, axis=-1)     

        pvar = get_smoothed_variance(pv_unconstrained)
        qvar = get_smoothed_variance(qv_unconstrained)
        kl_unweighted = compute_mvn_kl(qmean, qvar, pmean, pvar)
        z = sample_mvn_deterministic(eps, qmean, qvar)  

        z = self.zproj(z)
        x += (z + h)
        x = self.shared_block(x, cf_guidance_label)
        return x, kl_unweighted, sr_loss_p+sr_loss_q

class DecoderLevel(nn.Module):
    c: int
    zdim: int
    nlayers: int
    w_scale: float
    num_classes: int
    num_attention: int
    current_resolution: int
    max_resolution: int
    c_next: Any

    def setup(self):
        is_conv = (self.current_resolution >= 4)
        layer_list = []
        attention_indices = get_evenly_spaced_indices(self.nlayers, self.num_attention)

        for i in range(1, self.nlayers+1):
            layer_list.append(
                StochasticConvLayer(
                    c=self.c, 
                    zdim=self.zdim, 
                    resolution=self.current_resolution,
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes
                )
            )
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c))

        if self.current_resolution < self.max_resolution:
            strides = 2 if is_conv else 4
            layer_list.append(Upsample(self.c_next, strides))
        self.layer_list = layer_list

    def p_sample(self, rng, x_c, x_u=None, label=None, mweight=1.0, vweight=1.0, T=1.0):
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                x_c, x_u = layer.p_sample(rng, x_c, x_u, label, mweight=mweight, vweight=vweight, T=T)
            else:
                x_c = layer(x_c)
                if x_u is not None:
                    x_u = layer(x_u)

        return x_c, x_u

    def __call__(self, rng, x, acts, label=None):
        shape = (x.shape[0], self.current_resolution, self.current_resolution, self.zdim)

        KLs = []
        SR_Losses = 0.
        i = 0
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                eps = jax.random.normal(next(rng), shape=shape)
                x, kl, sr_loss = layer(eps, x, acts, label)
                KLs.append(kl)
                SR_Losses += sr_loss
                i += 1
            else:
                x = layer(x)
        return x, KLs, SR_Losses

class EncoderLevel(nn.Module):
    c: int
    nlayers: int
    current_resolution: int
    c_next: Any
    num_attention: int=0
    w_scale: float=1.
    num_classes: int=0

    def setup(self):
        nlayers = self.nlayers
        is_conv = (self.current_resolution >= 4)

        layer_list = []
        attention_indices = get_evenly_spaced_indices(nlayers, self.num_attention)

        for i in range(nlayers):
            layer_list.append(
            ResBlock(
                    self.c, 
                    self.c, 
                    self.w_scale, 
                    num_classes=self.num_classes, 
                    is_conv=is_conv,
                )
            )
            
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c))
            
        if self.current_resolution > 1:
            strides = 2 if self.current_resolution > 4 else 4
            layer_list.append(Downsample(self.c_next, strides)) #note: c_next might be None
        self.layer_list = layer_list

    def __call__(self, x, label=None):
        acts = None
        for layer in self.layer_list:
            if isinstance(layer, Downsample):
                acts = x
            x = layer(x, label)

        if acts is None: acts = x
        return x, acts

        
class Encoder(nn.Module):
    c: int
    c_mult: Iterable[int]
    nlayers: Iterable[int]
    resolution: int
    num_attention: Iterable[int]
    num_classes: int

    @nn.compact
    def __call__(self, img, label=None):
        C = [self.c * mult for mult in self.c_mult]
        nlayers = self.nlayers
        w_scale = 1/jnp.sqrt(sum(nlayers))
        num_resolutions = len(nlayers)
        resolutions = get_resolutions(self.resolution, num_resolutions)

        x = Conv2D(C[0], kernel_size=3)(img)
        acts = []
        for i in range(num_resolutions):
            c_next = C[i+1] if i<num_resolutions-1 else None
            if c_next == C[i]: c_next = None

            x, acts_i = EncoderLevel(
                C[i],
                nlayers[i],
                resolutions[i],
                c_next,
                self.num_attention[i],
                w_scale,
                self.num_classes
            )(x, label)
            acts.append(acts_i)
        return acts

class LowresEncodingUnet(nn.Module):
    c: int
    nlayers: int
    num_classes: int

    @nn.compact
    def __call__(self, x, label=None):
        c = self.c
        nlayers = self.nlayers
        w_scale = 1/np.sqrt(2. * sum(nlayers)).astype('float32')
        acts = []

        x = Conv2D(c, 1)(x)
        for i in range(len(nlayers)):
            for j in range(nlayers[i]):
                x = ResBlock(c, c, w_scale, num_classes=self.num_classes)(x, label)
            acts.append(x)
            if i<len(nlayers)-1: x = Downsample(None, strides=2)(x)
        

        for i in reversed(range(len(nlayers))):
            for j in range(nlayers[i]):
                concatted = jnp.concatenate((x, acts[i]), axis=-1)
                x = Conv2D(c, 3)(concatted)
                x = ResBlock(c, c, w_scale, num_classes=self.num_classes)(x, label)
            if i>0: x = Upsample(None, strides=2)(x)
        
        return x


class VAE(nn.Module):
    c: int
    c_enc: int
    c_mult: Iterable[int]
    zdim: int
    nlayers: Iterable[int]
    enc_nlayers: Iterable[int]
    num_attention: Iterable[int]
    resolution: int
    num_final_blocks: int
    num_classes: int=0
    is_superres: bool=False
    superres_nlayers: Any=None

    def setup(self):
        self.cond = self.num_classes > 0
        self.embed = nn.Embed(self.num_classes+1, self.c)
        self.encoder = Encoder(self.c_enc, self.c_mult, self.enc_nlayers, self.resolution, self.num_attention, self.num_classes)

        layer_list = []
        resolutions = get_resolutions(self.resolution, len(self.nlayers))
        C = [self.c * mult for mult in self.c_mult]
        w_scale = 1/np.sqrt(sum(self.nlayers))
        
        if self.is_superres:
            #do some processing of the lower-resolution image.
            self.process_lowres_image = LowresEncodingUnet(C[-1], self.superres_nlayers, self.num_classes)
        else:
            self.initial_x = self.param('initial_x', initializers.zeros, (1, resolutions[-1], resolutions[-1], C[-1]))

        for i in reversed(range(len(resolutions))):
            c_next = C[i-1] if i>0 else None
            if c_next == C[i]: c_next = None

            layer_list.append(
                DecoderLevel(
                    c=C[i],
                    zdim=self.zdim,
                    nlayers=self.nlayers[i],
                    w_scale=w_scale,
                    num_classes=self.num_classes,
                    num_attention=self.num_attention[i],
                    current_resolution=resolutions[i],
                    max_resolution=resolutions[0],
                    c_next=c_next
                )
            )
        self.layer_list = layer_list

        self.output_blocks = [ResBlock(C[0], C[0], w_scale=w_scale) for _ in range(self.num_final_blocks)]
        self.outproj = Conv2D(6, 1)

    def p_sample(self, rng, num=10, label=None, img_lr=None, mweight=1.0, vweight=1.0):
        rng = RngGen(rng)

        if self.num_classes:
            uncond_label = jnp.ones_like(label) * self.num_classes
        else:
            uncond_label = None
            
        is_guided = (mweight != 1 or vweight != 1)

        assert (img_lr is not None) == self.is_superres
        if self.is_superres: 
            c_aug = 0.1 if self.resolution==64 else 0.15 #TODO: get the c_aug from config
            img_lr += jax.random.normal(next(rng), img_lr.shape) * c_aug
            x_c = self.process_lowres_image(img_lr, label)
            x_u = self.process_lowres_image(img_lr, uncond_label) if is_guided else None
        else:
            x_c = jnp.tile(self.initial_x, [num, 1, 1, 1])
            x_u = jnp.tile(self.initial_x, [num, 1, 1, 1]) if is_guided else None

        #update cond'l and uncond'l hidden states at each stochastic layer.
        for decoderlevel in self.layer_list:
            x_c, x_u = decoderlevel.p_sample(rng, x_c, x_u, label, mweight=mweight, vweight=vweight)

        #now we only need to process the conditional part to output images.
        for block in self.output_blocks:
            x_c = block(x_c)

        x_c = self.outproj(x_c)

        return x_c[..., :3]
    
    def __call__(self, rng, img, label=None, img_lr=None):
        rng = RngGen(rng)

        #randomly drop class label in the prior w/ 10% probability to enable CFG
        if self.cond:
            uncond_label = jnp.full_like(label, self.num_classes)
            mask = jnp.greater(jax.random.uniform(next(rng), label.shape), 0.9).astype(jnp.int32)
            cf_guidance_label = label*(1-mask) + mask*uncond_label

            generator_label = (label, cf_guidance_label)
        else: 
            label, generator_label = None, None

        assert (img_lr is not None) == self.is_superres
        if self.is_superres:
            x = self.process_lowres_image(img_lr, cf_guidance_label)
        else:
            x = jnp.tile(self.initial_x, [img.shape[0], 1, 1, 1])

        acts = self.encoder(img, label=label)
        KLs = []
        SR_Losses = 0.

        for i, decoderlevel in enumerate(self.layer_list):
            j = -i+len(self.nlayers)-1
            x, kls, sr_losses = decoderlevel(rng, x, acts[j], generator_label)
            KLs.extend(kls)
            SR_Losses += sr_losses
                
        for block in self.output_blocks:
            x = block(x)     
        x = self.outproj(x)
        return x, KLs, SR_Losses