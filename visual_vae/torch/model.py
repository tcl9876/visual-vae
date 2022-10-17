import torch
import torch.nn as nn
import numpy as np

from .torch_utils import get_smoothed_variance, sample_diag_mvn, compute_mvn_kl, maybe_remat
from ..general_utils import get_resolutions, get_evenly_spaced_indices
from .nn import Conv2D, Downsample, Upsample, ResBlock, AttentionLayer

class StochasticConvLayer(nn.Module):
    def __init__(self, c, c_enc, zdim, resolution, w_scale, num_classes, checkpoint=False):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.zdim = zdim
        self.resolution = resolution
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.checkpoint = checkpoint

        sr_lam = float(self.resolution)
        is_conv = self.resolution>=4

        self.prior_block = ResBlock(c, c+zdim*2, w_scale=1e-10, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv)
        self.posterior_block = ResBlock(c, zdim*2, c_in=c+c_enc, w_scale=1.0, sr_lam=sr_lam, num_classes=self.num_classes, is_conv=is_conv)
        self.zproj = Conv2D(self.zdim, self.c, kernel_size=1, w_scale=w_scale, use_bias=False)
        self.shared_block = ResBlock(c, c, w_scale=w_scale, num_classes=self.num_classes, is_conv=is_conv)
    
    def p_sample(self, x_c, x_u=None, label=None, mweight=1.0, vweight=1.0, T=1.0):
        zdim = self.zdim
        is_guided = (x_u is not None)

        p_out_c, _ = self.prior_block(x_c.clone(), label)
        pmean_c, pv_unconstrained_c, h_c = p_out_c[:, :zdim, ...], p_out_c[:, zdim:zdim*2, ...], p_out_c[:, zdim*2:, ...]
        pvar_c = get_smoothed_variance(pv_unconstrained_c)
        
        if is_guided:
            uncond_label = torch.ones_like(label) * self.num_classes
            
            p_out_u, _ = self.prior_block(x_u, uncond_label)
            pmean_u, pv_unconstrained_u, h_u = p_out_u[:, :zdim, ...], p_out_u[:, zdim:zdim*2, ...], p_out_u[:, zdim*2:, ...]
            pvar_u = get_smoothed_variance(pv_unconstrained_u)

            #compute distribution means and variances according to the guided sampling formula.
            pmean = (pmean_u + mweight * (pmean_c - pmean_u))
            pvar = pvar_u * torch.exp(vweight * (pvar_c.log() - pvar_u.log()))
        else:
            #use the conditional mean and variance as usual.
            pmean, pvar = pmean_c, pvar_c

        z = sample_diag_mvn(pmean, pvar, temp=T)
        z = self.zproj(z)

        x_c += (z + h_c)
        x_c = self.shared_block(x_c, label)
        
        if is_guided:
            x_u += (z + h_u)
            x_u = self.shared_block(x_u, uncond_label)
        
        return x_c, x_u
    
    #@maybe_remat
    def forward(self, eps, x, acts, label=None):
        if label is not None: label, cf_guidance_label = label #q is always conditional, p is not
        else: cf_guidance_label = None
        zdim = self.zdim
        p_out, sr_loss_p = self.prior_block(x, cf_guidance_label)
        pmean, pv_unconstrained, h = p_out[..., :zdim], p_out[..., zdim:zdim*2], p_out[..., zdim*2:]

        concatted = torch.cat((x, acts), axis=-1)
        q_out, sr_loss_q = self.posterior_block(concatted, label)
        qmean, qv_unconstrained = torch.chunk(q_out, 2, axis=-1)     

        pvar = get_smoothed_variance(pv_unconstrained)
        qvar = get_smoothed_variance(qv_unconstrained)
        kl_unweighted = compute_mvn_kl(qmean, qvar, pmean, pvar)
        z = sample_diag_mvn(qmean, qvar, eps=eps)

        z = self.zproj(z)
        x += (z + h)
        x = self.shared_block(x, cf_guidance_label)
        return x, kl_unweighted, sr_loss_p+sr_loss_q

class DecoderLevel(nn.Module):
    def __init__(self, c, c_enc, zdim, nlayers, w_scale, num_classes, num_attention, current_resolution, max_resolution, c_next, checkpoint=False):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.zdim = zdim 
        self.nlayers = nlayers
        self.w_scale = w_scale
        self.num_classes = num_classes
        self.num_attention = num_attention
        self.current_resolution = current_resolution
        self.max_resolution = max_resolution
        self.c_next = c_next

        is_conv = (self.current_resolution >= 4)
        layer_list = nn.ModuleList([])
        attention_indices = get_evenly_spaced_indices(self.nlayers, self.num_attention)

        for i in range(1, self.nlayers+1):
            layer_list.append(
                StochasticConvLayer(
                    c=self.c, 
                    c_enc=self.c_enc,
                    zdim=self.zdim, 
                    resolution=self.current_resolution,
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes,
                    checkpoint=checkpoint
                )
            )
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c, checkpoint=checkpoint))

        if self.current_resolution < self.max_resolution:
            strides = 2 if is_conv else 4
            layer_list.append(Upsample(self.c, self.c_next, strides))
        self.layer_list = layer_list

    def p_sample(self, x_c, x_u=None, label=None, mweight=1.0, vweight=1.0, T=1.):
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                x_c, x_u = layer.p_sample(x_c, x_u, label, mweight=mweight, vweight=vweight, T=T) 
            else:
                x_c = layer(x_c)
                if x_u is not None:
                    x_u = layer(x_u)

        return x_c, x_u

    def forward(self, x, acts, label=None):
        shape = (x.shape[0], self.current_resolution, self.current_resolution, self.zdim)

        KLs = []
        SR_Losses = 0.
        i = 0
        for layer in self.layer_list:
            if isinstance(layer, StochasticConvLayer):
                eps = torch.randn(shape)
                x, kl, sr_loss = layer(eps, x, acts, label)
                KLs.append(kl)
                SR_Losses += sr_loss
                i += 1
            else:
                x = layer(x)
        return x, KLs, SR_Losses

class EncoderLevel(nn.Module):
    def __init__(self, c, nlayers, current_resolution, c_next, num_attention=0, w_scale=1.0, num_classes=0, checkpoint=False):
        super().__init__()
        self.c = c
        self.nlayers = nlayers
        self.current_resolution = current_resolution
        self.c_next = c_next
        self.num_attention = num_attention
        self.w_scale = w_scale
        self.num_classes = num_classes
        is_conv = (self.current_resolution >= 4)

        layer_list = nn.ModuleList([])
        attention_indices = get_evenly_spaced_indices(nlayers, self.num_attention)

        for i in range(nlayers):
            layer_list.append(
                ResBlock(
                    self.c, 
                    self.c, 
                    w_scale=self.w_scale, 
                    num_classes=self.num_classes, 
                    is_conv=is_conv,
                    checkpoint=checkpoint
                )
            )
            
            if i in attention_indices:
                layer_list.append(AttentionLayer(self.c, checkpoint=checkpoint))
            
        if self.current_resolution > 1:
            strides = 2 if self.current_resolution > 4 else 4
            layer_list.append(Downsample(self.c, self.c_next, strides)) #note: c_next might be None
        self.layer_list = layer_list

    def forward(self, x, label=None):
        acts = None
        for layer in self.layer_list:
            if isinstance(layer, Downsample):
                acts = x
            x = layer(x, label)

        if acts is None: acts = x
        return x, acts

class Encoder(nn.Module):
    def __init__(self, c, c_mult, nlayers, resolution, num_attention, num_classes, checkpoint_min_resolution):
        super().__init__()
        self.c = c
        self.c_mult = c_mult
        self.nlayers = nlayers
        self.resolution = resolution
        self.num_attention = num_attention
        self.num_classes = num_classes
    
        C = [self.c * mult for mult in self.c_mult]
        nlayers = self.nlayers
        w_scale = 1/np.sqrt(sum(nlayers))
        num_resolutions = len(nlayers)
        resolutions = get_resolutions(self.resolution, num_resolutions)

        self.in_proj = Conv2D(3, C[0], kernel_size=3)
        self.encoder_levels = nn.ModuleList([])
        for i in range(num_resolutions):
            c_next = None
            if i<num_resolutions-1:
                if C[i] != C[i+1]:
                    c_next = C[i+1] 
            
            self.encoder_levels.append(
                EncoderLevel(
                    C[i],
                    nlayers[i],
                    resolutions[i],
                    c_next,
                    self.num_attention[i],
                    w_scale,
                    self.num_classes,
                    checkpoint=(resolution >= checkpoint_min_resolution)
                )
            )
    
    def forward(self, img, label=None):
        x = self.in_proj(img)
        acts = []
        for layer in self.encoder_levels:
            x, acts_i = layer(x, label)
            acts.append(acts_i)
        
        return acts

class LowresEncodingUnet(nn.Module):
    def __init__(self, c, nlayers, num_classes):
        super().__init__()
        self.c = c
        self.nlayers = nlayers
        self.num_classes = num_classes
        self.w_scale = 1/np.sqrt(2. * sum(nlayers)).astype('float32')

        self.in_proj = Conv2D(3, c, kernel_size=1)
        self.down_resblocks = nn.ModuleList([])
        for i in range(len(nlayers)):
            modules = nn.ModuleList([])
            for j in range(nlayers[i]):
                modules.append(ResBlock(c, c, w_scale=self.w_scale, num_classes=self.num_classes))
            if i < len(nlayers)-1:
                modules.append(Downsample(None, None, strides=2))
            self.down_resblocks.append(modules)

        self.up_resblocks = nn.ModuleList([])
        for i in reversed(range(len(nlayers))):
            modules = nn.ModuleList([])
            for j in range(nlayers[i]):
                modules.append(Conv2D(c*2, c, kernel_size=3))
                modules.append(ResBlock(c, c, w_scale=self.w_scale, num_classes=self.num_classes))
            if i > 0:
                modules.append(Upsample(None, None, strides=2))
            self.up_resblocks.append(modules)
            
    def forward(self, x, label=None):
        x = self.in_proj(x)
        hs = []
        for level in self.down_resblocks:
            for layer in level:
                if isinstance(layer, Downsample):
                    hs.append(x)
                x = layer(x, label)
        hs.append(x)

        for level, i in zip(self.up_resblocks, reversed(range(len(self.up_resblocks)))):
            for layer in level:
                if isinstance(layer, nn.Conv2d):
                    concatted = torch.cat((x, hs[i]), dim=1)
                    x = layer(concatted)
                else:
                    x = layer(x, label)
        
        return x

class VAE(nn.Module):
    def __init__(self, c, c_enc, c_mult, zdim, nlayers, enc_nlayers, num_attention, resolution, num_final_blocks, num_classes=0, is_superres=False, superres_nlayers=None, checkpoint_min_resolution=32):
        super().__init__()
        self.c = c
        self.c_enc = c_enc
        self.c_mult = c_mult
        self.zdim = zdim
        self.nlayers = nlayers
        self.enc_nlayers = enc_nlayers
        self.num_attention = num_attention
        self.resolution = resolution
        self.num_final_blocks = num_final_blocks
        self.num_classes = num_classes
        self.is_superres = is_superres
        self.superres_nlayers = superres_nlayers
        self.cond = self.num_classes > 0
        self.encoder = Encoder(self.c_enc, self.c_mult, self.enc_nlayers, self.resolution, self.num_attention, self.num_classes, checkpoint_min_resolution=checkpoint_min_resolution)
    
        layer_list = []
        resolutions = get_resolutions(self.resolution, len(self.nlayers))
        C = [self.c * mult for mult in self.c_mult]
        C_enc = [self.c_enc * mult for mult in self.c_mult]
        w_scale = 1/np.sqrt(sum(self.nlayers))
        
        if self.is_superres:
            #do some processing of the lower-resolution image.
            self.process_lowres_image = LowresEncodingUnet(C[-1], self.superres_nlayers, self.num_classes)
        else:
            self.initial_x = nn.Parameter(torch.zeros([1, C[-1], resolutions[-1], resolutions[-1]])) #this is a permuted parameter. be sure to catch this one and permute it.

        for i in reversed(range(len(resolutions))):
            c_next = C[i-1] if i>0 else None
            if c_next == C[i]: c_next = None

            layer_list.append(
                DecoderLevel(
                    c=C[i],
                    c_enc=C_enc[i],
                    zdim=self.zdim,
                    nlayers=self.nlayers[i],
                    w_scale=w_scale,
                    num_classes=self.num_classes,
                    num_attention=self.num_attention[i],
                    current_resolution=resolutions[i],
                    max_resolution=resolutions[0],
                    c_next=c_next,
                    checkpoint=(resolutions[i] >= checkpoint_min_resolution)
                )
            )
        self.layer_list = nn.ModuleList(layer_list)

        self.output_blocks = nn.ModuleList([ResBlock(C[0], C[0], w_scale=w_scale) for _ in range(self.num_final_blocks)])
        self.outproj = Conv2D(self.c, 6, 1)

    @torch.no_grad()
    def p_sample(self, num=10, label=None, img_lr=None, mweight=1.0, vweight=1.0):

        if self.num_classes:
            uncond_label = torch.ones_like(label) * self.num_classes
        else:
            uncond_label = None

        is_guided = (mweight != 1 or vweight != 1)

        assert (img_lr is not None) == self.is_superres
        if self.is_superres: 
            c_aug = 0.1 if self.resolution==64 else 0.15 #TODO: get the c_aug from config
            img_lr += c_aug * torch.randn(img_lr.shape)
            x_c = self.process_lowres_image(img_lr, label)
            x_u = self.process_lowres_image(img_lr, uncond_label) if is_guided else None
        else:
            x_c = torch.tile(self.initial_x, [num, 1, 1, 1])
            x_u = torch.tile(self.initial_x, [num, 1, 1, 1]) if is_guided else None

        #update cond'l and uncond'l hidden states at each stochastic layer.
        for decoderlevel in self.layer_list:
            x_c, x_u = decoderlevel.p_sample(x_c, x_u, label, mweight=mweight, vweight=vweight)
            if torch.isnan(x_c.var()):
                raise RuntimeError()

        #now we only need to process the conditional part to output images.
        for block in self.output_blocks:
            x_c = block(x_c)

        x_c = self.outproj(x_c)

        return x_c[:, :3, ...] #ignore variance output of the model as it's not needed for visualizing images.
    
    def forward(self, img, label=None, img_lr=None):
        #randomly drop class label in the prior w/ 10% probability to enable CFG
        if self.cond:
            uncond_label = torch.full_like(label, self.num_classes)
            mask = torch.greater(torch.rand(label.shape), 0.9).int()
            cf_guidance_label = label*(1-mask) + mask*uncond_label

            generator_label = (label, cf_guidance_label)
        else: 
            label, generator_label = None, None

        assert (img_lr is not None) == self.is_superres
        if self.is_superres:
            x = self.process_lowres_image(img_lr, cf_guidance_label)
        else:
            x = torch.tile(self.initial_x, [img.shape[0], 1, 1, 1])

        acts = self.encoder(img, label=label)
        KLs = []
        SR_Losses = 0.

        for i, decoderlevel in enumerate(self.layer_list):
            j = -i+len(self.nlayers)-1
            x, kls, sr_losses = decoderlevel(x, acts[j], generator_label)
            KLs.extend(kls)
            SR_Losses += sr_losses
                
        for block in self.output_blocks:
            x = block(x)     
        x = self.outproj(x)
        return x, KLs, SR_Losses