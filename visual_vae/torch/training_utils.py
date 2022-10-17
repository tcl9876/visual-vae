from pyexpat import model
import torch
from visual_vae.torch.torch_utils import compute_mvn_kl, weighted_kl
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, Adamax
    
def training_losses_fn(train_inputs, model, device_count, global_sr_weight, sigma_q, rate_schedule): 
    img, label, img_lr = train_inputs

    model_out, unweighted_kls, sr_loss = model(img, label, img_lr)
    sr_loss *= global_sr_weight

    mean_output, var_output = torch.chunk(model_out, 2, axis=-1)
    qvar = (sigma_q ** 2)
    neg_logpx_z = compute_mvn_kl(img, qvar, mean_output, var_output.exp(), raxis=None) #shape is ()

    total_kl_per_image = torch.sum(torch.stack(unweighted_kls, axis=-1), axis=-1)  #(B, )
    KL_Loss = torch.float32(0.)
    for i, k in enumerate(unweighted_kls):
        w = rate_schedule[i]
        k_weighted = weighted_kl(k, total_kl_per_image, w, 2.*w)
        KL_Loss += k_weighted    

    per_replica_bs = img.shape[0] / device_count
    neg_logpx_z /= per_replica_bs
    KL_Loss /= per_replica_bs
    total_loss = neg_logpx_z + KL_Loss + sr_loss
    metrics = {'loss': total_loss, 'distortion term': neg_logpx_z, 'kl term': KL_Loss, 'sr loss': sr_loss}
    return total_loss, metrics

class TrainState:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def state_dict(self):
        return {
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(
            state_dict["model"]
        )
        self.optimizer.load_state_dict(
            state_dict["optimizer"]
        )
                
        self.scheduler.load_state_dict(
            state_dict["scheduler"]
        )
        
    def get_step(self):
        return self.scheduler.last_epoch

def make_optimizer(model, maxlr, opt_type, ema_decay, betas):
    optimizer_class = Adamax if opt_type.lower() == "adamax" else Adam
    if ema_decay is None:
        return optimizer_class(model.parameters(), maxlr, betas=betas)
    
    class EMAStateOptimizer(optimizer_class):
        def __init__(self, model, maxlr, betas):
            super().__init__(model.parameters(), maxlr, betas=betas)
            
            self.model = model
            self.ema_parameters = [p.data.clone() for p in self.model.parameters()]
            self.ema_decay = ema_decay

        def step(self):
            super().step()
            with torch.no_grad():
                for i, ema_param, model_param in zip(range(len(self.ema_parameters)), self.ema_parameters, self.model.parameters()):
                    self.ema_parameters[i] = self.ema_decay * ema_param + (1 - self.ema_decay) * model_param

        def state_dict(self):
            sd = super().state_dict()
            sd["ema_parameters"] = self.ema_parameters
            return sd
        
        def load_state_dict(self, state_dict):
            if "ema_parameters" in state_dict.keys():
                self.ema_parameters = [w for w in state_dict.pop("ema_parameters")]
                super().load_state_dict(state_dict)
            else:
                #the current state dict does not have EMA parameters, so we set the EMA parameters to be the restored weights.
                super().load_state_dict(state_dict)
                self.ema_parameters = [w.data.clone() for w in self.model_parameters]

    return EMAStateOptimizer(model, maxlr, betas)

#https://stackoverflow.com/questions/62724824/what-is-the-param-last-epoch-on-pytorch-optimizers-schedulers-is-for
#supposedly, "last_epoch" means "last iteration" as it supposedly increments upon a call to optimizer.step()
class CosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, startlr, maxlr, minlr, warmup_steps, decay_steps, last_epoch=-1):
        
        self.startlr = startlr
        self.maxlr = maxlr
        self.minlr = minlr
        self.warmup_steps = max(warmup_steps, 1)
        self.decay_steps = decay_steps

        super(CosineDecayScheduler, self).__init__(optimizer=optimizer, last_epoch=last_epoch)

    def step_func(self):
        step = self.last_epoch
        step = torch.minimum(step, self.decay_steps)
        startlr, maxlr, minlr = self.startlr, self.maxlr, self.minlr
        warmup = startlr + step/self.warmup_steps * (maxlr - startlr)

        decay_factor = 0.5 * (1 + torch.cos(3.1416 * step/self.decay_steps))
        decay_factor = (1 - minlr/maxlr) * decay_factor + minlr/maxlr
        lr = maxlr * decay_factor
        return torch.minimum(warmup, lr)

    def get_lr(self):
        return [self.step_func() for v in self.base_lrs]
