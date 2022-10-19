import torch
from visual_vae.torch.torch_utils import compute_mvn_kl, weighted_kl
    
def training_losses_fn(train_inputs, model, device_count, global_sr_weight, sigma_q, rate_schedule): 
    img, label, img_lr = train_inputs

    model_out, unweighted_kls, sr_loss = model(img, label, img_lr)
    sr_loss *= global_sr_weight

    mean_output, var_output = torch.chunk(model_out, 2, dim=1)
    qvar = (sigma_q ** 2)
    neg_logpx_z = compute_mvn_kl(img, qvar, mean_output, var_output.exp(), rdim=(0,1,2,3)) #shape is ()

    total_kl_per_image = torch.sum(torch.stack(unweighted_kls, dim=-1), dim=-1)  #(B, )
    KL_Loss = torch.zeros_like(neg_logpx_z)
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
    def __init__(self, model, optimizer, ema_decay, startlr, maxlr, minlr, warmup_steps, decay_steps):
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device
        self.iterations = torch.zeros((), device=self.device, dtype=torch.int32)
        self.skips = 0
        self.ema_decay = ema_decay
        self.startlr = startlr
        self.maxlr = maxlr
        self.minlr = minlr
        self.warmup_steps = warmup_steps
        self.decay_steps = torch.tensor(decay_steps, device=self.device, dtype=torch.int32)

        if self.ema_decay is not None:
            self.ema_parameters = [p.data.clone() for p in self.model.parameters()]
        else:
            self.ema_parameters = None
    
    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "ema_parameters": self.ema_parameters,
            "iterations": self.iterations,
            "skips": self.skips
        }

    def load_state_dict(self, state_dict):
        if sorted(state_dict.keys()) == sorted(["model", "optimizer", "ema_parameters", "iterations", "skips"]):
            self.model.load_state_dict(
                state_dict["model"]
            )
            if "param_groups" in state_dict["optimizer"].keys():
                self.optimizer.load_state_dict(
                    state_dict["optimizer"]
                )
            else:
                print("The restored optimizer had an empty state (there were no param_groups)")
            if state_dict["ema_parameters"] is not None:
                self.ema_parameters = [w for w in state_dict.pop("ema_parameters")]
            else:
                #the current state dict does not have EMA parameters, so we set the EMA parameters to be the restored weights.
                self.ema_parameters = [p.data.clone() for p in self.model.parameters()]

            self.iterations = state_dict["iterations"]
            self.skips = state_dict["skips"]
        else:
            #try to restore from a model-only state dict ()
            if sorted(self.model.state_dict().keys()) == sorted(state_dict.keys()):
                self.model.load_state_dict(state_dict)
                print("The loaded state restored only model parameters, but no optimizer state.")
            else:
                print("Could not load the model state dict. Continuing training without the restored state dict.")

    def rate(self):
        step = self.iterations
        step = torch.minimum(torch.tensor(step, device=self.device), self.decay_steps)
        startlr, maxlr, minlr = self.startlr, self.maxlr, self.minlr
        warmup = startlr + step/self.warmup_steps * (maxlr - startlr)

        decay_factor = 0.5 * (1 + torch.cos(3.1416 * step/self.decay_steps))
        decay_factor = (1 - minlr/maxlr) * decay_factor + minlr/maxlr
        lr = maxlr * decay_factor
        return torch.minimum(warmup, lr)
    
    def get_iteration(self):
        return self.iterations.item()

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.iterations += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate

        if self.ema_parameters is not None:
            for i, ema_param, model_param in zip(range(len(self.ema_parameters)), self.ema_parameters, self.model.parameters()):
                self.ema_parameters[i] = self.ema_decay * ema_param + (1 - self.ema_decay) * model_param
    
        self.optimizer.zero_grad()

#Metrics related utils
class MeanObject(object):
    def __init__(self):
        self.reset_states()
    
    def reset_states(self):
        self._mean = 0.
        self._count = 0
        
    def update(self, new_entry):
        if not isinstance(new_entry, (int, float)):
            assert new_entry.shape == ()
        self._count = self._count + 1
        self._mean = (1-1/self._count)*self._mean + new_entry/self._count
        
    def result(self):
        return self._mean
        
class Metrics(object):
    def __init__(self, metric_names, accelerator):
        self.names = metric_names
        self.accelerator = accelerator
        self._metric_dict = {
            name: MeanObject() for name in self.names
        }
        
    
    def __repr__(self):
        metric_dict = {}
        for k, v in self._metric_dict.items():
            result = self.accelerator.gather(v.result())
            if not isinstance(result, (int, float)):
                result = result.mean().item()
            metric_dict[k] = result

        return repr(metric_dict)

    def update(self, new_metrics):
        for name in self.names:
            if name in new_metrics.keys():
                self._metric_dict[name].update(new_metrics[name])
        
    def reset_states(self):
        for name in self.names:
            self._metric_dict[name].reset_states()