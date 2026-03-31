import torch
from torch.optim import Optimizer


class PerturbedGradientDescent(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        defaults = dict(lr=lr, mu=mu)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for param, global_param in zip(group["params"], global_params):
                if param.grad is None:
                    continue
                global_param = global_param.to(device)
                update = param.grad.data + group["mu"] * (param.data - global_param.data)
                param.data.add_(update, alpha=-group["lr"])
