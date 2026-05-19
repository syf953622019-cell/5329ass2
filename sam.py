import torch


class SAM(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimisation optimizer.

    This implementation uses two steps:
    1. first_step: move weights to a nearby high-loss point.
    2. second_step: return weights and update using the gradient at that point.
    """

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, "rho should be non-negative."

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()

        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["adaptive"]:
                    e_w = torch.pow(p, 2) * p.grad * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)

                p.add_(e_w)
                self.state[p]["e_w"] = e_w

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                p.sub_(self.state[p]["e_w"])

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    def step(self):
        raise NotImplementedError("SAM requires first_step and second_step.")

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if group["adaptive"]:
                        norms.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                    else:
                        norms.append(p.grad.norm(p=2).to(shared_device))

        return torch.norm(torch.stack(norms), p=2)
