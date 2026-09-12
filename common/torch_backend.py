"""Native PyTorch tensors and autograd. No JAX imports."""

import torch

from common.tree import flatten, map_tree, unflatten


class _IQLVirtualSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * 0.5 / torch.sqrt(x.clamp(min=1e-12))


class Backend:
    name = "torch"

    def __init__(self, device="cpu", float64=False):
        self.device = torch.device(device)

    def array(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, device=self.device)

    @staticmethod
    def numpy(value):
        return value.detach().cpu().numpy()

    @staticmethod
    def stop(value):
        return value.detach()

    @staticmethod
    def grad(fn, params):
        leaves = flatten(params)
        active = {
            k: (v if v.requires_grad else v.detach().requires_grad_(True))
            for k, v in leaves.items()
        }
        loss = fn(unflatten(active))
        grads = torch.autograd.grad(
            loss, tuple(active.values()), create_graph=True, allow_unused=True
        )
        return loss, unflatten(
            {
                k: torch.zeros_like(v) if g is None else g
                for (k, v), g in zip(active.items(), grads)
            }
        )

    @staticmethod
    def relu(x):
        return torch.relu(x)

    @staticmethod
    def linear(x, weight, bias):
        # Preserve the upstream nn.Linear fused bias addition.
        return torch.nn.functional.linear(x, weight.T, bias)

    @staticmethod
    def layer_norm(x, scale, offset, eps):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), scale, offset, eps)

    tanh = staticmethod(torch.tanh)
    cos = staticmethod(torch.cos)
    exp = staticmethod(torch.exp)
    log = staticmethod(torch.log)
    abs = staticmethod(torch.abs)
    minimum = staticmethod(torch.minimum)
    where = staticmethod(torch.where)
    float32 = staticmethod(lambda x: x.to(torch.float32))
    cast_like = staticmethod(lambda x, ref: x.to(dtype=ref.dtype))
    # Native torch.optim computes the scalar bias corrections in Python double.
    adam_correction = staticmethod(
        lambda beta, count: 1 - beta ** count.to(torch.float64)
    )
    virtual_sqrt = staticmethod(_IQLVirtualSqrt.apply)

    @staticmethod
    def clip(x, low=None, high=None):
        return torch.clamp(x, min=low, max=high)

    @staticmethod
    def cat(xs):
        return torch.cat(xs, dim=-1)

    @staticmethod
    def safe_sqrt(x):
        # Exact forward value, zero derivative at zero. Avoid hidden 0*inf.
        positive = x > 0
        safe = torch.where(positive, x, torch.ones_like(x))
        return torch.where(positive, torch.sqrt(safe), torch.zeros_like(x))

    @staticmethod
    def softplus(x):
        return torch.nn.functional.softplus(x)

    def finish(self, tree):
        return map_tree(lambda x: x.detach(), tree)

    @staticmethod
    def compile(fn):
        return fn
