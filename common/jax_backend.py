"""Native JAX arrays, autodiff and JIT. No PyTorch imports."""

import math

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def _iql_virtual_sqrt(x):
    return jnp.sqrt(x)


@_iql_virtual_sqrt.defjvp
def _iql_virtual_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return jnp.sqrt(x), dx * 0.5 / jnp.sqrt(jnp.maximum(x, 1e-12))


class Backend:
    name = "jax"

    def __init__(self, device="cpu", float64=False):
        if float64:
            # IQL+AMO's log temperatures and optimizer moments are float64.
            jax.config.update("jax_enable_x64", True)
        platform = "gpu" if device.startswith("cuda") else device.split(":")[0]
        index = int(device.split(":")[1]) if ":" in device else 0
        self.device = jax.devices(platform)[index]

    def array(self, value):
        return jax.device_put(jnp.asarray(value), self.device)

    @staticmethod
    def numpy(value):
        return np.asarray(jax.device_get(value))

    stop = staticmethod(jax.lax.stop_gradient)
    grad = staticmethod(lambda fn, params: jax.value_and_grad(fn)(params))
    linear = staticmethod(lambda x, weight, bias: x @ weight + bias)
    relu = staticmethod(jax.nn.relu)
    tanh = staticmethod(jnp.tanh)
    cos = staticmethod(jnp.cos)
    exp = staticmethod(jnp.exp)
    log = staticmethod(jnp.log)
    abs = staticmethod(jnp.abs)
    minimum = staticmethod(jnp.minimum)
    where = staticmethod(jnp.where)
    float32 = staticmethod(lambda x: x.astype(jnp.float32))
    cast_like = staticmethod(lambda x, ref: x.astype(ref.dtype))

    @staticmethod
    def adam_correction(beta, count):
        # Avoid cancellation in float32 (1 - 0.999**1) when x64 is disabled.
        # Compute the scalar log in Python double before casting during JAX math.
        return -jnp.expm1(count * math.log(beta)) if beta else jnp.ones_like(count)

    virtual_sqrt = staticmethod(_iql_virtual_sqrt)
    clip = staticmethod(lambda x, low=None, high=None: jnp.clip(x, low, high))
    cat = staticmethod(lambda xs: jnp.concatenate(xs, axis=-1))
    softplus = staticmethod(jax.nn.softplus)

    @staticmethod
    def safe_sqrt(x):
        positive = x > 0
        safe = jnp.where(positive, x, jnp.ones_like(x))
        return jnp.where(positive, jnp.sqrt(safe), jnp.zeros_like(x))

    @staticmethod
    def layer_norm(x, scale, offset, eps):
        centered = x - x.mean(axis=-1, keepdims=True)
        return (
            centered
            / Backend.safe_sqrt(
                (centered * centered).mean(axis=-1, keepdims=True) + eps
            )
            * scale
            + offset
        )

    @staticmethod
    def finish(tree):
        return tree

    @staticmethod
    def compile(fn):
        return jax.jit(fn, static_argnames=("actor_step", "meta_step"))
