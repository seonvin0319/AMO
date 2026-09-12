"""Differentiable Adam with exact zero handling for hypergradients."""

import numpy as np

from common.tree import map_tree


def init_adam(params):
    return {
        "count": np.asarray(0, np.int32),
        "m": map_tree(np.zeros_like, params),
        "v": map_tree(np.zeros_like, params),
    }


def adam(ops, params, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8, sqrt_fn=None):
    sqrt_fn = ops.safe_sqrt if sqrt_fn is None else sqrt_fn
    count = state["count"] + 1
    correction1 = ops.adam_correction(beta1, count)
    correction2 = ops.adam_correction(beta2, count)
    m = map_tree(lambda a, g: beta1 * a + (1 - beta1) * g, state["m"], grads)
    v = map_tree(lambda a, g: beta2 * a + (1 - beta2) * g * g, state["v"], grads)
    out = map_tree(
        lambda p, a, b: p
        - lr
        * (a / ops.cast_like(correction1, a))
        / (sqrt_fn(b / ops.cast_like(correction2, b)) + eps),
        params,
        m,
        v,
    )
    return out, {"count": count, "m": m, "v": v}


def target_update(ops, params, target, tau):
    return map_tree(lambda p, t: ops.stop(tau * p + (1 - tau) * t), params, target)
