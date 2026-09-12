"""Functional MLPs with identical parameter layouts on both backends."""

import numpy as np


def init_mlp(
    rng,
    input_dim,
    output_dim,
    hidden_dim=256,
    depth=2,
    layernorm=False,
    special_init=False,
    output_bound=0.003,
):
    params = {}
    dims = [input_dim] + [hidden_dim] * depth + [output_dim]
    for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:])):
        last = i == depth
        bound = output_bound if last and special_init else din**-0.5
        p = {
            "w": rng.uniform(-bound, bound, (din, dout)).astype(np.float32),
            "b": (
                np.full(dout, 0.1)
                if special_init and not last
                else rng.uniform(-bound, bound, dout)
            ).astype(np.float32),
        }
        if layernorm and not last:
            p.update(scale=np.ones(dout, np.float32), offset=np.zeros(dout, np.float32))
        params[f"layer{i}"] = p
    return params


def mlp(ops, params, x, tanh=False, ln_eps=1e-5):
    for i in range(len(params)):
        p = params[f"layer{i}"]
        x = ops.linear(x, p["w"], p["b"])
        if i < len(params) - 1:
            x = ops.relu(x)
            if "scale" in p:
                x = ops.layer_norm(x, p["scale"], p["offset"], ln_eps)
    return ops.tanh(x) if tanh else x
