"""Guard early Adam updates against float32 bias-correction cancellation."""

import importlib
from contextlib import contextmanager

import jax
import numpy as np
import pytest
import torch

from common.optim import adam, init_adam
from common.tree import map_tree


@contextmanager
def x64_disabled():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_float32_adam_matches_native_torch(backend):
    # A separate IQL agent must not be needed to enable x64 before TD3 is correct.
    with x64_disabled():
        ops = importlib.import_module(f"common.{backend}_backend").Backend()
        params = {"x": np.zeros(4, np.float32)}
        state = map_tree(ops.array, init_adam(params))
        params = map_tree(ops.array, params)
        original = torch.nn.Parameter(torch.zeros(4))
        optimizer = torch.optim.Adam([original], lr=3e-4)

        def update(p, g, s):
            return adam(ops, p, g, s, 3e-4)

        if backend == "jax":
            update = jax.jit(update)
        for step in range(5):
            gradient = np.asarray([1e-8, 3e-4, -0.9, 3.0], np.float32) * (0.8**step)
            params, state = update(params, {"x": ops.array(gradient)}, state)
            original.grad = torch.from_numpy(gradient.copy())
            optimizer.step()
            np.testing.assert_allclose(
                ops.numpy(params["x"]),
                original.detach().numpy(),
                atol=5e-11,
                rtol=1e-6,
            )
            assert int(ops.numpy(state["count"])) == step + 1
            assert ops.numpy(params["x"]).dtype == np.float32
