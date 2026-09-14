"""External target-noise injection for tape-driven TD3+BC updates."""

import copy

import numpy as np
import pytest

from common.agent import make_agent
from common.config import load_config
from common.tree import flatten, map_tree


def _batch(seed=9, n=6):
    rng = np.random.default_rng(seed)
    out = {
        k: rng.normal(size=shape).astype(np.float32)
        for k, shape in (
            ("observations", (n, 3)),
            ("next_observations", (n, 3)),
            ("actions", (n, 2)),
            ("next_actions", (n, 2)),
            ("rewards", (n, 1)),
        )
    }
    out["actions"] = np.tanh(out["actions"])
    out["next_actions"] = np.tanh(out["next_actions"])
    out["terminals"] = np.asarray([[0], [1], [0], [0], [1], [0]], np.float32)[:n]
    return out


def _config():
    return load_config(
        "td3_bc", "halfcheetah-medium-v2", overrides={"hidden_dim": 8, "batch_size": 6}
    )


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_external_noise_matches_internal_rng_draw(backend):
    """Pre-drawing noise with the agent RNG equals passing it via noise=."""
    steps = 8
    batches = [_batch(seed=20 + i) for i in range(steps)]

    internal = make_agent("td3_bc", backend, 3, 2, _config(), seed=11)
    external = make_agent("td3_bc", backend, 3, 2, _config(), seed=11)

    # Capture the exact noise sequence the internal agent would consume.
    probe = make_agent("td3_bc", backend, 3, 2, _config(), seed=11)
    noises = []
    for batch in batches:
        n = len(batch["actions"])
        noises.append(
            {
                "target": probe.rng.standard_normal((n, probe.action_dim)).astype(
                    np.float32
                )
            }
        )

    metrics_i = []
    metrics_e = []
    for batch, noise in zip(batches, noises):
        metrics_i.append(internal.update(batch, return_metrics=True))
        metrics_e.append(external.update(batch, return_metrics=True, noise=noise))

    for mi, me in zip(metrics_i, metrics_e):
        assert mi.keys() == me.keys()
        for key in mi:
            np.testing.assert_allclose(mi[key], me[key], rtol=0, atol=0)

    si = flatten(map_tree(internal.ops.numpy, internal.state))
    se = flatten(map_tree(external.ops.numpy, external.state))
    assert si.keys() == se.keys()
    for key in si:
        np.testing.assert_array_equal(si[key], se[key], err_msg=key)
    assert internal.steps == external.steps == steps
    # External path must not advance RNG; probe RNG (which only drew noise)
    # matches the internal agent's remaining RNG after the same draws.
    assert (
        internal.rng.bit_generator.state == probe.rng.bit_generator.state
    )
    # External agent RNG is still at the initial post-make_agent state of a twin.
    twin = make_agent("td3_bc", backend, 3, 2, _config(), seed=11)
    assert external.rng.bit_generator.state == twin.rng.bit_generator.state


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_external_noise_repeatability(backend):
    batch = _batch()
    noise = {"target": np.random.default_rng(0).standard_normal((6, 2)).astype(np.float32)}
    a = make_agent("td3_bc", backend, 3, 2, _config(), seed=3)
    b = make_agent("td3_bc", backend, 3, 2, _config(), seed=3)
    ma = a.update(copy.deepcopy(batch), return_metrics=True, noise=noise)
    mb = b.update(copy.deepcopy(batch), return_metrics=True, noise=noise)
    for key in ma:
        np.testing.assert_allclose(ma[key], mb[key], rtol=0, atol=0)
    sa = flatten(map_tree(a.ops.numpy, a.state))
    sb = flatten(map_tree(b.ops.numpy, b.state))
    for key in sa:
        np.testing.assert_array_equal(sa[key], sb[key], err_msg=key)


def test_external_noise_rejects_bad_shape():
    agent = make_agent("td3_bc", "torch", 3, 2, _config(), seed=0)
    with pytest.raises(ValueError, match="shape"):
        agent.update(
            _batch(),
            noise={"target": np.zeros((3, 2), np.float32)},
            return_metrics=False,
        )
