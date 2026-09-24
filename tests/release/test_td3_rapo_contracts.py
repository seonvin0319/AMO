"""TD3 RAPO: critic target from π_E (TD3+BC) and L_α_B = L2_RMS only."""

import numpy as np
import pytest

from common.agent import make_agent
from common.config import load_config
from common.tree import flatten, map_tree

from tests.release.test_contracts import batch, config as base_config


def _agent(backend, **overrides):
    c = base_config("td3_amo")
    c.update(overrides)
    return make_agent("td3_amo", backend, 3, 2, c, seed=7)


def _leaf_max_abs(left, right):
    diffs = []
    for key in left:
        diffs.append(np.max(np.abs(left[key] - right[key])))
    return max(diffs)


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_bootstrap_scale_l2_matches_logged_l2(backend):
    agent = _agent(backend, bootstrap_loss="l2_rms", meta_interval=2, policy_freq=1)
    metrics = None
    for _ in range(2):
        metrics = agent.update(batch(), batch(12))
    assert metrics is not None
    np.testing.assert_allclose(
        metrics["L_alpha_B"], metrics["L2_RMS_B"], rtol=2e-3, atol=3e-5
    )
    assert abs(metrics["L_alpha_B"] - (metrics["L1_B"] + metrics["L2_RMS_B"])) > 1e-4


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_bootstrap_scale_l1_l2_is_sum(backend):
    agent = _agent(backend, bootstrap_loss="l1_l2_rms", meta_interval=2, policy_freq=1)
    metrics = None
    for _ in range(2):
        metrics = agent.update(batch(), batch(12))
    np.testing.assert_allclose(
        metrics["L_alpha_B"],
        metrics["L1_B"] + metrics["L2_RMS_B"],
        rtol=2e-3,
        atol=3e-5,
    )


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_critic_target_actor_polyak_follows_execution(backend):
    agent = _agent(
        backend,
        critic_target="actor",
        tau=1.0,
        policy_freq=1,
        meta_interval=2,
        alpha_E=0.5,
        alpha_B=8.0,
    )
    for _ in range(8):
        agent.update(batch(), batch(12))
    to_np = lambda tree: flatten(map_tree(agent.ops.numpy, tree))
    target = to_np(agent.state["target"]["actor"])
    actor = to_np(agent.state["p"]["actor"])
    bootstrap = to_np(agent.state["p"]["bootstrap"])
    assert _leaf_max_abs(target, actor) < 1e-5
    assert _leaf_max_abs(actor, bootstrap) > 1e-5
    assert _leaf_max_abs(target, bootstrap) > 1e-5


@pytest.mark.parametrize("backend", ("torch", "jax"))
def test_critic_target_bootstrap_polyak_follows_bootstrap(backend):
    agent = _agent(
        backend,
        critic_target="bootstrap",
        tau=1.0,
        policy_freq=1,
        meta_interval=2,
        alpha_E=0.5,
        alpha_B=8.0,
    )
    for _ in range(8):
        agent.update(batch(), batch(12))
    to_np = lambda tree: flatten(map_tree(agent.ops.numpy, tree))
    target = to_np(agent.state["target"]["actor"])
    actor = to_np(agent.state["p"]["actor"])
    bootstrap = to_np(agent.state["p"]["bootstrap"])
    assert _leaf_max_abs(target, bootstrap) < 1e-5
    assert _leaf_max_abs(actor, bootstrap) > 1e-5
    assert _leaf_max_abs(target, actor) > 1e-5


def test_rapo_yaml_loads():
    c = load_config(
        "td3_amo",
        "halfcheetah-medium-v2",
        path="configs/td3_rapo_tinit5_tlr1e-3.yaml",
    )
    assert c["alpha_E"] == 5
    assert c["alpha_B"] == 5
    assert c["alpha_lr"] == 0.001
    assert c["critic_target"] == "actor"
    assert c["bootstrap_loss"] == "l2_rms"
    assert c["execution_meta_loss"] == "le"
    assert c["critic_depth"] == 2
    assert c["critic_layernorm"] is False
