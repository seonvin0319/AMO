"""Numerical contracts, data boundaries and exact resume behavior."""

import numpy as np
import pytest

from common.agent import make_agent
from common.config import load_config
from common.data import load_dataset, qlearning_dataset
from common.tree import flatten, map_tree

RUNNABLE = ("td3_bc", "wpc", "aspc", "a2pr", "rebrac", "iql", "td3_amo", "iql_amo")


def batch(seed=9, n=6):
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


def config(algorithm):
    c = load_config(
        algorithm, "halfcheetah-medium-v2", overrides={"hidden_dim": 8, "batch_size": 6}
    )
    if algorithm in ("aspc", "td3_amo"):
        c["meta_interval"] = 2
    if algorithm == "a2pr":
        c["vae_hidden_dim"] = 8
    if algorithm == "iql_amo":
        c.update(meta_warmup_steps=0, meta_interval=2, outer_batch_size=6)
    return c


@pytest.mark.parametrize("algorithm", RUNNABLE)
def test_backend_parity_and_resume(algorithm, tmp_path):
    agents = {
        b: make_agent(algorithm, b, 3, 2, config(algorithm), seed=7)
        for b in ("torch", "jax")
    }
    for _ in range(4):
        metrics = {b: a.update(batch(), batch(12)) for b, a in agents.items()}
        assert metrics["torch"].keys() == metrics["jax"].keys()
        for key in metrics["torch"]:
            np.testing.assert_allclose(
                metrics["torch"][key],
                metrics["jax"][key],
                rtol=2e-3,
                atol=3e-5,
                err_msg=f"{algorithm}:{key}",
            )
        states = {b: flatten(map_tree(a.ops.numpy, a.state)) for b, a in agents.items()}
        assert states["torch"].keys() == states["jax"].keys()
        for key in states["torch"]:
            if key.endswith("/count"):
                np.testing.assert_array_equal(states["jax"][key], states["torch"][key])
                continue
            np.testing.assert_allclose(
                states["jax"][key],
                states["torch"][key],
                rtol=3e-5,
                atol=5e-6,
                err_msg=f"{algorithm}:{key}",
            )
    for backend, agent in agents.items():
        if algorithm == "iql_amo":
            for key, value in flatten(agent.state["p"]).items():
                expected = np.float64 if key.startswith("scale_") else np.float32
                assert agent.ops.numpy(value).dtype == expected, key
        path = tmp_path / f"{algorithm}-{backend}.npz"
        agent.save(path, {"marker": 31})
        restored = make_agent(algorithm, backend, 3, 2, config(algorithm), seed=888)
        assert restored.load(path) == {"marker": 31}
        assert restored.steps == agent.steps
        for _ in range(2):
            agent.update(batch(3), batch(11))
            restored.update(batch(3), batch(11))
        before = flatten(map_tree(agent.ops.numpy, agent.state))
        after = flatten(map_tree(restored.ops.numpy, restored.state))
        for key in before:
            np.testing.assert_array_equal(
                before[key], after[key], err_msg=f"{algorithm}:{backend}:{key}"
            )


def test_outer_batch_is_required_before_mutation():
    agent = make_agent("td3_amo", "torch", 3, 2, config("td3_amo"))
    agent.update(batch())
    before = agent.steps
    with pytest.raises(ValueError, match="independently sampled"):
        agent.update(batch())
    assert agent.steps == before


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_zero_rms_has_zero_finite_gradient(backend):
    from importlib import import_module

    o = import_module(f"common.{backend}_backend").Backend()
    p = {"x": o.array(np.zeros(4, np.float32))}
    loss, grad = o.grad(lambda p: o.safe_sqrt((p["x"] ** 2).mean()), p)
    assert float(o.numpy(loss)) == 0
    np.testing.assert_array_equal(o.numpy(grad["x"]), 0)


def test_raw_timeouts_preserve_next_action_alignment():
    raw = {
        "observations": np.arange(18).reshape(6, 3),
        "actions": np.arange(12).reshape(6, 2),
        "rewards": np.ones(6),
        "terminals": np.array([0, 0, 0, 1, 0, 0]),
        "timeouts": np.array([0, 1, 0, 0, 0, 1]),
    }
    out = qlearning_dataset(raw)
    np.testing.assert_array_equal(out["actions"], raw["actions"][[0, 2, 3, 4]])
    np.testing.assert_array_equal(out["next_actions"], raw["actions"][[1, 3, 4, 5]])
    np.testing.assert_array_equal(out["terminals"], [0, 0, 1, 0])


def test_rebrac_rejects_missing_next_actions(tmp_path):
    data = batch()
    del data["next_actions"]
    path = tmp_path / "dataset.npz"
    np.savez(path, **data)
    with pytest.raises(ValueError, match="next_actions"):
        load_dataset(path, rebrac=True)


def test_unknown_config_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown options"):
        load_config("td3_amo", "hopper-medium-v2", overrides={"T_lrr": 0.001})


def test_iql_meta_warmup_and_independent_outer():
    c = config("iql_amo")
    c.update(meta_warmup_steps=3, meta_interval=2)
    agent = make_agent("iql_amo", "torch", 3, 2, c)
    for _ in range(4):
        agent.update(batch())
    assert float(agent.state["opt"]["scale_E"]["count"]) == 0
    with pytest.raises(ValueError, match="independently sampled"):
        agent.update(batch())
    assert agent.steps == 4
    agent.update(batch(), batch(12))
    assert float(agent.state["opt"]["scale_E"]["count"]) == 1
    assert agent.state["p"]["scale_E"]["rho"].dtype.is_floating_point
