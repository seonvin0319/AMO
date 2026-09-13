"""Execution optimizations must preserve updates, failure detection and resume."""

import copy
import json

import jax
import numpy as np
import pytest

from common.agent import make_agent
from common.data import ReplayBuffer
from common.tree import flatten
from tests.release.test_contracts import RUNNABLE, batch, config


def assert_state_equal(a, b):
    left, right = flatten(a.ops.numpy_tree(a.state)), flatten(b.ops.numpy_tree(b.state))
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)
    assert a.steps == b.steps
    assert a.rng.bit_generator.state == b.rng.bit_generator.state


@pytest.mark.parametrize("algorithm", RUNNABLE)
def test_deferred_metrics_preserve_updates_and_resume(algorithm, tmp_path):
    c = config(algorithm)
    sync = make_agent(algorithm, "jax", 3, 2, c, seed=7)
    deferred = make_agent(algorithm, "jax", 3, 2, c, seed=7)
    for step in range(4):
        expected = sync.update(batch(step), batch(step + 100))
        assert (
            deferred.update(batch(step), batch(step + 100), return_metrics=False)
            is None
        )
    assert deferred.get_metrics() == expected
    assert_state_equal(sync, deferred)
    deferred.update(batch(5), batch(105), return_metrics=False)
    path = tmp_path / "state.npz"
    deferred.save(path)
    restored = make_agent(algorithm, "jax", 3, 2, c, seed=999)
    restored.load(path)
    for step in range(6, 8):
        deferred.update(batch(step), batch(step + 100), return_metrics=False)
        restored.update(batch(step), batch(step + 100))
    assert_state_equal(deferred, restored)


@pytest.mark.parametrize("backend", ("jax", "torch"))
def test_unlogged_transient_nonfinite_is_reported_and_not_saved(backend, tmp_path):
    agent = make_agent("iql", backend, 3, 2, config("iql"))
    valid = tmp_path / "valid.npz"
    agent.save(valid)

    # The parameters stay finite and the final logged loss recovers. Checking
    # only the latest metrics would miss both the NaN at 2 and infinity at 3.
    def transient(state, batch, outer, noise, **flags):
        step = noise["step"]
        loss = agent.ops.where(
            step == 2,
            step * 0 + float("nan"),
            agent.ops.where(step == 3, step * 0 + float("inf"), step * 0 + 1),
        )
        return state, {"loss": loss}

    agent.step = transient
    for _ in range(4):
        agent.update(batch(), return_metrics=False)
    for check in (agent.get_metrics, lambda: agent.save(tmp_path / "invalid.npz")):
        with pytest.raises(FloatingPointError, match="at step 2;"):
            check()
    assert not (tmp_path / "invalid.npz").exists()
    agent.load(valid)
    assert agent.update(batch()) == {"loss": 1.0}


def test_actor_jit_uses_current_parameters_after_update_and_load(tmp_path):
    agent = make_agent("td3_amo", "jax", 3, 2, config("td3_amo"), seed=7)
    obs = batch()["observations"]
    initial = agent.act(obs)
    for step in range(4):
        agent.update(batch(step), batch(step + 100), return_metrics=False)
    trained = agent.act(obs)
    assert not np.array_equal(initial, trained)
    eager = agent.ops.numpy(
        agent.actor(agent.state["p"]["actor"], agent.ops.array(obs))
    )
    np.testing.assert_allclose(trained, eager, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(agent.act(obs[0]), trained[0], atol=2e-7, rtol=2e-6)
    path = tmp_path / "trained.npz"
    agent.save(path)
    restored = make_agent("td3_amo", "jax", 3, 2, config("td3_amo"), seed=999)
    restored.act(obs)  # Compile before load to catch accidentally captured weights.
    restored.load(path)
    np.testing.assert_array_equal(restored.act(obs), trained)


@pytest.mark.parametrize("backend", ("jax", "torch"))
def test_resident_replay_preserves_samples_rng_and_independent_outer(backend):
    agent = make_agent("td3_amo", backend, 3, 2, config("td3_amo"))
    data = batch()
    resident = agent.ops.batch(data)
    for seed in (1009, 100003):
        host = ReplayBuffer(data, seed)
        device = ReplayBuffer(resident, seed, sampler=agent.ops.sample)
        for _ in range(5):
            expected = host.sample(4)
            actual = agent.ops.numpy_tree(device.sample(4))
            for key in expected:
                np.testing.assert_array_equal(expected[key], actual[key])
            assert host.rng.bit_generator.state == device.rng.bit_generator.state
        resumed = ReplayBuffer(data, 0)
        resumed.rng.bit_generator.state = copy.deepcopy(device.rng.bit_generator.state)
        expected = resumed.sample(4)
        actual = agent.ops.numpy_tree(device.sample(4))
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])


def test_resident_update_does_not_fetch_training_arrays(monkeypatch):
    agent = make_agent("td3_amo", "jax", 3, 2, config("td3_amo"))
    b = agent.ops.batch(batch())
    outer = agent.ops.batch(batch(100))
    original = jax.device_get

    def no_fetch(*args, **kwargs):
        raise AssertionError("host fetch during deferred update")

    monkeypatch.setattr(jax, "device_get", no_fetch)
    agent.update(b, outer, return_metrics=False)
    agent.update(b, outer, return_metrics=False)
    monkeypatch.setattr(jax, "device_get", original)
    assert np.isfinite(list(agent.get_metrics().values())).all()


def test_cli_resume_can_switch_replay_location(tmp_path):
    import yaml

    from train import main

    dataset = tmp_path / "data.npz"
    np.savez(dataset, **batch())
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"defaults": {"hidden_dim": 8, "batch_size": 6}})
    )
    common = [
        "--algorithm",
        "td3_amo",
        "--backend",
        "jax",
        "--env",
        "synthetic",
        "--dataset",
        str(dataset),
        "--config",
        str(config_path),
        "--seed",
        "17",
        "--log-every",
        "7",
        "--save-every",
        "13",
    ]
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    main(
        common
        + ["--output", str(resumed), "--steps", "17", "--replay-device", "training"]
    )
    main(
        common
        + [
            "--output",
            str(resumed),
            "--steps",
            "41",
            "--resume",
            str(resumed / "checkpoint.npz"),
        ]
    )
    main(common + ["--output", str(full), "--steps", "41"])
    with np.load(resumed / "checkpoint.npz") as a, np.load(
        full / "checkpoint.npz"
    ) as b:
        assert set(a.files) == set(b.files)
        for key in a.files:
            if key == "__metadata__":
                assert json.loads(str(a[key])) == json.loads(str(b[key]))
            else:
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)
    assert sorted(
        int(p.stem.split("_")[1]) for p in (resumed / "checkpoints").glob("*.npz")
    ) == [13, 17, 26, 39, 41]
