#!/usr/bin/env python3
"""Short CPU check: alpha_B and its Adam state stay fixed; alpha_E and bootstrap move."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from algorithms.jax.td3_amo import Agent
from common.config import load_config
from pathlib import Path

CFG = Path(__file__).resolve().parents[1] / "configs" / "td3_amo_fixed1_rapo.yaml"


def leaves(tree):
    if isinstance(tree, dict):
        out = []
        for key in sorted(tree):
            out.extend(leaves(tree[key]))
        return out
    return [np.asarray(tree)]


def main():
    cfg = load_config(
        "td3_amo",
        "hopper-medium-v2",
        path=CFG,
        overrides={"tau": 1.0, "batch_size": 32, "hidden_dim": 16},
    )
    agent = Agent(4, 2, cfg, seed=0, device="cpu")
    host = agent.ops.numpy_tree(agent.state)
    q1 = host["p"]["critic"]["q1"]
    layers = sorted(q1)
    assert layers == ["layer0", "layer1", "layer2", "layer3"], layers
    for name in ("layer0", "layer1", "layer2"):
        assert "scale" in q1[name] and "offset" in q1[name], name
    assert "scale" not in q1["layer3"]
    rho_b = np.array(host["p"]["scale_B"]["rho"], copy=True)
    opt_b = leaves(host["opt"]["scale_B"])
    rho_e = np.array(host["p"]["scale_E"]["rho"], copy=True)
    boot = leaves(host["p"]["bootstrap"])
    actor = leaves(host["p"]["actor"])
    obs = np.zeros((32, 4), np.float32)
    act = np.zeros((32, 2), np.float32)
    batch = {
        "observations": obs,
        "actions": act,
        "rewards": np.zeros((32, 1), np.float32),
        "terminals": np.zeros((32, 1), np.float32),
        "next_observations": obs,
        "next_actions": act,
    }
    for _ in range(40):
        agent.update(batch, batch, return_metrics=False)
    host = agent.ops.numpy_tree(agent.state)
    assert np.allclose(host["p"]["scale_B"]["rho"], rho_b)
    after_opt = leaves(host["opt"]["scale_B"])
    assert all(np.allclose(a, b) for a, b in zip(after_opt, opt_b))
    assert not np.allclose(host["p"]["scale_E"]["rho"], rho_e)
    assert any(not np.allclose(a, b) for a, b in zip(leaves(host["p"]["bootstrap"]), boot))
    assert any(not np.allclose(a, b) for a, b in zip(leaves(host["p"]["actor"]), actor))
    # tau=1 copies the bootstrap actor into the Bellman target.
    assert all(
        np.allclose(a, b)
        for a, b in zip(leaves(host["target"]["actor"]), leaves(host["p"]["bootstrap"]))
    )
    assert any(
        not np.allclose(a, b)
        for a, b in zip(leaves(host["target"]["actor"]), leaves(host["p"]["actor"]))
    )
    print(
        "ok alpha_B",
        float(np.asarray(host["p"]["scale_B"]["rho"])),
        "alpha_E",
        float(agent.ops.numpy(agent.ops.softplus(host["p"]["scale_E"]["rho"]))),
        "critic_layers",
        layers,
    )


if __name__ == "__main__":
    main()
