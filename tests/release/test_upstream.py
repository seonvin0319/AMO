"""Compare updates with pinned, independently authored upstream trainers.

Initial weights and sampled target/VAE noise are controlled. This checks the
algorithm equations and update order, not environment returns or RNG identity.
"""

import copy
import importlib

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchopt

from common.agent import make_agent
from tests.release.test_contracts import batch, config


def numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def tensor(value):
    return torch.from_numpy(numpy(value).copy())


def to_module(params, tanh=False):
    layers = []
    for i in range(len(params)):
        p = params[f"layer{i}"]
        linear = nn.Linear(p["w"].shape[0], p["w"].shape[1])
        with torch.no_grad():
            linear.weight.copy_(tensor(p["w"]).T)
            linear.bias.copy_(tensor(p["b"]))
        layers.append(linear)
        if i < len(params) - 1:
            layers.append(nn.ReLU())
            if "scale" in p:
                norm = nn.LayerNorm(len(p["scale"]))
                with torch.no_grad():
                    norm.weight.copy_(tensor(p["scale"]))
                    norm.bias.copy_(tensor(p["offset"]))
                layers.append(norm)
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = to_module(p["net"], tanh=True)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.net = to_module(p)

    def forward(self, s, a):
        return self.net(torch.cat((s, a), dim=-1))


class Twin(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.q1, self.q2 = Critic(p["q1"]), Critic(p["q2"])

    def forward(self, s, a):
        return self.q1(s, a), self.q2(s, a)

    def Q1(self, s, a):
        return self.q1(s, a)


def check_mlp(params, module, atol=5e-6):
    linear = [m for m in module.modules() if isinstance(m, nn.Linear)]
    norms = iter(m for m in module.modules() if isinstance(m, nn.LayerNorm))
    assert len(linear) == len(params)
    for i, layer in enumerate(linear):
        p = params[f"layer{i}"]
        np.testing.assert_allclose(
            numpy(p["w"]), layer.weight.detach().T, atol=atol, rtol=3e-5
        )
        np.testing.assert_allclose(
            numpy(p["b"]), layer.bias.detach(), atol=atol, rtol=3e-5
        )
        if "scale" in p:
            norm = next(norms)
            np.testing.assert_allclose(
                numpy(p["scale"]), norm.weight.detach(), atol=atol, rtol=3e-5
            )
            np.testing.assert_allclose(
                numpy(p["offset"]), norm.bias.detach(), atol=atol, rtol=3e-5
            )


def assign_mlp(params, module):
    linear = [m for m in module.modules() if isinstance(m, nn.Linear)]
    assert len(linear) == len(params)
    for i, layer in enumerate(linear):
        with torch.no_grad():
            layer.weight.copy_(tensor(params[f"layer{i}"]["w"]).T)
            layer.bias.copy_(tensor(params[f"layer{i}"]["b"]))


def reference_td3(agent, algorithm):
    ref = importlib.import_module(f"tests.references.{algorithm}")
    p, c = agent.state["p"], agent.c
    actor, q1, q2 = (
        Actor(p["actor"]),
        Critic(p["critic"]["q1"]),
        Critic(p["critic"]["q2"]),
    )
    differentiable = algorithm in ("aspc", "td3_amo")
    actor_optimizer = (
        torchopt.adam(lr=c["actor_lr"], use_accelerated_op=True)
        if differentiable
        else torch.optim.Adam(actor.parameters(), lr=c["actor_lr"])
    )
    kw = dict(
        max_action=1.0,
        actor=actor,
        actor_optimizer=actor_optimizer,
        critic_1=q1,
        critic_2=q2,
        critic_1_optimizer=torch.optim.Adam(q1.parameters(), lr=c["critic_lr"]),
        critic_2_optimizer=torch.optim.Adam(q2.parameters(), lr=c["critic_lr"]),
    )
    if algorithm in ("wpc", "aspc", "td3_amo"):
        value = (
            to_module(p["value"]) if "value" in p else nn.Sequential(nn.Linear(3, 1))
        )
        kw.update(
            vnet=value, vnet_optimizer=torch.optim.Adam(value.parameters(), lr=0.0003)
        )
    if algorithm == "td3_amo":
        kw.update(
            T=c["T_E"],
            T_B=c["T_B"],
            T_lr=c["T_lr"],
            T_freq=c["meta_interval"] // 2,
            adaptive_multiscale=True,
        )
        model = ref.AMO(**kw)
        with torch.no_grad():
            model.log_T.copy_(tensor(p["scale_E"]["rho"]))
            model.log_T_B.copy_(tensor(p["scale_B"]["rho"]))
    elif algorithm == "aspc":
        kw.update(alpha=c["alpha"], alpha_freq=c["meta_interval"] // 2)
        model = ref.Adaptive_TD3_BC(**kw)
        model.reward_mean = 0.0  # Missing initialization in upstream main.
        # The reference scales alpha LR with alpha_freq.
        model.alpha_optimizer.param_groups[0]["lr"] = c["scale_lr"]
        with torch.no_grad():
            model.alpha.copy_(tensor(p["scale"]["rho"]))
    else:
        kw.update(alpha=c["alpha"])
        model = ref.TD3_BC(**kw)
    return model


@pytest.mark.parametrize("algorithm", ["td3_bc", "wpc", "aspc", "td3_amo"])
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_td3_updates_match_upstream(algorithm, backend, monkeypatch):
    c = config(algorithm)
    # Preserve the published scheduler denominator and update interval.
    if algorithm in ("aspc", "td3_amo"):
        c["meta_interval"] = 20
    agent = make_agent(algorithm, backend, 3, 2, c, seed=17)
    ref = reference_td3(agent, algorithm)
    b, out = batch(6), batch(33)
    tb = [
        torch.from_numpy(b[k])
        for k in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        )
    ]
    to = [
        torch.from_numpy(out[k])
        for k in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        )
    ]
    for step in range(22):
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
        noise = torch.from_numpy(rng.standard_normal((6, 2)).astype(np.float32))
        monkeypatch.setattr(torch, "randn_like", lambda x: noise.to(x))
        agent.update(b, out)
        ref.train(tb, to) if algorithm == "td3_amo" else ref.train(tb)
        p = agent.state["p"]
        check_mlp(p["actor"]["net"], ref.actor)
        for i in (1, 2):
            check_mlp(p["critic"][f"q{i}"], getattr(ref, f"critic_{i}"))
        if algorithm == "td3_amo":
            check_mlp(p["bootstrap"]["net"], ref.critic_bootstrap_actor())
            np.testing.assert_allclose(
                p["scale_E"]["rho"], ref.log_T.detach(), atol=2e-6, rtol=2e-5
            )
            np.testing.assert_allclose(
                p["scale_B"]["rho"], ref.log_T_B.detach(), atol=2e-6, rtol=2e-5
            )
        if algorithm == "aspc":
            np.testing.assert_allclose(
                p["scale"]["rho"], ref.alpha.detach(), atol=2e-6, rtol=2e-5
            )
        if algorithm == "wpc":
            check_mlp(p["value"], ref.vnet)
        for i in (1, 2):
            check_mlp(
                agent.state["target"]["critic"][f"q{i}"],
                getattr(ref, f"critic_{i}_target"),
            )
        check_mlp(agent.state["target"]["actor"]["net"], ref.actor_target)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_iql_matches_upstream(backend):
    from tests.references import iql as original

    c = config("iql")
    agent = make_agent("iql", backend, 3, 2, c, seed=19)
    p = agent.state["p"]
    actor = original.GaussianPolicy(3, 2, 1.0, hidden_dim=8)
    q = original.TwinQ(3, 2, hidden_dim=8)
    value = original.ValueFunction(3, hidden_dim=8)
    assign_mlp(p["actor"]["net"], actor)
    assign_mlp(p["critic"]["q1"], q.q1)
    assign_mlp(p["critic"]["q2"], q.q2)
    assign_mlp(p["value"], value)
    ref = original.ImplicitQLearning(
        1.0,
        actor,
        torch.optim.Adam(actor.parameters(), lr=c["actor_lr"]),
        q,
        torch.optim.Adam(q.parameters(), lr=c["critic_lr"]),
        value,
        torch.optim.Adam(value.parameters(), lr=c["value_lr"]),
        max_steps=c["max_steps"],
    )
    b = batch(4)
    tb = [
        torch.from_numpy(b[k])
        for k in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        )
    ]
    for _ in range(8):
        agent.update(b)
        ref.train(tb)
        p = agent.state["p"]
        check_mlp(p["actor"]["net"], ref.actor)
        np.testing.assert_allclose(
            p["actor"]["log_std"], ref.actor.log_std.detach(), atol=2e-6
        )
        check_mlp(p["value"], ref.vf)
        check_mlp(p["critic"]["q1"], ref.qf.q1)
        check_mlp(p["critic"]["q2"], ref.qf.q2)
        check_mlp(agent.state["target"]["critic"]["q1"], ref.q_target.q1)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_a2pr_matches_upstream(backend, monkeypatch):
    from tests.references import a2pr as original

    c = config("a2pr")
    agent = make_agent("a2pr", backend, 3, 2, c, seed=11)
    p = agent.state["p"]
    ref = original.A2PR(3, 2, 1.0, "cpu")
    ref.actor = Actor(p["actor"])
    ref.actor_target = copy.deepcopy(ref.actor)
    ref.actor_optimizer = torch.optim.Adam(ref.actor.parameters(), lr=c["actor_lr"])
    ref.critic = Twin(p["critic"])
    ref.critic_target = copy.deepcopy(ref.critic)
    ref.critic_optimizer = torch.optim.Adam(ref.critic.parameters(), lr=c["critic_lr"])
    ref.value = to_module(p["value"])
    ref.value_optimizer = torch.optim.Adam(ref.value.parameters(), lr=c["value_lr"])
    encoder = to_module(p["vae"]["encoder"])
    decoder = to_module(p["vae"]["decoder"])
    ref.vae.e1, ref.vae.e2 = encoder[0], encoder[2]
    ref.vae.d1, ref.vae.d2, ref.vae.d3 = decoder[0], decoder[2], decoder[4]
    ref.vae.mean = to_module(p["vae"]["mean"])[0]
    ref.vae.log_std = to_module(p["vae"]["log_std"])[0]
    ref.vae_optimizer = torch.optim.Adam(ref.vae.parameters(), lr=c["vae_lr"])
    b = batch(5)
    tb = [
        torch.from_numpy(b[k])
        for k in (
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "terminals",
        )
    ]

    class Buffer:
        def sample(self, size):
            return (*tb[:4], 1 - tb[4])

    for _ in range(6):
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
        noises = iter(
            torch.from_numpy(rng.standard_normal(shape).astype(np.float32))
            for shape in ((6, 2), (6, 4), (6, 4))
        )
        monkeypatch.setattr(torch, "randn_like", lambda x: next(noises).to(x))
        agent.update(b)
        ref.train(Buffer(), 6)
        p = agent.state["p"]
        check_mlp(p["actor"]["net"], ref.actor)
        check_mlp(p["critic"]["q1"], ref.critic.q1)
        check_mlp(p["critic"]["q2"], ref.critic.q2)
        check_mlp(p["value"], ref.value)
        check_mlp(p["vae"]["encoder"], nn.Sequential(ref.vae.e1, ref.vae.e2))
        check_mlp(
            p["vae"]["decoder"], nn.Sequential(ref.vae.d1, ref.vae.d2, ref.vae.d3)
        )
        check_mlp(p["vae"]["mean"], ref.vae.mean)
        check_mlp(p["vae"]["log_std"], ref.vae.log_std)
