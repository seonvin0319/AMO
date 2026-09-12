"""Compare complete IQL+AMO updates with the uploaded experiment trainer."""

import copy

import numpy as np
import pytest
import torch

from common.agent import make_agent
from common.tree import flatten
from tests.references import iql_amo as ref
from tests.release.test_contracts import batch, config
from tests.release.test_upstream import assign_mlp, check_mlp


def reference(agent, monkeypatch):
    c = agent.c
    twin, value, gaussian = ref.TwinQ, ref.ValueFunction, ref.GaussianPolicy
    monkeypatch.setattr(
        ref, "TwinQ", lambda s, a: twin(s, a, hidden_dim=c["hidden_dim"])
    )
    monkeypatch.setattr(
        ref, "ValueFunction", lambda s: value(s, hidden_dim=c["hidden_dim"])
    )

    class SmallGaussian(gaussian):
        def __init__(self, s, a, m, **kw):
            super().__init__(s, a, m, hidden_dim=c["hidden_dim"], **kw)

    monkeypatch.setattr(ref, "GaussianPolicy", SmallGaussian)
    corl = dict(
        discount=c["discount"],
        tau=c["tau"],
        iql_tau=c["expectile"],
        batch_size=c["batch_size"],
        max_timesteps=c["max_steps"],
        iql_deterministic=False,
        vf_lr=c["value_lr"],
        qf_lr=c["critic_lr"],
        actor_lr=c["actor_lr"],
        actor_dropout=None,
    )
    meta = ref.MetaConfig(
        amo_dual_bpi=True,
        beta_initial=c["beta_initial"],
        beta_fixed=c["beta_initial"],
        rho_lr=c["rho_lr"],
        meta_warmup_steps=c["meta_warmup_steps"],
        meta_interval=c["meta_interval"],
        beta_min=c["beta_min"],
        beta_max=c["beta_max"],
    )
    trainer = ref.IQLAdaptiveBetaTrainer(
        corl=corl,
        meta=meta,
        device="cpu",
        seed=0,
        max_action=1.0,
        state_dim=agent.observation_dim,
        action_dim=agent.action_dim,
    )
    p = agent.state["p"]
    for network in (trainer.actor_adaptive, trainer.actor_fixed):
        assign_mlp(p["actor"]["net"], network)
    assign_mlp(p["value"], trainer.vf)
    for i in (1, 2):
        assign_mlp(p["critic"][f"q{i}"], getattr(trainer.qf, f"q{i}"))
    trainer.q_target = copy.deepcopy(trainer.qf).requires_grad_(False)
    return trainer


@pytest.mark.parametrize("beta_initial", [1.0, 5.0])
@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_iql_amo_original_updates(monkeypatch, beta_initial, backend):
    c = config("iql_amo")
    # A short cosine horizon exposes use of the post-update beta0 learning rate.
    c.update(beta_initial=beta_initial, max_steps=20, meta_warmup_steps=3)
    agent = make_agent("iql_amo", backend, 3, 2, c, seed=19)
    trainer = reference(agent, monkeypatch)
    keys = ("observations", "actions", "rewards", "next_observations", "terminals")
    for step in range(1, 10):
        b, out = batch(step), batch(step + 17)
        metrics = agent.update(b, out)
        _, row = trainer.train_step(
            [torch.from_numpy(b[k]) for k in keys],
            [torch.from_numpy(out[k]) for k in keys],
            do_meta=ref.is_meta_step(step, c["meta_warmup_steps"], c["meta_interval"]),
        )
        p = agent.state["p"]
        for name, module in (
            ("actor", trainer.actor_adaptive),
            ("bootstrap", trainer.actor_fixed),
        ):
            check_mlp(p[name]["net"], module)
            np.testing.assert_allclose(
                p[name]["log_std"], module.log_std.detach(), atol=3e-6
            )
        check_mlp(p["value"], trainer.vf)
        for i in (1, 2):
            check_mlp(p["critic"][f"q{i}"], getattr(trainer.qf, f"q{i}"))
            check_mlp(
                agent.state["target"]["critic"][f"q{i}"],
                getattr(trainer.q_target, f"q{i}"),
            )
        for role, scale in (("E", trainer.rho), ("B", trainer.rho_B)):
            np.testing.assert_allclose(
                p[f"scale_{role}"]["rho"], scale.detach(), atol=3e-6, rtol=2e-5
            )
        if row:
            for key in ("L_E", "L_T_B", "L1_B", "L2_RMS_B"):
                np.testing.assert_allclose(metrics[key], row[key], atol=2e-6, rtol=2e-3)


def test_iql_critic_is_independent_of_both_actor_scales():
    a = make_agent("iql_amo", "torch", 3, 2, config("iql_amo"), seed=7)
    c = config("iql_amo")
    c.update(beta_initial=10.0, rho_lr=0.01)
    b = make_agent("iql_amo", "torch", 3, 2, c, seed=7)
    for step in range(6):
        a.update(batch(step), batch(step + 50))
        b.update(batch(step), batch(step + 50))
    for key in ("critic", "value"):
        for path, x in flatten(a.state["p"][key]).items():
            np.testing.assert_array_equal(x, flatten(b.state["p"][key])[path])


def test_beta0_branch_does_not_change_execution_learning():
    a = make_agent("iql_amo", "torch", 3, 2, config("iql_amo"), seed=7)
    b = make_agent("iql_amo", "torch", 3, 2, config("iql_amo"), seed=7)
    b.state["p"]["scale_B"]["rho"] = torch.tensor(np.log(10.0), dtype=torch.float64)
    for step in range(6):
        a.update(batch(step), batch(step + 50))
        b.update(batch(step), batch(step + 50))
    for key in ("actor", "scale_E", "critic", "value"):
        for path, x in flatten(a.state["p"][key]).items():
            np.testing.assert_array_equal(x, flatten(b.state["p"][key])[path])
