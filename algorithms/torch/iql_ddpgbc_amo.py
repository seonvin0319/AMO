"""IQL critic + DDPG+BC actor, AMO α_E adapted with L_E only.

Value learning is IQL (expectile V, Q Bellman with V). Policy extraction is
Park et al. 2024 (arXiv:2406.09329 / FQL IQL) DDPG+BC. There is no π_B.
α_E is the TD3-AMO-style inner horizon: BC weight = 1/α_E. Meta-loss is L_E
(BPI), not L_E+L2_RMS.
"""

import math

from common import torch_backend as backend
from common.agent import BaseAgent
from common.optim import adam, target_update
from common.tree import map_tree


class Agent(BaseAgent):
    Backend = backend.Backend

    def log_prob(self, p, obs, actions):
        o = self.ops
        mean = self.actor(p, obs)
        log_std = o.clip(p["log_std"], -20.0, 2.0)
        if self.c.get("const_std", True):
            log_std = o.stop(log_std)
        nll = (
            0.5 * ((actions - mean) / o.exp(log_std)) ** 2
            + log_std
            + 0.5 * math.log(2 * math.pi)
        ).sum(axis=-1, keepdims=True)
        return -nll

    def inner(self, actor_params, critic_params, batch, horizon):
        o = self.ops
        obs, actions = batch["observations"], batch["actions"]
        mean = self.actor(actor_params, obs)
        q = self.minq(critic_params, obs, mean)
        q_loss = -q.mean() / (self.q_scale(q) + self.c["smoothness_eps"])
        bc = -self.log_prob(actor_params, obs, actions).mean()
        return q_loss + bc / horizon

    def virtual(self, state, batch, horizon):
        p = state["p"]["actor"]
        critic = map_tree(self.ops.stop, state["p"]["critic"])
        _, grads = self.ops.grad(
            lambda params: self.inner(params, critic, batch, horizon), p
        )
        return adam(self.ops, p, grads, state["opt"]["actor"], self.c["actor_lr"])[0]

    def execution_outer(self, scale, state, inner_batch, outer_batch):
        o = self.ops
        horizon = o.softplus(scale["rho"])
        plus = self.virtual(state, inner_batch, horizon)
        obs = outer_batch["observations"]
        old = self.actor(state["p"]["actor"], obs)
        new = self.actor(plus, obs)
        target = map_tree(o.stop, state["target"]["critic"])

        def action_gradient(action):
            _, g = o.grad(
                lambda p: self.minq(target, obs, p["action"]).sum(),
                {"action": o.stop(action)},
            )
            return o.stop(g["action"])

        g0, g1 = action_gradient(old), action_gradient(new)
        d = new - old
        penalty = (
            0.5
            * o.safe_sqrt(((g1 - g0) ** 2).sum(axis=-1))
            * o.safe_sqrt((d**2).sum(axis=-1))
        )
        proxy = (g0 * d).sum(axis=-1) - penalty
        return -proxy.mean() / self.q_scale(self.minq(target, obs, new))

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        next_v = o.stop(self.value(state["p"]["value"], batch["next_observations"]))
        target_q = o.stop(self.minq(state["target"]["critic"], obs, actions))

        def value_loss(p):
            diff = target_q - self.value(p, obs)
            weight = o.where(
                diff > 0, diff * 0 + c["expectile"], diff * 0 + 1 - c["expectile"]
            )
            return (weight * diff**2).mean()

        state, vl = self.update_network(state, "value", value_loss, c["value_lr"])
        q_target = o.stop(
            batch["rewards"] + (1 - batch["terminals"]) * c["discount"] * next_v
        )
        state, ql = self.update_network(
            state,
            "critic",
            lambda p: sum(((q - q_target) ** 2).mean() for q in self.q(p, obs, actions))
            / 2,
        )
        state = {
            **state,
            "target": {
                **state["target"],
                "critic": target_update(
                    o, state["p"]["critic"], state["target"]["critic"], c["tau"]
                ),
            },
        }
        if not actor_step:
            return state, {"critic_loss": ql, "value_loss": vl}

        horizon_e = o.stop(o.softplus(state["p"]["scale_E"]["rho"]))
        logs = {"critic_loss": ql, "value_loss": vl}
        if meta_step:
            lr = c["alpha_lr"] * 0.01 ** (
                state["opt"]["scale_E"]["count"] * c["meta_interval"] / 1_000_000
            )
            frozen = state
            state, logs["L_alpha_E"] = self.update_network(
                state,
                "scale_E",
                lambda p: self.execution_outer(p, frozen, batch, outer),
                lr,
            )
        critic = map_tree(o.stop, state["p"]["critic"])
        state, logs["actor_loss"] = self.update_network(
            state,
            "actor",
            lambda p: self.inner(p, critic, batch, horizon_e),
            c["actor_lr"],
        )
        logs.update(alpha_E=o.softplus(state["p"]["scale_E"]["rho"]))
        return state, logs
