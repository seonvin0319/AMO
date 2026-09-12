"""Final TD3+AMO: execution -B_pi and bootstrap detached-TQ L1+L2_RMS.

Reference: seonvin0319/AMO a8c1e48, adaptive_multiscale=True.
The endpoint gradient proxy is empirical; it is not a certified return bound.
"""

from common import torch_backend as backend
from common.agent import BaseAgent
from common.optim import adam
from common.tree import map_tree


class Agent(BaseAgent):
    Backend = backend.Backend

    def inner(self, actor_params, critic_params, batch, horizon):
        action = self.actor(actor_params, batch["observations"])
        q = self.q(critic_params, batch["observations"], action)[0]
        return -q.mean() / self.q_scale(q) + (
            (action - batch["actions"]) ** 2
        ).mean() / (2 * horizon)

    def virtual(self, state, name, batch, horizon, sgd=False):
        p = state["p"][name]
        critic = map_tree(self.ops.stop, state["p"]["critic"])
        _, grads = self.ops.grad(
            lambda params: self.inner(params, critic, batch, horizon), p
        )
        if sgd:
            return map_tree(lambda a, g: a - self.c["actor_lr"] * g, p, grads)
        return adam(self.ops, p, grads, state["opt"][name], self.c["actor_lr"])[0]

    def bpi(self, target_critic, obs, old_action, new_action):
        o = self.ops
        target = map_tree(o.stop, target_critic)

        def gradient(action):
            _, g = o.grad(
                lambda p: self.q(target, obs, p["action"])[0].sum(),
                {"action": o.stop(action)},
            )
            return o.stop(g["action"])

        g0, g1 = gradient(old_action), gradient(new_action)
        displacement = new_action - old_action
        norm_d = o.safe_sqrt((displacement**2).sum(axis=-1))
        norm_g = o.safe_sqrt(((g1 - g0) ** 2).sum(axis=-1))
        proxy = (g0 * displacement).sum(axis=-1) - 0.5 * norm_g * norm_d
        scale = self.q_scale(self.q(target, obs, new_action)[0])
        return -proxy.mean() / scale

    def execution_outer(self, scale, state, inner_batch, outer_batch):
        horizon = self.ops.softplus(scale["rho"])
        plus = self.virtual(state, "actor", inner_batch, horizon)
        obs = outer_batch["observations"]
        old = self.actor(state["p"]["actor"], obs)
        new = self.actor(plus, obs)
        return self.bpi(state["target"]["critic"], obs, old, new)

    def bootstrap_terms(self, scale, state, inner_batch, outer_batch):
        o = self.ops
        horizon = o.softplus(scale["rho"])
        plus = self.virtual(state, "bootstrap", inner_batch, horizon, sgd=True)
        target = map_tree(o.stop, state["target"]["critic"])
        obs, next_obs = outer_batch["observations"], outer_batch["next_observations"]
        action = self.actor(plus, obs)
        q = self.minq(target, obs, action)
        q_scale = o.stop(o.abs(q).mean()) + self.c["smoothness_eps"]
        new_next = self.minq(target, next_obs, self.actor(plus, next_obs))
        old_next = o.stop(
            self.minq(target, next_obs, self.actor(state["p"]["bootstrap"], next_obs))
        )
        delta_y = (
            self.c["discount"] * (1 - outer_batch["terminals"]) * (new_next - old_next)
        )
        rms = o.safe_sqrt(((delta_y / q_scale) ** 2).mean())
        factor = 2 * o.stop(horizon)
        l1 = (
            -factor * q.mean() / q_scale
            + ((action - outer_batch["actions"]) ** 2).mean()
        )
        l2 = factor * rms
        return l1, l2

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, _, _ = self.td_critic(state, batch, noise)
        if not actor_step:
            return state, logs
        o, c = self.ops, self.c
        # Preserve reference timing: E actor uses the pre-meta horizon;
        # B actor uses the newly updated bootstrap horizon.
        horizon_e = o.stop(o.softplus(state["p"]["scale_E"]["rho"]))
        if meta_step:
            lr = c["T_lr"] * 0.01 ** (
                state["opt"]["scale_E"]["count"] * c["meta_interval"] / 1_000_000
            )
            frozen_state = state
            state, logs["L_T_E"] = self.update_network(
                state,
                "scale_E",
                lambda p: self.execution_outer(p, frozen_state, batch, outer),
                lr,
            )
            frozen_state = state
            l1, l2 = self.bootstrap_terms(state["p"]["scale_B"], state, batch, outer)
            state, logs["L_T_B"] = self.update_network(
                state,
                "scale_B",
                lambda p: sum(self.bootstrap_terms(p, frozen_state, batch, outer)),
                lr,
            )
            logs.update(L1_B=l1, L2_RMS_B=l2)
        horizon_b = o.stop(o.softplus(state["p"]["scale_B"]["rho"]))
        for name, horizon in (("bootstrap", horizon_b), ("actor", horizon_e)):
            critic = map_tree(o.stop, state["p"]["critic"])
            state, loss = self.update_network(
                state,
                name,
                lambda p: self.inner(p, critic, batch, horizon),
                c["actor_lr"],
            )
            logs[f"{name}_loss"] = loss
        logs.update(T_E=o.softplus(state["p"]["scale_E"]["rho"]), T_B=horizon_b)
        return self.update_targets(state), logs
