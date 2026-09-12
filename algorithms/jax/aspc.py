"""ASPC's differentiable Adam update and original L1+L2+L3 objective."""

from common import jax_backend as backend
from common.agent import BaseAgent
from common.optim import adam


class Agent(BaseAgent):
    Backend = backend.Backend

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, _, _ = self.td_critic(state, batch, noise)
        if not actor_step:
            return state, logs
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        old_actor = state["p"]["actor"]
        old_pi = self.actor(old_actor, obs)

        def virtual(scale):
            alpha = o.softplus(scale["rho"])

            def inner(p):
                pi = self.actor(p, obs)
                q = self.q(state["p"]["critic"], obs, pi)[0]
                return (
                    -alpha * q.mean() / self.q_scale(q) + ((pi - actions) ** 2).mean()
                )

            loss, grads = o.grad(inner, old_actor)
            pnew, optnew = adam(
                o, old_actor, grads, state["opt"]["actor"], c["actor_lr"]
            )
            return pnew, optnew, loss

        pnew, optnew, inner_loss = virtual(state["p"]["scale"])
        if meta_step:

            def outer_loss(scale):
                pplus, _, _ = virtual(scale)
                pi_new = self.actor(pplus, obs)
                q_new = self.q(state["p"]["critic"], obs, pi_new)[0]
                ema = (1 - c["ema_alpha"]) * q_new.mean() + c["ema_alpha"] * o.stop(
                    state["aux"]["ema_q"]
                )
                delta_q = ema - o.stop(state["aux"]["previous_ema_q"])
                bc_old = o.stop(((old_pi - actions) ** 2).mean(axis=-1))
                delta_bc = o.abs(bc_old - ((pi_new - actions) ** 2).mean(axis=-1)).max()
                l1 = (
                    -o.stop(o.softplus(scale["rho"]))
                    * q_new.mean()
                    / self.q_scale(q_new)
                    + ((pi_new - actions) ** 2).mean()
                )
                l2 = delta_q**2
                l3 = o.stop(l2) * bc_old.max() * delta_bc
                return l1 + l2 + l3

            lr = c["scale_lr"] * 0.01 ** (
                state["opt"]["scale"]["count"] * c["meta_interval"] / 1_000_000
            )
            state, logs["scale_loss"] = self.update_network(
                state, "scale", outer_loss, lr
            )
            q_new = self.q(state["p"]["critic"], obs, self.actor(pnew, obs))[0].mean()
            ema = o.stop(
                (1 - c["ema_alpha"]) * q_new + c["ema_alpha"] * state["aux"]["ema_q"]
            )
            state = {**state, "aux": {"ema_q": ema, "previous_ema_q": ema}}
        state = {
            **state,
            "p": {**state["p"], "actor": pnew},
            "opt": {**state["opt"], "actor": optnew},
        }
        logs.update(actor_loss=inner_loss, alpha=o.softplus(state["p"]["scale"]["rho"]))
        return self.update_targets(state), logs
