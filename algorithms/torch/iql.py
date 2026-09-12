"""IQL with an expectile value function and Gaussian AWR policy."""

import math

from common import torch_backend as backend
from common.agent import BaseAgent
from common.optim import target_update


class Agent(BaseAgent):
    Backend = backend.Backend

    def policy_loss(self, p, obs, actions, weights):
        mean = self.actor(p, obs)
        log_std = self.ops.clip(p["log_std"], -20.0, 2.0)
        nll = (
            0.5 * ((actions - mean) / self.ops.exp(log_std)) ** 2
            + log_std
            + 0.5 * math.log(2 * math.pi)
        ).sum(axis=-1, keepdims=True)
        return (weights * nll).mean()

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        next_v = o.stop(self.value(state["p"]["value"], batch["next_observations"]))
        target_q = o.stop(self.minq(state["target"]["critic"], obs, actions))
        adv = o.stop(target_q - self.value(state["p"]["value"], obs))

        def value_loss(p):
            diff = target_q - self.value(p, obs)
            weights = o.where(
                diff > 0, diff * 0 + c["expectile"], diff * 0 + 1 - c["expectile"]
            )
            return (weights * diff**2).mean()

        state, vl = self.update_network(state, "value", value_loss, c["value_lr"])
        target = o.stop(
            batch["rewards"] + (1 - batch["terminals"]) * c["discount"] * next_v
        )

        def q_loss(p):
            return sum(((q - target) ** 2).mean() for q in self.q(p, obs, actions)) / 2

        state, ql = self.update_network(state, "critic", q_loss)
        state = {
            **state,
            "target": {
                **state["target"],
                "critic": target_update(
                    o, state["p"]["critic"], state["target"]["critic"], c["tau"]
                ),
            },
        }
        weights = o.stop(o.exp(o.clip(c["beta"] * adv, high=math.log(c["weight_cap"]))))
        # Cosine schedule uses the number of COMPLETED actor updates.
        fraction = (noise["step"] - 1) / c["max_steps"]
        lr = c["actor_lr"] * 0.5 * (1 + self.cos_pi(fraction))
        state, al = self.update_network(
            state, "actor", lambda p: self.policy_loss(p, obs, actions, weights), lr
        )
        return state, {"critic_loss": ql, "value_loss": vl, "actor_loss": al}

    def cos_pi(self, x):
        return self.ops.cos(x * math.pi)
