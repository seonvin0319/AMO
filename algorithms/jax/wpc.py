"""wPC(RC): value regression and a positive-advantage BC mask."""

from common import jax_backend as backend
from common.agent import BaseAgent


class Agent(BaseAgent):
    Backend = backend.Backend

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, target, _ = self.td_critic(state, batch, noise)
        obs = batch["observations"]
        # The reference mask uses post-critic Q and PRE-value-update V.
        q_data = self.ops.stop(self.q(state["p"]["critic"], obs, batch["actions"])[0])
        old_v = self.ops.stop(self.value(state["p"]["value"], obs))
        state, logs["value_loss"] = self.update_network(
            state,
            "value",
            lambda p: ((self.value(p, obs) - target) ** 2).mean(),
            self.c["value_lr"],
        )
        if actor_step:
            mask = self.ops.stop(1.0 * (q_data > old_v))

            def actor_loss(p):
                action = self.actor(p, obs)
                q = self.q(state["p"]["critic"], obs, action)[0]
                return (
                    -self.c["alpha"] * q.mean() / self.q_scale(q)
                    + (mask * (action - batch["actions"]) ** 2).mean()
                )

            state, logs["actor_loss"] = self.update_network(
                state, "actor", actor_loss, self.c["actor_lr"]
            )
            logs["bc_mask_fraction"] = mask.mean()
            state = self.update_targets(state)
        return state, logs
