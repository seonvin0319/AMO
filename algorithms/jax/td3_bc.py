"""TD3+BC with the ASPC robust critic (three layers, LayerNorm)."""

from common import jax_backend as backend
from common.agent import BaseAgent


class Agent(BaseAgent):
    Backend = backend.Backend

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, _, _ = self.td_critic(state, batch, noise)
        if actor_step:

            def actor_loss(p):
                action = self.actor(p, batch["observations"])
                q = self.q(state["p"]["critic"], batch["observations"], action)[0]
                return (
                    -self.c["alpha"] * q.mean() / self.q_scale(q)
                    + ((action - batch["actions"]) ** 2).mean()
                )

            state, loss = self.update_network(
                state, "actor", actor_loss, self.c["actor_lr"]
            )
            logs["actor_loss"] = loss
            state = self.update_targets(state)
        return state, logs
