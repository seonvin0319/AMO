"""ReBRAC: separate actor and Bellman-target behavior penalties."""

from common import torch_backend as backend
from common.agent import BaseAgent


class Agent(BaseAgent):
    Backend = backend.Backend

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, _, _ = self.td_critic(state, batch, noise)
        if actor_step:
            old_actor = state["p"]["actor"]

            def actor_loss(p):
                action = self.actor(p, batch["observations"])
                q = self.minq(state["p"]["critic"], batch["observations"], action)
                bc = ((action - batch["actions"]) ** 2).sum(axis=-1).mean()
                scale = self.ops.stop(self.ops.abs(q).mean())
                return self.c["actor_bc_coef"] * bc - q.mean() / scale

            state, logs["actor_loss"] = self.update_network(
                state, "actor", actor_loss, self.c["actor_lr"]
            )
            # Match the native ReBRAC reference's pre-update actor target.
            state = self.update_targets(state, actor_params=old_actor)
        return state, logs
