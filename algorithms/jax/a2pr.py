"""A2PR with its original deep networks and advantage-weighted VAE."""

from common import jax_backend as backend
from common.agent import BaseAgent
from common.networks import mlp
from common.tree import map_tree


class Agent(BaseAgent):
    Backend = backend.Backend

    def vae(self, params, observations, actions, noise):
        o = self.ops
        hidden = o.relu(mlp(o, params["encoder"], o.cat((observations, actions))))
        mean = mlp(o, params["mean"], hidden)
        log_std = o.clip(mlp(o, params["log_std"], hidden), -4.0, 15.0)
        latent = mean + o.exp(log_std) * noise
        recon = mlp(o, params["decoder"], o.cat((observations, latent)), tanh=True)
        return recon, mean, log_std

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        state, logs, target, next_actions = self.td_critic(state, batch, noise)
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        new_q_next = self.minq(
            state["p"]["critic"], batch["next_observations"], next_actions
        )
        target_v = o.stop(
            o.minimum(
                target,
                batch["rewards"]
                + (1 - batch["terminals"]) * c["discount"] * new_q_next,
            )
        )
        old_v = o.stop(self.value(state["p"]["value"], obs))
        state, logs["value_loss"] = self.update_network(
            state,
            "value",
            lambda p: ((self.value(p, obs) - target_v) ** 2).mean(),
            c["value_lr"],
        )
        q_data = o.stop(self.q(state["p"]["critic"], obs, actions)[0])
        weight = o.stop(1.0 * (q_data > old_v)) * c["vae_weight"]

        def vae_loss(p):
            recon, mean, log_std = self.vae(p, obs, actions, noise["vae"])
            reconstruction = (weight * (recon - actions) ** 2).mean()
            kl = -0.5 * (1 + 2 * log_std - mean**2 - o.exp(2 * log_std)).mean()
            return reconstruction + 0.5 * kl

        state, logs["vae_loss"] = self.update_network(
            state, "vae", vae_loss, c["vae_lr"]
        )
        if actor_step:
            frozen_vae = map_tree(o.stop, state["p"]["vae"])
            frozen_q = map_tree(o.stop, state["p"]["critic"])

            def actor_loss(p):
                pi = self.actor(p, obs)
                q_pi = self.q(frozen_q, obs, pi)[0]
                recon, _, _ = self.vae(frozen_vae, obs, pi, noise["reconstruction"])
                q_recon = self.q(frozen_q, obs, recon)[0]
                # Keep the reconstruction's action gradient, as in the source.
                embedding = o.where(q_recon >= q_data, recon, actions)
                mask = c["mask"] * (1.0 * ((q_recon >= old_v) | (q_data > old_v)))
                return (
                    -c["alpha"] * q_pi.mean() / self.q_scale(q_pi)
                    + (o.stop(mask) * (pi - embedding) ** 2).mean()
                )

            state, logs["actor_loss"] = self.update_network(
                state, "actor", actor_loss, c["actor_lr"]
            )
            state = self.update_targets(state)
        return state, logs
