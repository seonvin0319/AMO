"""IQL+AMO dual log-beta adaptation, AMO 7068c765.

The shared critic retains IQL's V-based Bellman target. The second actor's
L2_RMS is a policy-action Q-displacement surrogate, not that Bellman target.
"""

import math

from algorithms.jax.iql import Agent as IQLAgent
from common.optim import adam, target_update
from common.tree import map_tree


class Agent(IQLAgent):
    def beta(self, scale):
        return self.ops.float32(self.ops.exp(scale["rho"]))

    def weights(self, beta, adv):
        o = self.ops
        cap = self.c["weight_cap"]
        return o.clip(o.exp(o.clip(beta * adv, high=math.log(cap))), high=cap)

    def virtual(self, state, name, batch, adv, beta, lr, sgd=False):
        p = state["p"][name]
        weights = self.weights(beta, adv)
        _, grads = self.ops.grad(
            lambda p: self.policy_loss(
                p, batch["observations"], batch["actions"], weights
            ),
            p,
        )
        if sgd:
            return map_tree(lambda a, g: a - lr * g, p, grads)
        # Source Adam uses an exact forward sqrt and 1e-12 derivative floor,
        # distinct from the exact RMS zero subgradient.
        return adam(
            self.ops,
            p,
            grads,
            state["opt"][name],
            lr,
            sqrt_fn=self.ops.virtual_sqrt,
        )[0]

    def execution_outer(self, scale, state, batch, outer, adv, lr):
        o = self.ops
        plus = self.virtual(state, "actor", batch, adv, self.beta(scale), lr)
        obs = outer["observations"]
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

    def bootstrap_terms(self, scale, state, batch, outer, adv, lr):
        o = self.ops
        beta = self.beta(scale)
        plus = self.virtual(state, "bootstrap", batch, adv, beta, lr, sgd=True)
        target = map_tree(o.stop, state["target"]["critic"])
        obs, next_obs = outer["observations"], outer["next_observations"]
        action = self.actor(plus, obs)
        q = self.minq(target, obs, action)
        scale_q = o.stop(o.abs(q).mean()) + self.c["smoothness_eps"]
        next_new = self.minq(target, next_obs, self.actor(plus, next_obs))
        next_old = o.stop(
            self.minq(target, next_obs, self.actor(state["p"]["bootstrap"], next_obs))
        )
        delta = self.c["discount"] * (1 - outer["terminals"]) * (next_new - next_old)
        rms = o.safe_sqrt(((delta / scale_q) ** 2).mean())
        factor = 2 * o.stop(beta)
        l1 = -factor * q.mean() / scale_q + ((action - outer["actions"]) ** 2).mean()
        return l1, factor * rms

    def update_scale(self, state, name, loss_fn):
        state, loss = self.update_network(
            state, name, loss_fn, self.c["rho_lr"], beta1=0.0, beta2=0.999
        )
        scale = {
            "rho": self.ops.clip(
                state["p"][name]["rho"],
                math.log(self.c["beta_min"]),
                math.log(self.c["beta_max"]),
            )
        }
        return {**state, "p": {**state["p"], name: scale}}, loss

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        # Cache Q/V quantities before either network is updated.
        next_v = o.stop(self.value(state["p"]["value"], batch["next_observations"]))
        target_q = o.stop(self.minq(state["target"]["critic"], obs, actions))
        adv = o.stop(target_q - self.value(state["p"]["value"], obs))

        def value_loss(p):
            diff = target_q - self.value(p, obs)
            weight = o.where(
                diff > 0, diff * 0 + c["expectile"], diff * 0 + 1 - c["expectile"]
            )
            return (weight * diff**2).mean()

        state, vl = self.update_network(state, "value", value_loss, c["value_lr"])
        target = o.stop(
            batch["rewards"] + c["discount"] * (1 - batch["terminals"]) * next_v
        )
        state, ql = self.update_network(
            state,
            "critic",
            lambda p: sum(((q - target) ** 2).mean() for q in self.q(p, obs, actions))
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
        beta_e = o.stop(self.beta(state["p"]["scale_E"]))
        beta_b = o.stop(self.beta(state["p"]["scale_B"]))
        lr = (
            c["actor_lr"]
            * 0.5
            * (1 + o.cos(math.pi * (noise["step"] - 1) / c["max_steps"]))
        )
        # The beta0 actor takes its real Adam step BEFORE its virtual SGD step.
        state, bl = self.update_network(
            state,
            "bootstrap",
            lambda p: self.policy_loss(p, obs, actions, self.weights(beta_b, adv)),
            lr,
        )
        logs = {
            "critic_loss": ql,
            "value_loss": vl,
            "bootstrap_loss": bl,
            "beta_E_used": beta_e,
            "beta_B_used": beta_b,
        }
        if meta_step:
            frozen = map_tree(o.stop, state)
            state, logs["L_E"] = self.update_scale(
                state,
                "scale_E",
                lambda p: self.execution_outer(p, frozen, batch, outer, adv, lr),
            )
            # Source advances the beta0 cosine scheduler after its actual step;
            # its virtual SGD therefore uses the NEXT learning rate.
            lr_b = (
                c["actor_lr"]
                * 0.5
                * (1 + o.cos(math.pi * noise["step"] / c["max_steps"]))
            )
            frozen = map_tree(o.stop, state)
            l1, l2 = self.bootstrap_terms(
                state["p"]["scale_B"], frozen, batch, outer, adv, lr_b
            )
            state, logs["L_T_B"] = self.update_scale(
                state,
                "scale_B",
                lambda p: sum(self.bootstrap_terms(p, frozen, batch, outer, adv, lr_b)),
            )
            logs.update(L1_B=l1, L2_RMS_B=l2)
        # Cached beta_E reproduces the real source step, including meta events.
        state, logs["actor_loss"] = self.update_network(
            state,
            "actor",
            lambda p: self.policy_loss(p, obs, actions, self.weights(beta_e, adv)),
            lr,
        )
        logs.update(
            beta_E=self.beta(state["p"]["scale_E"]),
            beta_B=self.beta(state["p"]["scale_B"]),
        )
        return state, logs
