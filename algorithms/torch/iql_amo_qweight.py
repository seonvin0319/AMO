"""IQL + π_B-weighted Q regression with optional β_B / β_E AMO.

Keeps IQL data-action expectile V and V-based Bellman targets. Bootstrap actor
π_B only reweights the Q regression on data actions (via π_B / μ_hat); it does
not inject actions into V/Q targets. β_B outer is unweighted TD MSE of a
virtual critic — not L1/L2_RMS and not a claimed return improvement.
"""

from __future__ import annotations

import math

from algorithms.torch.iql import Agent as IQLAgent
from common.optim import adam, target_update
from common.tree import map_tree


class Agent(IQLAgent):
    def beta(self, scale):
        return self.ops.float32(self.ops.exp(scale["rho"]))

    def awr_weights(self, beta, adv):
        o = self.ops
        cap = self.c["weight_cap"]
        return o.clip(o.exp(o.clip(beta * adv, high=math.log(cap))), high=cap)

    def log_prob(self, p, obs, actions):
        """Diagonal Gaussian log-density in dataset action coordinates (tanh mean).

        Matches ``policy_loss`` NLL (no tanh-Jacobian correction): shared
        ``log_std`` per dim, mean = tanh(MLP(s)).
        """
        o = self.ops
        mean = self.actor(p, obs)
        log_std = o.clip(p["log_std"], -20.0, 2.0)
        var_term = 0.5 * ((actions - mean) / o.exp(log_std)) ** 2
        return -(var_term + log_std + 0.5 * math.log(2 * math.pi)).sum(
            axis=-1, keepdims=True
        )

    def behavior_bc_loss(self, p, obs, actions):
        return -self.log_prob(p, obs, actions).mean()

    def q_data_weights(self, bootstrap_p, behavior_p, obs, actions, *, stop_w: bool):
        """Normalized clipped density ratio w = ratio / mean(ratio)."""
        o = self.ops
        c = self.c
        log_pi = self.log_prob(bootstrap_p, obs, actions)
        log_mu = self.log_prob(behavior_p, obs, actions)
        log_mu = o.stop(log_mu)
        lo = math.log(c["qweight_w_min"])
        hi = math.log(c["qweight_w_max"])
        log_ratio = o.clip(log_pi - log_mu, low=lo, high=hi)
        ratio = o.exp(log_ratio)
        mean_ratio = ratio.mean() + 1e-8
        w = ratio / mean_ratio
        clip_hi = o.float32(log_ratio >= hi - 1e-6)
        clip_lo = o.float32(log_ratio <= lo + 1e-6)
        sum_w = w.sum()
        ess = (sum_w * sum_w) / ((w * w).sum() + 1e-8)
        extras = dict(
            log_pi_mean=log_pi.mean(),
            log_mu_mean=log_mu.mean(),
            log_ratio_mean=log_ratio.mean(),
            log_ratio_min=log_ratio.min(),
            log_ratio_max=log_ratio.max(),
            w_min=w.min(),
            w_max=w.max(),
            w_clip_hi_frac=clip_hi.mean(),
            w_clip_lo_frac=clip_lo.mean(),
            ess=ess / (w.shape[0] + 0.0),
            ratio_mean=mean_ratio,
        )
        if stop_w:
            w = o.stop(w)
        return w, extras

    def weighted_q_loss(self, critic_p, obs, actions, target, w):
        """mean_i[ w_i * mean_j (Q_j - y)^2 ]."""
        o = self.ops
        qs = self.q(critic_p, obs, actions)
        twin = sum((q - target) ** 2 for q in qs) / 2
        return (w * twin).mean()

    def virtual_actor(self, state, name, batch, adv, beta, lr, sgd=False):
        p = state["p"][name]
        weights = self.awr_weights(beta, adv)
        _, grads = self.ops.grad(
            lambda params: self.policy_loss(
                params, batch["observations"], batch["actions"], weights
            ),
            p,
        )
        if sgd:
            return map_tree(lambda a, g: a - lr * g, p, grads)
        return adam(
            self.ops,
            p,
            grads,
            state["opt"][name],
            lr,
            sqrt_fn=self.ops.virtual_sqrt,
        )[0]

    def virtual_adam_params(self, params, grads, opt_state, lr):
        return adam(
            self.ops,
            params,
            grads,
            opt_state,
            lr,
            sqrt_fn=self.ops.virtual_sqrt,
        )[0]

    def execution_outer(self, scale, state, batch, outer, adv, lr):
        o = self.ops
        plus = self.virtual_actor(state, "actor", batch, adv, self.beta(scale), lr)
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

    def beta_b_outer_td(
        self,
        scale,
        state,
        batch,
        outer,
        adv,
        lr_b,
        critic_p,
        critic_opt,
        y_inner,
        y_outer,
    ):
        """Virtual π_B → w_plus → VirtualAdam critic → unweighted outer TD MSE."""
        o = self.ops
        obs, actions = batch["observations"], batch["actions"]
        beta = self.beta(scale)
        plus = self.virtual_actor(
            state, "bootstrap", batch, adv, beta, lr_b, sgd=True
        )
        w_plus, _ = self.q_data_weights(
            plus, state["p"]["behavior"], obs, actions, stop_w=False
        )

        def critic_loss(p):
            return self.weighted_q_loss(p, obs, actions, y_inner, w_plus)

        _, grads = o.grad(critic_loss, critic_p)
        q_plus = self.virtual_adam_params(critic_p, grads, critic_opt, self.c["critic_lr"])
        o_obs, o_act = outer["observations"], outer["actions"]
        qs = self.q(q_plus, o_obs, o_act)
        twin = sum((q - y_outer) ** 2 for q in qs) / 2
        return twin.mean()

    def _actor_lr(self, step_completed):
        c = self.c
        return (
            c["actor_lr"]
            * 0.5
            * (1 + self.ops.cos(math.pi * step_completed / c["max_steps"]))
        )

    def step(self, state, batch, outer, noise, *, actor_step, meta_step):
        o, c = self.ops, self.c
        obs, actions = batch["observations"], batch["actions"]
        # 1) Pre-update caches (stop-grad).
        next_v = o.stop(self.value(state["p"]["value"], batch["next_observations"]))
        q_data = o.stop(self.minq(state["target"]["critic"], obs, actions))
        v_pre = o.stop(self.value(state["p"]["value"], obs))
        adv = o.stop(q_data - v_pre)
        y_inner = o.stop(
            batch["rewards"] + c["discount"] * (1 - batch["terminals"]) * next_v
        )
        outer_next_v = o.stop(
            self.value(state["p"]["value"], outer["next_observations"])
        )
        y_outer = o.stop(
            outer["rewards"]
            + c["discount"] * (1 - outer["terminals"]) * outer_next_v
        )

        # 2) Real V (unweighted expectile; fixed expectile from config).
        def value_loss(p):
            diff = q_data - self.value(p, obs)
            weight = o.where(
                diff > 0, diff * 0 + c["expectile"], diff * 0 + 1 - c["expectile"]
            )
            return (weight * diff**2).mean()

        state, vl = self.update_network(state, "value", value_loss, c["value_lr"])

        beta_e = o.stop(self.beta(state["p"]["scale_E"]))
        beta_b = o.stop(self.beta(state["p"]["scale_B"]))
        lr = self._actor_lr(noise["step"] - 1)

        # 3) Real bootstrap actor Adam (cached β_B).
        state, bl = self.update_network(
            state,
            "bootstrap",
            lambda p: self.policy_loss(p, obs, actions, self.awr_weights(beta_b, adv)),
            lr,
        )

        # 4) Snapshots for meta: post-bootstrap actor; pre-Q critic + Adam.
        critic_snap = map_tree(o.stop, state["p"]["critic"])
        critic_opt_snap = map_tree(o.stop, state["opt"]["critic"])
        post_bootstrap = map_tree(o.stop, state["p"]["bootstrap"])

        # Weights from current (post-update) π_B for the real Q step.
        if c.get("qweight_enabled", True):
            w, w_stats = self.q_data_weights(
                state["p"]["bootstrap"],
                state["p"]["behavior"],
                obs,
                actions,
                stop_w=True,
            )
        else:
            w = y_inner * 0 + 1
            w_stats = {}

        td_unweighted = sum((q - y_inner) ** 2 for q in self.q(state["p"]["critic"], obs, actions)) / 2
        td_unweighted = td_unweighted.mean()
        td_weighted = self.weighted_q_loss(state["p"]["critic"], obs, actions, y_inner, w)

        # 5) Real Q + target critic.
        state, ql = self.update_network(
            state,
            "critic",
            lambda p: self.weighted_q_loss(p, obs, actions, y_inner, w),
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

        logs = {
            "critic_loss": ql,
            "value_loss": vl,
            "bootstrap_loss": bl,
            "beta_E_used": beta_e,
            "beta_B_used": beta_b,
            "td_unweighted": td_unweighted,
            "td_weighted": td_weighted,
            "Q_data_abs_mean": o.abs(q_data).mean(),
            "Q_data_abs_max": o.abs(q_data).max(),
            "V_abs_mean": o.abs(v_pre).mean(),
            "V_abs_max": o.abs(v_pre).max(),
            **{f"qw_{k}": v for k, v in w_stats.items()},
        }

        # AWR diagnostics (execution).
        awr_e = self.awr_weights(beta_e, adv)
        logs["awr_weight_clip_frac_E"] = o.float32(awr_e >= c["weight_cap"] - 1e-6).mean()
        logs["awr_weight_underflow_frac_E"] = o.float32(awr_e <= 1e-6).mean()

        if meta_step:
            frozen = map_tree(o.stop, state)
            # Restore snapshots into frozen view for virtual paths.
            frozen = {
                **frozen,
                "p": {
                    **frozen["p"],
                    "bootstrap": post_bootstrap,
                    "critic": critic_snap,
                },
                "opt": {**frozen["opt"], "critic": critic_opt_snap},
            }
            lr_b = self._actor_lr(noise["step"])

            if c.get("adapt_beta_E", False):
                state, logs["L_E"] = self.update_scale(
                    state,
                    "scale_E",
                    lambda p: self.execution_outer(p, frozen, batch, outer, adv, lr),
                    lr=c.get("rho_E_lr", c["rho_lr"]),
                )

            if c.get("adapt_beta_B", False) and c.get("qweight_enabled", True):
                # Diagnostic: outer TD with actual-B weights (no virtual actor).
                w_act, _ = self.q_data_weights(
                        post_bootstrap,
                        frozen["p"]["behavior"],
                        obs,
                        actions,
                        stop_w=False,
                    )

                def critic_loss_act(p):
                    return self.weighted_q_loss(p, obs, actions, y_inner, w_act)

                _, g_act = o.grad(critic_loss_act, critic_snap)
                q_plus_act = self.virtual_adam_params(
                        critic_snap, g_act, critic_opt_snap, c["critic_lr"]
                    )
                o_obs, o_act = outer["observations"], outer["actions"]
                twin_act = (
                        sum((q - y_outer) ** 2 for q in self.q(q_plus_act, o_obs, o_act))
                    / 2
                    )
                logs["L_outer_TD_actual_B"] = twin_act.mean()

                def outer_fn(scale):
                    return self.beta_b_outer_td(
                        scale,
                        frozen,
                        batch,
                        outer,
                        adv,
                        lr_b,
                        critic_snap,
                        critic_opt_snap,
                        y_inner,
                        y_outer,
                        )

                state, logs["L_outer_TD_virtual_B"] = self.update_scale(
                    state,
                    "scale_B",
                    outer_fn,
                    lr=c.get("rho_B_lr", c["rho_lr"]),
                    )

        # 7) Real execution actor (cached β_E).
        state, logs["actor_loss"] = self.update_network(
            state,
            "actor",
            lambda p: self.policy_loss(p, obs, actions, self.awr_weights(beta_e, adv)),
            lr,
        )
        logs.update(
            beta_E=self.beta(state["p"]["scale_E"]),
            beta_B=self.beta(state["p"]["scale_B"]),
        )
        return state, logs

    def meta_due(self, step):
        if not (self.c.get("adapt_beta_E") or self.c.get("adapt_beta_B")):
            return False
        return super().meta_due(step)

    def update_scale(self, state, name, loss_fn, lr=None):
        rho_lr = self.c["rho_lr"] if lr is None else lr
        state, loss = self.update_network(
            state, name, loss_fn, rho_lr, beta1=0.0, beta2=0.999
        )
        scale = {
            "rho": self.ops.clip(
                state["p"][name]["rho"],
                math.log(self.c["beta_min"]),
                math.log(self.c["beta_max"]),
            )
        }
        return {**state, "p": {**state["p"], name: scale}}, loss

    def load_behavior(self, path):
        """Overwrite frozen μ_hat from a behavior-only checkpoint."""
        import json
        from pathlib import Path

        import numpy as np

        from common.tree import flatten, unflatten

        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"]))
            flat = {k: data[k] for k in data.files if k != "__metadata__"}
        tree = unflatten(flat)
        if "behavior" in tree.get("p", {}):
            behavior = tree["p"]["behavior"]
        elif "p" in tree and "actor" in tree["p"]:
            # BC script may save under actor key.
            behavior = tree["p"]["actor"]
        else:
            raise ValueError(f"No behavior/actor params in {path}")
        behavior = self.ops.finish(map_tree(self.ops.array, behavior))
        self.state = {
            **self.state,
            "p": {**self.state["p"], "behavior": behavior},
        }
        return meta
