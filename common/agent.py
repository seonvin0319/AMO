"""Functional training state, portable checkpoints, and common TD3 operations."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np

from common.networks import init_mlp, mlp
from common.optim import adam, init_adam, target_update
from common.tree import flatten, map_tree, unflatten


class BaseAgent:
    def __init__(self, observation_dim, action_dim, config, seed=0, device="cpu"):
        self.c = dict(config)
        self.ops = self.Backend(device, float64=config["algorithm"] == "iql_amo")
        self.observation_dim, self.action_dim = observation_dim, action_dim
        self.rng = np.random.default_rng(seed)
        self.steps = 0
        self.state = self.ops.finish(map_tree(self.ops.array, self.initialize()))
        self._compiled_step = self.ops.compile(self.step)

    def net(
        self,
        input_dim,
        output_dim,
        depth=2,
        layernorm=False,
        special=False,
        bound=0.003,
        width=None,
    ):
        return init_mlp(
            self.rng,
            input_dim,
            output_dim,
            width or self.c["hidden_dim"],
            depth,
            layernorm,
            special,
            bound,
        )

    def initialize(self):
        c = self.c
        actor = {
            "net": self.net(
                self.observation_dim,
                self.action_dim,
                c["actor_depth"],
                special=c.get("actor_special_init", False),
                bound=0.001,
            )
        }
        if c.get("gaussian", False):
            actor["log_std"] = np.zeros(self.action_dim, np.float32)
        p = {
            "actor": actor,
            "critic": {
                f"q{i}": self.net(
                    self.observation_dim + self.action_dim,
                    1,
                    c["critic_depth"],
                    c["critic_layernorm"],
                    c.get("critic_special_init", False),
                )
                for i in (1, 2)
            },
        }
        if c.get("value_depth"):
            p["value"] = self.net(self.observation_dim, 1, c["value_depth"])
        if c["algorithm"] == "td3_amo":
            p["bootstrap"] = copy.deepcopy(actor)
            p["scale_E"] = {
                "rho": np.asarray(c["T_E"] + np.log(-np.expm1(-c["T_E"])), np.float32)
            }
            p["scale_B"] = {
                "rho": np.asarray(c["T_B"] + np.log(-np.expm1(-c["T_B"])), np.float32)
            }
        if c["algorithm"] == "iql_amo":
            p["bootstrap"] = copy.deepcopy(actor)
            for role in ("E", "B"):
                p[f"scale_{role}"] = {
                    "rho": np.asarray(np.log(c["beta_initial"]), np.float64)
                }
        if c["algorithm"] == "aspc":
            a = c["alpha"]
            p["scale"] = {"rho": np.asarray(a + np.log(-np.expm1(-a)), np.float32)}
        if c["algorithm"] == "a2pr":
            latent = 2 * self.action_dim
            width = c["vae_hidden_dim"]
            p["vae"] = {
                "encoder": self.net(
                    self.observation_dim + self.action_dim, width, 1, width=width
                ),
                "mean": self.net(width, latent, 0),
                "log_std": self.net(width, latent, 0),
                "decoder": self.net(
                    self.observation_dim + latent, self.action_dim, 2, width=width
                ),
            }
        target_actor = p.get("bootstrap", actor)
        optimizers = {k: init_adam(v) for k, v in p.items()}
        if c["algorithm"] == "iql_amo":
            for role in ("E", "B"):
                optimizers[f"scale_{role}"]["count"] = np.asarray(0, np.float64)
        return {
            "p": p,
            "opt": optimizers,
            "target": {
                "actor": copy.deepcopy(target_actor),
                "critic": copy.deepcopy(p["critic"]),
            },
            "aux": {
                "ema_q": np.asarray(0.0, np.float32),
                "previous_ema_q": np.asarray(0.0, np.float32),
            },
        }

    def actor(self, params, observations):
        return mlp(self.ops, params["net"], observations, tanh=True)

    def q(self, params, observations, actions):
        sa = self.ops.cat((observations, actions))
        return tuple(
            mlp(self.ops, params[f"q{i}"], sa, ln_eps=self.c.get("ln_eps", 1e-5))
            for i in (1, 2)
        )

    def minq(self, params, observations, actions):
        return self.ops.minimum(*self.q(params, observations, actions))

    def value(self, params, observations):
        return mlp(self.ops, params, observations)

    def q_scale(self, q):
        return self.ops.stop(self.ops.clip(self.ops.abs(q).mean(), low=1e-6))

    def update_network(self, state, name, loss_fn, lr=None, **optimizer_kwargs):
        loss, grads = self.ops.grad(loss_fn, state["p"][name])
        params, opt = adam(
            self.ops,
            state["p"][name],
            grads,
            state["opt"][name],
            self.c["critic_lr"] if lr is None else lr,
            **optimizer_kwargs,
        )
        return {
            **state,
            "p": {**state["p"], name: params},
            "opt": {**state["opt"], name: opt},
        }, loss

    def td_target(self, state, batch, noise):
        o, c = self.ops, self.c
        next_action = o.clip(
            self.actor(state["target"]["actor"], batch["next_observations"])
            + o.clip(
                noise["target"] * c["policy_noise"], -c["noise_clip"], c["noise_clip"]
            ),
            -1.0,
            1.0,
        )
        next_q = self.minq(
            state["target"]["critic"], batch["next_observations"], next_action
        )
        if c["algorithm"] == "rebrac":
            next_q = next_q - c["critic_bc_coef"] * (
                (next_action - batch["next_actions"]) ** 2
            ).sum(axis=-1, keepdims=True)
        target = batch["rewards"] + (1 - batch["terminals"]) * c["discount"] * next_q
        return o.stop(target), o.stop(next_action)

    def td_critic(self, state, batch, noise):
        target, next_action = self.td_target(state, batch, noise)

        def loss_fn(p):
            return sum(
                ((q - target) ** 2).mean()
                for q in self.q(p, batch["observations"], batch["actions"])
            )

        state, loss = self.update_network(state, "critic", loss_fn)
        return state, {"critic_loss": loss}, target, next_action

    def update_targets(self, state, actor_params=None):
        source = (
            state["p"].get("bootstrap", state["p"]["actor"])
            if actor_params is None
            else actor_params
        )
        return {
            **state,
            "target": {
                "actor": target_update(
                    self.ops, source, state["target"]["actor"], self.c["tau"]
                ),
                "critic": target_update(
                    self.ops,
                    state["p"]["critic"],
                    state["target"]["critic"],
                    self.c["tau"],
                ),
            },
        }

    def meta_due(self, step):
        warmup = self.c.get("meta_warmup_steps", 0)
        return step > warmup and (step - warmup) % self.c.get("meta_interval", 20) == 0

    def update(self, batch, outer=None):
        step = self.steps + 1
        # Original ReBRAC's epoch loop uses zero-based indices: first update
        # includes actor/targets, followed by every second critic update.
        offset = 1 if self.c["algorithm"] == "rebrac" else 0
        actor_step = (step - offset) % self.c.get("policy_freq", 1) == 0
        meta_step = actor_step and self.meta_due(step)
        if (
            self.c["algorithm"] in ("td3_amo", "iql_amo")
            and meta_step
            and outer is None
        ):
            raise ValueError(
                "AMO requires an independently sampled outer batch on scale updates."
            )
        batch = {k: np.asarray(v, np.float32) for k, v in batch.items()}
        outer = (
            batch
            if outer is None
            else {k: np.asarray(v, np.float32) for k, v in outer.items()}
        )
        n = len(batch["actions"])
        noise = {
            "target": self.rng.standard_normal((n, self.action_dim)).astype(np.float32),
            "step": np.asarray(step, np.float32),
        }
        if self.c["algorithm"] == "a2pr":
            noise.update(
                vae=self.rng.standard_normal((n, 2 * self.action_dim)).astype(
                    np.float32
                ),
                reconstruction=self.rng.standard_normal(
                    (n, 2 * self.action_dim)
                ).astype(np.float32),
            )
        new_state, metrics = self._compiled_step(
            self.state,
            map_tree(self.ops.array, batch),
            map_tree(self.ops.array, outer),
            map_tree(self.ops.array, noise),
            actor_step=actor_step,
            meta_step=meta_step,
        )
        self.state = self.ops.finish(new_state)
        self.steps = step
        result = {k: float(self.ops.numpy(v)) for k, v in metrics.items()}
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError(
                f"Non-finite training metric at step {step}: {result}"
            )
        return result

    def act(self, observations):
        obs = np.asarray(observations, np.float32)
        singleton = obs.ndim == 1
        action = self.actor(
            self.state["p"]["actor"], self.ops.array(obs[None] if singleton else obs)
        )
        result = self.ops.numpy(action)
        return result[0] if singleton else result

    def save(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "config": self.c,
            "backend": self.ops.name,
            "steps": self.steps,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "rng": self.rng.bit_generator.state,
            "extra": extra or {},
        }
        arrays = {k: self.ops.numpy(v) for k, v in flatten(self.state).items()}
        arrays["__metadata__"] = np.asarray(json.dumps(metadata))
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            np.savez(f, **arrays)
        tmp.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__metadata__"]))
            if meta["format_version"] != 1 or meta["config"] != self.c:
                raise ValueError("Checkpoint configuration does not match this agent.")
            if (meta["observation_dim"], meta["action_dim"]) != (
                self.observation_dim,
                self.action_dim,
            ):
                raise ValueError(
                    "Checkpoint observation/action dimensions do not match."
                )
            self.state = self.ops.finish(
                map_tree(
                    self.ops.array,
                    unflatten({k: data[k] for k in data.files if k != "__metadata__"}),
                )
            )
        self.steps = meta["steps"]
        self.rng.bit_generator.state = meta["rng"]
        return meta["extra"]


def make_agent(algorithm, backend, observation_dim, action_dim, config, **kwargs):
    if backend not in ("torch", "jax"):
        raise ValueError(f"Unknown backend: {backend}")
    from common.config import ALGORITHMS

    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    module = importlib.import_module(f"algorithms.{backend}.{algorithm}")
    return module.Agent(observation_dim, action_dim, config, **kwargs)
