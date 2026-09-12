"""Native ReBRAC JAX updater as an external oracle for the port."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import optax

from common.agent import make_agent
from tests.references import rebrac as ref
from tests.release.test_contracts import batch, config


def actor_vars(p):
    return {
        "params": {
            f"Dense_{i}": {"kernel": v["w"], "bias": v["b"]}
            for i, v in enumerate(p["net"].values())
        }
    }


def critic_vars(p):
    q = {}
    for i in range(len(p["q1"])):
        a, b = p["q1"][f"layer{i}"], p["q2"][f"layer{i}"]
        q[f"Dense_{i}"] = {
            "kernel": jnp.stack((a["w"], b["w"])),
            "bias": jnp.stack((a["b"], b["b"])),
        }
        if "scale" in a:
            q[f"LayerNorm_{i}"] = {
                "scale": jnp.stack((a["scale"], b["scale"])),
                "bias": jnp.stack((a["offset"], b["offset"])),
            }
    return {"params": {"VmapCritic_0": q}}


class Metrics:
    def update(self, values):
        return self


def test_native_rebrac_updates(monkeypatch):
    c = config("rebrac")
    agent = make_agent("rebrac", "jax", 3, 2, c, seed=23)
    actor_module = ref.DetActor(
        action_dim=2, hidden_dim=8, layernorm=False, n_hiddens=3
    )
    critic_module = ref.EnsembleCritic(
        hidden_dim=8, num_critics=2, layernorm=True, n_hiddens=3
    )
    av, cv = actor_vars(agent.state["p"]["actor"]), critic_vars(
        agent.state["p"]["critic"]
    )
    actor = ref.ActorTrainState.create(
        apply_fn=actor_module.apply,
        params=av,
        target_params=av,
        tx=optax.adam(c["actor_lr"]),
    )
    critic = ref.CriticTrainState.create(
        apply_fn=critic_module.apply,
        params=cv,
        target_params=cv,
        tx=optax.adam(c["critic_lr"]),
    )
    b = batch(3)
    native = {
        "states": jnp.asarray(b["observations"]),
        "next_states": jnp.asarray(b["next_observations"]),
        "actions": jnp.asarray(b["actions"]),
        "next_actions": jnp.asarray(b["next_actions"]),
        "rewards": jnp.asarray(b["rewards"].ravel()),
        "dones": jnp.asarray(b["terminals"].ravel()),
    }
    key = jax.random.PRNGKey(0)
    for step in range(1, 7):
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(agent.rng.bit_generator.state)
        noise = jnp.asarray(rng.standard_normal((6, 2)).astype(np.float32))
        monkeypatch.setattr(jax.random, "normal", lambda key, shape: noise)
        agent.update(b)
        key, critic, _ = ref.update_critic(
            key,
            actor,
            critic,
            native,
            c["discount"],
            c["critic_bc_coef"],
            c["tau"],
            c["policy_noise"],
            c["noise_clip"],
            Metrics(),
        )
        if (step - 1) % 2 == 0:
            key, actor, critic, _ = ref.update_actor(
                key,
                actor,
                critic,
                native,
                c["actor_bc_coef"],
                c["tau"],
                True,
                Metrics(),
            )
        from common.tree import flatten

        for actual, expected in (
            (actor_vars(agent.state["p"]["actor"]), actor.params),
            (critic_vars(agent.state["p"]["critic"]), critic.params),
            (actor_vars(agent.state["target"]["actor"]), actor.target_params),
            (critic_vars(agent.state["target"]["critic"]), critic.target_params),
        ):
            for k, v in flatten(actual).items():
                np.testing.assert_allclose(
                    v, flatten(expected)[k], atol=1e-5, rtol=5e-5, err_msg=k
                )
