"""Strict, minimal algorithm configs shared by both backends."""

from pathlib import Path

import yaml

ALGORITHMS = ("aspc", "wpc", "td3_bc", "a2pr", "rebrac", "iql", "td3_amo", "iql_amo")

SHARED = dict(
    batch_size=256,
    hidden_dim=256,
    discount=0.99,
    tau=0.005,
    actor_lr=0.0003,
    critic_lr=0.0003,
    max_steps=1_000_000,
    normalize=True,
    reward_transform="none",
    eval_freq=5000,
    eval_episodes=10,
    eval_seed=None,
    eval_first_step=5000,
    final_eval_repeats=1,
)
TD3 = dict(
    actor_depth=2,
    critic_depth=3,
    critic_layernorm=True,
    critic_special_init=True,
    policy_freq=2,
    policy_noise=0.2,
    noise_clip=0.5,
)
DEFAULTS = {
    "td3_bc": {**TD3, "alpha": 2.5},
    "wpc": {**TD3, "alpha": 2.5, "value_depth": 2, "value_lr": 0.0003},
    "aspc": {
        **TD3,
        "alpha": 2.5,
        "meta_interval": 20,
        "scale_lr": 0.002,
        "ema_alpha": 0.995,
    },
    "td3_amo": {
        **TD3,
        "T_E": 1.0,
        "T_B": 1.0,
        "T_lr": 0.001,
        "meta_interval": 20,
        "smoothness_eps": 1e-6,
    },
    "a2pr": {
        **TD3,
        "actor_depth": 3,
        "critic_depth": 4,
        "critic_layernorm": False,
        "critic_special_init": False,
        "value_depth": 4,
        "value_lr": 0.0003,
        "vae_lr": 0.001,
        "vae_hidden_dim": 750,
        "alpha": 2.5,
        "mask": 1.0,
        "vae_weight": 1.0,
    },
    "rebrac": {
        **TD3,
        "actor_depth": 3,
        "actor_special_init": True,
        "actor_lr": 0.001,
        "critic_lr": 0.001,
        "actor_bc_coef": 0.001,
        "critic_bc_coef": 0.001,
        "ln_eps": 1e-6,
        "normalize": False,
        "batch_size": 1024,
    },
    "iql": dict(
        actor_depth=2,
        critic_depth=2,
        critic_layernorm=False,
        gaussian=True,
        value_depth=2,
        value_lr=0.0003,
        expectile=0.7,
        beta=3.0,
        weight_cap=100.0,
        policy_freq=1,
    ),
    # AMO feat/iql-amo-bpi: log-beta, dual B_pi / L1+L2_RMS.
    "iql_amo": dict(
        actor_depth=2,
        critic_depth=2,
        critic_layernorm=False,
        gaussian=True,
        value_depth=2,
        value_lr=0.0003,
        expectile=0.7,
        beta_initial=1.0,
        beta_min=0.05,
        beta_max=100.0,
        rho_lr=0.002,
        weight_cap=100.0,
        meta_warmup_steps=100000,
        meta_interval=20,
        outer_batch_size=256,
        policy_freq=1,
        smoothness_eps=1e-6,
    ),
}


def load_config(algorithm, env, path=None, overrides=None):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    config = {**SHARED, **DEFAULTS[algorithm]}
    path = (
        Path(path)
        if path
        else Path(__file__).resolve().parents[1] / "configs" / f"{algorithm}.yaml"
    )
    document = yaml.safe_load(path.read_text()) or {}
    if set(document) - {"defaults", "antmaze", "environments"}:
        raise ValueError(
            "Config only supports defaults, antmaze and environments sections."
        )
    updates = dict(document.get("defaults", {}))
    if env.startswith("antmaze-"):
        updates.update(document.get("antmaze", {}))
    updates.update(document.get("environments", {}).get(env, {}))
    updates.update(overrides or {})
    unknown = set(updates) - set(config)
    if unknown:
        raise ValueError(f"Unknown options for {algorithm}: {sorted(unknown)}")
    config.update(updates)
    config["algorithm"] = algorithm
    if config["reward_transform"] not in (
        "none",
        "antmaze_shift",
        "antmaze_scale100",
        "return_range",
    ):
        raise ValueError("Unknown reward transform.")
    for key in (
        "batch_size",
        "hidden_dim",
        "actor_lr",
        "critic_lr",
        "max_steps",
        "eval_freq",
        "eval_episodes",
        "T_E",
        "T_B",
        "T_lr",
        "meta_interval",
        "outer_batch_size",
        "final_eval_repeats",
    ):
        if key in config and config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if "meta_interval" in config and config["meta_interval"] % config["policy_freq"]:
        raise ValueError("meta_interval must be divisible by policy_freq")
    if algorithm == "iql_amo":
        if not 0 < config["beta_min"] <= config["beta_initial"] <= config["beta_max"]:
            raise ValueError("Require 0 < beta_min <= beta_initial <= beta_max")
        if config["rho_lr"] <= 0 or config["meta_warmup_steps"] < 0:
            raise ValueError("rho_lr must be positive and warmup nonnegative")
    return config
