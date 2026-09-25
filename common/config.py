"""Strict, minimal algorithm configs shared by both backends."""

from pathlib import Path

import yaml

ALGORITHMS = (
    "aspc",
    "wpc",
    "td3_bc",
    "a2pr",
    "rebrac",
    "iql",
    "td3_amo",
    "iql_amo",
    "iql_ddpgbc_amo",
)

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
    # Aligned with default --save-every=20000 offline CPU eval cadence.
    # In-process MuJoCo eval is off unless train.py --eval is passed.
    eval_freq=20000,
    eval_episodes=10,
    eval_seed=None,
    eval_first_step=20000,
    final_eval_repeats=5,
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
        "alpha_E": 5.0,
        "alpha_B": 5.0,
        "bootstrap_loss": "l2_rms",
        "alpha_lr": 0.001,
        "meta_interval": 20,
        "smoothness_eps": 1e-6,
        # Ablation knobs (default dual-actor is L2_RMS-only).
        # bootstrap_loss: "l2_rms" | "l1_l2_rms" | "l1" | "l2"
        #   l2_rms = factor * RMS(Δy/q); l2 = factor * mean-square (no sqrt)
        # execution_score: "bpi" | "direct_q"  (q(s,a+)-q(s,a0))
        # execution_only: True → single π_E for env + Bellman; no π_B/α_B.
        #   α_E meta-loss is L_E (BPI) + L2_RMS(Δy of π_E).
        # execution_meta_loss (dual-actor): "le" | "le_l2_rms"
        #   le = L_E (BPI) only on π_E (BootRMS main).
        #   le_l2_rms = L_E + L2_RMS(Δy of π_E); π_B still uses bootstrap_loss.
        "execution_score": "bpi",
        "execution_only": False,
        "execution_meta_loss": "le",
        # freeze_scale_* skips that coefficient's Adam step. The bootstrap
        # actor still trains. critic_target selects the Bellman actor.
        "freeze_scale_E": False,
        "freeze_scale_B": False,
        "critic_target": "bootstrap",
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
        critic_special_init=False,
        gaussian=True,
        value_depth=2,
        value_layernorm=False,
        value_special_init=False,
        value_lr=0.0003,
        expectile=0.7,
        beta=3.0,
        weight_cap=100.0,
        policy_freq=1,
    ),
    # AMO feat/iql-amo-bpi: log-beta, dual B_pi / L1+L2_RMS.
    # Q/V match TD3-AMO critic RC: depth 3 · LN · special init.
    "iql_amo": dict(
        actor_depth=2,
        critic_depth=3,
        critic_layernorm=True,
        critic_special_init=True,
        gaussian=True,
        value_depth=3,
        value_layernorm=True,
        value_special_init=True,
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
        # le = original L_E (BPI) on π_E. le_l2_rms adds L2_RMS(Δy of π_E).
        # IQL Bellman still uses V; the auxiliary π_B branch is unchanged.
        execution_meta_loss="le",
    ),
    # IQL expectile V/Q + Park et al. 2024 DDPG+BC actor. Single π_E, no π_B.
    # α_E is the inner BC horizon (weight 1/α_E). Meta-loss is L_E (BPI) only.
    "iql_ddpgbc_amo": dict(
        actor_depth=2,
        critic_depth=2,
        critic_layernorm=False,
        critic_special_init=False,
        gaussian=True,
        const_std=True,
        value_depth=2,
        value_layernorm=False,
        value_special_init=False,
        value_lr=0.0003,
        expectile=0.7,
        alpha_E=1.0,
        alpha_lr=0.001,
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
        "alpha_E",
        "alpha_B",
        "alpha_lr",
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
        if config.get("execution_meta_loss") not in ("le", "le_l2_rms"):
            raise ValueError("execution_meta_loss must be 'le' or 'le_l2_rms'")
    if algorithm == "td3_amo":
        if config.get("bootstrap_loss") not in ("l1_l2_rms", "l1", "l2_rms", "l2"):
            raise ValueError(
                "bootstrap_loss must be 'l1_l2_rms', 'l1', 'l2_rms', or 'l2'"
            )
        if config.get("execution_score") not in ("bpi", "direct_q"):
            raise ValueError("execution_score must be 'bpi' or 'direct_q'")
        if not isinstance(config.get("execution_only"), bool):
            raise ValueError("execution_only must be a bool")
        if config["execution_only"] and config.get("execution_score") != "bpi":
            raise ValueError("execution_only ablation keeps execution_score=bpi")
        if config.get("execution_meta_loss") not in ("le", "le_l2_rms"):
            raise ValueError("execution_meta_loss must be 'le' or 'le_l2_rms'")
        if not isinstance(config.get("freeze_scale_E"), bool) or not isinstance(
            config.get("freeze_scale_B"), bool
        ):
            raise ValueError("freeze_scale_E and freeze_scale_B must be bools")
        if config.get("critic_target") not in ("bootstrap", "execution"):
            raise ValueError("critic_target must be 'bootstrap' or 'execution'")
    if algorithm == "iql_ddpgbc_amo":
        if config["alpha_lr"] <= 0 or config["meta_warmup_steps"] < 0:
            raise ValueError("alpha_lr must be positive and warmup nonnegative")
        if not config.get("gaussian", False):
            raise ValueError("iql_ddpgbc_amo requires gaussian=True for DDPG+BC log_prob")
        if not isinstance(config.get("const_std"), bool):
            raise ValueError("const_std must be a bool")
    return config
