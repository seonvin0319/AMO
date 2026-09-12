"""Offline transition loading with explicit terminal/timeout semantics."""

import hashlib
from pathlib import Path

import numpy as np


def qlearning_dataset(raw, max_episode_steps=1000):
    """D4RL transition construction, including ReBRAC's aligned next actions.

    Timeout transitions are omitted (not relabeled as terminal). Next actions
    are taken from the raw sequence before filtering, never shifted afterwards.
    """
    n = len(raw["rewards"])
    indices = []
    episode_step = 0
    for i in range(n - 1):
        timeout = (
            bool(raw["timeouts"][i])
            if "timeouts" in raw
            else episode_step == max_episode_steps - 1
        )
        if timeout:
            episode_step = 0
            continue
        if bool(raw["terminals"][i]):
            episode_step = 0
        indices.append(i)
        episode_step += 1
    idx = np.asarray(indices, np.int64)
    return {
        "observations": raw["observations"][idx],
        "actions": raw["actions"][idx],
        "rewards": raw["rewards"][idx],
        "terminals": raw["terminals"][idx],
        "next_observations": raw["observations"][idx + 1],
        "next_actions": raw["actions"][idx + 1],
    }


def load_dataset(path=None, env_name=None, rebrac=False):
    if path:
        path = Path(path)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                raw = {k: data[k] for k in data.files}
        elif path.suffix in (".h5", ".hdf5"):
            import h5py

            with h5py.File(path) as data:
                raw = {
                    k: data[k][:]
                    for k in (
                        "observations",
                        "actions",
                        "rewards",
                        "terminals",
                        "timeouts",
                        "next_observations",
                        "next_actions",
                    )
                    if k in data
                }
        else:
            raise ValueError("Dataset must be .npz, .h5 or .hdf5")
    else:
        import d4rl  # noqa: F401
        import gym

        env = gym.make(env_name)
        try:
            raw = env.get_dataset()
        finally:
            env.close()
    prepared = bool(path and path.suffix == ".npz" and "next_observations" in raw)
    if prepared and "timeouts" in raw and np.any(raw["timeouts"]):
        raise ValueError(
            "Transition NPZ contains unresolved timeouts; export filtered transitions or a raw sequence without next_observations"
        )
    data = raw if prepared else qlearning_dataset(raw)
    required = {"observations", "actions", "rewards", "terminals", "next_observations"}
    if rebrac:
        required.add("next_actions")
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"Missing transition arrays: {sorted(missing)}. For ReBRAC use raw D4RL data or supply aligned next_actions."
        )
    data = {k: np.asarray(data[k], np.float32).copy() for k in required}
    data["rewards"] = data["rewards"].reshape(-1, 1)
    data["terminals"] = data["terminals"].reshape(-1, 1)
    n = len(data["actions"])
    if n == 0 or any(len(v) != n for v in data.values()):
        raise ValueError("All transition arrays must be nonempty and aligned")
    if (
        data["observations"].ndim != 2
        or data["actions"].ndim != 2
        or data["observations"].shape != data["next_observations"].shape
    ):
        raise ValueError(
            "Expected matrix observations/actions with matching next-observation shape"
        )
    if any(not np.isfinite(v).all() for v in data.values()):
        raise ValueError("Dataset contains NaN or infinity")
    if np.any((data["terminals"] != 0) & (data["terminals"] != 1)):
        raise ValueError("terminals must be binary; timeouts are a separate field")
    if np.max(np.abs(data["actions"])) > 1.00001:
        raise ValueError("This release targets D4RL actions in [-1, 1]")
    if "next_actions" in data:
        if data["next_actions"].shape != data["actions"].shape:
            raise ValueError("next_actions must match actions shape")
        if np.max(np.abs(data["next_actions"])) > 1.00001:
            raise ValueError("next_actions must also lie in [-1, 1]")
    return data


def fingerprint(data):
    h = hashlib.sha256()
    for k in sorted(data):
        h.update(k.encode())
        h.update(str(data[k].shape).encode())
        h.update(np.ascontiguousarray(data[k]).tobytes())
    return h.hexdigest()


def preprocess(data, config):
    data = {k: v.copy() for k, v in data.items()}
    mean = (
        data["observations"].mean(axis=0)
        if config["normalize"]
        else np.zeros(data["observations"].shape[1], np.float32)
    )
    std = (
        data["observations"].std(axis=0) + 1e-3
        if config["normalize"]
        else np.ones_like(mean)
    )
    for k in ("observations", "next_observations"):
        data[k] = (data[k] - mean) / std
    transform = config["reward_transform"]
    if transform == "antmaze_shift":
        data["rewards"] -= 1
    elif transform == "antmaze_scale100":
        data["rewards"] *= 100
    elif transform == "return_range":
        returns, length, total = [], 0, 0.0
        for reward, done in zip(data["rewards"].ravel(), data["terminals"].ravel()):
            total += float(reward)
            length += 1
            if done or length == 1000:
                returns.append(total)
                length, total = 0, 0.0
        spread = np.ptp(returns) if returns else 0
        if spread <= 0:
            raise ValueError(
                "Reward return range is zero or no complete episode was found"
            )
        data["rewards"] *= 1000 / spread
    return data, mean, std


class ReplayBuffer:
    def __init__(self, data, seed):
        self.data = data
        self.rng = np.random.default_rng(seed)

    def sample(self, size):
        idx = self.rng.integers(len(self.data["actions"]), size=size)
        return {k: v[idx] for k, v in self.data.items()}
