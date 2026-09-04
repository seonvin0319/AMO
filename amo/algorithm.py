import copy
import json
import math
import os
import random
import signal
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
import wandb


TensorBatch = List[torch.Tensor]


BOOTSTRAP_OUTER_LOSS_VERSION = "tq_detached_rms_target_v1"
BOOTSTRAP_RMS_DIAGNOSTIC_SMALL_H = 1e-3
BOOTSTRAP_RMS_DIAGNOSTIC_LARGE_H = 5e-2


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "halfcheetah-medium-v2"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(5e3)  # How often (time steps) we evaluate
    n_episodes: int = 10  # How many episodes run during evaluation
    max_timesteps: int = int(1e6)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    save_freq: int = 0  # Checkpoint save interval (0 -> every eval, original behavior)
    metrics_log_freq: int = 200  # Steps between JSONL metric dumps (0 -> disabled)
    profile_runtime: bool = False  # synchronize CUDA and log per-iteration cost
    load_model: str = ""  # Model load file name, "" doesn't load
    resume_tag: str = (
        ""  # Reuse a matching AMO results directory and its latest checkpoint.
    )
    # TD3
    buffer_size: int = 10000000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    expl_noise: float = 0.1  # Std of Gaussian exploration noise
    tau: float = 0.005  # Target network update rate
    policy_noise: float = 0.2  # Noise added to target actor during critic update
    noise_clip: float = 0.5  # Range to clip target actor noise
    policy_freq: int = 2  # Frequency of delayed actor updates
    # Execution scale init (T_E). Also the single-scale T when adaptive is off.
    T_E: float = 1.25
    # Bootstrap scale init. None copies T_E. Used only with adaptive_multiscale.
    T_B: Optional[float] = None
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    T_freq: int = 10  # Frequency of outer T updates
    proximal_n_steps: int = 1  # N in T = N * tau
    # Use separate execution/bootstrap actors and learned scales. This mode is
    # distinct from the sequential proximal actor-chain above.
    adaptive_multiscale: bool = False
    bootstrap_outer_loss_version: str = field(
        default=BOOTSTRAP_OUTER_LOSS_VERSION, init=False
    )
    # Read-only finite-difference diagnostics; zero disables them.
    bootstrap_rms_diagnostic_freq: int = 0
    T_lr: float = 2e-4
    smoothness_eps: float = 1e-6
    smoothness_max: Optional[float] = None

    # Wandb logging
    project: str = "AMO"
    group: str = "amo"
    name: str = ""

    def __post_init__(self):
        if self.proximal_n_steps < 1:
            raise ValueError("proximal_n_steps must be >= 1")
        if self.T_E <= 0:
            raise ValueError("T_E must be > 0")
        if self.T_B is not None and self.T_B <= 0:
            raise ValueError("T_B must be > 0")
        if self.T_B is not None and not self.adaptive_multiscale:
            raise ValueError("T_B is only used with adaptive_multiscale")
        if self.T_lr <= 0:
            raise ValueError("T_lr must be > 0")
        if self.adaptive_multiscale and self.proximal_n_steps != 1:
            raise ValueError(
                "adaptive_multiscale cannot be combined with proximal_n_steps; "
                "leave proximal_n_steps at its default value of 1"
            )
        if self.bootstrap_rms_diagnostic_freq < 0:
            raise ValueError("bootstrap_rms_diagnostic_freq must be >= 0")
        root = self.checkpoints_path
        if self.resume_tag and root is not None:
            best_dir, best_step = None, -1
            for d in Path(root).glob(f"{self.resume_tag}-*"):
                if not d.is_dir():
                    continue
                m = d / "metrics.jsonl"
                step = 0
                if m.exists() and m.stat().st_size:
                    step = int(
                        json.loads(m.read_text().strip().splitlines()[-1]).get(
                            "step", 0
                        )
                    )
                if step > best_step:
                    best_step, best_dir = step, d
            if best_dir is not None:
                self.name = best_dir.name
                self.checkpoints_path = str(best_dir)
                return
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if root is not None:
            self.checkpoints_path = os.path.join(root, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def softplus_inverse(value: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of softplus for strictly positive values."""
    tiny = torch.finfo(value.dtype).tiny
    value = value.clamp_min(tiny)
    return value + torch.log(-torch.expm1(-value))


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
            state - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(
            (buffer_size, 1), dtype=torch.float32, device=device
        )
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        # Diagnostic hook for cross-fit unit tests (never used by training logic).
        self.last_indices = torch.as_tensor(indices, dtype=torch.long)
        self.sample_call_count = getattr(self, "sample_call_count", 0) + 1
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    try:
        wandb.init(
            config=config,
            project=config["project"],
            group=config["group"],
            name=config["name"],
            id=str(uuid.uuid4()),
        )
        if wandb.run is not None:
            wandb.run.save()
    except Exception as exc:
        print(f"[wandb] init failed: {exc}")


def wandb_log(payload: dict, step: int) -> None:
    try:
        if wandb.run is not None:
            wandb.log(payload, step=step)
    except Exception as exc:
        print(f"[wandb] log failed: {exc}")


@torch.no_grad()
def eval_actor(
    env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )

        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu") -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()


class Critic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        layernorm: bool = True,
        n_hiddens: int = 3,
    ):
        super(Critic, self).__init__()

        layers = [
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
        ]
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_hiddens - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))

        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)
        self._initialize_weights(state_dim, action_dim, hidden_dim)

    def _initialize_weights(self, state_dim, action_dim, hidden_dim):
        nn.init.uniform_(
            self.net[0].weight,
            -1.0 / np.sqrt(state_dim + action_dim),
            1.0 / np.sqrt(state_dim + action_dim),
        )
        nn.init.constant_(self.net[0].bias, 0.1)

        for i in range(1, len(self.net) - 2):
            if isinstance(self.net[i], nn.Linear):
                nn.init.uniform_(
                    self.net[i].weight,
                    -1.0 / np.sqrt(hidden_dim),
                    1.0 / np.sqrt(hidden_dim),
                )
                nn.init.constant_(self.net[i].bias, 0.1)

        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], 1)
        return self.net(sa)


class Vnet(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Vnet, self).__init__()

        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        value = F.relu(self.l1(state))
        value = F.relu(self.l2(value))
        value = self.l3(value)

        return value


class AMO:
    def __init__(
        self,
        max_action: float,
        actor: nn.Module,
        actor_optimizer: torchopt.Optimizer,
        critic_1: nn.Module,
        critic_1_optimizer: torch.optim.Optimizer,
        critic_2,
        critic_2_optimizer,
        vnet,
        vnet_optimizer,
        *,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        T=1.25,
        T_B=None,
        T_freq=10,
        T_lr=2e-4,
        proximal_n_steps=1,
        adaptive_multiscale=False,
        bootstrap_rms_diagnostic_freq=0,
        smoothness_eps=1e-6,
        smoothness_max=None,
        actor_lr=3e-4,
        device="cpu",
        **_ignored,
    ):
        if T <= 0 or proximal_n_steps < 1:
            raise ValueError("T and proximal_n_steps must be positive")
        if T_B is not None and T_B <= 0:
            raise ValueError("T_B must be > 0")
        if T_B is not None and not adaptive_multiscale:
            raise ValueError("T_B is only used with adaptive_multiscale")
        if adaptive_multiscale and proximal_n_steps != 1:
            raise ValueError(
                "adaptive_multiscale cannot be combined with proximal_n_steps; "
                "leave proximal_n_steps at its default value of 1"
            )
        self.actor, self.actor_optimizer = actor, actor_optimizer
        self.actor_opt_state = actor_optimizer.init(list(actor.parameters()))
        self.critic_1, self.critic_2 = critic_1, critic_2
        self.critic_1_optimizer, self.critic_2_optimizer = (
            critic_1_optimizer,
            critic_2_optimizer,
        )
        self.critic_1_target, self.critic_2_target = copy.deepcopy(
            critic_1
        ), copy.deepcopy(critic_2)
        self.vnet, self.vnet_optimizer = vnet, vnet_optimizer
        self.max_action, self.discount, self.tau = max_action, discount, tau
        self.policy_noise, self.noise_clip, self.policy_freq = (
            policy_noise,
            noise_clip,
            policy_freq,
        )
        self.T_freq, self.proximal_n_steps = T_freq, proximal_n_steps
        self.adaptive_multiscale = bool(adaptive_multiscale)
        self.bootstrap_rms_diagnostic_freq = int(bootstrap_rms_diagnostic_freq)
        self.T_lr = float(T_lr)
        self.actor_lr = float(actor_lr)
        self.smoothness_eps, self.smoothness_max, self.device = (
            smoothness_eps,
            smoothness_max,
            device,
        )
        initial_T = torch.tensor(T, device=device)
        # Keep the original T_E initialization expression bit-for-bit.
        self.log_T = nn.Parameter(torch.log(torch.expm1(initial_T)))
        self.T_optimizer = torch.optim.Adam([self.log_T], lr=T_lr)
        self.T_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.T_optimizer, gamma=0.01 ** (1 / int(1e6 / (policy_freq * T_freq)))
        )
        self.log_T_B = None
        self.T_B_optimizer = None
        self.T_B_scheduler = None
        self.bootstrap_outer_update_count = 0
        self.bootstrap_l1_l2_same_sign_count = 0
        self.bootstrap_l1_l2_nonzero_count = 0
        self._execution_last_delta_log_T = 0.0
        self._bootstrap_last = {}
        if self.adaptive_multiscale:
            tb = float(T if T_B is None else T_B)
            if tb == float(T):
                initial_T_B = initial_T
                self.log_T_B = nn.Parameter(self.log_T.detach().clone())
            else:
                initial_T_B = torch.tensor(tb, device=device)
                self.log_T_B = nn.Parameter(torch.log(torch.expm1(initial_T_B)))
            self.T_B_optimizer = torch.optim.Adam([self.log_T_B], lr=T_lr)
            self.T_B_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.T_B_optimizer,
                gamma=0.01 ** (1 / int(1e6 / (policy_freq * T_freq))),
            )
            self._bootstrap_last = {
                "T_B": initial_T_B.item(),
                "L1_B_Q": 0.0,
                "L1_B_BC": 0.0,
                "L1_B": 0.0,
                "L2_RMS_B": 0.0,
                "L_T_B": 0.0,
                "delta_y_B_mean": 0.0,
                "delta_y_B_mean_abs": 0.0,
                "delta_y_B_rms": 0.0,
                "delta_y_B_rms_relative": 0.0,
                "delta_y_B_rms_nonterminal_conditional": 0.0,
                "delta_y_B_max_abs": 0.0,
                "nonterminal_fraction": 0.0,
                "L2_squared_relative_diag": 0.0,
                "q_scale_B": 0.0,
                "grad_T_B_total": 0.0,
                "grad_T_B_from_L1_Q_chain": 0.0,
                "grad_T_B_from_L1_BC_chain": 0.0,
                "grad_T_B_from_L1": 0.0,
                "grad_T_B_from_L2_RMS": 0.0,
                "grad_T_B_L1_L2_same_sign": 0.0,
                "abs_grad_ratio_L1_L2": 0.0,
                "delta_log_T_B": 0.0,
                "target_critic_grad_norm": 0.0,
            }
        if self.adaptive_multiscale:
            # [bootstrap actor, execution actor]. Both anchor to a_D.
            self.actor_bank = [copy.deepcopy(actor), actor]
        else:
            self.actor_bank = [actor] + [
                copy.deepcopy(actor) for _ in range(proximal_n_steps - 1)
            ]
        self.actor_bank_states = [self.actor_opt_state] + [
            actor_optimizer.init(list(bank_actor.parameters()))
            for bank_actor in self.actor_bank[1:]
        ]
        if self.adaptive_multiscale:
            self.actor_bank_states = [
                actor_optimizer.init(list(bank_actor.parameters()))
                for bank_actor in self.actor_bank
            ]
            self.actor_opt_state = self.actor_bank_states[1]
        self.actor_target = copy.deepcopy(self.critic_bootstrap_actor())
        self.execution_actor_target = (
            copy.deepcopy(self.execution_actor()) if self.adaptive_multiscale else None
        )
        self.total_it = 0

    def execution_actor(self):
        return self.actor_bank[-1]

    def critic_bootstrap_actor(self):
        """Policy used for TD3 target actions / actor_target soft updates."""
        return self.actor_bank[0] if self.adaptive_multiscale else self.actor_bank[-1]

    def bootstrap_scale(self):
        if not self.adaptive_multiscale:
            raise RuntimeError(
                "bootstrap_scale is only defined for adaptive_multiscale"
            )
        return F.softplus(self.log_T_B)

    def _virtual_actor_update(self, bank_index, state, reference, horizon):
        bank_actor = self.actor_bank[bank_index]
        named = list(bank_actor.named_parameters())
        names = [name for name, _ in named]
        params = [parameter for _, parameter in named]
        parameter_map = dict(named)
        critic_required = [
            parameter.requires_grad for parameter in self.critic_1.parameters()
        ]
        if self.adaptive_multiscale:
            for parameter in self.critic_1.parameters():
                parameter.requires_grad_(False)
        try:
            pi = torch.func.functional_call(bank_actor, parameter_map, (state,))
            inner_loss, _ = self._inner_loss(state, pi, reference.detach(), horizon)
            grads = torch.autograd.grad(inner_loss, params, create_graph=True)
        finally:
            if self.adaptive_multiscale:
                for parameter, value in zip(
                    self.critic_1.parameters(), critic_required
                ):
                    parameter.requires_grad_(value)
        updates, _ = self.actor_optimizer.update(
            grads, self.actor_bank_states[bank_index], inplace=False
        )
        updated = torchopt.apply_updates(params, list(updates), inplace=False)
        updated_map = dict(zip(names, updated))
        return parameter_map, updated_map

    def _virtual_bootstrap_actor_update(self, state, reference, T_B):
        """Differentiable theta_B - actor_lr * grad(theta_B) update."""
        bank_actor = self.actor_bank[0]
        named = list(bank_actor.named_parameters())
        params = [parameter for _, parameter in named]
        critic_required = [
            parameter.requires_grad for parameter in self.critic_1.parameters()
        ]
        for parameter in self.critic_1.parameters():
            parameter.requires_grad_(False)
        try:
            pi = torch.func.functional_call(bank_actor, dict(named), (state,))
            inner_loss, _ = self._inner_loss(state, pi, reference.detach(), T_B)
            grads = torch.autograd.grad(inner_loss, params, create_graph=True)
        finally:
            for parameter, value in zip(self.critic_1.parameters(), critic_required):
                parameter.requires_grad_(value)
        return {
            name: parameter - self.actor_lr * grad
            for (name, parameter), grad in zip(named, grads)
        }

    @staticmethod
    def _exact_rms(values):
        """Exact population RMS with a finite zero subgradient at the origin."""
        if values.numel() == 0:
            raise ValueError("exact RMS requires at least one value")
        if not bool(torch.any(values.detach() != 0.0)):
            return values.sum() * 0.0
        return torch.linalg.vector_norm(values) / math.sqrt(values.numel())

    @staticmethod
    def _tq_scaled_bootstrap_terms(T_B, normalized_q, bc, R_B):
        """Build inverse and detached-common-factor TQ objectives."""
        detached_T_B = T_B.detach()
        L1_B_Q = -2.0 * detached_T_B * normalized_q
        L1_B_BC = bc
        L1_B = L1_B_Q + L1_B_BC
        L2_RMS_B = 2.0 * detached_T_B * R_B
        inverse_objective = -normalized_q + bc / (2.0 * detached_T_B) + R_B
        return {
            "L1_B_Q": L1_B_Q,
            "L1_B_BC": L1_B_BC,
            "L1_B": L1_B,
            "L2_RMS_B": L2_RMS_B,
            "L_T_B": L1_B + L2_RMS_B,
            "L_T_B_inverse": inverse_objective,
        }

    @classmethod
    def _bootstrap_target_rms(cls, delta_q, done, q_scale, discount):
        nonterminal = 1.0 - done
        delta_y = discount * nonterminal * delta_q
        relative_delta_y = delta_y / q_scale
        rms_relative = cls._exact_rms(relative_delta_y)
        rms = cls._exact_rms(delta_y)

        nonterminal_count = nonterminal.detach().sum()
        if bool(nonterminal_count > 0.0):
            rms_nonterminal = torch.linalg.vector_norm(delta_y) / torch.sqrt(
                nonterminal_count
            )
        else:
            rms_nonterminal = delta_y.sum() * 0.0

        # Detached diagnostic matching the previous squared-relative objective.
        squared_relative_diag = (
            (nonterminal * delta_q.square()).mean() / q_scale.square()
        ).detach()
        return {
            "delta_y_B": delta_y,
            "relative_delta_y_B": relative_delta_y,
            "R_B": rms_relative,
            "delta_y_B_rms": rms,
            "delta_y_B_rms_nonterminal_conditional": rms_nonterminal,
            "nonterminal_fraction": nonterminal.detach().mean(),
            "L2_squared_relative_diag": squared_relative_diag,
        }

    def _bootstrap_multiscale_objective(self, batch_inner, batch_outer, T_B):
        """Return detached-TQ L1 plus normalized target-displacement RMS."""
        inner_state, inner_action = batch_inner[:2]
        state, action, _, next_state, done = batch_outer
        virtual_parameters = self._virtual_bootstrap_actor_update(
            inner_state, inner_action, T_B
        )
        target_parameters = list(self.critic_1_target.parameters()) + list(
            self.critic_2_target.parameters()
        )
        required = [parameter.requires_grad for parameter in target_parameters]
        for parameter in target_parameters:
            parameter.requires_grad_(False)
        try:
            virtual_action = torch.func.functional_call(
                self.critic_bootstrap_actor(), virtual_parameters, (state,)
            )
            q_virtual = torch.minimum(
                self.critic_1_target(state, virtual_action),
                self.critic_2_target(state, virtual_action),
            )
            q_scale = q_virtual.abs().mean().detach() + self.smoothness_eps
            bc = F.mse_loss(virtual_action, action.detach())
            normalized_q = q_virtual.mean() / q_scale

            # Both branches use deterministic actions and the same frozen target
            # critics. Only the new virtual-action branch carries a hypergradient.
            virtual_next_action = torch.func.functional_call(
                self.critic_bootstrap_actor(), virtual_parameters, (next_state,)
            )
            q_new = torch.minimum(
                self.critic_1_target(next_state, virtual_next_action),
                self.critic_2_target(next_state, virtual_next_action),
            )
            with torch.no_grad():
                current_next_action = self.critic_bootstrap_actor()(next_state).detach()
                q_old = torch.minimum(
                    self.critic_1_target(next_state, current_next_action),
                    self.critic_2_target(next_state, current_next_action),
                )
            delta_q = q_new - q_old.detach()
            target_rms = self._bootstrap_target_rms(
                delta_q, done, q_scale, self.discount
            )
            R_B = target_rms["R_B"]
            terms = self._tq_scaled_bootstrap_terms(T_B, normalized_q, bc, R_B)
            # The detached common factor preserves inverse-form gradient geometry.
            outer_loss = terms["L_T_B"]
            target_grad_norm = T_B.new_zeros(())
            return outer_loss, {
                "bootstrap_outer_loss_version": BOOTSTRAP_OUTER_LOSS_VERSION,
                **terms,
                "R_B": R_B,
                "q_scale_B": q_scale,
                "delta_Q_B": delta_q,
                **target_rms,
                "Qbar_B_new": q_new,
                "Qbar_B_old": q_old,
                "virtual_action": virtual_action,
                "virtual_next_action": virtual_next_action,
                "current_next_action": current_next_action,
                "virtual_actor_parameters": virtual_parameters,
                "target_critic_grad_norm": target_grad_norm,
            }
        finally:
            for parameter, value in zip(target_parameters, required):
                parameter.requires_grad_(value)

    def _report_nonfinite_multiscale(self, stage, scalars):
        context = {
            "stage": stage,
            "step": self.total_it,
            "T_E": float(F.softplus(self.log_T).detach().cpu()),
            "T_B": float(self.bootstrap_scale().detach().cpu()),
        }
        context.update(
            {
                key: float(value.detach().cpu())
                if isinstance(value, torch.Tensor)
                else float(value)
                for key, value in scalars.items()
            }
        )
        print(f"[adaptive_multiscale] non-finite update skipped: {json.dumps(context)}")

    def _bootstrap_rms_nonlocal_diagnostics(self, batch_inner, batch_outer):
        """Read-only local/nonlocal slopes of R_B with respect to rho_B=log_T_B."""
        rho_center = self.log_T_B.detach().clone().requires_grad_(True)

        def evaluate(rho):
            _, details = self._bootstrap_multiscale_objective(
                batch_inner, batch_outer, F.softplus(rho)
            )
            return details

        center_details = evaluate(rho_center)
        center = center_details["R_B"]
        grad = torch.autograd.grad(center, rho_center, retain_graph=True)[0]
        old_squared = (
            (1.0 - batch_outer[4]) * center_details["delta_Q_B"].square()
        ).mean() / center_details["q_scale_B"].square()
        old_squared_grad = torch.autograd.grad(
            old_squared, rho_center, retain_graph=True
        )[0]
        rms_l2_grad = torch.autograd.grad(center_details["L2_RMS_B"], rho_center)[0]
        values = {}
        for label, h in (
            ("small", BOOTSTRAP_RMS_DIAGNOSTIC_SMALL_H),
            ("large", BOOTSTRAP_RMS_DIAGNOSTIC_LARGE_H),
        ):
            minus = evaluate((rho_center.detach() - h).requires_grad_(False))["R_B"]
            plus = evaluate((rho_center.detach() + h).requires_grad_(False))["R_B"]
            secant = (plus - minus) / (2.0 * h)
            grad_value = grad.detach().item()
            secant_value = secant.detach().item()
            sign_agreement = (grad_value == 0.0 and secant_value == 0.0) or (
                grad_value * secant_value > 0.0
            )
            if secant_value == 0.0:
                gradient_ratio = (
                    1.0
                    if grad_value == 0.0
                    else (grad_value / torch.finfo(grad.dtype).tiny)
                )
            else:
                gradient_ratio = grad_value / secant_value
            values.update(
                {
                    f"amo/R_B_secant_{label}_h": h,
                    f"amo/R_B_secant_slope_{label}": secant_value,
                    f"amo/R_B_autograd_secant_sign_agreement_{label}": float(
                        sign_agreement
                    ),
                    f"amo/R_B_autograd_over_secant_{label}": (gradient_ratio),
                }
            )
            values[f"_{label}_minus"] = minus.detach().item()
            values[f"_{label}_plus"] = plus.detach().item()

        small_minus = values.pop("_small_minus")
        small_plus = values.pop("_small_plus")
        large_minus = values.pop("_large_minus")
        large_plus = values.pop("_large_plus")
        center_value = center.detach().item()
        monotone_small = (
            small_minus <= center_value <= small_plus
            or small_minus >= center_value >= small_plus
        )
        monotone_large = (
            large_minus <= center_value <= large_plus
            or large_minus >= center_value >= large_plus
        )
        values.update(
            {
                "amo/R_B_autograd_grad_rho": grad.detach().item(),
                "amo/grad_T_B_from_L2_squared_relative_old_diag": (
                    old_squared_grad.detach().item()
                ),
                "amo/grad_T_B_from_L2_RMS_readonly_diag": (rms_l2_grad.detach().item()),
                "amo/abs_grad_ratio_RMS_over_squared_old_diag": (
                    abs(rms_l2_grad.detach().item())
                    / max(
                        abs(old_squared_grad.detach().item()),
                        torch.finfo(grad.dtype).tiny,
                    )
                ),
                "amo/RMS_grad_larger_than_squared_old_diag": float(
                    abs(rms_l2_grad.detach().item())
                    > abs(old_squared_grad.detach().item())
                ),
                "amo/R_B_nonlocal_center": center_value,
                "amo/R_B_monotone_small": float(monotone_small),
                "amo/R_B_monotone_large": float(monotone_large),
            }
        )
        return values

    def _update_bootstrap_scale(
        self,
        batch_inner,
        batch_outer,
        T_B,
    ):
        outer_loss, details = self._bootstrap_multiscale_objective(
            batch_inner, batch_outer, T_B
        )
        delta_y = details["delta_y_B"]
        finite_values = [
            outer_loss,
            details["L1_B_Q"],
            details["L1_B_BC"],
            details["L1_B"],
            details["L2_RMS_B"],
            details["R_B"],
            details["q_scale_B"],
            delta_y,
        ]
        values_finite = all(
            bool(torch.isfinite(value).all()) for value in finite_values
        )
        grad_l1_q = None
        grad_l1_bc = None
        grad_l1 = None
        grad_l2 = None
        grad_total = None
        if values_finite:
            grad_l1_q = torch.autograd.grad(
                details["L1_B_Q"], self.log_T_B, retain_graph=True
            )[0]
            grad_l1_bc = torch.autograd.grad(
                details["L1_B_BC"], self.log_T_B, retain_graph=True
            )[0]
            grad_l1 = torch.autograd.grad(
                details["L1_B"], self.log_T_B, retain_graph=True
            )[0]
            grad_l2 = torch.autograd.grad(details["L2_RMS_B"], self.log_T_B)[0]
            grad_total = grad_l1 + grad_l2
        grads_finite = (
            grad_total is not None
            and bool(torch.isfinite(grad_l1_q).all())
            and bool(torch.isfinite(grad_l1_bc).all())
            and bool(torch.isfinite(grad_l1).all())
            and bool(torch.isfinite(grad_l2).all())
            and bool(torch.isfinite(grad_total).all())
        )
        log_T_B_before = self.log_T_B.detach().clone()
        if grads_finite:
            self.T_B_optimizer.zero_grad(set_to_none=True)
            self.log_T_B.grad = grad_total.detach().clone()
            self.T_B_optimizer.step()
            self.T_B_scheduler.step()
        else:
            self._report_nonfinite_multiscale(
                "bootstrap_scale",
                {
                    "L_T_B": outer_loss,
                    "L1_B": details["L1_B"],
                    "L2_RMS_B": details["L2_RMS_B"],
                    "R_B": details["R_B"],
                    "q_scale_B": details["q_scale_B"],
                },
            )
        delta_log_T_B = (self.log_T_B.detach() - log_T_B_before).item()
        self.bootstrap_outer_update_count += 1
        same_sign = bool(
            grad_l1 is not None
            and grad_l2 is not None
            and (grad_l1 * grad_l2).detach().item() > 0.0
        )
        both_nonzero = bool(
            grad_l1 is not None
            and grad_l2 is not None
            and grad_l1.detach().item() != 0.0
            and grad_l2.detach().item() != 0.0
        )
        if both_nonzero:
            self.bootstrap_l1_l2_nonzero_count += 1
            self.bootstrap_l1_l2_same_sign_count += int(same_sign)
        grad_l1_value = 0.0 if grad_l1 is None else grad_l1.detach().item()
        grad_l2_value = 0.0 if grad_l2 is None else grad_l2.detach().item()
        ratio_floor = torch.finfo(T_B.dtype).tiny
        self._bootstrap_last = {
            "T_B": self.bootstrap_scale().detach().item(),
            "L1_B_Q": details["L1_B_Q"].detach().item(),
            "L1_B_BC": details["L1_B_BC"].detach().item(),
            "L1_B": details["L1_B"].detach().item(),
            "L2_RMS_B": details["L2_RMS_B"].detach().item(),
            "L_T_B": outer_loss.detach().item(),
            "delta_y_B_mean": delta_y.detach().mean().item(),
            "delta_y_B_mean_abs": delta_y.detach().abs().mean().item(),
            "delta_y_B_rms": details["delta_y_B_rms"].detach().item(),
            "delta_y_B_rms_relative": details["R_B"].detach().item(),
            "delta_y_B_rms_nonterminal_conditional": details[
                "delta_y_B_rms_nonterminal_conditional"
            ]
            .detach()
            .item(),
            "delta_y_B_max_abs": delta_y.detach().abs().max().item(),
            "nonterminal_fraction": details["nonterminal_fraction"].item(),
            "L2_squared_relative_diag": details["L2_squared_relative_diag"].item(),
            "q_scale_B": details["q_scale_B"].detach().item(),
            "grad_T_B_total": 0.0 if grad_total is None else grad_total.detach().item(),
            "grad_T_B_from_L1_Q_chain": (
                0.0 if grad_l1_q is None else grad_l1_q.detach().item()
            ),
            "grad_T_B_from_L1_BC_chain": (
                0.0 if grad_l1_bc is None else grad_l1_bc.detach().item()
            ),
            "grad_T_B_from_L1": grad_l1_value,
            "grad_T_B_from_L2_RMS": grad_l2_value,
            "grad_T_B_L1_L2_same_sign": float(same_sign),
            "abs_grad_ratio_L1_L2": (
                abs(grad_l1_value) / max(abs(grad_l2_value), ratio_floor)
            ),
            "delta_log_T_B": delta_log_T_B,
            "target_critic_grad_norm": details["target_critic_grad_norm"]
            .detach()
            .item(),
        }
        return outer_loss.detach()

    def _bootstrap_scale_logs(self):
        T_E = F.softplus(self.log_T).detach()
        T_B = self.bootstrap_scale().detach()
        return {
            "amo/T_E": T_E.item(),
            "amo/T_B": T_B.item(),
            "amo/T_B_over_T_E": (T_B / T_E).item(),
            "amo/L1_B_Q": self._bootstrap_last["L1_B_Q"],
            "amo/L1_B_BC": self._bootstrap_last["L1_B_BC"],
            "amo/L1_B": self._bootstrap_last["L1_B"],
            "amo/L2_RMS_B": self._bootstrap_last["L2_RMS_B"],
            "amo/L_T_B": self._bootstrap_last["L_T_B"],
            "amo/delta_y_B_mean": self._bootstrap_last["delta_y_B_mean"],
            "amo/delta_y_B_mean_abs": self._bootstrap_last["delta_y_B_mean_abs"],
            "amo/delta_y_B_rms": self._bootstrap_last["delta_y_B_rms"],
            "amo/delta_y_B_rms_relative": self._bootstrap_last[
                "delta_y_B_rms_relative"
            ],
            "amo/delta_y_B_rms_nonterminal_conditional": self._bootstrap_last[
                "delta_y_B_rms_nonterminal_conditional"
            ],
            "amo/delta_y_B_max_abs": self._bootstrap_last["delta_y_B_max_abs"],
            "amo/nonterminal_fraction": self._bootstrap_last["nonterminal_fraction"],
            "amo/L2_squared_relative_diag": self._bootstrap_last[
                "L2_squared_relative_diag"
            ],
            "amo/q_scale_B": self._bootstrap_last["q_scale_B"],
            "amo/grad_T_B_total": self._bootstrap_last["grad_T_B_total"],
            "amo/grad_T_B_from_L1_Q_chain": self._bootstrap_last[
                "grad_T_B_from_L1_Q_chain"
            ],
            "amo/grad_T_B_from_L1_BC_chain": self._bootstrap_last[
                "grad_T_B_from_L1_BC_chain"
            ],
            "amo/grad_T_B_from_L1": self._bootstrap_last["grad_T_B_from_L1"],
            "amo/grad_T_B_from_L2_RMS": self._bootstrap_last["grad_T_B_from_L2_RMS"],
            "amo/grad_T_B_L1_L2_same_sign": self._bootstrap_last[
                "grad_T_B_L1_L2_same_sign"
            ],
            "amo/abs_grad_ratio_L1_L2": self._bootstrap_last["abs_grad_ratio_L1_L2"],
            "amo/delta_log_T_B": self._bootstrap_last["delta_log_T_B"],
            "amo/delta_log_T_E": self._execution_last_delta_log_T,
            "amo/target_critic_grad_norm": self._bootstrap_last[
                "target_critic_grad_norm"
            ],
            "amo/grad_T_B_explicit_TQ_coeff": 0.0,
            "amo/bootstrap_L1_L2_same_sign_rate": (
                float(self.bootstrap_l1_l2_same_sign_count)
                / max(1, self.bootstrap_l1_l2_nonzero_count)
            ),
            "amo/bootstrap_outer_loss_version": BOOTSTRAP_OUTER_LOSS_VERSION,
            "amo/adaptive_multiscale": 1.0,
            "amo/critic_bootstrap_actor_role": 0.0,
            "amo/evaluation_actor_role": 1.0,
            "amo/bootstrap_target_polyak_source": 0.0,
            "amo/execution_target_polyak_source": 1.0,
        }

    def _inner_loss(self, state, pi, reference, T):
        q = self.critic_1(state, pi)
        q_abs_mean = q.abs().mean().detach().clamp_min(1e-6)
        # L_inner = -(1 / mean|Q|) Q + (1 / (2T)) ||pi - a||^2.
        loss = -(q.mean() / q_abs_mean) + F.mse_loss(pi, reference) / (2.0 * T)
        return loss, q_abs_mean

    def _outer_loss(self, state, pi, pi_new):
        """Return -B_PI / mean|Q_target| with target parameters frozen."""
        target = self.critic_1_target
        required = [p.requires_grad for p in target.parameters()]
        for p in target.parameters():
            p.requires_grad_(False)
        try:
            d = pi_new - pi
            a0 = pi.detach().clone().requires_grad_(True)
            q0 = target(state, a0)
            g0 = torch.autograd.grad(q0.sum(), a0)[0].detach()
            a1 = pi_new.detach().clone().requires_grad_(True)
            q1 = target(state, a1)
            g1 = torch.autograd.grad(q1.sum(), a1)[0].detach()
            d_norm = d.norm(dim=1)
            delta_g = (g1 - g0).norm(dim=1)
            # ε=0 descent term: (L/2)||d||^2 with L=||Δg||/||d|| → (1/2)||Δg||||d||
            penalty = 0.5 * delta_g * d_norm
            if self.smoothness_max is not None:
                penalty = torch.minimum(
                    penalty, 0.5 * self.smoothness_max * d_norm.square()
                )
            implied_l = delta_g / d_norm.detach().clamp_min(self.smoothness_eps)
            b_pi = (g0 * d).sum(dim=1) - penalty
            q_scale = target(state, pi_new).abs().mean().detach().clamp_min(1e-6)
            outer_loss = -b_pi.mean() / q_scale
            return outer_loss, {
                "amo/B_PI_target": b_pi.detach().mean().item(),
                "amo/q_scale_target": q_scale.item(),
                "amo/outer_loss": outer_loss.detach().item(),
                "amo/mean_Lhat_target": implied_l.mean().item(),
            }
        finally:
            for p, value in zip(target.parameters(), required):
                p.requires_grad_(value)

    def _update_actor_bank(self, state, action, T, T_B=None):
        if self.adaptive_multiscale:
            logs = {"amo/T": T.detach().item()}
            horizons = [T_B, T.detach()]
            for index, (bank_actor, horizon) in enumerate(
                zip(self.actor_bank, horizons)
            ):
                reference = action.detach()
                params = list(bank_actor.parameters())
                pi = bank_actor(state)
                inner_loss, q_scale = self._inner_loss(state, pi, reference, horizon)
                grads = torch.autograd.grad(inner_loss, params)
                updates, opt_state = self.actor_optimizer.update(
                    grads, self.actor_bank_states[index], inplace=False
                )
                self.actor_bank_states[index] = torchopt.pytree.tree_map(
                    lambda x: x.detach() if isinstance(x, torch.Tensor) else x,
                    opt_state,
                )
                with torch.no_grad():
                    for parameter, updated in zip(
                        params,
                        torchopt.apply_updates(params, list(updates), inplace=False),
                    ):
                        parameter.copy_(updated)
                tag = "T_B" if index == 0 else "T"
                logs[f"amo/inner_loss_{tag}"] = inner_loss.item()
                logs[f"amo/q_abs_mean_{tag}"] = q_scale.item()
            self.actor_opt_state = self.actor_bank_states[1]
            return logs

        tau = T.detach() / float(self.proximal_n_steps)
        logs = {
            "amo/T": T.detach().item(),
            "amo/N": float(self.proximal_n_steps),
            "amo/tau": tau.item(),
        }
        for index, bank_actor in enumerate(self.actor_bank):
            reference = (
                action.detach()
                if index == 0
                else self.actor_bank[index - 1](state).detach()
            )
            params = list(bank_actor.parameters())
            pi = bank_actor(state)
            inner_loss, q_scale = self._inner_loss(state, pi, reference, tau)
            grads = torch.autograd.grad(inner_loss, params)
            updates, opt_state = self.actor_optimizer.update(
                grads, self.actor_bank_states[index], inplace=False
            )
            self.actor_bank_states[index] = torchopt.pytree.tree_map(
                lambda x: x.detach() if isinstance(x, torch.Tensor) else x, opt_state
            )
            with torch.no_grad():
                for parameter, updated in zip(
                    params, torchopt.apply_updates(params, list(updates), inplace=False)
                ):
                    parameter.copy_(updated)
            logs[f"amo/inner_loss_hop_{index + 1}"] = inner_loss.item()
            logs[f"amo/q_abs_mean_hop_{index + 1}"] = q_scale.item()
        self.actor_opt_state = self.actor_bank_states[0]
        return logs

    def _virtual_actor_bank_update(self, state, action, outer_state, T):
        """Differentiate actor update(s) w.r.t. T for the outer objective."""
        if self.adaptive_multiscale:
            # Outer fits T through the final-policy one-step update with horizon T.
            bank_actor = self.actor_bank[1]
            parameter_map, updated_map = self._virtual_actor_update(1, state, action, T)
            outer_pi = torch.func.functional_call(
                bank_actor, parameter_map, (outer_state,)
            )
            outer_pi_new = torch.func.functional_call(
                bank_actor, updated_map, (outer_state,)
            )
            return outer_pi, outer_pi_new

        tau = T / float(self.proximal_n_steps)
        virtual_parameters = []
        reference = action.detach()

        for index, bank_actor in enumerate(self.actor_bank):
            named = list(bank_actor.named_parameters())
            names = [name for name, _ in named]
            params = [parameter for _, parameter in named]
            parameter_map = dict(named)
            pi = torch.func.functional_call(bank_actor, parameter_map, (state,))
            inner_loss, _ = self._inner_loss(state, pi, reference, tau)
            grads = torch.autograd.grad(inner_loss, params, create_graph=True)
            updates, _ = self.actor_optimizer.update(
                grads, self.actor_bank_states[index], inplace=False
            )
            updated = torchopt.apply_updates(params, list(updates), inplace=False)
            updated_map = dict(zip(names, updated))
            virtual_parameters.append((parameter_map, updated_map))
            reference = torch.func.functional_call(
                bank_actor, updated_map, (state,)
            ).detach()

        initial_map, updated_map = virtual_parameters[-1]
        final_actor = self.actor_bank[-1]
        outer_pi = torch.func.functional_call(final_actor, initial_map, (outer_state,))
        outer_pi_new = torch.func.functional_call(
            final_actor, updated_map, (outer_state,)
        )
        return outer_pi, outer_pi_new

    def train(self, batch, batch_outer=None):
        self.total_it += 1
        state, action, reward, next_state, done = batch
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            target_q = reward + (1 - done) * self.discount * torch.minimum(
                self.critic_1_target(next_state, next_action),
                self.critic_2_target(next_state, next_action),
            )
        critic_loss = F.mse_loss(self.critic_1(state, action), target_q) + F.mse_loss(
            self.critic_2(state, action), target_q
        )
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        logs = {"amo/critic_loss": critic_loss.item()}
        if self.total_it % self.policy_freq:
            return logs
        T = F.softplus(self.log_T)
        if self.total_it % (self.policy_freq * self.T_freq) == 0:
            if batch_outer is None:
                raise ValueError(
                    "AMO requires an independently sampled outer batch for every T update"
                )
            outer_state = batch_outer[0]
            outer_pi, outer_pi_new = self._virtual_actor_bank_update(
                state, action, outer_state, T
            )
            outer_loss, outer_logs = self._outer_loss(
                outer_state, outer_pi, outer_pi_new
            )
            # Chain mode differentiates its last T/N hop; adaptive multiscale
            # differentiates the full execution scale directly.
            n_scale = 1.0 if self.adaptive_multiscale else float(self.proximal_n_steps)
            log_T_before = self.log_T.detach().clone()
            self.T_optimizer.zero_grad()
            scaled_outer_loss = outer_loss * n_scale
            if self.adaptive_multiscale:
                grad_T_E = torch.autograd.grad(scaled_outer_loss, self.log_T)[0]
                if bool(torch.isfinite(scaled_outer_loss).all()) and bool(
                    torch.isfinite(grad_T_E).all()
                ):
                    self.log_T.grad = grad_T_E.detach().clone()
                    self.T_optimizer.step()
                    self.T_scheduler.step()
                else:
                    self._report_nonfinite_multiscale(
                        "execution_scale",
                        {"outer_loss": scaled_outer_loss, "grad_T_E": grad_T_E},
                    )
                self._execution_last_delta_log_T = (
                    self.log_T.detach() - log_T_before
                ).item()
            else:
                scaled_outer_loss.backward()
            logs.update(outer_logs)
            if not self.adaptive_multiscale:
                logs["amo/outer_N_scale"] = n_scale
            logs["amo/T_grad"] = (
                0.0 if self.log_T.grad is None else self.log_T.grad.item()
            )
            if self.adaptive_multiscale:
                logs["amo/grad_T_E"] = logs["amo/T_grad"]
                logs["amo/grad_T_E_BPI"] = logs["amo/T_grad"]
                logs["amo/B_PI_E"] = outer_logs["amo/B_PI_target"]
                logs["amo/L_T_E"] = scaled_outer_loss.detach().item()
            if not self.adaptive_multiscale:
                self.T_optimizer.step()
                self.T_scheduler.step()
            else:
                T_B = self.bootstrap_scale()
                bootstrap_loss = self._update_bootstrap_scale(batch, batch_outer, T_B)
                logs["amo/bootstrap_outer_loss"] = bootstrap_loss.item()
                if (
                    self.bootstrap_rms_diagnostic_freq > 0
                    and self.total_it % self.bootstrap_rms_diagnostic_freq == 0
                ):
                    logs.update(
                        self._bootstrap_rms_nonlocal_diagnostics(batch, batch_outer)
                    )
        actor_T_B = (
            self.bootstrap_scale().detach() if self.adaptive_multiscale else None
        )
        logs.update(self._update_actor_bank(state, action, T, T_B=actor_T_B))
        if self.adaptive_multiscale:
            logs.update(self._bootstrap_scale_logs())
        soft_update(self.critic_1_target, self.critic_1, self.tau)
        soft_update(self.critic_2_target, self.critic_2, self.tau)
        soft_update(self.actor_target, self.critic_bootstrap_actor(), self.tau)
        if self.adaptive_multiscale:
            soft_update(self.execution_actor_target, self.execution_actor(), self.tau)
        return logs

    def state_dict(self):
        state = {
            "checkpoint_version": 10 if self.adaptive_multiscale else 5,
            "actor": self.actor.state_dict(),
            "actor_bank": [a.state_dict() for a in self.actor_bank],
            "actor_bank_states": self.actor_bank_states,
            "actor_target": self.actor_target.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "critic_1_target": self.critic_1_target.state_dict(),
            "critic_2_target": self.critic_2_target.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "log_T": self.log_T.detach().cpu(),
            "T_optimizer": self.T_optimizer.state_dict(),
            "T_scheduler": self.T_scheduler.state_dict(),
            "total_it": self.total_it,
        }
        if self.adaptive_multiscale:
            state.update(
                {
                    "adaptive_multiscale": True,
                    "bootstrap_outer_loss_version": BOOTSTRAP_OUTER_LOSS_VERSION,
                    "T_E_raw": self.log_T.detach().cpu(),
                    "T_E_effective": F.softplus(self.log_T).detach().cpu(),
                    "log_T_B": self.log_T_B.detach().cpu(),
                    "T_B": self.bootstrap_scale().detach().cpu(),
                    "T_B_optimizer": self.T_B_optimizer.state_dict(),
                    "T_B_scheduler": self.T_B_scheduler.state_dict(),
                    "bootstrap_outer_update_count": self.bootstrap_outer_update_count,
                    "bootstrap_l1_l2_same_sign_count": (
                        self.bootstrap_l1_l2_same_sign_count
                    ),
                    "bootstrap_l1_l2_nonzero_count": (
                        self.bootstrap_l1_l2_nonzero_count
                    ),
                    "execution_last_delta_log_T": self._execution_last_delta_log_T,
                    "bootstrap_last": self._bootstrap_last,
                    "execution_actor_target": self.execution_actor_target.state_dict(),
                }
            )
        return state

    def load_state_dict(self, state):
        checkpoint_adaptive = bool(state.get("adaptive_multiscale", False))
        if checkpoint_adaptive != self.adaptive_multiscale:
            raise ValueError(
                f"checkpoint adaptive_multiscale={checkpoint_adaptive} "
                f"!= trainer adaptive_multiscale={self.adaptive_multiscale}"
            )
        if self.adaptive_multiscale:
            checkpoint_loss_version = state.get("bootstrap_outer_loss_version")
            if checkpoint_loss_version != BOOTSTRAP_OUTER_LOSS_VERSION:
                raise ValueError(
                    "adaptive_multiscale bootstrap outer-loss mismatch: "
                    f"checkpoint={checkpoint_loss_version!r}, "
                    f"trainer={BOOTSTRAP_OUTER_LOSS_VERSION!r}; start a fresh "
                    "run instead of silently resuming old-loss state"
                )
            if int(state.get("checkpoint_version", 0)) < 10:
                raise ValueError(
                    "adaptive_multiscale checkpoint v10 or newer is required "
                    "for independent T_E and T_B learning"
                )
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "critic_1_target",
            "critic_2_target",
            "actor_target",
        ):
            getattr(self, name).load_state_dict(state[name])
        for module, values in zip(
            self.actor_bank, state.get("actor_bank", [state["actor"]])
        ):
            module.load_state_dict(values)
        self.actor_bank_states = state.get("actor_bank_states", self.actor_bank_states)
        self.critic_1_optimizer.load_state_dict(state["critic_1_optimizer"])
        self.critic_2_optimizer.load_state_dict(state["critic_2_optimizer"])
        with torch.no_grad():
            self.log_T.copy_(state["log_T"].to(self.log_T.device))
        self.T_optimizer.load_state_dict(state["T_optimizer"])
        self.T_scheduler.load_state_dict(state["T_scheduler"])
        if self.adaptive_multiscale:
            with torch.no_grad():
                self.log_T_B.copy_(state["log_T_B"].to(self.log_T_B.device))
            self.T_B_optimizer.load_state_dict(state["T_B_optimizer"])
            self.T_B_scheduler.load_state_dict(state["T_B_scheduler"])
            self.execution_actor_target.load_state_dict(state["execution_actor_target"])
            self.bootstrap_outer_update_count = int(
                state.get("bootstrap_outer_update_count", 0)
            )
            self.bootstrap_l1_l2_same_sign_count = int(
                state.get("bootstrap_l1_l2_same_sign_count", 0)
            )
            self.bootstrap_l1_l2_nonzero_count = int(
                state.get("bootstrap_l1_l2_nonzero_count", 0)
            )
            self._execution_last_delta_log_T = float(
                state.get("execution_last_delta_log_T", 0.0)
            )
            self._bootstrap_last = state.get("bootstrap_last", self._bootstrap_last)
        self.total_it = int(state["total_it"])
        legacy_dual_key = "dual_" + "proximal"
        if bool(state.get(legacy_dual_key, False)):
            raise ValueError(
                "fixed-ratio dual checkpoints are deprecated and cannot be resumed"
            )


@pyrallis.wrap()
def train(config: TrainConfig):
    # D4RL imports MuJoCo at module import time. Keep it local so the public
    # AMO API and unit tests remain usable without a configured MuJoCo runtime.
    import d4rl

    env = gym.make(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env)

    if config.normalize_reward:
        modify_reward(dataset, config.env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)
    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    actor = Actor(state_dim, action_dim, max_action).to(config.device)
    actor_optimizer = torchopt.adam(lr=3e-4, use_accelerated_op=True)

    critic_1 = Critic(state_dim, action_dim).to(config.device)
    critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
    critic_2 = Critic(state_dim, action_dim).to(config.device)
    critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)

    vnet = Vnet(state_dim).to(config.device)
    vnet_optimizer = torch.optim.Adam(vnet.parameters(), lr=3e-4)
    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "critic_1": critic_1,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2": critic_2,
        "critic_2_optimizer": critic_2_optimizer,
        "vnet": vnet,
        "vnet_optimizer": vnet_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # TD3
        "policy_noise": config.policy_noise * max_action,
        "noise_clip": config.noise_clip * max_action,
        "policy_freq": config.policy_freq,
        # TD3 + BC
        "T": config.T_E,
        "T_B": config.T_B,
        "T_freq": config.T_freq,
        "smoothness_eps": config.smoothness_eps,
        "smoothness_max": config.smoothness_max,
        "proximal_n_steps": config.proximal_n_steps,
        "adaptive_multiscale": config.adaptive_multiscale,
        "bootstrap_rms_diagnostic_freq": config.bootstrap_rms_diagnostic_freq,
        "actor_lr": 3e-4,
        "T_lr": config.T_lr,
    }

    print("---------------------------------------")
    print(f"Training AMO, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = AMO(**kwargs)

    resume_ckpt: Optional[Path] = None
    if config.load_model != "":
        resume_ckpt = Path(config.load_model)
    elif config.resume_tag and config.checkpoints_path is not None:
        ckpts = list(Path(config.checkpoints_path).glob("checkpoint_*.pt"))
        if ckpts:
            resume_ckpt = max(ckpts, key=lambda p: int(p.stem.split("_")[1]))

    start_step = 0
    if resume_ckpt is not None:
        ckpt = torch.load(resume_ckpt, map_location=config.device, weights_only=False)
        ckpt_ver = ckpt.get("checkpoint_version", 1)
        if ckpt_ver < 2:
            print(
                f"WARNING: {resume_ckpt} is checkpoint v{ckpt_ver} (incomplete state). "
                f"Resume skipped — starting from step 0."
            )
        else:
            trainer.load_state_dict(ckpt)
            start_step = int(trainer.total_it)
            actor = trainer.actor
            print(
                f"Resumed from {resume_ckpt} at step {start_step} (checkpoint v{ckpt_ver})"
            )

    wandb_init(asdict(config))

    # Additive local metric logging for AMO diagnostics. Does not affect training.
    metrics_file = None
    eval_file = None
    if config.checkpoints_path is not None and config.metrics_log_freq > 0:
        metrics_file = open(os.path.join(config.checkpoints_path, "metrics.jsonl"), "a")
        eval_file = open(os.path.join(config.checkpoints_path, "eval.jsonl"), "a")
    save_freq = config.save_freq if config.save_freq > 0 else config.eval_freq

    stop_requested = {"flag": False}

    def _on_stop(signum, _frame):
        stop_requested["flag"] = True
        print(
            f"[signal] received {signal.Signals(signum).name}; finishing current step"
        )

    signal.signal(signal.SIGTERM, _on_stop)
    signal.signal(signal.SIGINT, _on_stop)

    evaluations = []
    if config.profile_runtime and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for t in range(start_step, int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        batch_outer = None
        if (trainer.total_it + 1) % (config.policy_freq * config.T_freq) == 0:
            batch_outer = replay_buffer.sample(config.batch_size)
            batch_outer = [b.to(config.device) for b in batch_outer]
        if config.profile_runtime and torch.cuda.is_available():
            torch.cuda.synchronize()
        iteration_started = time.perf_counter()
        log_dict = trainer.train(batch, batch_outer=batch_outer)
        if config.profile_runtime:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            log_dict["runtime/iteration_ms"] = (
                time.perf_counter() - iteration_started
            ) * 1000.0
            if torch.cuda.is_available():
                log_dict[
                    "runtime/current_allocated_mb"
                ] = torch.cuda.memory_allocated() / (1024.0**2)
                log_dict[
                    "runtime/current_reserved_mb"
                ] = torch.cuda.memory_reserved() / (1024.0**2)
                log_dict[
                    "runtime/peak_allocated_mb"
                ] = torch.cuda.max_memory_allocated() / (1024.0**2)
                log_dict[
                    "runtime/peak_reserved_mb"
                ] = torch.cuda.max_memory_reserved() / (1024.0**2)
        wandb_log(log_dict, step=trainer.total_it)
        if (
            metrics_file is not None
            and log_dict
            and (trainer.total_it % config.metrics_log_freq == 0)
        ):
            rec = {"step": trainer.total_it}
            rec.update(
                {
                    k: float(v) if isinstance(v, (int, float, torch.Tensor)) else v
                    for k, v in log_dict.items()
                }
            )
            metrics_file.write(json.dumps(rec) + "\n")
            metrics_file.flush()
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                trainer.execution_actor(),
                device=config.device,
                n_episodes=config.n_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            evaluations.append(normalized_eval_score)
            print("---------------------------------------")
            print(
                f"Evaluation over {config.n_episodes} episodes: "
                f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
            )
            print("---------------------------------------")

            if eval_file is not None:
                eval_file.write(
                    json.dumps(
                        {
                            "step": trainer.total_it,
                            "t": t + 1,
                            "d4rl_normalized_score": float(normalized_eval_score),
                            "eval_return": float(eval_score),
                        }
                    )
                    + "\n"
                )
                eval_file.flush()

            if config.checkpoints_path is not None and (t + 1) % save_freq == 0:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            wandb_log(
                {"d4rl_normalized_score": normalized_eval_score},
                step=trainer.total_it,
            )

        if stop_requested["flag"]:
            on_schedule = (
                config.checkpoints_path is not None
                and (t + 1) % config.eval_freq == 0
                and (t + 1) % save_freq == 0
            )
            if config.checkpoints_path is not None and not on_schedule:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )
                print(f"[signal] emergency save at step={t + 1}")
            else:
                print(f"[signal] already on save schedule at step={t + 1}")
            break

    if metrics_file is not None:
        metrics_file.close()
    if eval_file is not None:
        eval_file.close()


if __name__ == "__main__":
    train()
