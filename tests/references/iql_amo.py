# Extracted without equation edits from AMO 7068c765; see SOURCE_PROVENANCE.md.
from __future__ import annotations
from typing import *
from dataclasses import dataclass, field, asdict
from pathlib import Path
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.func import functional_call
EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Normal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = dist.mean if not self.training else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )

class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=True)
        self.q2 = MLP(dims, squeeze_output=True)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state, action))

class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)

def safe_exp_weights(
    beta: torch.Tensor,
    adv: torch.Tensor,
    cap: float = EXP_ADV_MAX,
) -> torch.Tensor:
    """w(beta, A) = min(exp(beta * A), cap) with overflow-safe forward.

    Caps the exponent at log(cap) before exp. In the non-clipped regime this
    matches ``torch.exp(beta * adv).clamp(max=cap)`` in value and gradient.
    Clipped samples have zero gradient w.r.t. beta / adv through the weight.
    """
    log_cap = math.log(cap)
    x = beta * adv
    # Cap exponent first (overflow-safe), then clamp to guarantee <= cap
    # even when exp(log(cap)) is slightly above due to float rounding.
    return torch.clamp(torch.exp(torch.clamp(x, max=log_cap)), max=cap)

class _SafeSqrt(torch.autograd.Function):
    """Forward exact sqrt; backward avoids 0/0 NaNs for higher-order grads."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        return torch.sqrt(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        denom = torch.sqrt(x.clamp(min=1e-12))
        return grad_output * (0.5 / denom)

def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    return _SafeSqrt.apply(x)

def functional_adam_step(
    named_params: Dict[str, torch.Tensor],
    named_grads: Dict[str, Optional[torch.Tensor]],
    adam_state: Dict[str, Dict[str, Any]],
    *,
    lr: float,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, Any]]]:
    """One Adam update matching ``torch.optim.Adam`` (decoupled WD not used).

    Returns ``(theta_plus, new_state)``. ``new_state`` is detached/copied for
    inspection; ``theta_plus`` keeps the autograd graph through ``grads``.
    """
    b1, b2 = betas
    theta_plus: Dict[str, torch.Tensor] = {}
    new_state: Dict[str, Dict[str, Any]] = {}

    for name, p in named_params.items():
        g = named_grads.get(name, None)
        st = adam_state.get(name)

        if g is None:
            # Match PyTorch: parameters with grad=None are left unchanged and
            # their optimizer state is not advanced.
            theta_plus[name] = p
            if st is not None:
                new_state[name] = {
                    "step": int(st["step"]),
                    "exp_avg": st["exp_avg"].clone(),
                    "exp_avg_sq": st["exp_avg_sq"].clone(),
                }
            continue

        if weight_decay != 0.0:
            g = g + weight_decay * p

        if st is None:
            step = 0
            exp_avg = torch.zeros_like(p)
            exp_avg_sq = torch.zeros_like(p)
        else:
            step = int(st["step"])
            exp_avg = st["exp_avg"]
            exp_avg_sq = st["exp_avg_sq"]

        step_plus = step + 1
        # Keep graph on moment updates that depend on g.
        exp_avg_p = exp_avg * b1 + (1.0 - b1) * g
        exp_avg_sq_p = exp_avg_sq * b2 + (1.0 - b2) * (g * g)

        bias_c1 = 1.0 - b1**step_plus
        bias_c2 = 1.0 - b2**step_plus
        m_hat = exp_avg_p / bias_c1
        v_hat = exp_avg_sq_p / bias_c2
        denom = safe_sqrt(v_hat) + eps
        theta_plus[name] = p - lr * m_hat / denom

        new_state[name] = {
            "step": step_plus,
            "exp_avg": exp_avg_p.detach().clone(),
            "exp_avg_sq": exp_avg_sq_p.detach().clone(),
            "v_hat": v_hat.detach().clone(),
            "m_hat": m_hat.detach().clone(),
        }

    return theta_plus, new_state

def adam_state_from_optimizer(
    optimizer: torch.optim.Optimizer,
    named_params: Dict[str, nn.Parameter],
) -> Dict[str, Dict[str, Any]]:
    """Extract per-parameter Adam state keyed by parameter name."""
    id_to_name = {id(p): n for n, p in named_params.items()}
    out: Dict[str, Dict[str, Any]] = {}
    for p in optimizer.param_groups[0]["params"]:
        name = id_to_name[id(p)]
        if p not in optimizer.state or len(optimizer.state[p]) == 0:
            continue
        st = optimizer.state[p]
        out[name] = {
            "step": int(st["step"]),
            "exp_avg": st["exp_avg"].detach().clone(),
            "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
        }
    return out

def named_parameters_dict(module: nn.Module) -> Dict[str, nn.Parameter]:
    return dict(module.named_parameters())

def actor_ell(actor: nn.Module, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Per-sample imitation loss ell; shape [B]."""
    policy_out = actor(states)
    if isinstance(policy_out, torch.distributions.Distribution):
        ell = -policy_out.log_prob(actions).sum(-1)
    elif torch.is_tensor(policy_out):
        if policy_out.shape != actions.shape:
            raise RuntimeError(
                f"Actions shape mismatch: policy {tuple(policy_out.shape)} "
                f"vs actions {tuple(actions.shape)}"
            )
        ell = torch.sum((policy_out - actions) ** 2, dim=1)
    else:
        raise NotImplementedError(type(policy_out))
    assert_vec1d(ell, "ell")
    return ell

def actor_ell_functional(
    actor: nn.Module,
    params: Dict[str, torch.Tensor],
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """ell(theta; s, a) via functional_call (supports Parameter log_std)."""
    policy_out = functional_call(actor, params, (states,))
    if isinstance(policy_out, torch.distributions.Distribution):
        ell = -policy_out.log_prob(actions).sum(-1)
    elif torch.is_tensor(policy_out):
        if policy_out.shape != actions.shape:
            raise RuntimeError(
                f"Actions shape mismatch: policy {tuple(policy_out.shape)} "
                f"vs actions {tuple(actions.shape)}"
            )
        ell = torch.sum((policy_out - actions) ** 2, dim=1)
    else:
        raise NotImplementedError(type(policy_out))
    assert_vec1d(ell, "ell_functional")
    return ell

def actor_action_mean(actor: nn.Module, states: torch.Tensor) -> torch.Tensor:
    """Deterministic action mean: GaussianPolicy.mean or Deterministic tensor."""
    policy_out = actor(states)
    if isinstance(policy_out, torch.distributions.Distribution):
        return policy_out.mean
    if torch.is_tensor(policy_out):
        return policy_out
    raise NotImplementedError(type(policy_out))

def actor_action_mean_functional(
    actor: nn.Module,
    params: Dict[str, torch.Tensor],
    states: torch.Tensor,
) -> torch.Tensor:
    """Functional counterpart of ``actor_action_mean``."""
    policy_out = functional_call(actor, params, (states,))
    if isinstance(policy_out, torch.distributions.Distribution):
        return policy_out.mean
    if torch.is_tensor(policy_out):
        return policy_out
    raise NotImplementedError(type(policy_out))

def exact_rms(values: torch.Tensor) -> torch.Tensor:
    """Exact population RMS with a finite zero subgradient at the origin."""
    if values.numel() == 0:
        raise ValueError("exact RMS requires at least one value")
    if not bool(torch.any(values.detach() != 0.0)):
        return values.sum() * 0.0
    return torch.linalg.vector_norm(values) / math.sqrt(values.numel())

def tq_scaled_bootstrap_terms(
    T_B: torch.Tensor,
    normalized_q: torch.Tensor,
    bc: torch.Tensor,
    R_B: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """AMO ``tq_detached_rms_target_v1`` L1_B + L2_RMS_B terms."""
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

def bootstrap_target_rms(
    delta_q: torch.Tensor,
    done: torch.Tensor,
    q_scale: torch.Tensor,
    discount: float,
) -> Dict[str, torch.Tensor]:
    """Normalized target-displacement RMS used by AMO TB outer."""
    nonterminal = 1.0 - done
    delta_y = discount * nonterminal * delta_q
    relative_delta_y = delta_y / q_scale
    rms_relative = exact_rms(relative_delta_y)
    rms = exact_rms(delta_y)

    nonterminal_count = nonterminal.detach().sum()
    if bool(nonterminal_count > 0.0):
        rms_nonterminal = torch.linalg.vector_norm(delta_y) / torch.sqrt(
            nonterminal_count
        )
    else:
        rms_nonterminal = delta_y.sum() * 0.0

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

def b_pi_outer_loss(
    q_target: nn.Module,
    state: torch.Tensor,
    pi: torch.Tensor,
    pi_new: torch.Tensor,
    *,
    smoothness_eps: float = 1e-6,
    smoothness_max: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """AMO -B_PI / mean|Q| outer using TwinQ ``q_target(state, action)`` (min)."""
    target_params = list(q_target.parameters())
    required = [p.requires_grad for p in target_params]
    for p in target_params:
        p.requires_grad_(False)
    try:
        d = pi_new - pi
        a0 = pi.detach().clone().requires_grad_(True)
        q0 = q_target(state, a0)
        g0 = torch.autograd.grad(q0.sum(), a0)[0].detach()
        a1 = pi_new.detach().clone().requires_grad_(True)
        q1 = q_target(state, a1)
        g1 = torch.autograd.grad(q1.sum(), a1)[0].detach()
        d_norm = d.norm(dim=1)
        delta_g = (g1 - g0).norm(dim=1)
        penalty = 0.5 * delta_g * d_norm
        if smoothness_max is not None:
            penalty = torch.minimum(
                penalty, 0.5 * float(smoothness_max) * d_norm.square()
            )
        implied_l = delta_g / d_norm.detach().clamp_min(smoothness_eps)
        b_pi = (g0 * d).sum(dim=1) - penalty
        q_scale = q_target(state, pi_new).abs().mean().detach().clamp_min(1e-6)
        outer_loss = -b_pi.mean() / q_scale
        logs = {
            "B_PI_target": float(b_pi.detach().mean().item()),
            "q_scale_target": float(q_scale.item()),
            "outer_loss": float(outer_loss.detach().item()),
            "mean_Lhat_target": float(implied_l.mean().item()),
        }
        return outer_loss, logs
    finally:
        for p, value in zip(target_params, required):
            p.requires_grad_(value)

def assert_vec1d(t: torch.Tensor, name: str) -> None:
    if t.ndim != 1:
        raise AssertionError(f"{name} must be rank-1 [B], got shape {tuple(t.shape)}")

def squeeze_to_1d(t: torch.Tensor, name: str) -> torch.Tensor:
    if t.ndim == 2 and t.shape[-1] == 1:
        t = t.squeeze(-1)
    assert_vec1d(t, name)
    return t

def is_meta_step(step: int, meta_warmup_steps: int, meta_interval: int) -> bool:
    return step > meta_warmup_steps and (step - meta_warmup_steps) % meta_interval == 0

def get_rng_state(outer_rng: Optional[np.random.RandomState] = None) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "outer_rng": outer_rng.get_state() if outer_rng is not None else None,
    }
    return state

def set_rng_state(state: Dict[str, Any], outer_rng: Optional[np.random.RandomState] = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])

    def _byte_cpu(t: Any) -> torch.Tensor:
        if not isinstance(t, torch.Tensor):
            t = torch.as_tensor(t, dtype=torch.uint8)
        return t.detach().to(device="cpu", dtype=torch.uint8).contiguous()

    torch.set_rng_state(_byte_cpu(state["torch_cpu"]))
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        # Older saves may have one state or a list; coerce each to CPU ByteTensor.
        if isinstance(cuda_states, torch.Tensor):
            cuda_states = [_byte_cpu(cuda_states)]
        else:
            cuda_states = [_byte_cpu(s) for s in cuda_states]
        # Match current device count: pad/truncate if GPU count changed.
        n = torch.cuda.device_count()
        if len(cuda_states) < n:
            cuda_states = cuda_states + [cuda_states[-1]] * (n - len(cuda_states))
        torch.cuda.set_rng_state_all(cuda_states[:n])
    if outer_rng is not None and state.get("outer_rng") is not None:
        outer_rng.set_state(state["outer_rng"])

def ess_stats(w: torch.Tensor) -> Dict[str, float]:
    w = w.detach().float().reshape(-1)
    s = float(w.sum().item())
    s2 = float((w * w).sum().item())
    if s2 <= 0.0 or not math.isfinite(s2):
        return {
            "ess": float("nan"),
            "ess_fraction": float("nan"),
            "ess_undefined": True,
            "weight_sum": s,
            "weight_sumsq": s2,
        }
    ess = (s * s) / s2
    return {
        "ess": float(ess),
        "ess_fraction": float(ess / max(w.numel(), 1)),
        "ess_undefined": False,
        "weight_sum": s,
        "weight_sumsq": s2,
    }

def weight_diagnostics(w: torch.Tensor, cap: float = EXP_ADV_MAX) -> Dict[str, float]:
    w = w.detach().float().reshape(-1)
    qs = torch.quantile(w, torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=w.device)).cpu().numpy()
    clipped = (w >= cap - 1e-6).float().mean().item()
    near_zero = (w <= 1e-12).float().mean().item()
    out = {
        "weight_mean": float(w.mean().item()),
        "weight_max": float(w.max().item()),
        "weight_q0": float(qs[0]),
        "weight_q25": float(qs[1]),
        "weight_q50": float(qs[2]),
        "weight_q75": float(qs[3]),
        "weight_q100": float(qs[4]),
        "weight_clipped_fraction": float(clipped),
        "weight_numerical_zero_fraction": float(near_zero),
    }
    out.update(ess_stats(w))
    return out

def advantage_diagnostics(adv: torch.Tensor) -> Dict[str, float]:
    a = adv.detach().float().reshape(-1)
    qs = torch.quantile(
        a, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0], device=a.device)
    ).cpu().numpy()
    return {
        "adv_mean": float(a.mean().item()),
        "adv_std": float(a.std(unbiased=False).item()),
        "adv_q0": float(qs[0]),
        "adv_q10": float(qs[1]),
        "adv_q25": float(qs[2]),
        "adv_q50": float(qs[3]),
        "adv_q75": float(qs[4]),
        "adv_q90": float(qs[5]),
        "adv_q100": float(qs[6]),
    }

@dataclass
class MetaConfig:
    adaptive_enabled: bool = True
    beta_initial: float = 3.0
    beta_fixed: float = 3.0
    beta_min: float = 0.05
    beta_max: float = 100.0
    rho_parameterization: str = "log_beta"
    rho_dtype: str = "float64"
    rho_lr: float = 1e-4
    rho_adam_betas: List[float] = field(default_factory=lambda: [0.0, 0.999])
    rho_adam_eps: float = 1e-8
    rho_weight_decay: float = 0.0
    meta_warmup_steps: int = 100000
    meta_interval: int = 20
    outer_batch_size: int = 256
    outer_reference: str = "current_beta_stopgrad"
    weight_cap: float = 100.0
    amo_dual_bpi: bool = False
    smoothness_eps: float = 1e-6

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "MetaConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def project_rho(rho: torch.Tensor, beta_min: float, beta_max: float) -> Tuple[torch.Tensor, bool]:
    lo = math.log(beta_min)
    hi = math.log(beta_max)
    projected = False
    with torch.no_grad():
        if float(rho.item()) < lo:
            rho.fill_(lo)
            projected = True
        elif float(rho.item()) > hi:
            rho.fill_(hi)
            projected = True
    return rho, projected

def beta_from_rho(rho: torch.Tensor) -> torch.Tensor:
    """Preserve graph: do not use .item() / float() / new tensor from Python float."""
    return torch.exp(rho)

def make_actor(
    *,
    deterministic: bool,
    state_dim: int,
    action_dim: int,
    max_action: float,
    actor_dropout: Optional[float],
    device: str,
) -> nn.Module:
    if deterministic:
        actor: nn.Module = DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=actor_dropout
        )
    else:
        actor = GaussianPolicy(
            state_dim, action_dim, max_action, dropout=actor_dropout
        )
    return actor.to(device)

def gaussian_log_std_stats(actor: nn.Module) -> Dict[str, float]:
    if not isinstance(actor, GaussianPolicy):
        return {}
    from tests.references.iql_amo import LOG_STD_MAX, LOG_STD_MIN

    ls = actor.log_std.detach()
    clamped = ls.clamp(LOG_STD_MIN, LOG_STD_MAX)
    hit_lo = (ls <= LOG_STD_MIN).float().mean().item()
    hit_hi = (ls >= LOG_STD_MAX).float().mean().item()
    return {
        "log_std_mean": float(ls.mean().item()),
        "log_std_min": float(ls.min().item()),
        "log_std_max": float(ls.max().item()),
        "log_std_clamp_lo_fraction": float(hit_lo),
        "log_std_clamp_hi_fraction": float(hit_hi),
        "log_std_clamped_mean": float(clamped.mean().item()),
    }

class IQLAdaptiveBetaTrainer:
    def __init__(
        self,
        *,
        corl: Dict[str, Any],
        meta: MetaConfig,
        device: str,
        seed: int,
        max_action: float,
        state_dim: int,
        action_dim: int,
    ):
        self.device = device
        self.seed = seed
        self.corl = corl
        self.meta = meta
        self.discount = float(corl["discount"])
        self.tau = float(corl["tau"])
        self.iql_tau = float(corl["iql_tau"])
        self.batch_size = int(corl["batch_size"])
        self.max_timesteps = int(corl["max_timesteps"])
        self.iql_deterministic = bool(corl["iql_deterministic"])
        actor_dropout = corl.get("actor_dropout", None)

        self.qf = TwinQ(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = ValueFunction(state_dim).to(device)

        # Preserve CORL init order: Q, V, then actor; second actor = deepcopy.
        actor0 = make_actor(
            deterministic=self.iql_deterministic,
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            actor_dropout=actor_dropout,
            device=device,
        )
        self.actor_fixed = actor0
        self.actor_adaptive = copy.deepcopy(actor0)

        lr_v = float(corl["vf_lr"])
        lr_q = float(corl["qf_lr"])
        lr_a = float(corl["actor_lr"])
        self.v_optimizer = torch.optim.Adam(self.vf.parameters(), lr=lr_v, betas=(0.9, 0.999), eps=1e-8)
        self.q_optimizer = torch.optim.Adam(self.qf.parameters(), lr=lr_q, betas=(0.9, 0.999), eps=1e-8)
        self.actor_fixed_optimizer = torch.optim.Adam(
            self.actor_fixed.parameters(), lr=lr_a, betas=(0.9, 0.999), eps=1e-8
        )
        self.actor_adaptive_optimizer = torch.optim.Adam(
            self.actor_adaptive.parameters(), lr=lr_a, betas=(0.9, 0.999), eps=1e-8
        )
        self.actor_fixed_lr_schedule = CosineAnnealingLR(
            self.actor_fixed_optimizer, T_max=self.max_timesteps
        )
        self.actor_adaptive_lr_schedule = CosineAnnealingLR(
            self.actor_adaptive_optimizer, T_max=self.max_timesteps
        )

        beta0 = float(meta.beta_initial)
        self.beta_fixed = float(meta.beta_fixed)
        self.weight_cap = float(meta.weight_cap)
        dtype = torch.float64 if meta.rho_dtype == "float64" else torch.float32
        self.rho = torch.nn.Parameter(
            torch.tensor(math.log(beta0), dtype=dtype, device=device)
        )
        betas_rho = tuple(meta.rho_adam_betas)
        self.rho_optimizer = torch.optim.Adam(
            [self.rho],
            lr=float(meta.rho_lr),
            betas=betas_rho,
            eps=float(meta.rho_adam_eps),
            weight_decay=float(meta.rho_weight_decay),
        )
        self.rho_B: Optional[torch.nn.Parameter] = None
        self.rho_B_optimizer: Optional[torch.optim.Adam] = None
        if meta.amo_dual_bpi:
            beta_B = float(meta.beta_fixed) if meta.beta_fixed > 0 else beta0
            self.rho_B = torch.nn.Parameter(
                torch.tensor(math.log(beta_B), dtype=dtype, device=device)
            )
            self.rho_B_optimizer = torch.optim.Adam(
                [self.rho_B],
                lr=float(meta.rho_lr),
                betas=betas_rho,
                eps=float(meta.rho_adam_eps),
                weight_decay=float(meta.rho_weight_decay),
            )

        self.total_it = 0
        self.completed_meta_events = 0
        self.outer_rng = np.random.RandomState(seed + 100003)
        self.max_action = max_action

    # ---- beta helpers ----
    def current_beta_tensor(self) -> torch.Tensor:
        if not self.meta.adaptive_enabled:
            # Keep as tensor for API uniformity; no graph needed.
            return torch.tensor(
                self.beta_fixed, dtype=torch.float32, device=self.device
            )
        # Cast for model math while preserving graph from float64 rho.
        return beta_from_rho(self.rho).to(dtype=torch.float32)

    def current_beta_value(self) -> float:
        with torch.no_grad():
            return float(self.current_beta_tensor().detach().float().cpu().item())

    def current_beta0_tensor(self) -> torch.Tensor:
        """Learnable beta0 = exp(rho_B) when amo_dual_bpi; else frozen beta_fixed."""
        if self.meta.amo_dual_bpi and self.rho_B is not None:
            return beta_from_rho(self.rho_B).to(dtype=torch.float32)
        return torch.tensor(self.beta_fixed, dtype=torch.float32, device=self.device)

    def current_beta0_value(self) -> float:
        with torch.no_grad():
            return float(self.current_beta0_tensor().detach().float().cpu().item())

    # ---- one training step ----
    def train_step(
        self,
        batch_b: List[torch.Tensor],
        batch_c: Optional[List[torch.Tensor]],
        *,
        do_meta: bool,
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:
        self.total_it += 1
        step = self.total_it

        states, actions, rewards, next_states, dones = batch_b
        rewards_b = squeeze_to_1d(rewards, "rewards_B")
        dones_b = squeeze_to_1d(dones, "dones_B")

        # Cache advantages BEFORE Q/V updates.
        with torch.no_grad():
            next_v_b = self.vf(next_states)
            next_v_b = squeeze_to_1d(next_v_b, "next_v_B")
            target_q_b = self.q_target(states, actions)
            target_q_b = squeeze_to_1d(target_q_b, "target_q_B")
            v_b = self.vf(states)
            v_b = squeeze_to_1d(v_b, "v_B")
            adv_b = target_q_b - v_b
            assert_vec1d(adv_b, "adv_B")

            adv_c = None
            if do_meta:
                assert batch_c is not None
                s_c, a_c, _, _, _ = batch_c
                target_q_c = squeeze_to_1d(self.q_target(s_c, a_c), "target_q_C")
                v_c = squeeze_to_1d(self.vf(s_c), "v_C")
                adv_c = target_q_c - v_c
                assert_vec1d(adv_c, "adv_C")

        # --- V update (once) ---
        # Recompute graph for V: adv = target_q - V(states) with target_q stopped.
        with torch.no_grad():
            target_q_for_v = self.q_target(states, actions)
            target_q_for_v = squeeze_to_1d(target_q_for_v, "target_q_for_v")
        v = squeeze_to_1d(self.vf(states), "v_online")
        adv_for_v = target_q_for_v - v
        v_loss = asymmetric_l2_loss(adv_for_v, self.iql_tau)
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # --- Q update (once) ---
        y_b = rewards_b + self.discount * (1.0 - dones_b) * next_v_b
        assert_vec1d(y_b, "y_B")
        qs = self.qf.both(states, actions)
        q1 = squeeze_to_1d(qs[0], "q1")
        q2 = squeeze_to_1d(qs[1], "q2")
        # Match original: sum(mse)/len(qs) == 0.5*(mse1+mse2)
        q_loss = sum(F.mse_loss(q, y_b) for q in (q1, q2)) / 2
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        soft_update(self.q_target, self.qf, self.tau)

        adv_detached = adv_b.detach()

        # --- Fixed actor (always normal Adam) ---
        if self.meta.amo_dual_bpi:
            beta_fixed_t = self.current_beta0_tensor().detach()
        else:
            beta_fixed_t = torch.tensor(
                self.beta_fixed, dtype=torch.float32, device=self.device
            )
        w_fixed = safe_exp_weights(beta_fixed_t, adv_detached, cap=self.weight_cap)
        ell_fixed = actor_ell(self.actor_fixed, states, actions)
        loss_fixed = torch.mean(w_fixed * ell_fixed)
        self.actor_fixed_optimizer.zero_grad()
        loss_fixed.backward()
        self.actor_fixed_optimizer.step()
        self.actor_fixed_lr_schedule.step()

        # --- Adaptive actor ---
        meta_row: Optional[Dict[str, Any]] = None
        beta_used_t = self.current_beta_tensor()
        beta_used_val = float(beta_used_t.detach().float().cpu().item())

        if do_meta and self.meta.adaptive_enabled:
            if self.meta.amo_dual_bpi:
                meta_row = self._meta_amo_dual_update(
                    states=states,
                    actions=actions,
                    adv_b=adv_detached,
                    batch_c=batch_c,
                    beta_used_t=beta_used_t,
                )
            else:
                meta_row = self._meta_adaptive_update(
                    states=states,
                    actions=actions,
                    adv_b=adv_detached,
                    batch_c=batch_c,
                    adv_c=adv_c,
                    beta_used_t=beta_used_t,
                )
            self.completed_meta_events += 1
        else:
            # Normal CORL Adam update with beta_used (no graph through rho).
            w_ad = safe_exp_weights(
                beta_used_t.detach(), adv_detached, cap=self.weight_cap
            )
            ell_ad = actor_ell(self.actor_adaptive, states, actions)
            loss_ad = torch.mean(w_ad * ell_ad)
            self.actor_adaptive_optimizer.zero_grad()
            loss_ad.backward()
            self.actor_adaptive_optimizer.step()
            self.actor_adaptive_lr_schedule.step()

        beta_next_val = self.current_beta_value()
        actor_lr = float(self.actor_adaptive_optimizer.param_groups[0]["lr"])

        with torch.no_grad():
            q_data = self.qf(states, actions)
            q_data = squeeze_to_1d(q_data, "q_data")
            v_data = squeeze_to_1d(self.vf(states), "v_data")

        metrics: Dict[str, float] = {
            "step": float(step),
            "q_loss": float(q_loss.item()),
            "value_loss": float(v_loss.item()),
            "actor_fixed_loss": float(loss_fixed.item()),
            "actor_adaptive_loss": float(
                meta_row["L_inner"] if meta_row is not None else loss_ad.item()
            ),
            "beta_fixed": float(
                self.current_beta0_value()
                if self.meta.amo_dual_bpi
                else self.beta_fixed
            ),
            "beta_used": beta_used_val,
            "beta_next": beta_next_val,
            "actor_lr": actor_lr,
            "Q_data_mean": float(q_data.mean().item()),
            "Q_data_std": float(q_data.std(unbiased=False).item()),
            "V_data_mean": float(v_data.mean().item()),
            "V_data_std": float(v_data.std(unbiased=False).item()),
        }
        if self.meta.amo_dual_bpi:
            metrics["beta0"] = float(self.current_beta0_value())
            metrics["beta_adapt"] = beta_used_val
        metrics.update({f"{k}": v for k, v in advantage_diagnostics(adv_detached).items()})
        metrics.update(gaussian_log_std_stats(self.actor_adaptive))
        return metrics, meta_row

    def _meta_adaptive_update(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_b: torch.Tensor,
        batch_c: List[torch.Tensor],
        adv_c: torch.Tensor,
        beta_used_t: torch.Tensor,
    ) -> Dict[str, Any]:
        s_c, a_c, _, _, _ = batch_c
        named = named_parameters_dict(self.actor_adaptive)
        params = {k: v for k, v in named.items()}

        # Inner loss with beta graph preserved.
        w_inner = safe_exp_weights(beta_used_t, adv_b, cap=self.weight_cap)
        ell_inner = actor_ell(self.actor_adaptive, states, actions)
        L_inner = torch.mean(w_inner * ell_inner)

        grads = torch.autograd.grad(
            L_inner,
            list(params.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        named_grads = {n: g for n, g in zip(params.keys(), grads)}

        adam_state = adam_state_from_optimizer(self.actor_adaptive_optimizer, named)
        lr = float(self.actor_adaptive_optimizer.param_groups[0]["lr"])
        betas = self.actor_adaptive_optimizer.param_groups[0]["betas"]
        eps = self.actor_adaptive_optimizer.param_groups[0]["eps"]

        theta_plus, _new_st = functional_adam_step(
            params,
            named_grads,
            adam_state,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
        )

        # Outer reference weights: stopgrad on beta and weights.
        w_ref_c = safe_exp_weights(
            beta_used_t.detach(), adv_c.detach(), cap=self.weight_cap
        ).detach()
        ell_outer = actor_ell_functional(self.actor_adaptive, theta_plus, s_c, a_c)
        L_meta = torch.mean(w_ref_c * ell_outer)

        g_rho = torch.autograd.grad(L_meta, self.rho, retain_graph=False)[0]

        # Real adaptive actor update: ONE Adam.step with detached inner grads.
        self.actor_adaptive_optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params.values(), grads):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach()
        self.actor_adaptive_optimizer.step()
        self.actor_adaptive_lr_schedule.step()

        # Rho update + projection.
        rho_before = float(self.rho.detach().cpu().item())
        self.rho_optimizer.zero_grad(set_to_none=True)
        self.rho.grad = g_rho.detach()
        self.rho_optimizer.step()
        _, projected = project_rho(self.rho, self.meta.beta_min, self.meta.beta_max)
        rho_after = float(self.rho.detach().cpu().item())

        # Rho Adam diagnostics.
        rho_st = self.rho_optimizer.state.get(self.rho, {})
        exp_avg = float(rho_st["exp_avg"].detach().cpu().item()) if rho_st else 0.0
        exp_avg_sq = float(rho_st["exp_avg_sq"].detach().cpu().item()) if rho_st else 0.0
        step_rho = int(rho_st["step"]) if rho_st else 0
        b1, b2 = self.rho_optimizer.param_groups[0]["betas"]
        if step_rho > 0:
            v_hat = exp_avg_sq / (1.0 - b2**step_rho)
            sqrt_v_hat = math.sqrt(max(v_hat, 0.0))
        else:
            sqrt_v_hat = 0.0

        row: Dict[str, Any] = {
            "step": self.total_it,
            "completed_meta_events": self.completed_meta_events + 1,
            "rho_used": rho_before,
            "rho_next": rho_after,
            "beta_used": float(math.exp(rho_before)),
            "beta_next": float(math.exp(rho_after)),
            "L_inner": float(L_inner.detach().cpu().item()),
            "L_meta": float(L_meta.detach().cpu().item()),
            "g_rho": float(g_rho.detach().cpu().item()),
            "delta_rho": rho_after - rho_before,
            "rho_exp_avg": exp_avg,
            "rho_sqrt_v_hat": sqrt_v_hat,
            "rho_optimizer_step": step_rho,
            "actor_optimizer_step": int(
                next(iter(self.actor_adaptive_optimizer.state.values()))["step"]
            )
            if self.actor_adaptive_optimizer.state
            else 0,
            "actor_learning_rate": lr,
            "projection_applied": bool(projected),
        }
        row.update({f"inner_{k}": v for k, v in weight_diagnostics(w_inner, self.weight_cap).items()})
        row.update({f"outer_{k}": v for k, v in weight_diagnostics(w_ref_c, self.weight_cap).items()})
        row.update({f"inner_{k}": v for k, v in advantage_diagnostics(adv_b).items()})
        row.update({f"outer_{k}": v for k, v in advantage_diagnostics(adv_c).items()})
        return row

    def _meta_amo_dual_update(
        self,
        *,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_b: torch.Tensor,
        batch_c: List[torch.Tensor],
        beta_used_t: torch.Tensor,
    ) -> Dict[str, Any]:
        """AMO dual-scale: B_PI on beta_adapt (rho) + TB bootstrap on beta0 (rho_B)."""
        assert self.rho_B is not None and self.rho_B_optimizer is not None
        s_c, a_c, _r_c, next_s_c, dones_c = batch_c
        dones_c = squeeze_to_1d(dones_c, "dones_C")
        smoothness_eps = float(self.meta.smoothness_eps)

        # ---- beta_adapt / B_PI path (actor_adaptive) ----
        named_ad = named_parameters_dict(self.actor_adaptive)
        params_ad = {k: v for k, v in named_ad.items()}

        w_inner = safe_exp_weights(beta_used_t, adv_b, cap=self.weight_cap)
        ell_inner = actor_ell(self.actor_adaptive, states, actions)
        L_inner = torch.mean(w_inner * ell_inner)

        grads_ad = torch.autograd.grad(
            L_inner,
            list(params_ad.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        named_grads_ad = {n: g for n, g in zip(params_ad.keys(), grads_ad)}

        adam_state_ad = adam_state_from_optimizer(
            self.actor_adaptive_optimizer, named_ad
        )
        lr_ad = float(self.actor_adaptive_optimizer.param_groups[0]["lr"])
        betas_ad = self.actor_adaptive_optimizer.param_groups[0]["betas"]
        eps_ad = self.actor_adaptive_optimizer.param_groups[0]["eps"]

        theta_plus, _ = functional_adam_step(
            params_ad,
            named_grads_ad,
            adam_state_ad,
            lr=lr_ad,
            betas=betas_ad,
            eps=eps_ad,
            weight_decay=0.0,
        )

        pi = actor_action_mean(self.actor_adaptive, s_c)
        pi_new = actor_action_mean_functional(self.actor_adaptive, theta_plus, s_c)
        L_E, b_pi_logs = b_pi_outer_loss(
            self.q_target,
            s_c,
            pi,
            pi_new,
            smoothness_eps=smoothness_eps,
        )
        g_rho = torch.autograd.grad(L_E, self.rho, retain_graph=False)[0]

        self.actor_adaptive_optimizer.zero_grad(set_to_none=True)
        for p, g in zip(params_ad.values(), grads_ad):
            if g is None:
                p.grad = None
            else:
                p.grad = g.detach()
        self.actor_adaptive_optimizer.step()
        self.actor_adaptive_lr_schedule.step()

        rho_before = float(self.rho.detach().cpu().item())
        self.rho_optimizer.zero_grad(set_to_none=True)
        self.rho.grad = g_rho.detach()
        self.rho_optimizer.step()
        _, projected_E = project_rho(self.rho, self.meta.beta_min, self.meta.beta_max)
        rho_after = float(self.rho.detach().cpu().item())

        # ---- beta0 / TB path (virtual actor_fixed; AMO SGD-style) ----
        beta0_t = self.current_beta0_tensor()
        named_fx = named_parameters_dict(self.actor_fixed)
        params_fx = {k: v for k, v in named_fx.items()}

        w_B = safe_exp_weights(beta0_t, adv_b, cap=self.weight_cap)
        ell_B = actor_ell(self.actor_fixed, states, actions)
        L_inner_B = torch.mean(w_B * ell_B)
        grads_fx = torch.autograd.grad(
            L_inner_B,
            list(params_fx.values()),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        lr_fx = float(self.actor_fixed_optimizer.param_groups[0]["lr"])
        virtual_params: Dict[str, torch.Tensor] = {}
        for (name, p), g in zip(params_fx.items(), grads_fx):
            if g is None:
                virtual_params[name] = p
            else:
                virtual_params[name] = p - lr_fx * g

        # Freeze target Q params while differentiating through actions only.
        target_params = list(self.q_target.parameters())
        target_req = [p.requires_grad for p in target_params]
        for p in target_params:
            p.requires_grad_(False)
        try:
            virtual_action = actor_action_mean_functional(
                self.actor_fixed, virtual_params, s_c
            )
            q_virtual = squeeze_to_1d(
                self.q_target(s_c, virtual_action), "q_virtual"
            )
            q_scale = q_virtual.abs().mean().detach() + smoothness_eps
            bc = F.mse_loss(virtual_action, a_c.detach())
            normalized_q = q_virtual.mean() / q_scale

            virtual_next = actor_action_mean_functional(
                self.actor_fixed, virtual_params, next_s_c
            )
            q_new = squeeze_to_1d(
                self.q_target(next_s_c, virtual_next), "q_new"
            )
            with torch.no_grad():
                current_next = actor_action_mean(self.actor_fixed, next_s_c).detach()
                q_old = squeeze_to_1d(
                    self.q_target(next_s_c, current_next), "q_old"
                )
            delta_q = q_new - q_old.detach()
            target_rms = bootstrap_target_rms(
                delta_q, dones_c, q_scale, self.discount
            )
            R_B = target_rms["R_B"]
            terms = tq_scaled_bootstrap_terms(beta0_t, normalized_q, bc, R_B)
            L_T_B = terms["L_T_B"]
        finally:
            for p, value in zip(target_params, target_req):
                p.requires_grad_(value)

        g_rho_B = torch.autograd.grad(L_T_B, self.rho_B, retain_graph=False)[0]

        rho_B_before = float(self.rho_B.detach().cpu().item())
        self.rho_B_optimizer.zero_grad(set_to_none=True)
        self.rho_B.grad = g_rho_B.detach()
        self.rho_B_optimizer.step()
        _, projected_B = project_rho(
            self.rho_B, self.meta.beta_min, self.meta.beta_max
        )
        rho_B_after = float(self.rho_B.detach().cpu().item())

        def _rho_adam_diag(opt: torch.optim.Adam, param: torch.nn.Parameter):
            st = opt.state.get(param, {})
            exp_avg = float(st["exp_avg"].detach().cpu().item()) if st else 0.0
            exp_avg_sq = float(st["exp_avg_sq"].detach().cpu().item()) if st else 0.0
            step_rho = int(st["step"]) if st else 0
            b1, b2 = opt.param_groups[0]["betas"]
            if step_rho > 0:
                v_hat = exp_avg_sq / (1.0 - b2**step_rho)
                sqrt_v_hat = math.sqrt(max(v_hat, 0.0))
            else:
                sqrt_v_hat = 0.0
            return exp_avg, sqrt_v_hat, step_rho

        exp_avg_E, sqrt_v_E, step_E = _rho_adam_diag(self.rho_optimizer, self.rho)
        exp_avg_B, sqrt_v_B, step_B = _rho_adam_diag(self.rho_B_optimizer, self.rho_B)

        row: Dict[str, Any] = {
            "step": self.total_it,
            "completed_meta_events": self.completed_meta_events + 1,
            "amo_dual_bpi": True,
            "rho_used": rho_before,
            "rho_next": rho_after,
            "beta_used": float(math.exp(rho_before)),
            "beta_next": float(math.exp(rho_after)),
            "rho_B_used": rho_B_before,
            "rho_B_next": rho_B_after,
            "beta0_used": float(math.exp(rho_B_before)),
            "beta0_next": float(math.exp(rho_B_after)),
            "L_inner": float(L_inner.detach().cpu().item()),
            "L_E": float(L_E.detach().cpu().item()),
            "L_meta": float(L_E.detach().cpu().item()),  # alias for metrics compat
            "L_inner_B": float(L_inner_B.detach().cpu().item()),
            "L_T_B": float(L_T_B.detach().cpu().item()),
            "L1_B": float(terms["L1_B"].detach().cpu().item()),
            "L2_RMS_B": float(terms["L2_RMS_B"].detach().cpu().item()),
            "R_B": float(R_B.detach().cpu().item()),
            "q_scale_B": float(q_scale.detach().cpu().item()),
            "g_rho": float(g_rho.detach().cpu().item()),
            "g_rho_B": float(g_rho_B.detach().cpu().item()),
            "delta_rho": rho_after - rho_before,
            "delta_rho_B": rho_B_after - rho_B_before,
            "rho_exp_avg": exp_avg_E,
            "rho_sqrt_v_hat": sqrt_v_E,
            "rho_optimizer_step": step_E,
            "rho_B_exp_avg": exp_avg_B,
            "rho_B_sqrt_v_hat": sqrt_v_B,
            "rho_B_optimizer_step": step_B,
            "actor_optimizer_step": int(
                next(iter(self.actor_adaptive_optimizer.state.values()))["step"]
            )
            if self.actor_adaptive_optimizer.state
            else 0,
            "actor_learning_rate": lr_ad,
            "actor_fixed_learning_rate": lr_fx,
            "projection_applied": bool(projected_E),
            "projection_applied_B": bool(projected_B),
            "bootstrap_outer_loss_version": "tq_detached_rms_target_v1",
        }
        row.update({f"b_pi_{k}": v for k, v in b_pi_logs.items()})
        row.update(
            {f"inner_{k}": v for k, v in weight_diagnostics(w_inner, self.weight_cap).items()}
        )
        row.update({f"inner_{k}": v for k, v in advantage_diagnostics(adv_b).items()})
        return row

    # ---- checkpoint ----
    def state_dict(self) -> Dict[str, Any]:
        fmt = (
            "corl_iql_amo_bpi_v1"
            if self.meta.amo_dual_bpi
            else "corl_iql_adaptive_beta_v1"
        )
        out: Dict[str, Any] = {
            "qf": self.qf.state_dict(),
            "q_target": self.q_target.state_dict(),
            "vf": self.vf.state_dict(),
            "actor_fixed": self.actor_fixed.state_dict(),
            "actor_adaptive": self.actor_adaptive.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor_fixed_optimizer": self.actor_fixed_optimizer.state_dict(),
            "actor_adaptive_optimizer": self.actor_adaptive_optimizer.state_dict(),
            "actor_fixed_lr_schedule": self.actor_fixed_lr_schedule.state_dict(),
            "actor_adaptive_lr_schedule": self.actor_adaptive_lr_schedule.state_dict(),
            "rho": self.rho.detach().cpu(),
            "rho_optimizer": self.rho_optimizer.state_dict(),
            "total_it": self.total_it,
            "completed_meta_events": self.completed_meta_events,
            "rng": get_rng_state(self.outer_rng),
            "meta": self.meta.to_dict(),
            "corl": self.corl,
            "format": fmt,
        }
        if self.meta.amo_dual_bpi and self.rho_B is not None and self.rho_B_optimizer is not None:
            out["rho_B"] = self.rho_B.detach().cpu()
            out["rho_B_optimizer"] = self.rho_B_optimizer.state_dict()
        return out

    def load_state_dict(self, ckpt: Dict[str, Any], *, strict_target: bool = True) -> None:
        fmt = ckpt.get("format")
        allowed = {"corl_iql_adaptive_beta_v1", "corl_iql_amo_bpi_v1"}
        if fmt not in allowed:
            raise ValueError(
                f"Incompatible checkpoint format: {fmt!r}; "
                "smoke checkpoints must not be resumed as production."
            )
        if fmt == "corl_iql_amo_bpi_v1" and not self.meta.amo_dual_bpi:
            raise ValueError(
                "Checkpoint is amo_dual_bpi but trainer meta.amo_dual_bpi is False"
            )
        if fmt == "corl_iql_adaptive_beta_v1" and self.meta.amo_dual_bpi:
            raise ValueError(
                "Checkpoint is non-amo adaptive-beta but trainer meta.amo_dual_bpi is True"
            )
        self.qf.load_state_dict(ckpt["qf"])
        if "q_target" not in ckpt:
            raise ValueError("Checkpoint missing target Q; refuse online-Q overwrite")
        self.q_target.load_state_dict(ckpt["q_target"])
        self.vf.load_state_dict(ckpt["vf"])
        self.actor_fixed.load_state_dict(ckpt["actor_fixed"])
        self.actor_adaptive.load_state_dict(ckpt["actor_adaptive"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.v_optimizer.load_state_dict(ckpt["v_optimizer"])
        self.actor_fixed_optimizer.load_state_dict(ckpt["actor_fixed_optimizer"])
        self.actor_adaptive_optimizer.load_state_dict(ckpt["actor_adaptive_optimizer"])
        self.actor_fixed_lr_schedule.load_state_dict(ckpt["actor_fixed_lr_schedule"])
        self.actor_adaptive_lr_schedule.load_state_dict(ckpt["actor_adaptive_lr_schedule"])
        with torch.no_grad():
            self.rho.copy_(ckpt["rho"].to(device=self.rho.device, dtype=self.rho.dtype))
        self.rho_optimizer.load_state_dict(ckpt["rho_optimizer"])
        if self.meta.amo_dual_bpi:
            if "rho_B" not in ckpt or "rho_B_optimizer" not in ckpt:
                raise ValueError("amo_dual_bpi checkpoint missing rho_B / rho_B_optimizer")
            assert self.rho_B is not None and self.rho_B_optimizer is not None
            with torch.no_grad():
                self.rho_B.copy_(
                    ckpt["rho_B"].to(device=self.rho_B.device, dtype=self.rho_B.dtype)
                )
            self.rho_B_optimizer.load_state_dict(ckpt["rho_B_optimizer"])
        self.total_it = int(ckpt["total_it"])
        self.completed_meta_events = int(ckpt["completed_meta_events"])
        set_rng_state(ckpt["rng"], self.outer_rng)
