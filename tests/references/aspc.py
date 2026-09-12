# Extracted without semantic changes from ASPC/algorithms/offline/aspc.py; see SOURCE_PROVENANCE.md.
import copy
import math
import json
from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchopt
from torch.optim.lr_scheduler import CosineAnnealingLR
TensorBatch = List[torch.Tensor]
BOOTSTRAP_OUTER_LOSS_VERSION = "tq_detached_rms_target_v1"
BOOTSTRAP_RMS_DIAGNOSTIC_SMALL_H = 1e-3
BOOTSTRAP_RMS_DIAGNOSTIC_LARGE_H = 5e-2
EXP_ADV_MAX=100.
LOG_STD_MIN=-20.
LOG_STD_MAX=2.

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

class Actor(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, action_dim), nn.Tanh())
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str='cpu') -> np.ndarray:
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return self(state).cpu().data.numpy().flatten()

class Critic(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=256, layernorm: bool=True, use_dyt: bool=True, n_hiddens: int=3):
        super(Critic, self).__init__()
        layers = [nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU()]
        if layernorm:
            layers.append(nn.LayerNorm(hidden_dim))
        for _ in range(n_hiddens - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._initialize_weights(state_dim, action_dim, hidden_dim)

    def _initialize_weights(self, state_dim, action_dim, hidden_dim):
        nn.init.uniform_(self.net[0].weight, -1.0 / np.sqrt(state_dim + action_dim), 1.0 / np.sqrt(state_dim + action_dim))
        nn.init.constant_(self.net[0].bias, 0.1)
        for i in range(1, len(self.net) - 2):
            if isinstance(self.net[i], nn.Linear):
                nn.init.uniform_(self.net[i].weight, -1.0 / np.sqrt(hidden_dim), 1.0 / np.sqrt(hidden_dim))
                nn.init.constant_(self.net[i].bias, 0.1)
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        nn.init.uniform_(self.net[-1].bias, -0.003, 0.003)

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

class Adaptive_TD3_BC:

    def __init__(self, max_action: float, actor: nn.Module, actor_optimizer: torchopt.Optimizer, critic_1: nn.Module, critic_1_optimizer: torch.optim.Optimizer, critic_2: nn.Module, critic_2_optimizer: torch.optim.Optimizer, vnet: nn.Module, vnet_optimizer: torch.optim.Optimizer, discount: float=0.99, tau: float=0.005, policy_noise: float=0.2, noise_clip: float=0.5, policy_freq: int=2, alpha: float=2.5, alpha_freq: int=10, ema_alpha: float=0.995, loss_function: int=4, device: str='cpu', action_dim: int=1):
        self.actor = actor
        self.actor_target = copy.deepcopy(actor)
        self.actor_optimizer = actor_optimizer
        named_params = list(self.actor.named_parameters())
        params = [p for _, p in named_params]
        self.actor_opt_state = self.actor_optimizer.init(params)
        self.critic_1 = critic_1
        self.critic_1_target = copy.deepcopy(critic_1)
        self.critic_1_optimizer = critic_1_optimizer
        self.critic_2 = critic_2
        self.critic_2_target = copy.deepcopy(critic_2)
        self.critic_2_optimizer = critic_2_optimizer
        self.vnet = vnet
        self.vnet_optimizer = vnet_optimizer
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha_freq = alpha_freq
        self.alpha = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha)) - 1.0))
        self.alpha_optimizer = torch.optim.Adam([self.alpha], lr=0.002 / 10 * self.alpha_freq)
        self.gamma = 0.01 ** (1 / int(1000000.0 / (self.policy_freq * self.alpha_freq)))
        self.alpha_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.alpha_optimizer, gamma=self.gamma)
        self.total_it = 0
        self.device = device
        self.ema_alpha = ema_alpha
        self.ema_q = torch.tensor(0.0, device=device)
        self.previous_ema_q = 0.0
        self.loss_function = loss_function
        self.action_dim = action_dim

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        log_dict = {}
        self.total_it += 1
        state, action, reward, next_state, done = batch
        not_done = 1 - done
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward - self.reward_mean + not_done * self.discount * target_q
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        if self.total_it % self.policy_freq == 0:
            alpha = F.softplus(self.alpha)
            log_dict['alpha'] = alpha.item()
            named_params = list(self.actor.named_parameters())
            param_names = [n for n, _ in named_params]
            params = [p for _, p in named_params]
            actor_param_dict = {n: p for n, p in zip(param_names, params)}
            pi = torch.func.functional_call(self.actor, actor_param_dict, (state,))
            q = self.critic_1(state, pi)
            log_dict['q'] = q.mean().item()
            q_abs_mean = q.abs().mean().detach()
            lmbda = alpha / q_abs_mean
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            grads = torch.autograd.grad(actor_loss, params, create_graph=True)
            updates, self.actor_opt_state = self.actor_optimizer.update(grads, self.actor_opt_state, inplace=False)
            self.actor_opt_state = torchopt.pytree.tree_map(lambda x: x.detach() if isinstance(x, torch.Tensor) else x, self.actor_opt_state)
            updated_params = torchopt.apply_updates(params, list(updates), inplace=False)
            if self.total_it % (self.policy_freq * self.alpha_freq) == 0:
                updated_actor_param_dict = {n: p_new for n, p_new in zip(param_names, updated_params)}
                pi_new = torch.func.functional_call(self.actor, updated_actor_param_dict, (state,))
                q_new = self.critic_1(state, pi_new)
                q_new_mean = q_new.mean()
                self.ema_q = (1 - self.ema_alpha) * q_new_mean + self.ema_alpha * self.ema_q.detach()
                delta_q = self.ema_q - self.previous_ema_q
                bc_sup = F.mse_loss(pi, action, reduction='none').mean(dim=1).max().detach()
                delta_bc_sup = (F.mse_loss(pi, action, reduction='none').mean(dim=1).detach() - F.mse_loss(pi_new, action, reduction='none').mean(dim=1)).abs().max()
                eq1 = -alpha.detach() * (q_new.mean() / q_new.abs().mean().detach()) + F.mse_loss(pi_new, action)
                eq2 = delta_q ** 2
                eq3 = eq2.detach() * bc_sup * delta_bc_sup
                if self.loss_function == 1:
                    alpha_loss = eq1
                elif self.loss_function == 2:
                    alpha_loss = eq1 + eq2
                elif self.loss_function == 3:
                    alpha_loss = eq1 + eq3
                elif self.loss_function == 4:
                    alpha_loss = eq1 + eq2 + eq3
                log_dict['bc_loss_sup'] = bc_sup.item()
                log_dict['delta_Q'] = torch.abs(delta_q).item()
                log_dict['delta_bc_sup'] = delta_bc_sup.item()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha_scheduler.step()
                self.previous_ema_q = self.ema_q.detach().item()
            with torch.no_grad():
                for (n, _), p_new in zip(named_params, updated_params):
                    p_old = dict(self.actor.named_parameters())[n]
                    p_old.copy_(p_new)
            soft_update(self.critic_1_target, self.critic_1, self.tau)
            soft_update(self.critic_2_target, self.critic_2, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)
        return log_dict

    def compute_actor_loss(self, actor_params, state, action):
        pi = torch.func.functional_call(self.actor, actor_params, (state,))
        q = self.critic_1(state, pi)
        self.old_q_mean = q.mean().detach()
        q_abs_mean = q.abs().mean().detach()
        lmbda = self.alpha / q_abs_mean
        actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
        return actor_loss

    def state_dict(self) -> Dict[str, Any]:
        return {'critic_1': self.critic_1.state_dict(), 'critic_1_optimizer': self.critic_1_optimizer.state_dict(), 'critic_2': self.critic_2.state_dict(), 'critic_2_optimizer': self.critic_2_optimizer.state_dict(), 'actor': self.actor.state_dict(), 'actor_optimizer': self.actor_optimizer.state_dict(), 'total_it': self.total_it}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.critic_1.load_state_dict(state_dict['critic_1'])
        self.critic_1_optimizer.load_state_dict(state_dict['critic_1_optimizer'])
        self.critic_1_target = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(state_dict['critic_2'])
        self.critic_2_optimizer.load_state_dict(state_dict['critic_2_optimizer'])
        self.critic_2_target = copy.deepcopy(self.critic_2)
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.actor_target = copy.deepcopy(self.actor)
        self.total_it = state_dict['total_it']
