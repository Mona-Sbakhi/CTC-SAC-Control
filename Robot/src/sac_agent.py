#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Agent
==============================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
Affiliation: Islamic University of Gaza / University of Seville
Conference: CoDIT 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SACConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr:    float = 3e-4
    critic_lr:   float = 3e-4
    alpha_lr:    float = 3e-4
    gamma:       float = 0.99
    tau:         float = 0.005
    init_alpha:  float = 0.2
    batch_size:  int   = 256
    buffer_size: int   = 1_000_000
    learning_starts: int = 1000
    action_low:  float = -5.0
    action_high: float = 5.0
    target_entropy: Optional[float] = None


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros(capacity,               dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros(capacity,               dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = float(reward)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).unsqueeze(1).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
            torch.FloatTensor(self.dones[idx]).unsqueeze(1).to(device),
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────

class GaussianActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2

    def __init__(self, state_dim, action_dim, hidden_dims, action_low, action_high):
        super().__init__()
        self.action_scale = (action_high - action_low) / 2
        self.action_bias  = (action_high + action_low) / 2

        layers, dims = [], [state_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.net          = nn.Sequential(*layers)
        self.mean_head    = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, state):
        x       = self.net(state)
        mean    = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std    = log_std.exp()
        x_t    = Normal(mean, std).rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = Normal(mean, std).log_prob(x_t) \
                   - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def get_action(self, state, deterministic=False):
        mean, _ = self.forward(state)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        return self.sample(state)[0]


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        in_dim = state_dim + action_dim

        def build():
            dims   = [in_dim] + list(hidden_dims) + [1]
            layers = []
            for i in range(len(dims) - 2):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
            layers.append(nn.Linear(dims[-2], dims[-1]))
            return nn.Sequential(*layers)

        self.q1, self.q2 = build(), build()

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ─────────────────────────────────────────────
# SAC AGENT
# ─────────────────────────────────────────────

class SACAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 config: Optional[SACConfig] = None,
                 device: str = 'cpu'):
        self.config     = config or SACConfig()
        self.device     = torch.device(device)
        self.state_dim  = state_dim
        self.action_dim = action_dim

        self.actor = GaussianActor(
            state_dim, action_dim, self.config.hidden_dims,
            self.config.action_low, self.config.action_high
        ).to(self.device)

        self.critic        = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.log_alpha      = torch.tensor(np.log(self.config.init_alpha),
                                           requires_grad=True, device=self.device)
        self.target_entropy = self.config.target_entropy or -action_dim

        self.actor_optimizer  = optim.Adam(self.actor.parameters(),  lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.alpha_optimizer  = optim.Adam([self.log_alpha],          lr=self.config.alpha_lr)

        self.replay_buffer = ReplayBuffer(self.config.buffer_size, state_dim, action_dim)
        self.total_steps   = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.actor.get_action(s, deterministic).cpu().numpy()[0]

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.config.learning_starts:
            return

        s, a, r, ns, d = self.replay_buffer.sample(self.config.batch_size, self.device)

        with torch.no_grad():
            na, log_pi = self.actor.sample(ns)
            q1_t, q2_t = self.critic_target(ns, na)
            q_target = r + self.config.gamma * (1 - d) * (torch.min(q1_t, q2_t) - self.alpha * log_pi)

        q1, q2 = self.critic(s, a)
        critic_loss = ((q1 - q_target).pow(2) + (q2 - q_target).pow(2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_a, log_pi = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_a)
        actor_loss = (self.alpha * log_pi - torch.min(q1_new, q2_new)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.config.tau * p.data + (1 - self.config.tau) * tp.data)

        self.total_steps += 1

    def save(self, path: str):
        torch.save({
            'actor':       self.actor.state_dict(),
            'critic':      self.critic.state_dict(),
            'log_alpha':   self.log_alpha,
            'total_steps': self.total_steps,
        }, path)
        print(f"  💾 Saved → {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic'])
        if 'log_alpha'   in ckpt: self.log_alpha   = ckpt['log_alpha']
        if 'total_steps' in ckpt: self.total_steps = ckpt['total_steps']
        print(f"  ✅ Loaded ← {path}")
