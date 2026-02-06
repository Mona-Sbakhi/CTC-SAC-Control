#!/usr/bin/env python3
"""
Soft Actor-Critic (SAC) Agent
CoDIT 2026 Paper - Compatible with Colab checkpoint
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SACConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    target_entropy: float = None
    init_alpha: float = 0.2
    batch_size: int = 256
    buffer_size: int = 1000000
    learning_starts: int = 1000
    action_low: float = -5.0
    action_high: float = 5.0


class GaussianActor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -20, 2
    
    def __init__(self, state_dim, action_dim, hidden_dims, action_low, action_high):
        super().__init__()
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2
        
        layers = []
        dims = [state_dim] + list(hidden_dims)
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
    
    def forward(self, state):
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)
    
    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        if deterministic:
            return torch.tanh(mean) * self.action_scale + self.action_bias
        return self.sample(state)[0]


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        input_dim = state_dim + action_dim
        
        def build_q():
            layers = []
            dims = [input_dim] + list(hidden_dims) + [1]
            for i in range(len(dims) - 2):
                layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
            layers.append(nn.Linear(dims[-2], dims[-1]))
            return nn.Sequential(*layers)
        
        self.q1, self.q2 = build_q(), build_q()
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, config=None, device='cpu'):
        self.config = config or SACConfig()
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor
        self.actor = GaussianActor(
            state_dim, action_dim, self.config.hidden_dims,
            self.config.action_low, self.config.action_high
        ).to(self.device)
        
        # Critic
        self.critic = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy
        self.log_alpha = torch.tensor(np.log(self.config.init_alpha), requires_grad=True, device=self.device)
        self.target_entropy = self.config.target_entropy or -action_dim
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        
        self.total_steps = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_t, deterministic)
            return action.cpu().numpy()[0]
    
    def load(self, path):
        """Load model - compatible with Colab checkpoint format"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        print("✅ Model loaded successfully!")
    
    def save(self, path):
        """Save model"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'total_steps': self.total_steps
        }, path)


print("✅ sac_agent.py loaded!")
