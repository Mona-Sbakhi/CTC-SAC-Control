#!/usr/bin/env python3
"""
Train SAC Agent directly in CoppeliaSim
CoDIT 2026 Paper
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from dataclasses import dataclass
from typing import Tuple
from collections import deque

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_dynamics import create_nominal_robot
from ctc_controller import CTCController, CTCGains

# ==========================================
# SAC COMPONENTS (Inline for simplicity)
# ==========================================

@dataclass
class SACConfig:
    hidden_dims: Tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    init_alpha: float = 0.2
    batch_size: int = 128
    buffer_size: int = 100000
    learning_starts: int = 500
    action_low: float = -5.0
    action_high: float = 5.0


class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, device='cpu'):
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.FloatTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }


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
        
        self.actor = GaussianActor(
            state_dim, action_dim, self.config.hidden_dims,
            self.config.action_low, self.config.action_high
        ).to(self.device)
        
        self.critic = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target = TwinQNetwork(state_dim, action_dim, self.config.hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.log_alpha = torch.tensor(np.log(self.config.init_alpha), requires_grad=True, device=self.device)
        self.target_entropy = -action_dim
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        
        self.buffer = ReplayBuffer(self.config.buffer_size, state_dim, action_dim)
        self.total_steps = 0
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor.get_action(state_t, deterministic)
            return action.cpu().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self):
        if self.buffer.size < self.config.learning_starts:
            return {}
        
        batch = self.buffer.sample(self.config.batch_size, self.device)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(-1)
        next_states = batch['next_states']
        dones = batch['dones'].unsqueeze(-1)
        
        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new, _ = self.critic(states, new_actions)
        actor_loss = (self.alpha.detach() * log_probs - q1_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}
    
    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha,
            'total_steps': self.total_steps
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


# ==========================================
# TRAINING CONFIGURATION
# ==========================================
NUM_EPISODES = 200
STEPS_PER_EPISODE = 200  # Shorter episodes for faster training
DT = 0.02  # 20ms per step
SAVE_EVERY = 50

# ==========================================
# CONNECT TO COPPELIASIM
# ==========================================
print("="*60)
print("CTC+SAC Training in CoppeliaSim")
print("="*60)

print("\nConnecting to CoppeliaSim...")
client = RemoteAPIClient(host='host.docker.internal')
sim = client.require('sim')
print("✅ Connected!")

# ==========================================
# SETUP ROBOT
# ==========================================
print("Setting up robot...")
joint1 = sim.getObject('/Mico/joint')
joint2 = sim.getObject('/Mico/joint/link/joint')

# Enable torque mode
sim.setObjectInt32Param(joint1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
sim.setObjectInt32Param(joint2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
print("✅ Robot ready!")

# ==========================================
# INITIALIZE CONTROLLERS
# ==========================================
robot = create_nominal_robot()
gains = CTCGains(Kp=np.array([100.0, 50.0]), Kd=np.array([20.0, 10.0]))
ctc = CTCController(robot, gains, include_friction=True)

# SAC Agent
sac_config = SACConfig(
    hidden_dims=(256, 256),
    action_low=-5.0,
    action_high=5.0,
    batch_size=128,
    learning_starts=500
)
agent = SACAgent(state_dim=8, action_dim=2, config=sac_config, device='cpu')
print("✅ Controllers initialized!")

# ==========================================
# TRAJECTORY FUNCTION
# ==========================================
def generate_trajectory(t, freq=0.5, amp=0.5):
    w = 2 * np.pi * freq
    q_d = np.array([amp * np.sin(w * t), amp * np.sin(w * t + np.pi/4)])
    q_dot_d = np.array([amp * w * np.cos(w * t), amp * w * np.cos(w * t + np.pi/4)])
    q_ddot_d = np.array([-amp * w**2 * np.sin(w * t), -amp * w**2 * np.sin(w * t + np.pi/4)])
    return q_d, q_dot_d, q_ddot_d

# ==========================================
# REWARD FUNCTION
# ==========================================
def compute_reward(e, e_dot, tau_rl):
    r = -10.0 * np.sum(e**2)      # Position error
    r += -1.0 * np.sum(e_dot**2)   # Velocity error
    r += -0.01 * np.sum(tau_rl**2) # Control effort
    if np.linalg.norm(e) < 0.1:
        r += 1.0  # Success bonus
    return r

# ==========================================
# TRAINING LOOP
# ==========================================
print("\n" + "="*60)
print(f"Starting Training: {NUM_EPISODES} episodes")
print("="*60)

os.makedirs('results', exist_ok=True)

episode_rewards = []
episode_errors = []
best_reward = -np.inf

for episode in range(NUM_EPISODES):
    # Start simulation
    sim.startSimulation()
    time.sleep(0.1)  # Let simulation settle
    
    # Reset robot to initial position
    sim.setJointPosition(joint1, 0.0)
    sim.setJointPosition(joint2, 0.0)
    sim.setJointTargetVelocity(joint1, 0.0)
    sim.setJointTargetVelocity(joint2, 0.0)
    time.sleep(0.1)
    
    episode_reward = 0
    errors = []
    t = 0.0
    
    for step in range(STEPS_PER_EPISODE):
        # Get current state
        q = np.array([sim.getJointPosition(joint1), sim.getJointPosition(joint2)])
        q_dot = np.array([sim.getJointVelocity(joint1), sim.getJointVelocity(joint2)])
        
        # Generate trajectory
        q_d, q_dot_d, q_ddot_d = generate_trajectory(t)
        
        # Compute errors
        e = q_d - q
        e_dot = q_dot_d - q_dot
        
        # Build state
        obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
        
        # CTC torque
        tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
        
        # SAC action
        tau_rl = agent.select_action(obs, deterministic=False)
        tau_rl = np.clip(tau_rl, -5.0, 5.0)
        
        # Total torque
        tau_total = np.clip(tau_ctc + tau_rl, -20.0, 20.0)
        
        # Apply torque
        sim.setJointTargetForce(joint1, float(tau_total[0]))
        sim.setJointTargetForce(joint2, float(tau_total[1]))
        
        # Step simulation
        time.sleep(DT)
        t += DT
        
        # Get next state
        q_next = np.array([sim.getJointPosition(joint1), sim.getJointPosition(joint2)])
        q_dot_next = np.array([sim.getJointVelocity(joint1), sim.getJointVelocity(joint2)])
        q_d_next, q_dot_d_next, _ = generate_trajectory(t)
        e_next = q_d_next - q_next
        e_dot_next = q_dot_d_next - q_dot_next
        next_obs = np.concatenate([q_next, q_dot_next, e_next, e_dot_next]).astype(np.float32)
        
        # Compute reward
        reward = compute_reward(e, e_dot, tau_rl)
        
        # Store transition
        done = (step == STEPS_PER_EPISODE - 1)
        agent.store_transition(obs, tau_rl, reward, next_obs, done)
        
        # Update agent
        agent.update()
        
        episode_reward += reward
        errors.append(np.linalg.norm(e))
    
    # Stop simulation
    sim.stopSimulation()
    time.sleep(0.1)
    
    # Log
    mean_error = np.mean(errors)
    episode_rewards.append(episode_reward)
    episode_errors.append(mean_error)
    
    # Save best model
    if episode_reward > best_reward:
        best_reward = episode_reward
        agent.save('results/best_ctc_sac_coppeliasim.pt')
    
    # Print progress
    if (episode + 1) % 10 == 0:
        recent_reward = np.mean(episode_rewards[-10:])
        recent_error = np.mean(episode_errors[-10:])
        print(f"Episode {episode+1:4d}/{NUM_EPISODES} | Reward: {recent_reward:8.2f} | Error: {recent_error:.4f} | Steps: {agent.total_steps}")
    
    # Save checkpoint
    if (episode + 1) % SAVE_EVERY == 0:
        agent.save(f'results/checkpoint_ep{episode+1}.pt')
        print(f"  💾 Saved checkpoint")

# ==========================================
# SAVE FINAL MODEL & RESULTS
# ==========================================
agent.save('results/final_ctc_sac_coppeliasim.pt')

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Smooth function
def smooth(data, window=10):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

ax1 = axes[0]
ax1.plot(smooth(episode_rewards), 'b-', linewidth=2)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Episode Reward')
ax1.set_title('Training Reward')
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(smooth(episode_errors), 'r-', linewidth=2)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Mean Tracking Error [rad]')
ax2.set_title('Tracking Error')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves_coppeliasim.png', dpi=150)
print("\n✅ Saved: results/training_curves_coppeliasim.png")

# Save training data
training_data = {
    'episode_rewards': episode_rewards,
    'episode_errors': episode_errors,
    'num_episodes': NUM_EPISODES,
    'steps_per_episode': STEPS_PER_EPISODE,
    'total_steps': agent.total_steps
}
with open('results/training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)

print("\n" + "="*60)
print("🎉 Training Complete!")
print("="*60)
print(f"\nFinal Results:")
print(f"  Episodes: {NUM_EPISODES}")
print(f"  Total Steps: {agent.total_steps}")
print(f"  Best Reward: {best_reward:.2f}")
print(f"  Final Error: {np.mean(episode_errors[-10:]):.4f} rad")
print(f"\nSaved models:")
print(f"  📦 results/best_ctc_sac_coppeliasim.pt")
print(f"  📦 results/final_ctc_sac_coppeliasim.pt")
