#!/usr/bin/env python3
"""
RL-Only (SAC without CTC) Training for Mico Robot
CoDIT 2026 Paper - Baseline Comparison
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

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# ==========================================
# SAC COMPONENTS (Same as CTC+SAC)
# ==========================================
class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size = 0, 0
    
    def add(self, s, a, r, ns, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = ns
        self.dones[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (torch.FloatTensor(self.states[idx]),
                torch.FloatTensor(self.actions[idx]),
                torch.FloatTensor(self.rewards[idx]),
                torch.FloatTensor(self.next_states[idx]),
                torch.FloatTensor(self.dones[idx]))


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256, action_scale=20.0):
        super().__init__()
        self.action_scale = action_scale  # Larger for RL-Only (needs full torque)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
    
    def forward(self, s):
        x = self.net(s)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mean, log_std
    
    def sample(self, s):
        mean, log_std = self.forward(s)
        std = log_std.exp()
        normal = Normal(mean, std)
        x = normal.rsample()
        action = torch.tanh(x) * self.action_scale
        log_prob = normal.log_prob(x) - torch.log(self.action_scale * (1 - torch.tanh(x).pow(2)) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)
    
    def get_action(self, s, deterministic=False):
        mean, log_std = self.forward(s)
        if deterministic:
            return torch.tanh(mean) * self.action_scale
        return self.sample(s)[0]


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
    
    def forward(self, s, a):
        x = torch.cat([s, a], -1)
        return self.q1(x), self.q2(x)


class SAC:
    def __init__(self, state_dim, action_dim, action_scale=20.0, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.gamma, self.tau = gamma, tau
        self.action_scale = action_scale
        
        self.actor = Actor(state_dim, action_dim, action_scale=action_scale)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.target_entropy = -action_dim
        
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        
        self.buffer = ReplayBuffer(100000, state_dim, action_dim)
    
    @property
    def alpha(self):
        return self.log_alpha.exp()
    
    def select_action(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.FloatTensor(s).unsqueeze(0)
            return self.actor.get_action(s, deterministic).numpy()[0]
    
    def update(self, batch_size=128):
        if self.buffer.size < 500:
            return
        
        s, a, r, ns, d = self.buffer.sample(batch_size)
        
        with torch.no_grad():
            na, log_prob = self.actor.sample(ns)
            tq1, tq2 = self.critic_target(ns, na)
            target = r.unsqueeze(-1) + self.gamma * (1 - d.unsqueeze(-1)) * (torch.min(tq1, tq2) - self.alpha * log_prob)
        
        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        new_a, log_prob = self.actor.sample(s)
        q1, _ = self.critic(s, new_a)
        actor_loss = (self.alpha.detach() * log_prob - q1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
    
    def save(self, path):
        torch.save({'actor': self.actor.state_dict(), 
                    'critic': self.critic.state_dict(),
                    'log_alpha': self.log_alpha}, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])


# ==========================================
# TRAJECTORY (Same as CTC+SAC)
# ==========================================
def generate_trajectory(t, freq=0.5, amp=0.5):
    w = 2 * np.pi * freq
    q_d = np.array([amp * np.sin(w * t), amp * np.sin(w * t + np.pi/4)])
    q_dot_d = np.array([amp * w * np.cos(w * t), amp * w * np.cos(w * t + np.pi/4)])
    q_ddot_d = np.array([-amp * w**2 * np.sin(w * t), -amp * w**2 * np.sin(w * t + np.pi/4)])
    return q_d, q_dot_d, q_ddot_d


def compute_reward(e, e_dot, tau):
    """Same reward function as CTC+SAC for fair comparison"""
    r = -10.0 * np.sum(e**2) - 1.0 * np.sum(e_dot**2) - 0.01 * np.sum(tau**2)
    if np.linalg.norm(e) < 0.1:
        r += 1.0
    return r


# ==========================================
# MAIN TRAINING
# ==========================================
def main():
    print("="*60)
    print("RL-Only (SAC without CTC) - Mico Robot")
    print("="*60)
    print("\n⚠️  NO CTC - SAC must learn FULL control from scratch!")
    print("    Action space: ±20 N·m (vs ±5 N·m for CTC+SAC)")
    
    # Connect
    print("\nConnecting to CoppeliaSim...")
    client = RemoteAPIClient(host='host.docker.internal')
    sim = client.require('sim')
    print("✅ Connected!")
    
    # Joints (Mico robot)
    j1 = sim.getObject('/Mico/joint')
    j2 = sim.getObject('/Mico/joint/link/joint')
    print(f"✅ Joints: j1={j1}, j2={j2}")
    
    # Torque mode
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    print("✅ Torque mode enabled!")
    
    # SAC Agent (NO CTC!)
    # Action scale = 20 N·m (full torque range, unlike 5 N·m for residual)
    sac = SAC(state_dim=8, action_dim=2, action_scale=20.0)
    print("✅ SAC Agent ready (NO CTC)!")
    
    # Training params
    NUM_EPISODES = 200
    STEPS_PER_EP = 200
    DT = 0.02
    
    os.makedirs('results', exist_ok=True)
    
    episode_rewards = []
    episode_errors = []
    best_reward = -np.inf
    
    print(f"\n🚀 Starting RL-Only training: {NUM_EPISODES} episodes")
    print("-"*60)
    
    for ep in range(NUM_EPISODES):
        sim.startSimulation()
        time.sleep(0.1)
        
        # Reset
        sim.setJointPosition(j1, 0.0)
        sim.setJointPosition(j2, 0.0)
        sim.setJointTargetVelocity(j1, 0.0)
        sim.setJointTargetVelocity(j2, 0.0)
        time.sleep(0.05)
        
        ep_reward = 0
        errors = []
        t = 0.0
        
        for step in range(STEPS_PER_EP):
            # State
            q = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot = np.array([sim.getJointVelocity(j1), sim.getJointVelocity(j2)])
            
            # Trajectory
            q_d, q_dot_d, q_ddot_d = generate_trajectory(t)
            e = q_d - q
            e_dot = q_dot_d - q_dot
            
            # State vector (same as CTC+SAC)
            obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
            
            # RL-Only: NO CTC, just SAC action
            tau = sac.select_action(obs)
            tau = np.clip(tau, -20.0, 20.0)  # Full torque range
            
            # Apply
            sim.setJointTargetForce(j1, float(tau[0]))
            sim.setJointTargetForce(j2, float(tau[1]))
            
            time.sleep(DT)
            t += DT
            
            # Next state
            q_next = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot_next = np.array([sim.getJointVelocity(j1), sim.getJointVelocity(j2)])
            q_d_next, q_dot_d_next, _ = generate_trajectory(t)
            e_next = q_d_next - q_next
            e_dot_next = q_dot_d_next - q_dot_next
            next_obs = np.concatenate([q_next, q_dot_next, e_next, e_dot_next]).astype(np.float32)
            
            # Reward & store
            reward = compute_reward(e, e_dot, tau)
            done = (step == STEPS_PER_EP - 1)
            sac.buffer.add(obs, tau, reward, next_obs, float(done))
            sac.update()
            
            ep_reward += reward
            errors.append(np.linalg.norm(e))
        
        sim.stopSimulation()
        time.sleep(0.1)
        
        mean_error = np.mean(errors)
        episode_rewards.append(ep_reward)
        episode_errors.append(mean_error)
        
        if ep_reward > best_reward:
            best_reward = ep_reward
            sac.save('results/best_rl_only.pt')
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:4d}/{NUM_EPISODES} | Reward: {np.mean(episode_rewards[-10:]):10.1f} | Error: {np.mean(episode_errors[-10:]):.4f} rad")
        
        if (ep + 1) % 50 == 0:
            sac.save(f'results/rl_only_checkpoint_ep{ep+1}.pt')
            print(f"  💾 Checkpoint saved")
    
    # Save final
    sac.save('results/final_rl_only.pt')
    
    # Save training data
    with open('results/rl_only_training_data.json', 'w') as f:
        json.dump({
            'rewards': episode_rewards, 
            'errors': episode_errors,
            'method': 'RL-Only (SAC without CTC)',
            'action_scale': 20.0
        }, f, indent=2)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('RL-Only (SAC without CTC) Training', fontsize=14, fontweight='bold')
    
    def smooth(x, w=10):
        return np.convolve(x, np.ones(w)/w, 'valid') if len(x) >= w else x
    
    axes[0].plot(smooth(episode_rewards), 'b-', lw=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Reward')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(smooth(episode_errors), 'r-', lw=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Mean Error [rad]')
    axes[1].set_title('Tracking Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/rl_only_training_curves.png', dpi=150)
    
    print("\n" + "="*60)
    print("🎉 RL-Only Training Complete!")
    print("="*60)
    print(f"  Best Reward: {best_reward:.1f}")
    print(f"  Final Error: {np.mean(episode_errors[-10:]):.4f} rad")
    print(f"  Saved: results/best_rl_only.pt")
    print(f"  Saved: results/rl_only_training_curves.png")


if __name__ == "__main__":
    main()
