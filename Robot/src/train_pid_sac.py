#!/usr/bin/env python3
"""
Train PID+SAC Agent in CoppeliaSim
=====================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Baseline comparison: PID as base controller + SAC residual learning.
This directly addresses Reviewer 8's request for a PID+RL comparison.

Key difference from CTC+SAC:
  - PID provides ~50-60% of required torque (no dynamics knowledge)
  - SAC must learn a larger residual → action range ±10 N·m (vs ±5 for CTC+SAC)

Usage (inside Docker):
    cd /workspace/src
    python train_pid_sac.py

Saves best model to: ../models/best_pid_sac.pt

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import numpy as np
import json, time, os
import torch

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from pid_controller import PIDController, PIDGains
from sac_agent import SACAgent, SACConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
COPPELIASIM_HOST = os.environ.get("COPPELIASIM_HOST", "host.docker.internal")
NUM_EPISODES     = 200
STEPS_PER_EP     = 200
DT               = 0.02        # 20 ms / step → 4 s per episode
SAVE_EVERY       = 50
MODELS_DIR       = "../models"
RESULTS_DIR      = "../results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── TRAJECTORY ────────────────────────────────────────────────────────────────
def trajectory(t, freq=0.5, amp=0.5):
    w        = 2 * np.pi * freq
    q_d      = np.array([amp * np.sin(w * t),         amp * np.sin(w * t + np.pi / 4)])
    q_dot_d  = np.array([amp * w * np.cos(w * t),     amp * w * np.cos(w * t + np.pi / 4)])
    q_ddot_d = np.array([-amp * w**2 * np.sin(w * t), -amp * w**2 * np.sin(w * t + np.pi / 4)])
    return q_d, q_dot_d, q_ddot_d

# ── REWARD ────────────────────────────────────────────────────────────────────
def compute_reward(e, e_dot, tau_rl, alpha=1.0, beta=0.01, gamma=0.001):
    r = (-alpha * np.linalg.norm(e)**2
         - beta  * np.linalg.norm(e_dot)**2
         - gamma * np.linalg.norm(tau_rl)**2)
    if np.linalg.norm(e) < 0.1:
        r += 1.0
    return float(r)

# ── SAFE RESET ────────────────────────────────────────────────────────────────
def setup_joints(sim):
    mico       = sim.getObject('/Mico')
    all_joints = sim.getObjectsInTree(mico, sim.object_joint_type, 0)
    for jh in all_joints:
        sim.setObjectInt32Param(jh, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setJointTargetPosition(jh, 0.0)
    j1 = sim.getObject('/Mico/joint/link/joint')             # shoulder
    j2 = sim.getObject('/Mico/joint/link/joint/link/joint')  # elbow
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    return j1, j2, all_joints

def reset_episode(sim, j1, j2, all_joints):
    sim.stopSimulation();  time.sleep(0.3)
    sim.startSimulation(); time.sleep(0.2)
    for jh in all_joints:
        sim.setObjectInt32Param(jh, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
        sim.setJointTargetPosition(jh, 0.0)
    time.sleep(0.4)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    time.sleep(0.1)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PID+SAC Training — CoDIT 2026")
    print("  (Baseline for Reviewer 8 comparison)")
    print("=" * 60)

    print(f"\nConnecting to CoppeliaSim at {COPPELIASIM_HOST}...")
    client = RemoteAPIClient(host=COPPELIASIM_HOST)
    sim    = client.require('sim')
    print("✅ Connected!")

    j1, j2, all_joints = setup_joints(sim)
    print("✅ All 6 joints found — joints 1,4,5,6 locked; shoulder+elbow in torque mode")

    # Controllers
    # PID gains match the PD terms inside CTC for a fair comparison
    gains = PIDGains(
        Kp=np.array([100.0, 50.0]),
        Ki=np.array([5.0,   2.0]),
        Kd=np.array([20.0,  10.0]),
    )
    pid = PIDController(gains=gains, dt=DT, tau_max=15.0)

    # SAC needs larger action range: PID supplies ~50% torque, RL must cover more
    config = SACConfig(action_low=-10.0, action_high=10.0)
    agent  = SACAgent(state_dim=8, action_dim=2, config=config)
    print("✅ PID + SAC initialised")
    print(f"   PID gains: Kp={gains.Kp}, Ki={gains.Ki}, Kd={gains.Kd}")
    print(f"   SAC action range: ±10 N·m  (vs ±5 for CTC+SAC)")

    print(f"\n{'='*60}")
    print(f"  Starting: {NUM_EPISODES} episodes × {STEPS_PER_EP} steps")
    print(f"{'='*60}\n")

    rewards, errors = [], []
    best_reward = -np.inf

    for ep in range(NUM_EPISODES):
        pid.reset()                                   # clear PID integral windup
        reset_episode(sim, j1, j2, all_joints)        # reset robot to q=[0,0]

        ep_reward, ep_errors = 0.0, []
        t = 0.0

        for step in range(STEPS_PER_EP):
            q     = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot = np.array([sim.getJointVelocity(j1),  sim.getJointVelocity(j2)])

            q_d, q_dot_d, _ = trajectory(t)
            e     = q_d - q
            e_dot = q_dot_d - q_dot
            obs   = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)

            # PID base torque
            tau_pid = pid.compute_torque(e, e_dot)

            # SAC residual
            tau_rl  = agent.select_action(obs, deterministic=False)
            tau_rl  = np.clip(tau_rl, -10.0, 10.0)

            # Total torque (clipped to safe range)
            tau = np.clip(tau_pid + tau_rl, -20.0, 20.0)

            sim.setJointTargetForce(j1, float(tau[0]))
            sim.setJointTargetForce(j2, float(tau[1]))
            time.sleep(DT)
            t += DT

            # Next state
            q_n     = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot_n = np.array([sim.getJointVelocity(j1),  sim.getJointVelocity(j2)])
            q_d_n, q_dot_d_n, _ddot_n = trajectory(t)
            obs_n   = np.concatenate([q_n, q_dot_n, q_d_n - q_n, q_dot_d_n - q_dot_n]).astype(np.float32)

            reward = compute_reward(e, e_dot, tau_rl)
            done   = (step == STEPS_PER_EP - 1)
            agent.store_transition(obs, tau_rl, reward, obs_n, done)
            agent.update()

            ep_reward += reward
            ep_errors.append(np.linalg.norm(e))

        rewards.append(ep_reward)
        errors.append(np.mean(ep_errors))

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(f"{MODELS_DIR}/best_pid_sac.pt")

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1:3d}/{NUM_EPISODES} | "
                  f"Reward: {np.mean(rewards[-10:]):8.1f} | "
                  f"Error:  {np.mean(errors[-10:]):.4f} rad | "
                  f"Steps:  {agent.total_steps}")

        if (ep + 1) % SAVE_EVERY == 0:
            agent.save(f"{MODELS_DIR}/pid_sac_checkpoint_ep{ep+1}.pt")

    agent.save(f"{MODELS_DIR}/final_pid_sac.pt")
    json.dump({'rewards': rewards, 'errors': errors},
              open(f"{RESULTS_DIR}/training_pid_sac.json", 'w'), indent=2)
    print("\n✅ PID+SAC Training complete!")
    print(f"   Best model → {MODELS_DIR}/best_pid_sac.pt")
    print(f"   Final mean error (last 10 ep): {np.mean(errors[-10:]):.4f} rad")

if __name__ == "__main__":
    main()
