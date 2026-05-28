#!/usr/bin/env python3
"""
Train CTC+SAC Agent in CoppeliaSim  (v2 — 2-DOF scene)
========================================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Changes from v1:
  - Joint paths updated for 2-joint scene (/Mico/joint, /Mico/joint/link/joint)
  - No wrist joints to lock — simpler setup_joints
  - SAC action range widened to ±10 N·m (was ±5) for faster convergence
  - NUM_EPISODES = 300 for better convergence

Usage (inside Docker):
    cd /workspace/src
    python train_ctc_sac_v2.py

Saves best model to: ../models/best_ctc_sac_v2.pt

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import numpy as np
import json, time, os
import torch

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_dynamics import create_nominal_robot
from ctc_controller import CTCController, CTCGains
from sac_agent import SACAgent, SACConfig

# ── CONFIG ────────────────────────────────────────────────────────
COPPELIASIM_HOST = os.environ.get("COPPELIASIM_HOST", "host.docker.internal")
NUM_EPISODES     = 300
STEPS_PER_EP     = 200
DT               = 0.02        # 20 ms / step → 4 s per episode
SAVE_EVERY       = 50
MODELS_DIR       = "../models"
RESULTS_DIR      = "../results"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── JOINT SETUP (2-DOF scene) ─────────────────────────────────────
def setup_joints(sim):
    """
    2-joint scene: shoulder (/Mico/joint) and elbow (/Mico/joint/link/joint).
    Both set to torque mode. No extra joints to lock.
    """
    j1 = sim.getObject('/Mico/joint')              # shoulder → handle 21
    j2 = sim.getObject('/Mico/joint/link/joint')   # elbow    → handle 23
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    return j1, j2

def reset_episode(sim, j1, j2):
    """Stop/start simulation, zero joints in position mode, then switch to torque."""
    sim.stopSimulation();  time.sleep(0.3)
    sim.startSimulation(); time.sleep(0.2)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_position)
    sim.setJointTargetPosition(j1, 0.0)
    sim.setJointTargetPosition(j2, 0.0)
    time.sleep(0.4)
    sim.setObjectInt32Param(j1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    sim.setObjectInt32Param(j2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
    time.sleep(0.1)

# ── TRAJECTORY ────────────────────────────────────────────────────
def trajectory(t, freq=0.5, amp=0.5):
    w = 2 * np.pi * freq
    q_d      = np.array([amp * np.sin(w * t),         amp * np.sin(w * t + np.pi / 4)])
    q_dot_d  = np.array([amp * w * np.cos(w * t),     amp * w * np.cos(w * t + np.pi / 4)])
    q_ddot_d = np.array([-amp * w**2 * np.sin(w * t), -amp * w**2 * np.sin(w * t + np.pi / 4)])
    return q_d, q_dot_d, q_ddot_d

# ── REWARD ────────────────────────────────────────────────────────
def compute_reward(e, e_dot, tau_rl, alpha=1.0, beta=0.01, gamma=0.001):
    r = (-alpha * np.linalg.norm(e)**2
         - beta  * np.linalg.norm(e_dot)**2
         - gamma * np.linalg.norm(tau_rl)**2)
    if np.linalg.norm(e) < 0.1:
        r += 1.0
    return float(r)

# ── MAIN ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CTC+SAC Training v2 — CoDIT 2026")
    print("  (2-DOF scene, action range ±10 N·m, 300 episodes)")
    print("=" * 60)

    print(f"\nConnecting to CoppeliaSim at {COPPELIASIM_HOST}...")
    client = RemoteAPIClient(host=COPPELIASIM_HOST)
    sim    = client.require('sim')
    print("✅ Connected!")

    j1, j2 = setup_joints(sim)
    print("✅ 2 joints found — shoulder (/Mico/joint) + elbow (/Mico/joint/link/joint)")

    robot  = create_nominal_robot()
    gains  = CTCGains(Kp=np.array([100.0, 50.0]), Kd=np.array([20.0, 10.0]))
    ctc    = CTCController(robot, gains, include_friction=True)
    config = SACConfig(action_low=-10.0, action_high=10.0)   # ±10 N·m (was ±5)
    agent  = SACAgent(state_dim=8, action_dim=2, config=config)
    print("✅ Controllers initialised")
    print(f"   CTC gains: Kp={gains.Kp}, Kd={gains.Kd}")
    print(f"   SAC action range: ±10 N·m")

    print(f"\n{'='*60}")
    print(f"  Starting: {NUM_EPISODES} episodes × {STEPS_PER_EP} steps")
    print(f"{'='*60}\n")

    rewards, errors = [], []
    best_reward = -np.inf

    for ep in range(NUM_EPISODES):
        reset_episode(sim, j1, j2)

        ep_reward, ep_errors = 0.0, []
        t = 0.0

        for step in range(STEPS_PER_EP):
            q     = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot = np.array([sim.getJointVelocity(j1),  sim.getJointVelocity(j2)])

            q_d, q_dot_d, q_ddot_d = trajectory(t)
            e     = q_d - q
            e_dot = q_dot_d - q_dot
            obs   = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)

            tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
            tau_rl  = agent.select_action(obs, deterministic=False)
            tau_rl  = np.clip(tau_rl, -10.0, 10.0)
            tau     = np.clip(tau_ctc + tau_rl, -20.0, 20.0)

            sim.setJointTargetForce(j1, float(tau[0]))
            sim.setJointTargetForce(j2, float(tau[1]))
            time.sleep(DT)
            t += DT

            q_n     = np.array([sim.getJointPosition(j1), sim.getJointPosition(j2)])
            q_dot_n = np.array([sim.getJointVelocity(j1),  sim.getJointVelocity(j2)])
            q_d_n, q_dot_d_n, _ = trajectory(t)
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
            agent.save(f"{MODELS_DIR}/best_ctc_sac_v2.pt")

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1:3d}/{NUM_EPISODES} | "
                  f"Reward: {np.mean(rewards[-10:]):8.1f} | "
                  f"Error:  {np.mean(errors[-10:]):.4f} rad | "
                  f"Steps:  {agent.total_steps}")

        if (ep + 1) % SAVE_EVERY == 0:
            agent.save(f"{MODELS_DIR}/ctc_sac_v2_checkpoint_ep{ep+1}.pt")

    agent.save(f"{MODELS_DIR}/final_ctc_sac_v2.pt")
    json.dump({'rewards': rewards, 'errors': errors},
              open(f"{RESULTS_DIR}/training_ctc_sac_v2.json", 'w'), indent=2)
    print("\n✅ CTC+SAC v2 Training complete!")
    print(f"   Best model → {MODELS_DIR}/best_ctc_sac_v2.pt")
    print(f"   Final mean error (last 10 ep): {np.mean(errors[-10:]):.4f} rad")

if __name__ == "__main__":
    main()
