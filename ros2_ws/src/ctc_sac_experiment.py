#!/usr/bin/env python3
"""
CTC+SAC Experiment for CoppeliaSim
CoDIT 2026 Paper
Saves results to results/ folder with CSV data
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
import csv
import torch

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from robot_dynamics import RobotDynamics, RobotParameters, create_nominal_robot
from ctc_controller import CTCController, CTCGains
from sac_agent import SACAgent, SACConfig

# ==========================================
# CREATE RESULTS DIRECTORY
# ==========================================
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
print(f"✅ Results will be saved to: {results_dir}/")

# ==========================================
# CONNECT TO COPPELIASIM
# ==========================================
print("Connecting to CoppeliaSim...")
client = RemoteAPIClient(host='host.docker.internal')
sim = client.require('sim')
print("✅ Connected!")

# ==========================================
# SETUP ROBOT
# ==========================================
print("Setting up robot...")
joint1 = sim.getObject('/Mico/joint')
joint2 = sim.getObject('/Mico/joint/link/joint')
print(f"✅ Found joints!")

# Enable torque mode
sim.setObjectInt32Param(joint1, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
sim.setObjectInt32Param(joint2, sim.jointintparam_dynctrlmode, sim.jointdynctrl_force)
print("✅ Torque mode enabled!")

# ==========================================
# LOAD CONTROLLERS
# ==========================================
robot = create_nominal_robot()
gains = CTCGains(Kp=np.array([100.0, 50.0]), Kd=np.array([20.0, 10.0]))
ctc = CTCController(robot, gains, include_friction=True)

# Load SAC agent
print("Loading SAC agent...")
sac_config = SACConfig(hidden_dims=(256, 256), action_low=-5.0, action_high=5.0)
sac_agent = SACAgent(state_dim=8, action_dim=2, config=sac_config, device='cpu')
sac_agent.load('results/best_ctc_sac_coppeliasim.pt')
print("✅ SAC agent loaded!")

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
# RUN EXPERIMENT
# ==========================================
print("\n" + "="*50)
print("Running CTC+SAC Experiment")
print("="*50)

# Data recording
time_data = []
q_desired_data = []
q_actual_data = []
q_dot_actual_data = []
tau_ctc_data = []
tau_rl_data = []
tau_total_data = []
error_data = []
error_dot_data = []

# Start simulation
sim.startSimulation()
start_time = time.time()
duration = 10.0

try:
    while (time.time() - start_time) < duration:
        t = time.time() - start_time
        
        # Get current state
        q = np.array([sim.getJointPosition(joint1), sim.getJointPosition(joint2)])
        q_dot = np.array([sim.getJointVelocity(joint1), sim.getJointVelocity(joint2)])
        
        # Generate trajectory
        q_d, q_dot_d, q_ddot_d = generate_trajectory(t)
        
        # Compute errors
        e = q_d - q
        e_dot = q_dot_d - q_dot
        
        # CTC with full feedback (not feedforward only)
        tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
        
        # SAC residual
        obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
        tau_rl = sac_agent.select_action(obs, deterministic=True)
        tau_rl = np.clip(tau_rl, -5.0, 5.0)
        
        # Total torque
        tau_total = np.clip(tau_ctc + tau_rl, -20.0, 20.0)
        
        # Apply torque
        sim.setJointTargetForce(joint1, float(tau_total[0]))
        sim.setJointTargetForce(joint2, float(tau_total[1]))
        
        # Record data
        time_data.append(t)
        q_desired_data.append(q_d.copy())
        q_actual_data.append(q.copy())
        q_dot_actual_data.append(q_dot.copy())
        tau_ctc_data.append(tau_ctc.copy())
        tau_rl_data.append(tau_rl.copy())
        tau_total_data.append(tau_total.copy())
        error_data.append(e.copy())
        error_dot_data.append(e_dot.copy())
        
        # Log every second
        if int(t * 10) % 10 == 0:
            print(f"  t={t:.1f}s | e=[{e[0]:.3f}, {e[1]:.3f}] | τ_RL=[{tau_rl[0]:.2f}, {tau_rl[1]:.2f}]")
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n⚠️ Interrupted")
finally:
    sim.stopSimulation()

# ==========================================
# CONVERT TO NUMPY ARRAYS
# ==========================================
time_arr = np.array(time_data)
q_desired = np.array(q_desired_data)
q_actual = np.array(q_actual_data)
q_dot_actual = np.array(q_dot_actual_data)
tau_ctc = np.array(tau_ctc_data)
tau_rl = np.array(tau_rl_data)
tau_total = np.array(tau_total_data)
errors = np.array(error_data)
errors_dot = np.array(error_dot_data)

# ==========================================
# COMPUTE METRICS
# ==========================================
metrics = {
    'rms_error_q1': float(np.sqrt(np.mean(errors[:, 0]**2))),
    'rms_error_q2': float(np.sqrt(np.mean(errors[:, 1]**2))),
    'max_error_q1': float(np.max(np.abs(errors[:, 0]))),
    'max_error_q2': float(np.max(np.abs(errors[:, 1]))),
    'mean_error': float(np.mean(np.linalg.norm(errors, axis=1))),
    'mean_tau_rl_1': float(np.mean(np.abs(tau_rl[:, 0]))),
    'mean_tau_rl_2': float(np.mean(np.abs(tau_rl[:, 1]))),
    'mean_tau_ctc_1': float(np.mean(np.abs(tau_ctc[:, 0]))),
    'mean_tau_ctc_2': float(np.mean(np.abs(tau_ctc[:, 1]))),
    'ctc_contribution_q1': float(np.mean(np.abs(tau_ctc[:, 0]) / (np.abs(tau_total[:, 0]) + 1e-6)) * 100),
    'ctc_contribution_q2': float(np.mean(np.abs(tau_ctc[:, 1]) / (np.abs(tau_total[:, 1]) + 1e-6)) * 100)
}

print("\n" + "="*50)
print("CTC+SAC Results:")
print("="*50)
print(f"  RMS Error q1:      {metrics['rms_error_q1']:.4f} rad")
print(f"  RMS Error q2:      {metrics['rms_error_q2']:.4f} rad")
print(f"  Max Error q1:      {metrics['max_error_q1']:.4f} rad")
print(f"  Max Error q2:      {metrics['max_error_q2']:.4f} rad")
print(f"  Mean Error:        {metrics['mean_error']:.4f} rad")
print(f"  CTC Contribution:  q1={metrics['ctc_contribution_q1']:.1f}%, q2={metrics['ctc_contribution_q2']:.1f}%")

# ==========================================
# SAVE JSON METRICS
# ==========================================
json_path = os.path.join(results_dir, 'ctc_sac_metrics.json')
with open(json_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Saved: {json_path}")

# ==========================================
# SAVE CSV DATA
# ==========================================
csv_path = os.path.join(results_dir, 'ctc_sac_data.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    # Header
    writer.writerow([
        'time', 
        'q1_desired', 'q2_desired', 
        'q1_actual', 'q2_actual',
        'q1_dot', 'q2_dot',
        'error_q1', 'error_q2',
        'error_dot_q1', 'error_dot_q2',
        'tau_ctc_1', 'tau_ctc_2',
        'tau_rl_1', 'tau_rl_2',
        'tau_total_1', 'tau_total_2'
    ])
    # Data
    for i in range(len(time_arr)):
        writer.writerow([
            time_arr[i],
            q_desired[i, 0], q_desired[i, 1],
            q_actual[i, 0], q_actual[i, 1],
            q_dot_actual[i, 0], q_dot_actual[i, 1],
            errors[i, 0], errors[i, 1],
            errors_dot[i, 0], errors_dot[i, 1],
            tau_ctc[i, 0], tau_ctc[i, 1],
            tau_rl[i, 0], tau_rl[i, 1],
            tau_total[i, 0], tau_total[i, 1]
        ])
print(f"✅ Saved: {csv_path}")

# ==========================================
# PLOT RESULTS
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CTC+SAC - Experiment Results', fontsize=14, fontweight='bold')

# 1. Trajectory tracking
ax1 = axes[0, 0]
ax1.plot(time_arr, q_desired[:, 0], 'b--', label='q1_desired', linewidth=2)
ax1.plot(time_arr, q_actual[:, 0], 'b-', label='q1_actual', linewidth=1)
ax1.plot(time_arr, q_desired[:, 1], 'r--', label='q2_desired', linewidth=2)
ax1.plot(time_arr, q_actual[:, 1], 'r-', label='q2_actual', linewidth=1)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Joint Angle [rad]')
ax1.set_title('Trajectory Tracking')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Torque decomposition
ax2 = axes[0, 1]
ax2.plot(time_arr, tau_ctc[:, 0], 'b-', label='τ_CTC_1', linewidth=2)
ax2.plot(time_arr, tau_rl[:, 0], 'b--', label='τ_RL_1', linewidth=1)
ax2.plot(time_arr, tau_ctc[:, 1], 'r-', label='τ_CTC_2', linewidth=2)
ax2.plot(time_arr, tau_rl[:, 1], 'r--', label='τ_RL_2', linewidth=1)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Torque [N·m]')
ax2.set_title('Torque Decomposition')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Tracking error
ax3 = axes[1, 0]
ax3.plot(time_arr, errors[:, 0], 'b-', label='Error_q1', linewidth=2)
ax3.plot(time_arr, errors[:, 1], 'r-', label='Error_q2', linewidth=2)
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Error [rad]')
ax3.set_title('Tracking Error')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. CTC contribution
ax4 = axes[1, 1]
ctc_ratio = np.abs(tau_ctc) / (np.abs(tau_total) + 1e-6) * 100
ax4.plot(time_arr, ctc_ratio[:, 0], 'b-', label='CTC ratio q1', linewidth=2)
ax4.plot(time_arr, ctc_ratio[:, 1], 'r-', label='CTC ratio q2', linewidth=2)
ax4.axhline(y=90, color='g', linestyle='--', label='90% target')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('CTC Contribution [%]')
ax4.set_title('CTC Torque Contribution')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 110])

plt.tight_layout()
png_path = os.path.join(results_dir, 'ctc_sac_results.png')
plt.savefig(png_path, dpi=150)
print(f"✅ Saved: {png_path}")

# ==========================================
# SAVE COMPARISON JSON (for comparing with CTC-only)
# ==========================================
comparison_path = os.path.join(results_dir, 'comparison.json')

# Load existing comparison if exists
if os.path.exists(comparison_path):
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)
else:
    comparison = {}

# Add CTC+SAC results
comparison['ctc_sac'] = {
    'rms_error_q1': metrics['rms_error_q1'],
    'rms_error_q2': metrics['rms_error_q2'],
    'max_error_q1': metrics['max_error_q1'],
    'max_error_q2': metrics['max_error_q2'],
    'mean_error': metrics['mean_error']
}

with open(comparison_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"✅ Saved: {comparison_path}")

print("\n" + "="*50)
print("🎉 CTC+SAC Experiment Complete!")
print("="*50)
print(f"\nFiles saved in '{results_dir}/':")
print(f"  📊 ctc_sac_metrics.json  - Performance metrics")
print(f"  📈 ctc_sac_data.csv      - Time series data")
print(f"  🖼️  ctc_sac_results.png   - Plots")
print(f"  📋 comparison.json       - Comparison data")
