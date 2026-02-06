#!/usr/bin/env python3
"""
CoppeliaSim Experiment Runner for CTC+SAC
==========================================
CoDIT 2026 Paper Implementation

This script runs experiments in CoppeliaSim comparing:
1. CTC-Only (baseline)
2. CTC+SAC (proposed)
3. RL-Only (baseline)
4. PID+RL (baseline)

Can run WITHOUT ROS2 (direct ZMQ connection)

Author: Mohammed
Date: January 2026
"""

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from robot_dynamics import RobotDynamics, RobotParameters, create_nominal_robot
from ctc_controller import CTCController, SimplePDController, CTCGains

# CoppeliaSim ZMQ API
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    COPPELIASIM_AVAILABLE = True
except ImportError:
    print("⚠️ CoppeliaSim API not available. Install with:")
    print("   pip install coppeliasim-zmqremoteapi-client")
    COPPELIASIM_AVAILABLE = False


class ExperimentRunner:
    """
    Run experiments in CoppeliaSim for paper results.
    """
    
    def __init__(self, experiment_name: str = "experiment"):
        """Initialize experiment runner."""
        self.experiment_name = experiment_name
        self.results_dir = f"../results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Robot dynamics
        self.robot = create_nominal_robot()
        
        # Controllers
        gains = CTCGains(
            Kp=np.array([100.0, 50.0]),
            Kd=np.array([20.0, 10.0])
        )
        self.ctc = CTCController(self.robot, gains, include_friction=True)
        self.pid = SimplePDController(self.robot, gains)
        
        # CoppeliaSim
        self.client = None
        self.sim = None
        self.joint_handles = []
        
        # Data recording
        self.reset_data()
    
    def reset_data(self):
        """Reset data buffers."""
        self.time_data = []
        self.q_desired_data = []
        self.q_actual_data = []
        self.q_dot_data = []
        self.tau_ctc_data = []
        self.tau_rl_data = []
        self.tau_total_data = []
        self.error_data = []
    
    def connect_coppeliasim(self) -> bool:
        """Connect to CoppeliaSim."""
        if not COPPELIASIM_AVAILABLE:
            return False
        
        print("Connecting to CoppeliaSim...")
        try:
            self.client = RemoteAPIClient(host='host.docker.internal')
            self.sim = self.client.require('sim')
            print("✅ Connected to CoppeliaSim!")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def setup_robot(self) -> bool:
        """Setup robot joint handles."""
        print("Setting up robot...")
        
        joint_paths = [
            '/Mico/joint',
            '/Mico/joint/link/joint'
        ]
        
        self.joint_handles = []
        
        for i, path in enumerate(joint_paths):
            try:
                handle = self.sim.getObject(path)
                self.joint_handles.append(handle)
                print(f"  Found Joint {i+1}: {path}")
            except:
                print(f"  ❌ Could not find: {path}")
        
        if len(self.joint_handles) >= 2:
            print(f"✅ Found {len(self.joint_handles)} joints!")
            return True
        else:
            print("❌ Could not find enough joints!")
            return False
    
    def set_torque_mode(self):
        """Enable torque control mode."""
        print("Enabling torque mode...")
        for i, handle in enumerate(self.joint_handles):
            self.sim.setObjectInt32Param(
                handle,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_force
            )
            print(f"  Joint {i+1}: Torque mode enabled")
        print("✅ Torque mode enabled!")
    
    def get_joint_states(self):
        """Get current joint states."""
        q = np.zeros(2)
        q_dot = np.zeros(2)
        
        for i, handle in enumerate(self.joint_handles):
            q[i] = self.sim.getJointPosition(handle)
            q_dot[i] = self.sim.getJointVelocity(handle)
        
        return q, q_dot
    
    def set_joint_torques(self, tau):
        """Set joint torques."""
        for i, handle in enumerate(self.joint_handles):
            self.sim.setJointTargetForce(handle, float(tau[i]))
    
    def generate_trajectory(self, t: float, traj_type: str = 'sinusoidal',
                           freq: float = 0.5, amp: float = 0.5):
        """Generate reference trajectory."""
        w = 2 * np.pi * freq
        
        if traj_type == 'sinusoidal':
            q_d = np.array([
                amp * np.sin(w * t),
                amp * np.sin(w * t + np.pi/4)
            ])
            q_dot_d = np.array([
                amp * w * np.cos(w * t),
                amp * w * np.cos(w * t + np.pi/4)
            ])
            q_ddot_d = np.array([
                -amp * w**2 * np.sin(w * t),
                -amp * w**2 * np.sin(w * t + np.pi/4)
            ])
        elif traj_type == 'circular':
            q_d = np.array([
                amp * np.cos(w * t),
                amp * np.sin(w * t)
            ])
            q_dot_d = np.array([
                -amp * w * np.sin(w * t),
                amp * w * np.cos(w * t)
            ])
            q_ddot_d = np.array([
                -amp * w**2 * np.cos(w * t),
                -amp * w**2 * np.sin(w * t)
            ])
        else:  # point_to_point
            T = 1.0 / freq
            t_mod = t % T
            if t_mod < T/2:
                q_d = np.array([amp, amp/2])
            else:
                q_d = np.array([-amp, -amp/2])
            q_dot_d = np.zeros(2)
            q_ddot_d = np.zeros(2)
        
        return q_d, q_dot_d, q_ddot_d
    
    def run_experiment(self, method: str = 'ctc_only',
                       duration: float = 10.0,
                       traj_type: str = 'sinusoidal',
                       sac_agent=None):
        """
        Run single experiment.
        
        Args:
            method: 'ctc_only', 'ctc_sac', 'rl_only', 'pid_rl'
            duration: Experiment duration [s]
            traj_type: Trajectory type
            sac_agent: Trained SAC agent (for RL methods)
        """
        print(f"\n{'='*50}")
        print(f"Running Experiment: {method.upper()}")
        print(f"Trajectory: {traj_type}, Duration: {duration}s")
        print(f"{'='*50}")
        
        self.reset_data()
        
        # Start simulation
        self.sim.startSimulation()
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration:
                t = time.time() - start_time
                
                # Get current state
                q, q_dot = self.get_joint_states()
                
                # Generate trajectory
                q_d, q_dot_d, q_ddot_d = self.generate_trajectory(t, traj_type)
                
                # Compute errors
                e = q_d - q
                e_dot = q_dot_d - q_dot
                
                # ============================================
                # COMPUTE CONTROL TORQUE
                # ============================================
                
                tau_ctc = np.zeros(2)
                tau_rl = np.zeros(2)
                
                if method == 'ctc_only':
                    # CTC + PD feedback
                    tau_ctc = self.ctc.compute_torque(
                        q, q_dot, q_d, q_dot_d, q_ddot_d
                    )
                    tau_total = tau_ctc
                    
                elif method == 'ctc_sac':
                    # CTC feedforward + SAC residual
                    tau_ctc = self.ctc.compute_feedforward_only(
                        q, q_dot, q_ddot_d
                    )
                    if sac_agent is not None:
                        obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
                        tau_rl = sac_agent.select_action(obs, deterministic=True)
                        tau_rl = np.clip(tau_rl, -5.0, 5.0)
                    tau_total = tau_ctc + tau_rl
                    
                elif method == 'rl_only':
                    # RL only (no CTC)
                    if sac_agent is not None:
                        obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
                        tau_rl = sac_agent.select_action(obs, deterministic=True)
                        tau_rl = np.clip(tau_rl, -20.0, 20.0)
                    tau_total = tau_rl
                    
                elif method == 'pid_rl':
                    # PID + SAC residual
                    tau_pid = self.pid.compute_torque(
                        q, q_dot, q_d, q_dot_d
                    )
                    tau_ctc = tau_pid  # Store as "ctc" for consistency
                    if sac_agent is not None:
                        obs = np.concatenate([q, q_dot, e, e_dot]).astype(np.float32)
                        tau_rl = sac_agent.select_action(obs, deterministic=True)
                        tau_rl = np.clip(tau_rl, -5.0, 5.0)
                    tau_total = tau_pid + tau_rl
                
                # Clip and apply
                tau_total = np.clip(tau_total, -20.0, 20.0)
                self.set_joint_torques(tau_total)
                
                # Record data
                self.time_data.append(t)
                self.q_desired_data.append(q_d.copy())
                self.q_actual_data.append(q.copy())
                self.q_dot_data.append(q_dot.copy())
                self.tau_ctc_data.append(tau_ctc.copy())
                self.tau_rl_data.append(tau_rl.copy())
                self.tau_total_data.append(tau_total.copy())
                self.error_data.append(e.copy())
                
                # Log
                if int(t * 10) % 10 == 0:
                    print(f"  t={t:.1f}s | e=[{e[0]:.3f}, {e[1]:.3f}] | "
                          f"τ_total=[{tau_total[0]:.2f}, {tau_total[1]:.2f}]")
                
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            self.sim.stopSimulation()
            print("✅ Experiment completed!")
        
        return self.compute_metrics()
    
    def compute_metrics(self):
        """Compute experiment metrics."""
        errors = np.array(self.error_data)
        
        metrics = {
            'rms_error_q1': np.sqrt(np.mean(errors[:, 0]**2)),
            'rms_error_q2': np.sqrt(np.mean(errors[:, 1]**2)),
            'max_error_q1': np.max(np.abs(errors[:, 0])),
            'max_error_q2': np.max(np.abs(errors[:, 1])),
            'mean_error': np.mean(np.linalg.norm(errors, axis=1))
        }
        
        print(f"\nMetrics:")
        print(f"  RMS Error q1: {metrics['rms_error_q1']:.4f} rad")
        print(f"  RMS Error q2: {metrics['rms_error_q2']:.4f} rad")
        print(f"  Mean Error: {metrics['mean_error']:.4f} rad")
        
        return metrics
    
    def plot_results(self, method: str):
        """Plot experiment results."""
        time_arr = np.array(self.time_data)
        q_desired = np.array(self.q_desired_data)
        q_actual = np.array(self.q_actual_data)
        tau_ctc = np.array(self.tau_ctc_data)
        tau_rl = np.array(self.tau_rl_data)
        error = np.array(self.error_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{method.upper()} - Experiment Results', fontsize=14)
        
        # Plot 1: Trajectory tracking
        ax1 = axes[0, 0]
        ax1.plot(time_arr, q_desired[:, 0], 'b--', label='q1_desired', linewidth=2)
        ax1.plot(time_arr, q_actual[:, 0], 'b-', label='q1_actual', linewidth=1)
        ax1.plot(time_arr, q_desired[:, 1], 'r--', label='q2_desired', linewidth=2)
        ax1.plot(time_arr, q_actual[:, 1], 'r-', label='q2_actual', linewidth=1)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('Joint Angle [rad]')
        ax1.set_title('Trajectory Tracking')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Torque decomposition
        ax2 = axes[0, 1]
        ax2.plot(time_arr, tau_ctc[:, 0], 'b-', label='τ_CTC_1', linewidth=2)
        ax2.plot(time_arr, tau_rl[:, 0], 'b--', label='τ_RL_1', linewidth=1)
        ax2.plot(time_arr, tau_ctc[:, 1], 'r-', label='τ_CTC_2', linewidth=2)
        ax2.plot(time_arr, tau_rl[:, 1], 'r--', label='τ_RL_2', linewidth=1)
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Torque [N·m]')
        ax2.set_title('Torque Decomposition')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Tracking error
        ax3 = axes[1, 0]
        ax3.plot(time_arr, error[:, 0], 'b-', label='Error_q1', linewidth=2)
        ax3.plot(time_arr, error[:, 1], 'r-', label='Error_q2', linewidth=2)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('Error [rad]')
        ax3.set_title('Tracking Error')
        ax3.legend()
        ax3.grid(True)
        
        # Plot 4: Torque ratio
        ax4 = axes[1, 1]
        tau_total = tau_ctc + tau_rl
        ctc_ratio = np.abs(tau_ctc) / (np.abs(tau_total) + 1e-6) * 100
        ax4.plot(time_arr, ctc_ratio[:, 0], 'b-', label='CTC ratio q1', linewidth=2)
        ax4.plot(time_arr, ctc_ratio[:, 1], 'r-', label='CTC ratio q2', linewidth=2)
        ax4.axhline(y=90, color='g', linestyle='--', label='90% target')
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('CTC Contribution [%]')
        ax4.set_title('CTC Torque Contribution')
        ax4.legend()
        ax4.grid(True)
        ax4.set_ylim([0, 110])
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/{method}_results.png', dpi=150)
        print(f"✅ Saved: {self.results_dir}/{method}_results.png")
        plt.show()
    
    def save_data(self, method: str):
        """Save experiment data."""
        data = {
            'time': self.time_data,
            'q_desired': [q.tolist() for q in self.q_desired_data],
            'q_actual': [q.tolist() for q in self.q_actual_data],
            'tau_ctc': [t.tolist() for t in self.tau_ctc_data],
            'tau_rl': [t.tolist() for t in self.tau_rl_data],
            'error': [e.tolist() for e in self.error_data]
        }
        
        filename = f'{self.results_dir}/{method}_data.json'
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"✅ Saved: {filename}")


def run_all_experiments():
    """Run all experiments for paper comparison."""
    runner = ExperimentRunner("paper_experiments")
    
    if not runner.connect_coppeliasim():
        print("❌ Could not connect to CoppeliaSim!")
        print("Make sure CoppeliaSim is running with the Mico robot loaded.")
        return
    
    if not runner.setup_robot():
        return
    
    runner.set_torque_mode()
    
    # Run experiments
    results = {}
    
    # 1. CTC-Only (baseline)
    input("\nPress Enter to run CTC-Only experiment...")
    metrics = runner.run_experiment('ctc_only', duration=10.0, traj_type='sinusoidal')
    results['ctc_only'] = metrics
    runner.plot_results('ctc_only')
    runner.save_data('ctc_only')
    
    # 2. CTC+SAC (if agent available)
    # TODO: Load trained SAC agent
    # input("\nPress Enter to run CTC+SAC experiment...")
    # metrics = runner.run_experiment('ctc_sac', duration=10.0, sac_agent=agent)
    # results['ctc_sac'] = metrics
    
    # Save comparison
    with open(f'{runner.results_dir}/comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*50)
    print("All experiments completed!")
    print("="*50)


if __name__ == '__main__':
    run_all_experiments()
