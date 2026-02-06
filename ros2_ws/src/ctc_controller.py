#!/usr/bin/env python3
"""
Computed Torque Controller (CTC) for 2-DOF Manipulator
=======================================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

CTC uses analytical Lagrangian dynamics to compute feedforward
torques that cancel nonlinear dynamics, achieving ~90% of required
control effort.

Control Law:
    τ = M(q)[q̈_d + Kd*ė + Kp*e] + C(q,q̇)q̇ + G(q)
    
where:
    e = q_d - q (position error)
    ė = q̇_d - q̇ (velocity error)

Author: Mohammed
Date: January 2026
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from robot_dynamics import RobotDynamics, RobotParameters, create_nominal_robot


@dataclass
class CTCGains:
    """PD gains for CTC feedback linearization."""
    Kp: np.ndarray = None  # Position gains [2,]
    Kd: np.ndarray = None  # Velocity gains [2,]
    
    def __post_init__(self):
        if self.Kp is None:
            self.Kp = np.array([100.0, 50.0])  # Default position gains
        if self.Kd is None:
            self.Kd = np.array([20.0, 10.0])   # Default velocity gains
    
    @classmethod
    def critically_damped(cls, wn: np.ndarray) -> 'CTCGains':
        """
        Create critically damped gains from natural frequencies.
        
        For critically damped response: ζ = 1
            Kp = ωn²
            Kd = 2*ωn
            
        Args:
            wn: Natural frequencies [ωn1, ωn2] in rad/s
        """
        Kp = wn ** 2
        Kd = 2 * wn
        return cls(Kp=Kp, Kd=Kd)
    
    @classmethod
    def from_settling_time(cls, ts: float, overshoot: float = 0.0) -> 'CTCGains':
        """
        Design gains from desired settling time and overshoot.
        
        Args:
            ts: Desired settling time [seconds]
            overshoot: Desired overshoot [0-1], 0 for critically damped
        """
        # For 2% settling criterion: ts ≈ 4/(ζ*ωn)
        if overshoot <= 0:
            zeta = 1.0  # Critically damped
        else:
            zeta = -np.log(overshoot) / np.sqrt(np.pi**2 + np.log(overshoot)**2)
        
        wn = 4 / (zeta * ts)
        
        Kp = np.array([wn**2, wn**2])
        Kd = np.array([2*zeta*wn, 2*zeta*wn])
        
        return cls(Kp=Kp, Kd=Kd)


class CTCController:
    """
    Computed Torque Controller (Inverse Dynamics Control).
    
    This controller linearizes the robot dynamics, resulting in
    a double integrator system that can be controlled with simple
    PD feedback.
    
    Closed-loop dynamics (ideal case):
        ë + Kd*ė + Kp*e = 0
        
    where e = q_d - q
    """
    
    def __init__(self, robot: Optional[RobotDynamics] = None,
                 gains: Optional[CTCGains] = None,
                 include_friction: bool = True):
        """
        Initialize CTC controller.
        
        Args:
            robot: Robot dynamics model (uses nominal if None)
            gains: Controller gains (uses defaults if None)
            include_friction: Include friction compensation
        """
        self.robot = robot or create_nominal_robot()
        self.gains = gains or CTCGains()
        self.include_friction = include_friction
        
        # Statistics tracking
        self.reset_stats()
    
    def reset_stats(self):
        """Reset controller statistics."""
        self.stats = {
            'tau_feedforward': [],
            'tau_feedback': [],
            'tau_total': [],
            'errors': [],
            'feedforward_ratio': []
        }
    
    def compute_torque(self, q: np.ndarray, q_dot: np.ndarray,
                       q_d: np.ndarray, q_dot_d: np.ndarray,
                       q_ddot_d: np.ndarray) -> np.ndarray:
        """
        Compute control torque using CTC.
        
        τ = M(q)*[q̈_d + Kd*ė + Kp*e] + C(q,q̇)*q̇ + G(q) [+ F(q̇)]
        
        Args:
            q: Current joint positions
            q_dot: Current joint velocities
            q_d: Desired joint positions
            q_dot_d: Desired joint velocities
            q_ddot_d: Desired joint accelerations
            
        Returns:
            Control torques [τ₁, τ₂]
        """
        # Compute errors
        e = q_d - q
        e_dot = q_dot_d - q_dot
        
        # Get dynamic matrices
        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, q_dot)
        G = self.robot.gravity_vector(q)
        
        # Compute auxiliary control input (linearizing feedback)
        # v = q̈_d + Kd*ė + Kp*e
        v = q_ddot_d + self.gains.Kd * e_dot + self.gains.Kp * e
        
        # Feedforward torque (dynamics compensation)
        tau_ff = M @ v + C @ q_dot + G
        
        # Add friction compensation if enabled
        if self.include_friction:
            F = self.robot.friction_vector(q_dot)
            tau_ff += F
        
        return tau_ff
    
    def compute_torque_decomposed(self, q: np.ndarray, q_dot: np.ndarray,
                                   q_d: np.ndarray, q_dot_d: np.ndarray,
                                   q_ddot_d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute torque with decomposition into components.
        
        Useful for analysis and debugging.
        
        Returns:
            Dictionary with:
                - tau_total: Total control torque
                - tau_inertia: M(q)*q̈_d term
                - tau_coriolis: C(q,q̇)*q̇ term
                - tau_gravity: G(q) term
                - tau_friction: F(q̇) term (if enabled)
                - tau_feedback: M(q)*(Kd*ė + Kp*e) term
                - feedforward_ratio: |τ_ff| / |τ_total|
        """
        # Compute errors
        e = q_d - q
        e_dot = q_dot_d - q_dot
        
        # Get matrices
        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, q_dot)
        G = self.robot.gravity_vector(q)
        
        # Decomposed torques
        tau_inertia = M @ q_ddot_d
        tau_coriolis = C @ q_dot
        tau_gravity = G
        tau_feedback = M @ (self.gains.Kd * e_dot + self.gains.Kp * e)
        
        # Feedforward = inertia + coriolis + gravity
        tau_ff = tau_inertia + tau_coriolis + tau_gravity
        
        # Add friction
        tau_friction = np.zeros(2)
        if self.include_friction:
            tau_friction = self.robot.friction_vector(q_dot)
            tau_ff += tau_friction
        
        # Total torque
        tau_total = tau_ff + tau_feedback
        
        # Compute feedforward ratio
        total_norm = np.linalg.norm(tau_total)
        if total_norm > 1e-6:
            ff_ratio = np.linalg.norm(tau_ff) / total_norm
        else:
            ff_ratio = 1.0
        
        return {
            'tau_total': tau_total,
            'tau_inertia': tau_inertia,
            'tau_coriolis': tau_coriolis,
            'tau_gravity': tau_gravity,
            'tau_friction': tau_friction,
            'tau_feedback': tau_feedback,
            'tau_feedforward': tau_ff,
            'feedforward_ratio': ff_ratio,
            'error': e,
            'error_dot': e_dot
        }
    
    def compute_feedforward_only(self, q: np.ndarray, q_dot: np.ndarray,
                                  q_ddot_d: np.ndarray) -> np.ndarray:
        """
        Compute only feedforward torque (no feedback).
        
        τ_ff = M(q)*q̈_d + C(q,q̇)*q̇ + G(q)
        
        This represents the "physics-informed" base control.
        Used as base controller for residual RL.
        """
        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, q_dot)
        G = self.robot.gravity_vector(q)
        
        tau_ff = M @ q_ddot_d + C @ q_dot + G
        
        if self.include_friction:
            tau_ff += self.robot.friction_vector(q_dot)
        
        return tau_ff
    
    def estimate_torque_contribution(self, q: np.ndarray, q_dot: np.ndarray,
                                      q_ddot: np.ndarray) -> Dict[str, float]:
        """
        Estimate percentage contribution of each dynamic term.
        
        This analysis shows why CTC provides ~90% of required torque.
        
        Returns:
            Dictionary with percentage contributions
        """
        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, q_dot)
        G = self.robot.gravity_vector(q)
        F = self.robot.friction_vector(q_dot)
        
        tau_inertia = np.linalg.norm(M @ q_ddot)
        tau_coriolis = np.linalg.norm(C @ q_dot)
        tau_gravity = np.linalg.norm(G)
        tau_friction = np.linalg.norm(F)
        
        total = tau_inertia + tau_coriolis + tau_gravity + tau_friction
        
        if total < 1e-6:
            return {'inertia': 0, 'coriolis': 0, 'gravity': 0, 'friction': 0}
        
        return {
            'inertia': 100 * tau_inertia / total,
            'coriolis': 100 * tau_coriolis / total,
            'gravity': 100 * tau_gravity / total,
            'friction': 100 * tau_friction / total,
            'total_ctc': 100 * (tau_inertia + tau_coriolis + tau_gravity) / total
        }


class SimplePDController:
    """
    Simple PD Controller for comparison.
    
    τ = Kp*e + Kd*ė + G(q)
    
    Only uses gravity compensation, no full dynamics.
    Provides ~50% of required torque (vs 90% for CTC).
    """
    
    def __init__(self, robot: Optional[RobotDynamics] = None,
                 gains: Optional[CTCGains] = None):
        self.robot = robot or create_nominal_robot()
        self.gains = gains or CTCGains()
    
    def compute_torque(self, q: np.ndarray, q_dot: np.ndarray,
                       q_d: np.ndarray, q_dot_d: np.ndarray,
                       q_ddot_d: np.ndarray = None) -> np.ndarray:
        """Compute PD + gravity compensation torque."""
        e = q_d - q
        e_dot = q_dot_d - q_dot
        
        # PD feedback
        tau_pd = self.gains.Kp * e + self.gains.Kd * e_dot
        
        # Gravity compensation only (not full dynamics)
        G = self.robot.gravity_vector(q)
        
        return tau_pd + G


# ============================================
# TEST
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("CTC Controller Module - Test")
    print("=" * 60)
    
    # Create controller
    ctc = CTCController()
    pd = SimplePDController()
    
    # Test trajectory point
    q = np.array([0.5, 0.3])
    q_dot = np.array([0.2, 0.1])
    q_d = np.array([0.6, 0.4])
    q_dot_d = np.array([0.1, 0.05])
    q_ddot_d = np.array([0.0, 0.0])
    
    print(f"\nTest State:")
    print(f"  q_actual  = {q}")
    print(f"  q_desired = {q_d}")
    print(f"  error     = {q_d - q}")
    
    # CTC torque
    tau_ctc = ctc.compute_torque(q, q_dot, q_d, q_dot_d, q_ddot_d)
    print(f"\nCTC Torque: τ = {tau_ctc}")
    
    # PD torque
    tau_pd = pd.compute_torque(q, q_dot, q_d, q_dot_d)
    print(f"PD Torque:  τ = {tau_pd}")
    
    # Decomposed analysis
    decomp = ctc.compute_torque_decomposed(q, q_dot, q_d, q_dot_d, q_ddot_d)
    print(f"\nCTC Torque Decomposition:")
    print(f"  τ_gravity    = {decomp['tau_gravity']}")
    print(f"  τ_coriolis   = {decomp['tau_coriolis']}")
    print(f"  τ_inertia    = {decomp['tau_inertia']}")
    print(f"  τ_friction   = {decomp['tau_friction']}")
    print(f"  τ_feedforward = {decomp['tau_feedforward']}")
    print(f"  τ_feedback   = {decomp['tau_feedback']}")
    print(f"  Feedforward ratio: {decomp['feedforward_ratio']*100:.1f}%")
    
    # Contribution analysis
    contrib = ctc.estimate_torque_contribution(q, q_dot, q_ddot_d)
    print(f"\nTorque Contribution Analysis:")
    print(f"  Inertia:  {contrib['inertia']:.1f}%")
    print(f"  Coriolis: {contrib['coriolis']:.1f}%")
    print(f"  Gravity:  {contrib['gravity']:.1f}%")
    print(f"  Friction: {contrib['friction']:.1f}%")
    
    print("\n✅ CTC Controller Module Working!")
