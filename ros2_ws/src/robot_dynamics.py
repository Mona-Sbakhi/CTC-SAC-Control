#!/usr/bin/env python3
"""
Robot Dynamics Module for 2-DOF Planar Manipulator
===================================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Based on Assignment 3 dynamic model with enhancements for:
- Friction compensation
- Parameter uncertainty handling
- Efficient matrix computations

Robot Parameters (Kinova Mico - First 2 Joints):
- L1 = L2 = 0.15 m
- m1 = 2.072 kg, m2 = 1.072 kg
- I1 = 0.001 kg·m², I2 = 0.00098 kg·m²

Author: Mohammed
Date: January 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RobotParameters:
    """Robot physical parameters."""
    # Link lengths [m]
    L1: float = 0.15
    L2: float = 0.15
    
    # Link masses [kg]
    m1: float = 2.072
    m2: float = 1.072
    
    # Link inertias [kg·m²]
    I1: float = 0.001
    I2: float = 0.00098
    
    # Center of mass positions [m]
    lc1: float = 0.075  # L1/2
    lc2: float = 0.075  # L2/2
    
    # Gravity [m/s²]
    g: float = 9.81
    
    # Friction coefficients (for realistic simulation)
    b1: float = 0.1  # Viscous friction joint 1 [N·m·s/rad]
    b2: float = 0.1  # Viscous friction joint 2 [N·m·s/rad]
    fc1: float = 0.2  # Coulomb friction joint 1 [N·m]
    fc2: float = 0.2  # Coulomb friction joint 2 [N·m]
    
    def with_uncertainty(self, mass_error: float = 0.0, 
                         inertia_error: float = 0.0) -> 'RobotParameters':
        """
        Create parameters with specified uncertainty.
        
        Args:
            mass_error: Percentage error in mass (e.g., 0.1 = 10%)
            inertia_error: Percentage error in inertia
            
        Returns:
            New RobotParameters with perturbed values
        """
        return RobotParameters(
            L1=self.L1, L2=self.L2,
            m1=self.m1 * (1 + mass_error),
            m2=self.m2 * (1 + mass_error),
            I1=self.I1 * (1 + inertia_error),
            I2=self.I2 * (1 + inertia_error),
            lc1=self.lc1, lc2=self.lc2,
            g=self.g,
            b1=self.b1, b2=self.b2,
            fc1=self.fc1, fc2=self.fc2
        )


class RobotDynamics:
    """
    Complete dynamic model for 2-DOF planar manipulator.
    
    Equation of motion:
        M(q)q̈ + C(q,q̇)q̇ + G(q) + F(q̇) = τ
    
    where:
        M(q)  : Inertia matrix (2x2)
        C(q,q̇): Coriolis/centrifugal matrix (2x2)
        G(q)  : Gravity vector (2x1)
        F(q̇)  : Friction vector (2x1)
        τ     : Joint torques (2x1)
    """
    
    def __init__(self, params: Optional[RobotParameters] = None):
        """
        Initialize robot dynamics.
        
        Args:
            params: Robot parameters (uses defaults if None)
        """
        self.params = params or RobotParameters()
        self._precompute_constants()
    
    def _precompute_constants(self):
        """Precompute constant terms for efficiency."""
        p = self.params
        
        # Constants for M(q)
        self.a1 = p.m1 * p.lc1**2 + p.I1 + p.m2 * p.L1**2 + p.m2 * p.lc2**2 + p.I2
        self.a2 = p.m2 * p.L1 * p.lc2  # Coupling term coefficient
        self.a3 = p.m2 * p.lc2**2 + p.I2
        
        # Constants for G(q)
        self.g1 = (p.m1 * p.lc1 + p.m2 * p.L1) * p.g
        self.g2 = p.m2 * p.lc2 * p.g
    
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Compute inertia/mass matrix M(q).
        
        M(q) = | M11  M12 |
               | M12  M22 |
        
        Args:
            q: Joint positions [θ₁, θ₂]
            
        Returns:
            2x2 symmetric positive definite mass matrix
        """
        c2 = np.cos(q[1])
        
        M11 = self.a1 + 2 * self.a2 * c2
        M12 = self.a3 + self.a2 * c2
        M22 = self.a3
        
        return np.array([[M11, M12],
                         [M12, M22]])
    
    def coriolis_matrix(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute Coriolis/centrifugal matrix C(q, q̇).
        
        Using Christoffel symbols formulation.
        
        Args:
            q: Joint positions [θ₁, θ₂]
            q_dot: Joint velocities [θ̇₁, θ̇₂]
            
        Returns:
            2x2 Coriolis matrix
        """
        s2 = np.sin(q[1])
        h = self.a2 * s2
        
        C11 = -h * q_dot[1]
        C12 = -h * (q_dot[0] + q_dot[1])
        C21 = h * q_dot[0]
        C22 = 0.0
        
        return np.array([[C11, C12],
                         [C21, C22]])
    
    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector G(q).
        
        For vertical planar robot, gravity acts along Y-axis.
        
        Args:
            q: Joint positions [θ₁, θ₂]
            
        Returns:
            2x1 gravity vector
        """
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        
        G1 = self.g1 * c1 + self.g2 * c12
        G2 = self.g2 * c12
        
        return np.array([G1, G2])
    
    def friction_vector(self, q_dot: np.ndarray) -> np.ndarray:
        """
        Compute friction forces F(q̇).
        
        F = b*q̇ + fc*sign(q̇)
        
        Args:
            q_dot: Joint velocities [θ̇₁, θ̇₂]
            
        Returns:
            2x1 friction vector
        """
        p = self.params
        
        # Viscous + Coulomb friction
        F1 = p.b1 * q_dot[0] + p.fc1 * np.sign(q_dot[0])
        F2 = p.b2 * q_dot[1] + p.fc2 * np.sign(q_dot[1])
        
        return np.array([F1, F2])
    
    def inverse_dynamics(self, q: np.ndarray, q_dot: np.ndarray, 
                         q_ddot: np.ndarray, include_friction: bool = True) -> np.ndarray:
        """
        Compute required torques for given motion (Inverse Dynamics).
        
        τ = M(q)q̈ + C(q,q̇)q̇ + G(q) [+ F(q̇)]
        
        Args:
            q: Joint positions
            q_dot: Joint velocities
            q_ddot: Joint accelerations
            include_friction: Whether to include friction compensation
            
        Returns:
            Required joint torques
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        
        tau = M @ q_ddot + C @ q_dot + G
        
        if include_friction:
            tau += self.friction_vector(q_dot)
        
        return tau
    
    def forward_dynamics(self, q: np.ndarray, q_dot: np.ndarray, 
                         tau: np.ndarray, include_friction: bool = True) -> np.ndarray:
        """
        Compute accelerations for given torques (Forward Dynamics).
        
        q̈ = M⁻¹(q)[τ - C(q,q̇)q̇ - G(q) - F(q̇)]
        
        Args:
            q: Joint positions
            q_dot: Joint velocities
            tau: Applied joint torques
            include_friction: Whether to include friction effects
            
        Returns:
            Joint accelerations
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        
        rhs = tau - C @ q_dot - G
        
        if include_friction:
            rhs -= self.friction_vector(q_dot)
        
        # Solve M * q_ddot = rhs
        q_ddot = np.linalg.solve(M, rhs)
        
        return q_ddot
    
    def get_all_matrices(self, q: np.ndarray, q_dot: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Get all dynamic matrices at once.
        
        Returns:
            Tuple of (M, C, G, F)
        """
        M = self.mass_matrix(q)
        C = self.coriolis_matrix(q, q_dot)
        G = self.gravity_vector(q)
        F = self.friction_vector(q_dot)
        
        return M, C, G, F
    
    def energy(self, q: np.ndarray, q_dot: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute kinetic and potential energy.
        
        Returns:
            Tuple of (kinetic_energy, potential_energy, total_energy)
        """
        M = self.mass_matrix(q)
        p = self.params
        
        # Kinetic energy: K = 0.5 * q̇ᵀ M q̇
        K = 0.5 * q_dot @ M @ q_dot
        
        # Potential energy (with respect to horizontal reference)
        s1 = np.sin(q[0])
        s12 = np.sin(q[0] + q[1])
        P = (p.m1 * p.lc1 + p.m2 * p.L1) * p.g * s1 + p.m2 * p.lc2 * p.g * s12
        
        return K, P, K + P


# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_nominal_robot() -> RobotDynamics:
    """Create robot with nominal (ideal) parameters."""
    return RobotDynamics(RobotParameters())


def create_uncertain_robot(mass_error: float = 0.1, 
                           inertia_error: float = 0.1) -> RobotDynamics:
    """
    Create robot with parameter uncertainties.
    
    Args:
        mass_error: Percentage mass uncertainty (default 10%)
        inertia_error: Percentage inertia uncertainty (default 10%)
    """
    nominal = RobotParameters()
    uncertain = nominal.with_uncertainty(mass_error, inertia_error)
    return RobotDynamics(uncertain)


# ============================================
# TEST
# ============================================

if __name__ == '__main__':
    print("=" * 60)
    print("Robot Dynamics Module - Test")
    print("=" * 60)
    
    # Create robot
    robot = create_nominal_robot()
    
    # Test state
    q = np.array([np.pi/4, np.pi/6])  # 45°, 30°
    q_dot = np.array([0.5, 0.3])
    q_ddot = np.array([0.1, 0.1])
    
    # Compute matrices
    M, C, G, F = robot.get_all_matrices(q, q_dot)
    
    print(f"\nTest State:")
    print(f"  q = [{np.degrees(q[0]):.1f}°, {np.degrees(q[1]):.1f}°]")
    print(f"  q̇ = {q_dot}")
    
    print(f"\nMass Matrix M(q):")
    print(f"  {M}")
    
    print(f"\nCoriolis Matrix C(q,q̇):")
    print(f"  {C}")
    
    print(f"\nGravity Vector G(q):")
    print(f"  {G}")
    
    print(f"\nFriction Vector F(q̇):")
    print(f"  {F}")
    
    # Inverse dynamics
    tau = robot.inverse_dynamics(q, q_dot, q_ddot)
    print(f"\nInverse Dynamics τ = M*q̈ + C*q̇ + G + F:")
    print(f"  τ = {tau}")
    
    # Verify with forward dynamics
    q_ddot_check = robot.forward_dynamics(q, q_dot, tau)
    print(f"\nForward Dynamics Verification:")
    print(f"  q̈_original = {q_ddot}")
    print(f"  q̈_computed = {q_ddot_check}")
    print(f"  Error = {np.linalg.norm(q_ddot - q_ddot_check):.2e}")
    
    print("\n✅ Robot Dynamics Module Working!")
