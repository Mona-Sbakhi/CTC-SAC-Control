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

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
Affiliation: Islamic University of Gaza / University of Seville
Conference: CoDIT 2026
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

    # Friction coefficients
    b1: float = 0.1   # Viscous friction joint 1 [N·m·s/rad]
    b2: float = 0.1   # Viscous friction joint 2 [N·m·s/rad]
    fc1: float = 0.2  # Coulomb friction joint 1 [N·m]
    fc2: float = 0.2  # Coulomb friction joint 2 [N·m]

    def with_uncertainty(self, mass_error: float = 0.0,
                         inertia_error: float = 0.0) -> 'RobotParameters':
        """Create parameters with specified uncertainty (e.g. 0.1 = 10%)."""
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
    """

    def __init__(self, params: Optional[RobotParameters] = None):
        self.params = params or RobotParameters()
        self._precompute_constants()

    def _precompute_constants(self):
        p = self.params
        self.a1 = p.m1 * p.lc1**2 + p.I1 + p.m2 * p.L1**2 + p.m2 * p.lc2**2 + p.I2
        self.a2 = p.m2 * p.L1 * p.lc2
        self.a3 = p.m2 * p.lc2**2 + p.I2
        self.g1 = (p.m1 * p.lc1 + p.m2 * p.L1) * p.g
        self.g2 = p.m2 * p.lc2 * p.g

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        c2 = np.cos(q[1])
        M11 = self.a1 + 2 * self.a2 * c2
        M12 = self.a3 + self.a2 * c2
        M22 = self.a3
        return np.array([[M11, M12], [M12, M22]])

    def coriolis_matrix(self, q: np.ndarray, q_dot: np.ndarray) -> np.ndarray:
        s2 = np.sin(q[1])
        h = self.a2 * s2
        return np.array([[-h * q_dot[1], -h * (q_dot[0] + q_dot[1])],
                         [h * q_dot[0], 0.0]])

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        c1  = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        return np.array([self.g1 * c1 + self.g2 * c12,
                         self.g2 * c12])

    def friction_vector(self, q_dot: np.ndarray) -> np.ndarray:
        p = self.params
        return np.array([p.b1 * q_dot[0] + p.fc1 * np.sign(q_dot[0]),
                         p.b2 * q_dot[1] + p.fc2 * np.sign(q_dot[1])])

    def inverse_dynamics(self, q, q_dot, q_ddot,
                         include_friction: bool = True) -> np.ndarray:
        """τ = M(q)q̈ + C(q,q̇)q̇ + G(q) [+ F(q̇)]"""
        tau = self.mass_matrix(q) @ q_ddot + \
              self.coriolis_matrix(q, q_dot) @ q_dot + \
              self.gravity_vector(q)
        if include_friction:
            tau += self.friction_vector(q_dot)
        return tau

    def get_all_matrices(self, q, q_dot) -> Tuple[np.ndarray, ...]:
        return (self.mass_matrix(q),
                self.coriolis_matrix(q, q_dot),
                self.gravity_vector(q),
                self.friction_vector(q_dot))


def create_nominal_robot() -> RobotDynamics:
    return RobotDynamics(RobotParameters())


def create_uncertain_robot(mass_error: float = 0.1,
                           inertia_error: float = 0.1) -> RobotDynamics:
    return RobotDynamics(RobotParameters().with_uncertainty(mass_error, inertia_error))
