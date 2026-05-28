#!/usr/bin/env python3
"""
Computed Torque Controller (CTC) for 2-DOF Manipulator
=======================================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Control Law:
    τ = M(q)[q̈_d + Kd*ė + Kp*e] + C(q,q̇)q̇ + G(q) [+ F(q̇)]

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
Affiliation: Islamic University of Gaza / University of Seville
Conference: CoDIT 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from robot_dynamics import RobotDynamics, create_nominal_robot


@dataclass
class CTCGains:
    """PD gains for CTC feedback linearization."""
    Kp: np.ndarray = field(default_factory=lambda: np.array([100.0, 50.0]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([20.0, 10.0]))


class CTCController:
    """
    Computed Torque Controller (Inverse Dynamics Control).

    Linearizes robot dynamics so the closed-loop error satisfies:
        ë + Kd*ė + Kp*e = 0
    """

    def __init__(self, robot: Optional[RobotDynamics] = None,
                 gains: Optional[CTCGains] = None,
                 include_friction: bool = True):
        self.robot = robot or create_nominal_robot()
        self.gains = gains or CTCGains()
        self.include_friction = include_friction

    def compute_torque(self, q, q_dot, q_d, q_dot_d, q_ddot_d) -> np.ndarray:
        """
        Compute CTC torque.

        τ = M(q)*[q̈_d + Kd*ė + Kp*e] + C(q,q̇)*q̇ + G(q) [+ F(q̇)]
        """
        e     = q_d - q
        e_dot = q_dot_d - q_dot

        M = self.robot.mass_matrix(q)
        C = self.robot.coriolis_matrix(q, q_dot)
        G = self.robot.gravity_vector(q)

        v   = q_ddot_d + self.gains.Kd * e_dot + self.gains.Kp * e
        tau = M @ v + C @ q_dot + G

        if self.include_friction:
            tau += self.robot.friction_vector(q_dot)

        return tau


class SimplePIDController:
    """
    PID + gravity compensation baseline.

    τ = Kp*e + Ki*∫e dt + Kd*ė + G(q)

    Provides ~50–60% of required torque (vs ~90% for CTC).
    Used as baseline comparison in the paper.
    """

    def __init__(self, robot: Optional[RobotDynamics] = None,
                 Kp=None, Ki=None, Kd=None, dt: float = 0.01):
        self.robot  = robot or create_nominal_robot()
        self.Kp     = np.array(Kp) if Kp is not None else np.array([100.0, 50.0])
        self.Ki     = np.array(Ki) if Ki is not None else np.array([5.0, 2.0])
        self.Kd     = np.array(Kd) if Kd is not None else np.array([20.0, 10.0])
        self.dt     = dt
        self.integral = np.zeros(2)

    def reset(self):
        self.integral = np.zeros(2)

    def compute_torque(self, q, q_dot, q_d, q_dot_d, q_ddot_d=None) -> np.ndarray:
        e     = q_d - q
        e_dot = q_dot_d - q_dot
        self.integral += e * self.dt

        tau_pid = self.Kp * e + self.Ki * self.integral + self.Kd * e_dot
        G       = self.robot.gravity_vector(q)

        return tau_pid + G
