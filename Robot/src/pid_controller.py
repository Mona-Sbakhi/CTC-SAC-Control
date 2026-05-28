#!/usr/bin/env python3
"""
PID Controller for 2-DOF Manipulator
======================================
CoDIT 2026 Paper: CTC + SAC Hybrid Control

Decentralised joint-space PID — the standard baseline in residual RL literature.
Unlike CTC, this controller does NOT incorporate inertia, Coriolis, or gravity
explicitly; it reacts to tracking errors only.

Authors: Mona Alsbakhi, Majed Tabash, Anas Alsalool, Asma Sbaih
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class PIDGains:
    Kp: np.ndarray = field(default_factory=lambda: np.array([100.0, 50.0]))
    Ki: np.ndarray = field(default_factory=lambda: np.array([5.0,   2.0]))
    Kd: np.ndarray = field(default_factory=lambda: np.array([20.0,  10.0]))


class PIDController:
    """
    Decentralised joint-space PID controller.

    τ_PID = Kp·e + Ki·∫e dt + Kd·ė

    Gains are intentionally set to match the PD gains used inside the CTC
    controller (Kp, Kd) plus a small integral term (Ki), ensuring a fair
    comparison between CTC and PID as base controllers for residual RL.
    """

    def __init__(self, gains: PIDGains = None, dt: float = 0.02,
                 tau_max: float = 15.0):
        self.gains   = gains or PIDGains()
        self.dt      = dt
        self.tau_max = tau_max          # saturation limit for base torque
        self.integral = np.zeros(2)

    def reset(self):
        """Call at the start of each episode to clear integral windup."""
        self.integral = np.zeros(2)

    def compute_torque(self, e: np.ndarray, e_dot: np.ndarray) -> np.ndarray:
        """
        Compute PID torque.

        Args:
            e     : position error  (q_d - q),    shape (2,)
            e_dot : velocity error  (q_dot_d - q_dot), shape (2,)

        Returns:
            tau   : joint torques,  shape (2,)
        """
        self.integral += e * self.dt
        # anti-windup: clip integral contribution
        self.integral = np.clip(self.integral, -5.0, 5.0)

        tau = (self.gains.Kp * e
               + self.gains.Ki * self.integral
               + self.gains.Kd * e_dot)

        return np.clip(tau, -self.tau_max, self.tau_max)
