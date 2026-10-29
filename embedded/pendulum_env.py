"""Inverted pendulum. The contract (state, action, 20 Hz) is frozen in policy_contract.py.

Raw observation (3,): cos(θ), sin(θ), angular velocity, not normalized.
θ = 0 is upright. An episode lasts 200 steps.
step(torque) saturates the torque at ±MAX_TORQUE and returns (observation, reward, done).
"""

import numpy as np

from policy_contract import DT, MAX_SPEED, MAX_TORQUE


class Pendulum:
    def __init__(self, seed=0, max_steps=200):
        self.max_speed = MAX_SPEED
        self.max_torque = MAX_TORQUE
        self.dt = DT
        self.g = 10.0
        self.m = 1.0
        self.l = 1.0
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(2)
        self.steps = 0

    def reset(self):
        high = np.array([np.pi, 1.0])
        self.state = self.rng.uniform(-high, high)
        self.steps = 0
        return self._obs()

    def step(self, torque):
        theta, theta_dot = self.state
        torque = float(np.clip(np.asarray(torque, dtype=np.float64).reshape(-1)[0], -self.max_torque, self.max_torque))

        # cost: angle error, speed, and effort
        theta_norm = ((theta + np.pi) % (2 * np.pi)) - np.pi
        cost = theta_norm**2 + 0.1 * theta_dot**2 + 0.001 * torque**2

        new_theta_dot = theta_dot + (3 * self.g / (2 * self.l) * np.sin(theta) + 3.0 / (self.m * self.l**2) * torque) * self.dt
        new_theta_dot = float(np.clip(new_theta_dot, -self.max_speed, self.max_speed))
        self.state = np.array([theta + new_theta_dot * self.dt, new_theta_dot])
        self.steps += 1
        return self._obs(), float(-cost), self.steps >= self.max_steps

    def _obs(self):
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
