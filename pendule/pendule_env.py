"""Pendule inversé. Le contrat (état, action, 20 Hz) est figé dans interface.py.

Observation brute (3,) : cos(θ), sin(θ), vitesse angulaire non normalisée.
θ = 0 est la position haute. L'épisode dure 200 pas.
step(couple) sature le couple à ±MAX_COUPLE et renvoie (observation, récompense, fini).
"""

import numpy as np

from interface import DT, MAX_COUPLE, VITESSE_MAX


class Pendule:
    def __init__(self, seed=0, max_steps=200):
        self.max_speed = VITESSE_MAX
        self.max_torque = MAX_COUPLE
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

    def step(self, u):
        th, thdot = self.state
        u = float(np.clip(np.asarray(u, dtype=np.float64).reshape(-1)[0], -self.max_torque, self.max_torque))

        # coût : écart à la verticale, vitesse, et effort
        th_norm = ((th + np.pi) % (2 * np.pi)) - np.pi
        cost = th_norm**2 + 0.1 * thdot**2 + 0.001 * u**2

        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l**2) * u) * self.dt
        newthdot = float(np.clip(newthdot, -self.max_speed, self.max_speed))
        self.state = np.array([th + newthdot * self.dt, newthdot])
        self.steps += 1
        return self._obs(), float(-cost), self.steps >= self.max_steps

    def _obs(self):
        th, thdot = self.state
        return np.array([np.cos(th), np.sin(th), thdot], dtype=np.float32)
