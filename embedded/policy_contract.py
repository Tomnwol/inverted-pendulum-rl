"""Frozen contract of the embedded policy.

Raw input: float32[3] = cos θ, sin θ, θ̇ (rad/s). θ = 0 is upright.
Preprocessing: θ̇ / MAX_SPEED.
Reference network: 3 → 64 → Tanh → 64 → Tanh → 1, mean action, no noise.
Output: torque before saturation. The ±MAX_TORQUE clip stays in Pendulum.step.
Period: 0.05 s (20 Hz). One step is one control tick.
Reference weights: weights/actor.pt, actor state_dict only, float32.
Distilled student: weights/student.pt, same input and output, 3 → 16 → 16 → 1.
Int8 student: weights/student_int8.pt. Network input and output stay float32.
ONNX student: weights/student.onnx. Raw observation in, torque out. The θ̇ / MAX_SPEED
scale is inside the graph.

Run the scripts from this directory.

Changing the reference network or the preprocessing means retraining PPO.
Changing the student width means rerunning distill.py, then quantize.py.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ            # 0.05 s
OBS_DIM = 3
ACTION_DIM = 1
MAX_SPEED = 8.0                  # physical clip, and divisor of θ̇
MAX_TORQUE = 2.0
HIDDEN = 64                      # reference actor, not the distilled student
HIDDEN_STUDENT = 16              # 3 → 16 → 16 → 1, same input and output
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
REFERENCE_WEIGHTS = WEIGHTS_DIR / "actor.pt"
STUDENT_WEIGHTS = WEIGHTS_DIR / "student.pt"
STUDENT_INT8_WEIGHTS = WEIGHTS_DIR / "student_int8.pt"
STUDENT_ONNX = WEIGHTS_DIR / "student.onnx"
EVAL_EPISODES, EVAL_SEED = 50, 123


def to_network_input(obs):
    """Raw observation → network input. Only change: θ̇ / MAX_SPEED."""
    network_input = torch.tensor(np.array(obs), dtype=torch.float32)
    network_input[..., 2] /= MAX_SPEED
    return network_input


def reference_network():
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, ACTION_DIM),
    )


def load_reference(path=REFERENCE_WEIGHTS):
    if not Path(path).exists():
        raise SystemExit("No saved weights. Run first: python ppo.py")
    network = reference_network()
    network.load_state_dict(torch.load(path, weights_only=True))
    network.eval()
    return network


def student_network():
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN_STUDENT), nn.Tanh(),
        nn.Linear(HIDDEN_STUDENT, HIDDEN_STUDENT), nn.Tanh(),
        nn.Linear(HIDDEN_STUDENT, ACTION_DIM),
    )


def load_student(path=STUDENT_WEIGHTS):
    if not Path(path).exists():
        raise SystemExit("No saved student. Run first: python distill.py")
    network = student_network()
    network.load_state_dict(torch.load(path, weights_only=True))
    network.eval()
    return network


def policy(network, obs):
    """Raw observation (3,) → torque before the clip."""
    with torch.no_grad():
        return float(network(to_network_input(obs)[None]).item())


def evaluate(network, n=EVAL_EPISODES, seed=EVAL_SEED):
    """Mean return. Deterministic policy, same seed as PPO."""
    from pendulum_env import Pendulum
    env, returns = Pendulum(seed=seed), []
    for _ in range(n):
        obs, total, done = env.reset(), 0.0, False
        while not done:
            torque = policy(network, obs)
            obs, reward, done = env.step(torque)
            total += reward
        returns.append(total)
    return float(np.mean(returns))
