"""Distill the float32 PPO policy into a small MLP.

    python distill.py

Imitates weights/actor.pt by least squares on trajectories.
The target is always the teacher's torque. On some steps the applied
torque is noisy: the pendulum visits the neighborhood of the trajectory,
and the student learns the teacher's action on those states.
Evaluation follows the contract: 50 episodes, seed 123.
"""

import numpy as np
import torch
import torch.nn as nn

from policy_contract import STUDENT_WEIGHTS, evaluate, load_reference, policy, reference_network, student_network, to_network_input
from pendulum_env import Pendulum

N_EPISODES = 200
NOISE = 0.4                  # standard deviation added to the torque, in N·m
NOISE_FRACTION = 0.35        # fraction of steps that leave the exact trajectory
EPOCHS, BATCH, LR = 40, 256, 1e-3
SEED = 0


def parameter_count(network):
    return sum(p.numel() for p in network.parameters())


def collect(teacher):
    rng = np.random.default_rng(SEED)
    inputs, targets = [], []
    for i in range(N_EPISODES):
        env = Pendulum(seed=1_000 + i)       # seed 123 stays reserved for evaluation
        obs, done = env.reset(), False
        while not done:
            torque = policy(teacher, obs)
            inputs.append(to_network_input(obs))
            targets.append(torque)
            if rng.random() < NOISE_FRACTION:
                torque = torque + float(rng.normal(0.0, NOISE))
            obs, _, done = env.step(torque)
    return torch.stack(inputs), torch.tensor(targets, dtype=torch.float32)


def train(inputs, targets):
    torch.manual_seed(SEED)
    student = student_network()
    opt = torch.optim.Adam(student.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    n = len(targets)
    for epoch in range(1, EPOCHS + 1):
        total = 0.0
        for idx in torch.randperm(n).split(BATCH):
            loss = loss_fn(student(inputs[idx]).squeeze(-1), targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            print(f"epoch {epoch:3d} | mse {total / n:.6f}")
    student.eval()
    return student


def main():
    teacher = load_reference()
    print(f"teacher {parameter_count(reference_network())} parameters | "
          f"student {parameter_count(student_network())} parameters")
    print("collecting trajectories...")
    inputs, targets = collect(teacher)
    print(f"{len(targets)} pairs (prepared input, teacher torque)")
    student = train(inputs, targets)
    with torch.no_grad():
        mse = ((student(inputs).squeeze(-1) - targets) ** 2).mean().item()
    teacher_return = evaluate(teacher)
    student_return = evaluate(student)
    STUDENT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), STUDENT_WEIGHTS)
    print(f"teacher | eval {teacher_return:8.1f}")
    print(f"student | eval {student_return:8.1f} | mse {mse:.6f}")
    print(f"Student saved: weights/{STUDENT_WEIGHTS.name}")


if __name__ == "__main__":
    main()
