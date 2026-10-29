import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from policy_contract import EVAL_EPISODES, EVAL_SEED, REFERENCE_WEIGHTS, evaluate as evaluate_policy, reference_network, to_network_input
from pendulum_env import Pendulum

# --- PPO algorithm ---
GAMMA, LAM, CLIP = 0.95, 0.95, 0.2
EPOCHS, MINIBATCH = 10, 256
LR_ACTOR, LR_CRITIC = 3e-4, 1e-3
ADV_EPS = 1e-8                               # avoids a division by zero when the advantage is constant
# --- Collection and network ---
N_ENVS, T, N_ITER = 16, 200, 100
REWARD_SCALE = 0.1
LOG_STD_RANGE = (-2.0, 0.5)
SEED = 0
# --- Evaluation ---
MILESTONES = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
if len(sys.argv) > 1:
    N_ITER = int(sys.argv[1])                    # python ppo.py 50  -> 50 iterations instead of N_ITER
torch.manual_seed(SEED)

actor = reference_network()
critic = reference_network()                 # same shape, independent weights
log_std = nn.Parameter(torch.zeros(1))       # learned standard deviation
opt = torch.optim.Adam([{"params": list(actor.parameters()) + [log_std], "lr": LR_ACTOR},
                        {"params": critic.parameters(), "lr": LR_CRITIC}])

def evaluate(n=EVAL_EPISODES):               # deterministic policy (mean, no noise)
    return evaluate_policy(actor, n=n, seed=EVAL_SEED)

envs = [Pendulum(seed=i + 1) for i in range(N_ENVS)]
print(f"iter {0:3d} | eval {evaluate():8.1f} | std action {log_std.exp().item():.2f}")
for it in range(1, N_ITER + 1):
    # 1) collection: each env plays a full episode
    obs = np.array([e.reset() for e in envs])
    O, A, LP, V, R = [], [], [], [], []
    with torch.no_grad():
        for t in range(T):
            o = to_network_input(obs)
            dist = Normal(actor(o).squeeze(-1), log_std.exp())
            a = dist.sample()
            O.append(o); A.append(a); LP.append(dist.log_prob(a)); V.append(critic(o).squeeze(-1))
            steps = [e.step(x) for e, x in zip(envs, a.numpy())]
            obs = np.array([p[0] for p in steps])
            R.append(torch.tensor([p[1] for p in steps], dtype=torch.float32))
        last_value = critic(to_network_input(obs)).squeeze(-1)
    O, A, LP, V, R = map(torch.stack, (O, A, LP, V, R))          # shapes (T, N_ENVS, ...)
    mean_return = R.sum(0).mean().item()
    R = R * REWARD_SCALE                                        # reward scale

    # 2) advantages (GAE) and critic targets
    adv, g = torch.zeros(T, N_ENVS), torch.zeros(N_ENVS)
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else V[t + 1]
        g = R[t] + GAMMA * next_value - V[t] + GAMMA * LAM * g
        adv[t] = g
    ret = adv + V
    O, A, LP, adv, ret = (x.reshape(T * N_ENVS, *x.shape[2:]) for x in (O, A, LP, adv, ret))
    adv = (adv - adv.mean()) / (adv.std() + ADV_EPS)

    # 3) PPO update: several passes over the same data
    for _ in range(EPOCHS):
        for idx in torch.randperm(T * N_ENVS).split(MINIBATCH):
            dist = Normal(actor(O[idx]).squeeze(-1), log_std.exp())
            ratio = (dist.log_prob(A[idx]) - LP[idx]).exp()
            loss_pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
            loss_v = 0.5 * ((critic(O[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            opt.zero_grad()
            (loss_pi + loss_v).backward()
            opt.step()
            with torch.no_grad():
                log_std.clamp_(*LOG_STD_RANGE)

    if it in MILESTONES:
        print(f"iter {it:3d} | train return {mean_return:8.1f} "
              f"| eval {evaluate():8.1f} | std action {log_std.exp().item():.2f}")

REFERENCE_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
torch.save(actor.state_dict(), REFERENCE_WEIGHTS)
print(f"Actor saved: weights/{REFERENCE_WEIGHTS.name}")
