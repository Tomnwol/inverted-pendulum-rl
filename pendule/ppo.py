from pathlib import Path

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from interface import EVAL_EPISODES, EVAL_SEED, evaluer as evaluer_politique, prep, reseau_reference
from pendule_env import Pendule

# --- Algorithme PPO ---
GAMMA, LAM, CLIP = 0.95, 0.95, 0.2
EPOCHS, MINIBATCH = 10, 256
LR_ACTEUR, LR_CRITIQUE = 3e-4, 1e-3
ADV_EPS = 1e-8                               # évite une division par zéro quand l'avantage est constant
# --- Collecte et réseau ---
N_ENVS, T, N_ITER = 16, 200, 100
REWARD_SCALE = 0.1
LOG_STD_RANGE = (-2.0, 0.5)
SEED = 0
# --- Évaluation ---
PALIERS = (1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100)
if len(sys.argv) > 1:
    N_ITER = int(sys.argv[1])                    # python ppo.py 50  -> 50 itérations au lieu de N_ITER
torch.manual_seed(SEED)

actor = reseau_reference()
critic = reseau_reference()                  # même forme, poids indépendants
log_std = nn.Parameter(torch.zeros(1))       # écart-type appris
opt = torch.optim.Adam([{"params": list(actor.parameters()) + [log_std], "lr": LR_ACTEUR},
                        {"params": critic.parameters(), "lr": LR_CRITIQUE}])

def evaluer(n=EVAL_EPISODES):                # politique déterministe (moyenne, sans bruit)
    return evaluer_politique(actor, n=n, seed=EVAL_SEED)

envs = [Pendule(seed=i + 1) for i in range(N_ENVS)]
print(f"iter {0:3d} | éval {evaluer():8.1f} | std action {log_std.exp().item():.2f}")
for it in range(1, N_ITER + 1):
    # 1) collecte : chaque env joue un épisode complet
    obs = np.array([e.reset() for e in envs])
    O, A, LP, V, R = [], [], [], [], []
    with torch.no_grad():
        for t in range(T):
            o = prep(obs)
            dist = Normal(actor(o).squeeze(-1), log_std.exp())
            a = dist.sample()
            O.append(o); A.append(a); LP.append(dist.log_prob(a)); V.append(critic(o).squeeze(-1))
            pas = [e.step(x) for e, x in zip(envs, a.numpy())]
            obs = np.array([p[0] for p in pas])
            R.append(torch.tensor([p[1] for p in pas], dtype=torch.float32))
        derniere_valeur = critic(prep(obs)).squeeze(-1)
    O, A, LP, V, R = map(torch.stack, (O, A, LP, V, R))          # formes (T, N_ENVS, ...)
    retour_moyen = R.sum(0).mean().item()
    R = R * REWARD_SCALE                                        # échelle des récompenses

    # 2) avantages (GAE) et cibles du critique
    adv, g = torch.zeros(T, N_ENVS), torch.zeros(N_ENVS)
    for t in reversed(range(T)):
        v_suivant = derniere_valeur if t == T - 1 else V[t + 1]
        g = R[t] + GAMMA * v_suivant - V[t] + GAMMA * LAM * g
        adv[t] = g
    ret = adv + V
    O, A, LP, adv, ret = (x.reshape(T * N_ENVS, *x.shape[2:]) for x in (O, A, LP, adv, ret))
    adv = (adv - adv.mean()) / (adv.std() + ADV_EPS)

    # 3) mise à jour PPO : plusieurs passes sur les mêmes données
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

    if it in PALIERS:
        print(f"iter {it:3d} | retour (entraînement) {retour_moyen:8.1f} "
              f"| éval {evaluer():8.1f} | std action {log_std.exp().item():.2f}")

torch.save(actor.state_dict(), Path(__file__).with_name("pendule_acteur.pt"))
print("Acteur enregistré : pendule_acteur.pt")