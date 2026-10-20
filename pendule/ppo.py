from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from pendule_env import Pendule

N, T, GAMMA, LAM, CLIP = 16, 200, 0.95, 0.95, 0.2    # envs, pas/épisode, γ, λ (GAE), ε (clip)
torch.manual_seed(0)

def mlp():
    return nn.Sequential(nn.Linear(3, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))

actor, critic = mlp(), mlp()                 # moyenne de l'action / valeur de l'état
log_std = nn.Parameter(torch.zeros(1))       # écart-type appris
opt = torch.optim.Adam([{"params": list(actor.parameters()) + [log_std], "lr": 3e-4},
                        {"params": critic.parameters(), "lr": 1e-3}])

def prep(obs):                               # vitesse ramenée à ~[-1, 1]
    o = torch.tensor(np.array(obs), dtype=torch.float32)
    o[..., 2] /= 8
    return o

def evaluer(n=50):                           # politique déterministe (moyenne, sans bruit)
    env, retours = Pendule(seed=123), []
    for _ in range(n):
        o, total, fini = env.reset(), 0.0, False
        while not fini:
            with torch.no_grad():
                u = actor(prep(o)[None]).item()
            o, r, fini = env.step(u)
            total += r
        retours.append(total)
    return np.mean(retours)

envs = [Pendule(seed=i + 1) for i in range(N)]
for it in range(1, 101):
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
    O, A, LP, V, R = map(torch.stack, (O, A, LP, V, R))          # formes (T, N, ...)
    retour_moyen = R.sum(0).mean().item()
    R = R * 0.1                                                  # échelle des récompenses

    # 2) avantages (GAE) et cibles du critique
    adv, g = torch.zeros(T, N), torch.zeros(N)
    for t in reversed(range(T)):
        v_suivant = derniere_valeur if t == T - 1 else V[t + 1]
        g = R[t] + GAMMA * v_suivant - V[t] + GAMMA * LAM * g
        adv[t] = g
    ret = adv + V
    O, A, LP, adv, ret = (x.reshape(T * N, *x.shape[2:]) for x in (O, A, LP, adv, ret))
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    # 3) mise à jour PPO : plusieurs passes sur les mêmes données
    for _ in range(10):
        for idx in torch.randperm(T * N).split(256):
            dist = Normal(actor(O[idx]).squeeze(-1), log_std.exp())
            ratio = (dist.log_prob(A[idx]) - LP[idx]).exp()
            loss_pi = -torch.min(ratio * adv[idx], ratio.clamp(1 - CLIP, 1 + CLIP) * adv[idx]).mean()
            loss_v = 0.5 * ((critic(O[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
            opt.zero_grad()
            (loss_pi + loss_v).backward()
            opt.step()
            with torch.no_grad():
                log_std.clamp_(-2.0, 0.5)

    if it % 10 == 0 or it == 1:
        print(f"iter {it:3d} | retour (entraînement) {retour_moyen:8.1f} "
              f"| éval {evaluer():8.1f} | std action {log_std.exp().item():.2f}")

torch.save(actor.state_dict(), Path(__file__).with_name("pendule_acteur.pt"))
print("Acteur enregistré : pendule_acteur.pt")