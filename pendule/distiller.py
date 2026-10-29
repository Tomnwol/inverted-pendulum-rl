"""Distille la PPO float32 en un petit MLP.

    python distiller.py

Imite pendule_acteur.pt par moindres carrés sur des trajectoires.
La cible est toujours le couple du professeur. Sur une partie des pas,
le couple appliqué est bruité : le pendule visite le voisinage de la
trajectoire, et l'élève apprend l'action du professeur sur ces états.
L'évaluation est celle du contrat : 50 épisodes, graine 123.
"""

import numpy as np
import torch
import torch.nn as nn

from interface import POIDS_ELEVE, charger_reference, evaluer, politique, prep, reseau_eleve, reseau_reference
from pendule_env import Pendule

N_EPISODES = 200
BRUIT = 0.4                  # écart-type ajouté au couple, en N·m
PART_BRUIT = 0.35            # fraction de pas qui quittent la trajectoire exacte
EPOCHS, BATCH, LR = 40, 256, 1e-3
SEED = 0


def nombre_de_parametres(reseau):
    return sum(p.numel() for p in reseau.parameters())


def collecter(professeur):
    rng = np.random.default_rng(SEED)
    entrees, cibles = [], []
    for i in range(N_EPISODES):
        env = Pendule(seed=1_000 + i)       # la graine 123 reste réservée à l'évaluation
        o, fini = env.reset(), False
        while not fini:
            u = politique(professeur, o)
            entrees.append(prep(o))
            cibles.append(u)
            if rng.random() < PART_BRUIT:
                u = u + float(rng.normal(0.0, BRUIT))
            o, _, fini = env.step(u)
    return torch.stack(entrees), torch.tensor(cibles, dtype=torch.float32)


def entrainer(entrees, cibles):
    torch.manual_seed(SEED)
    eleve = reseau_eleve()
    opt = torch.optim.Adam(eleve.parameters(), lr=LR)
    perte = nn.MSELoss()
    n = len(cibles)
    for epoque in range(1, EPOCHS + 1):
        total = 0.0
        for idx in torch.randperm(n).split(BATCH):
            loss = perte(eleve(entrees[idx]).squeeze(-1), cibles[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if epoque == 1 or epoque % 10 == 0 or epoque == EPOCHS:
            print(f"époque {epoque:3d} | mse {total / n:.6f}")
    eleve.eval()
    return eleve


def main():
    professeur = charger_reference()
    print(f"professeur {nombre_de_parametres(reseau_reference())} paramètres | "
          f"élève {nombre_de_parametres(reseau_eleve())} paramètres")
    print("collecte des trajectoires...")
    entrees, cibles = collecter(professeur)
    print(f"{len(cibles)} paires (entrée préparée, couple du professeur)")
    eleve = entrainer(entrees, cibles)
    with torch.no_grad():
        mse = ((eleve(entrees).squeeze(-1) - cibles) ** 2).mean().item()
    retour_prof = evaluer(professeur)
    retour_eleve = evaluer(eleve)
    torch.save(eleve.state_dict(), POIDS_ELEVE)
    print(f"professeur | éval {retour_prof:8.1f}")
    print(f"élève      | éval {retour_eleve:8.1f} | mse {mse:.6f}")
    print(f"Élève enregistré : {POIDS_ELEVE.name}")


if __name__ == "__main__":
    main()
