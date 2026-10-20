#Dans cette exemple, on étudie une recherche sans gradient, juste de l'aléatoire pour tenter d'améliorer J, le retour moyen d'une politique.

from pathlib import Path

import numpy as np
from pendule_env import Pendule


def evaluer_J(politique, n_episodes=20, seed=100):
    """J = retour moyen sur n_episodes (somme des récompenses, sans gamma).
    Même seed => mêmes états de départ => comparaison équitable entre politiques."""
    env = Pendule(seed=seed)
    retours = []
    for _ in range(n_episodes):
        obs, total, fini = env.reset(), 0.0, False
        while not fini:
            obs, r, fini = env.step(politique(obs))
            total += r
        retours.append(total)
    return np.mean(retours)


def politique_depuis(w):
    """Petit réseau 3 -> 8 -> 1 dont les 41 paramètres sont rangés dans le vecteur w."""
    W1, b1 = w[:24].reshape(3, 8), w[24:32]
    W2, b2 = w[32:40], w[40]
    return lambda obs: 1.0 * np.tanh(np.tanh(obs @ W1 + b1) @ W2 + b2)   # couple dans [-2, 2]


if __name__ == "__main__":
    DIM = 41

    # 1) J est une ESTIMATION : elle change selon les états de départ tirés
    rng = np.random.default_rng(0)
    w = rng.standard_normal(DIM) * 0.1
    for n in (5, 100):
        estimations = [evaluer_J(politique_depuis(w), n_episodes=n, seed=s) for s in range(8)]
        print(f"J estimé avec n = {n:3d} épisodes, sur 8 jeux de départs : "
              f"min {min(estimations):.0f}, max {max(estimations):.0f}, écart-type {np.std(estimations):.0f}")

    # 2) Recherche aléatoire : on perturbe, on garde si c'est mieux
    rng = np.random.default_rng(0)
    w = rng.standard_normal(DIM) * 0.1
    meilleur = evaluer_J(politique_depuis(w))
    print(f"\nessai   0 | J = {meilleur:8.1f}")
    for essai in range(1, 15000):
        candidat = w + 0.3* rng.standard_normal(DIM)         # petite perturbation aléatoire
        j = evaluer_J(politique_depuis(candidat))
        if j > meilleur:                                       # on ne garde que les améliorations
            w, meilleur = candidat, j
        if essai % 50 == 0:
            print(f"essai {essai:3d} | meilleur J = {meilleur:8.1f}")

    chemin = Path(__file__).with_name("ch3_poids.npy")
    np.save(chemin, w)
    print(f"\npoids enregistrés : {chemin}")
    print(f"J sur les 20 départs de la recherche : {meilleur:.1f}")

    # 3) Vérification sur des départs jamais vus pendant la recherche
    #print(f"\nJ sur 100 départs jamais vus : {evaluer_J(politique_depuis(w), 100, seed=999):.1f}")