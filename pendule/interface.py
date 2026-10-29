"""Contrat figé de la politique embarquée.

Entrée brute : float32[3] = cos θ, sin θ, θ̇ (rad/s). θ = 0 en haut.
Prétraitement : θ̇ / VITESSE_MAX.
Réseau de référence : 3 → 64 → Tanh → 64 → Tanh → 1, moyenne, sans bruit.
Sortie : couple avant saturation. Le clip ±MAX_COUPLE reste dans Pendule.step.
Période : 0,05 s (20 Hz). Un step = un tick de contrôle.
Poids de référence : pendule_acteur.pt, state_dict de l'acteur seul, float32.
Élève distillé : pendule_eleve.pt, même entrée et même sortie, 3 → 16 → 16 → 1.
Élève int8 : pendule_eleve_int8.pt. L'entrée et la sortie du réseau restent des float32.

Modifier le réseau de référence ou le prétraitement impose de réentraîner la PPO.
Modifier la largeur de l'élève impose de relancer distiller.py, puis quantifier.py.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

FREQUENCE_HZ = 20
DT = 1.0 / FREQUENCE_HZ          # 0,05 s
OBS_DIM = 3
ACTION_DIM = 1
VITESSE_MAX = 8.0                # clip physique, et diviseur de θ̇
MAX_COUPLE = 2.0
HIDDEN = 64                      # acteur de référence, pas l'élève distillé
HIDDEN_ELEVE = 16                # 3 → 16 → 16 → 1, même entrée et même sortie
POIDS_REFERENCE = Path(__file__).with_name("pendule_acteur.pt")
POIDS_ELEVE = Path(__file__).with_name("pendule_eleve.pt")
POIDS_ELEVE_INT8 = Path(__file__).with_name("pendule_eleve_int8.pt")
EVAL_EPISODES, EVAL_SEED = 50, 123


def prep(obs):
    """Observation brute → entrée du réseau. Seule modification : θ̇ / 8."""
    o = torch.tensor(np.array(obs), dtype=torch.float32)
    o[..., 2] /= VITESSE_MAX
    return o


def reseau_reference():
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
        nn.Linear(HIDDEN, ACTION_DIM),
    )


def charger_reference(chemin=POIDS_REFERENCE):
    if not Path(chemin).exists():
        raise SystemExit("Pas de poids enregistrés. Lance d'abord : python ppo.py")
    reseau = reseau_reference()
    reseau.load_state_dict(torch.load(chemin, weights_only=True))
    reseau.eval()
    return reseau


def reseau_eleve():
    return nn.Sequential(
        nn.Linear(OBS_DIM, HIDDEN_ELEVE), nn.Tanh(),
        nn.Linear(HIDDEN_ELEVE, HIDDEN_ELEVE), nn.Tanh(),
        nn.Linear(HIDDEN_ELEVE, ACTION_DIM),
    )


def charger_eleve(chemin=POIDS_ELEVE):
    if not Path(chemin).exists():
        raise SystemExit("Pas d'élève enregistré. Lance d'abord : python distiller.py")
    reseau = reseau_eleve()
    reseau.load_state_dict(torch.load(chemin, weights_only=True))
    reseau.eval()
    return reseau


def politique(reseau, obs):
    """Observation brute (3,) → couple avant clip."""
    with torch.no_grad():
        return float(reseau(prep(obs)[None]).item())


def evaluer(reseau, n=EVAL_EPISODES, seed=EVAL_SEED):
    """Retour moyen. Politique déterministe, même graine que la PPO."""
    from pendule_env import Pendule
    env, retours = Pendule(seed=seed), []
    for _ in range(n):
        o, total, fini = env.reset(), 0.0, False
        while not fini:
            u = politique(reseau, o)
            o, r, fini = env.step(u)
            total += r
        retours.append(total)
    return float(np.mean(retours))
