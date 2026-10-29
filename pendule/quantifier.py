"""Quantifie l'élève en int8 et mesure l'écart avec le float32.

    python quantifier.py

Quantification symétrique par tenseur. L'échelle de chaque tenseur vient
du maximum absolu vu sur les trajectoires du distiller : un cran vaut
max / 127. Les poids et les activations sont des entiers de −127 à 127.
tanh est calculée sur la valeur reconvertie en float, puis le résultat
est à nouveau quantifié : c'est l'équivalent d'une table entière.
L'entrée préparée et le couple rendu restent des float32.
"""

import torch
import torch.nn as nn

from distiller import collecter
from interface import (
    POIDS_ELEVE_INT8,
    charger_eleve,
    charger_reference,
    evaluer,
)

CRAN_MAX = 127


class EleveInt8(nn.Module):
    """Même signature que l'élève float32 : float préparé en entrée, couple en sortie."""

    def __init__(self, blob):
        super().__init__()
        self.n = len(blob["poids"])
        for i, (poids, biais) in enumerate(zip(blob["poids"], blob["biais"])):
            self.register_buffer(f"w{i}", poids.to(torch.int8))
            self.register_buffer(f"b{i}", biais.to(torch.int32))
            self.register_buffer(f"sw{i}", blob["echelles_poids"][i].float().reshape(()))
            self.register_buffer(f"se{i}", blob["echelles_entree"][i].float().reshape(()))
            self.register_buffer(f"ss{i}", blob["echelles_sortie"][i].float().reshape(()))

    def _entier(self, valeur, echelle):
        return torch.clamp(torch.round(valeur / echelle), -CRAN_MAX, CRAN_MAX)

    def forward(self, x):
        q = self._entier(x, self.se0)
        for i in range(self.n):
            poids = getattr(self, f"w{i}").float()
            echelle_entree = getattr(self, f"se{i}")
            echelle_poids = getattr(self, f"sw{i}")
            echelle_sortie = getattr(self, f"ss{i}")
            accumulateur = nn.functional.linear(q, poids)
            couple = (accumulateur + getattr(self, f"b{i}").float()) * (echelle_entree * echelle_poids)
            couple = self._entier(couple, echelle_sortie) * echelle_sortie
            if i < self.n - 1:
                q = self._entier(torch.tanh(couple), getattr(self, f"se{i + 1}"))
            else:
                q = couple
        return q


def _echelle(tenseur):
    maximum = tenseur.detach().abs().amax().clamp(min=1e-8)
    return maximum / CRAN_MAX


def _entier_scale(tenseur, echelle):
    return torch.clamp(torch.round(tenseur / echelle), -CRAN_MAX, CRAN_MAX)


def _biais_entier(biais, echelle):
    """Le biais est ajouté dans l'accumulateur int32, pas codé sur 8 bits."""
    return torch.round(biais / echelle).to(torch.int32)


def quantifier(eleve, entrees):
    """Calibre les échelles sur `entrees` (déjà passées par prep) et fige les entiers."""
    activations = [entrees]
    h = entrees
    with torch.no_grad():
        for couche in eleve:
            h = couche(h)
            activations.append(h)

    lineaires = [couche for couche in eleve if isinstance(couche, nn.Linear)]
    poids, biais = [], []
    echelles_poids, echelles_entree, echelles_sortie = [], [], []
    # activations : entrée, sortie du linéaire, tanh, sortie du linéaire, tanh, couple
    indices_entree = (0, 2, 4)
    indices_sortie = (1, 3, 5)
    for i, couche in enumerate(lineaires):
        echelle_poids = _echelle(couche.weight)
        echelle_entree = _echelle(activations[indices_entree[i]])
        echelle_sortie = _echelle(activations[indices_sortie[i]])
        echelle_acc = echelle_entree * echelle_poids
        poids.append(_entier_scale(couche.weight, echelle_poids).to(torch.int8))
        biais.append(_biais_entier(couche.bias, echelle_acc))
        echelles_poids.append(echelle_poids.reshape(()))
        echelles_entree.append(echelle_entree.reshape(()))
        echelles_sortie.append(echelle_sortie.reshape(()))
    return {
        "poids": poids,
        "biais": biais,
        "echelles_poids": echelles_poids,
        "echelles_entree": echelles_entree,
        "echelles_sortie": echelles_sortie,
    }


def charger_int8(chemin=POIDS_ELEVE_INT8):
    if not chemin.exists():
        raise SystemExit("Pas d'élève int8. Lance d'abord : python quantifier.py")
    blob = torch.load(chemin, weights_only=True)
    modele = EleveInt8(blob)
    modele.eval()
    return modele


def main():
    eleve = charger_eleve()
    print("collecte des trajectoires de calibration...")
    entrees, _ = collecter(charger_reference())
    blob = quantifier(eleve, entrees)
    torch.save(blob, POIDS_ELEVE_INT8)
    entier = charger_int8()
    with torch.no_grad():
        ecart = (eleve(entrees) - entier(entrees)).squeeze(-1).abs()
    print(f"professeur    | éval {evaluer(charger_reference()):8.1f}")
    print(f"élève float32 | éval {evaluer(eleve):8.1f}")
    print(f"élève int8    | éval {evaluer(entier):8.1f} "
          f"| écart moyen {ecart.mean():.4f} | écart max {ecart.max():.4f}")
    print(f"Élève int8 enregistré : {POIDS_ELEVE_INT8.name}")


if __name__ == "__main__":
    main()
