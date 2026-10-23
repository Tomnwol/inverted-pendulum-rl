"""Courbes de retour moyen à partir du tableau de resultats.md."""
import re
from pathlib import Path

import matplotlib.pyplot as plt

SEUIL = -300
COLONNES = (
    "Recherche aléatoire",
    "REINFORCE",
    "Monte Carlo",
    "Différence temporelle",
    "PPO",
)
COULEURS = ("#7a5c3e", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")

lignes = Path(__file__).with_name("resultats.md").read_text().splitlines()
budgets, series = [], [[] for _ in COLONNES]
for ligne in lignes:
    cellules = [c.strip() for c in ligne.strip().strip("|").split("|")]
    if not cellules or not cellules[0].isdigit():
        continue
    budgets.append(int(cellules[0]))
    for i, cellule in enumerate(cellules[1:]):
        nombre = re.search(r"-?\d+(?:\.\d+)?", cellule)
        series[i].append(float(nombre.group()) if nombre and "NOSIM" not in cellule else None)

fig, ax = plt.subplots(figsize=(11, 6))
for nom, ys, couleur in zip(COLONNES, series, COULEURS):
    xs = [x for x, y in zip(budgets, ys) if y is not None]
    ys = [y for y in ys if y is not None]
    ax.plot(xs, ys, marker="o", linewidth=2, color=couleur, label=nom)

ax.axhline(SEUIL, color="0.25", linestyle="--", linewidth=1.2, label="seuil d'entraînement (−300)")
ax.axhspan(SEUIL, 0, color="#2ca02c", alpha=0.08, zorder=0)

ax.set_xscale("symlog", linthresh=50)
ax.set_xlim(-2, 60000)
ax.set_ylim(-1600, 0)
ax.set_xlabel("Itérations")
ax.set_ylabel("Retour moyen")
ax.set_title("Comparaison des entraînements")
ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()

sortie = Path(__file__).with_name("results.png")
fig.savefig(sortie, dpi=150, bbox_inches="tight")
print(f"graphique enregistré : {sortie}")
