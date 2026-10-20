import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

rng = np.random.default_rng(0)

def vrai(x):
    return np.sin(2 * np.pi * x)     # la vraie loi, inconnue du modèle

# Peu de données d'entraînement, beaucoup de test, avec du bruit (σ = 0.2)
bruit = 0.5
x_train = rng.uniform(0, 1, 15)
y_train = vrai(x_train) + rng.normal(0, bruit, 15)
x_test = rng.uniform(0, 1, 200)
y_test = vrai(x_test) + rng.normal(0, bruit, 200)

def mse(a, b):
    return np.mean((a - b) ** 2)

# On ajuste des polynômes de degré 0 à 12
degres = range(0, 13)
err_train, err_test = [], []

for d in degres:
    p = Polynomial.fit(x_train, y_train, d)
    err_train.append(mse(p(x_train), y_train)) # Erreur sur les données d'entraînement
    err_test.append(mse(p(x_test), y_test))    # Erreur sur les données de test
    # La conclusion est qu'il se passe deux choses:
    # 1. Plus le modèle apprend, plus l'erreur sur les données d'entraînement diminue.
    # 2. Plus le modèle apprend, plus l'erreur sur les données de test diminue.
    # 3. Cependant, il y a un point d'inflexion où l'erreur sur les données de test augmente. C'est le surapprentissage.
    # 4. En effet, le modèle commence à apprendre le bruit des données de test.
    print(f"degré {d:2d} | train = {err_train[-1]:.4f} | test = {err_test[-1]:.4f}")

# --- Graphique 1 : erreurs en fonction du degré ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(list(degres), err_train, "o-", label="train")
plt.plot(list(degres), err_test, "o-", label="test")
plt.yscale("log")
plt.xlabel("Degré du polynôme (capacité du modèle)")
plt.ylabel("Erreur (MSE)")
plt.legend()

# --- Graphique 2 : trois modèles face aux données ---
plt.subplot(1, 2, 2)
xs = np.linspace(0, 1, 300)
plt.scatter(x_train, y_train, color="black", zorder=3, label="train")
plt.plot(xs, vrai(xs), "k--", label="vraie courbe")
for d in (1, 3, 12):
    plt.plot(xs, Polynomial.fit(x_train, y_train, d)(xs), label=f"degré {d}")
plt.ylim(-2, 2)
plt.legend()
plt.tight_layout()
plt.show()