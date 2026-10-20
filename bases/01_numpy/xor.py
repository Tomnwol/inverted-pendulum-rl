import numpy as np

np.random.seed(0)

# --- Données : table de vérité du XOR ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- Architecture : 2 entrées -> 4 neurones cachés -> 1 sortie ---
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

lr = 1.0          # taux d'apprentissage
epochs = 1000

for epoch in range(epochs):
    # 1) Forward
    a1 = sigmoid(X @ W1 + b1)          # couche cachée
    y_hat = sigmoid(a1 @ W2 + b2)      # sortie (probabilité)

    # 2) Loss : entropie croisée binaire
    eps = 1e-9 # pour éviter les log(0)
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


    # 3) Backward (rétropropagation)
    n = X.shape[0]
    dz2 = (y_hat - y) / n              # gradient à la sortie
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1) # on remonte à travers la couche cachée
    dW1 = X.T @ dz1
    db1 = dz1.sum(axis=0, keepdims=True)

    # 4) Mise à jour : descente de gradient
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"\n\nÉpoque {epoch:5d}")
        print("--------------------------------")
        print("a1: \n", a1)
        print("y_hat: \n", y_hat)
        print(f"loss = {loss:.4f}")
        print("W1: \n", W1)
        print("b1: \n", b1)
        print("W2: \n", W2)
        print("b2: \n", b2)

# --- Résultat ---
print("\nPrédictions finales :")
for xi, p in zip(X, y_hat):
    print(xi, "->", round(float(p[0]), 3))