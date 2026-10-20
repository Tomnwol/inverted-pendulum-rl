import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Pour chaque activation : (fonction, dérivée locale)
activations = {
    "sigmoid": (sigmoid,                lambda z, a: a * (1 - a)),
    "tanh":    (np.tanh,                lambda z, a: 1 - a ** 2),
    "relu":    (lambda z: np.maximum(0, z), lambda z, a: (z > 0).astype(float)),
}

def train(name, seed, hidden=4, lr=0.5, epochs=3000):
    f, df = activations[name]
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((2, hidden)); b1 = np.zeros((1, hidden))
    W2 = rng.standard_normal((hidden, 1)); b2 = np.zeros((1, 1))
    eps = 1e-9

    for epoch in range(epochs):
        # Forward
        z1 = X @ W1 + b1
        a1 = f(z1)
        y_hat = sigmoid(a1 @ W2 + b2)       # la sortie reste une sigmoïde (probabilité)

        # Loss
        loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

        # Backward
        dz2 = (y_hat - y) / len(y)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ W2.T) * df(z1, a1)     # ← seule ligne qui dépend de l'activation
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)

        # Mise à jour
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1

    # À la fin de train(), remplace "return loss" par :
    return loss, y_hat, a1, z1

    # Puis, pour les graines qui échouent :


# Comparaison : 5 graines différentes par activation
for name in activations:
    loss, y_hat, a1, z1 = train("relu", seed=4)
    print("y_hat :", y_hat.round(3).ravel())
    print("activations cachées :\n", a1.round(2))
    print("neurones qui valent 0 partout :", (a1.sum(axis=0) == 0).sum())
    #resultats = [train(name, seed) for seed in range(5)]
    #print(f"{name:8s}", " ".join(f"{r:.4f}" for r in resultats))