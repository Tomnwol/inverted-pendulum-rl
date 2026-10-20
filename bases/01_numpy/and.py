import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float) # Dimension 4x2
y = np.array([[0], [0], [0], [1]], dtype=float)   # porte ET

w = np.zeros((2, 1)) # Dimension 2x1
print("w =", w)
b = 0.0
lr = 1.0
eps = 1e-9

for epoch in range(50001):
    # 1) Forward
    z = X @ w + b # Dimension 4x1
    y_hat = sigmoid(z)

    # 2) Loss : entropie croisée binaire
    loss = -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))

    # 3) Gradients
    
    # Démonstration faisable mais complexe
    # Dérivation en chaîne : 
    #    dloss/dw = dloss/dz * dz/dw, 
    # or dloss/dz = dloss/dy_hat * dy_hat/dz

    # Calcul des termes:
    #   dloss/dy_hat = (y_hat - y) / (y_hat * (1 - y_hat))   dérivée de l'entropie croisée
    #   dy_hat/dz    = y_hat * (1 - y_hat)                   dérivée de la sigmoïde
    #   dz/dw        = x                                     z = w·x + b est linéaire
    #   dz/db        = 1
    
    # Donc finalement :
    #   dloss/dw = (y_hat - y) * x
    #   dloss/dz = y_hat - y
    
    dz = y_hat - y          # c'est pas vraiment dz mais dloss/dz
    dw = X.T @ dz / len(y)  # c'est pas vraiment dw mais dloss/dw
    db = np.mean(dz)

    # 4) Mise à jour
    w -= lr * dw
    b -= lr * db

    if epoch % 500 == 0:
        print(f"Époque {epoch:4d} | loss = {loss:.4f}")

print("\nw =", w.round(2), "| b =", round(b, 2))
for xi, p in zip(X, y_hat):
    print(xi, "->", round(p.item(), 3))