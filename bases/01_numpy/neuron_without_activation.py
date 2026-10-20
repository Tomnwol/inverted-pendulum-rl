import numpy as np

# Données : y = -3x + 5
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([5, 2, -1, -4, -7], dtype=float)

#Bruit
y = y + np.random.randn(5) * 0.5
print(y)

w = 0.0
b = 0.0
lr = 0.1

for epoch in range(101):
    # 1) Forward
    # Prédiction, au début elle est mauvaise, elle vaut [0., 0., 0., 0., 0.]
    y_hat = w * x + b
    

    # 2) Loss : erreur quadratique moyenne (MSE)
    # Cette mauvaise prédiction entraine une grande erreur (loss)
    loss = np.mean((y_hat - y) ** 2)

    # 3) Gradients
    # Gradients sont les dérivées de la fonction de loss par rapport à w et b
    # Détail du calcul : 
    #  (y_hat - y) * x est la dérivée de la fonction (y_hat - y)² par rapport à w
    #  (y_hat - y) est la dérivée de la fonction (y_hat - y)² par rapport à b
    dw = np.mean(2 * (y_hat - y) * x)
    db = np.mean(2 * (y_hat - y))

    # 4) Mise à jour
    w -= lr * dw
    b -= lr * db

    if epoch % 10 == 0:
        print(f"Époque {epoch:3d} | loss = {loss:.5f} | w = {w:.8f} | b = {b:.3f}")