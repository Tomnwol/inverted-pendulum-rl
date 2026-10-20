import numpy as np

# Données : y = x² - 3x + 5
x = np.array([0, 1, 2, 3, 4], dtype=float)
y = np.array([5, 3, 3, 5, 9], dtype=float)


print(y)

w1 = 0.0
w2 = 0.0
b = 0.0
lr = 0.01

for epoch in range(10001):
    # 1) Forward
    # Prédiction, au début elle est mauvaise, elle vaut [0., 0., 0., 0., 0.]
    y_hat = w1 * x**2 + w2 * x + b
    

    # 2) Loss : erreur quadratique moyenne (MSE)
    # Cette mauvaise prédiction entraine une grande erreur (loss)
    loss = np.mean((y_hat - y) ** 2)

    # 3) Gradients
    # Gradients sont les dérivées de la fonction de loss par rapport à w et b
    # Détail du calcul : 
    #  (y_hat - y) * x est la dérivée de la fonction (y_hat - y)² par rapport à w1
    #  (y_hat - y) * x est la dérivée de la fonction (y_hat - y)² par rapport à w2
    #  (y_hat - y) est la dérivée de la fonction (y_hat - y)² par rapport à b
    dw1 = np.mean(2 * (y_hat - y) * x**2)
    dw2 = np.mean(2 * (y_hat - y) * x)
    db = np.mean(2 * (y_hat - y))

    # 4) Mise à jour
    w1 -= lr * dw1
    w2 -= lr * dw2
    b -= lr * db

    if epoch % 10 == 0:
        print(f"Époque {epoch:3d} | loss = {loss:.5f} | w1 = {w1:.8f} | w2 = {w2:.8f} | b = {b:.3f}")   