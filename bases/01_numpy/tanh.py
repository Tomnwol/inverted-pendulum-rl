import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(params, x):
    w1, b1, w2, b2 = params
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    y_hat = w2 * a1 + b2
    return z1, a1, y_hat

def loss_fn(params, x, y):
    _, _, y_hat = forward(params, x)
    return (y_hat - y) ** 2


x = np.linspace(-2, 2, 20)
y = np.tanh(2 * x)

params = [0.5, 0.0, 0.5, 0.0]       # w1, b1, w2, b2
lr = 0.1

for epoch in range(20001):
    w1, b1, w2, b2 = params

    # Forward (sur les 20 points en même temps)
    z1, a1, y_hat = forward(params, x)

    # Loss : MSE
    loss = np.mean((y_hat - y) ** 2)

    # Backward : mêmes formules, avec une moyenne sur les exemples
    dy_hat = 2 * (y_hat - y)
    dw2 = np.mean(dy_hat * a1)
    db2 = np.mean(dy_hat)
    da1 = dy_hat * w2
    dz1 = da1 * a1 * (1 - a1)
    dw1 = np.mean(dz1 * x)
    db1 = np.mean(dz1)

    # Mise à jour
    params = [w1 - lr * dw1, b1 - lr * db1, w2 - lr * dw2, b2 - lr * db2]

    if epoch % 2000 == 0:
        print(f"Époque {epoch:5d} | loss = {loss:.5f}")

print("\nParamètres appris (w1, b1, w2, b2) :", np.round(params, 2))