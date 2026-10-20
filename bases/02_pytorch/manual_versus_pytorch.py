import torch
import torch.nn as nn

torch.manual_seed(0)

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Même architecture : 2 -> 4 -> 1, sigmoïde partout
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()                                   # entropie croisée binaire
optimizer = torch.optim.SGD(model.parameters(), lr=1.0)

for epoch in range(10001):
    y_hat = model(X)                # 1) Forward
    loss = loss_fn(y_hat, y)        # 2) Loss

    optimizer.zero_grad()           # remet les gradients à zéro
    loss.backward()                 # 3) Gradients (rétropropagation automatique)
    optimizer.step()                # 4) Mise à jour : w -= lr * gradient

    if epoch % 1000 == 0:
        print(f"Époque {epoch:5d} | loss = {loss.item():.4f}")

print(model(X).detach().round(decimals=3).squeeze())

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(2, 4), nn.Sigmoid(), nn.Linear(4, 1), nn.Sigmoid())
loss_fn = nn.BCELoss()

# --- Gradients de PyTorch ---
loss = loss_fn(model(X), y)
loss.backward()

W1, b1 = model[0].weight, model[0].bias
W2, b2 = model[2].weight, model[2].bias

# --- Tes formules manuelles ---
with torch.no_grad():
    a1 = torch.sigmoid(X @ W1.T + b1)
    y_hat = torch.sigmoid(a1 @ W2.T + b2)

    dz2 = (y_hat - y) / len(y)
    dW2 = dz2.T @ a1                 # (1,4)
    db2 = dz2.sum(dim=0)             # (1,)
    dz1 = (dz2 @ W2) * a1 * (1 - a1) # (4,4)
    dW1 = dz1.T @ X                  # (4,2)
    db1 = dz1.sum(dim=0)             # (4,)

print("dW1 :", torch.allclose(dW1, W1.grad, atol=1e-6))
print("db1 :", torch.allclose(db1, b1.grad, atol=1e-6))
print("dW2 :", torch.allclose(dW2, W2.grad, atol=1e-6))
print("db2 :", torch.allclose(db2, b2.grad, atol=1e-6))