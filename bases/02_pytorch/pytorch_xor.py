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