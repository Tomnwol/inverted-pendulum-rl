# Exercice tanh avec x neurones cachés

import torch
import torch.nn as nn

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

def essai(hidden, seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(2, hidden), nn.Tanh(), nn.Linear(hidden, 1), nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = nn.BCELoss()
    for _ in range(2000):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

for hidden in (1, 2, 4, 8):
    reussites = sum(essai(hidden, s) < 0.05 for s in range(20))
    print(f"{hidden} neurones cachés : {reussites}/20 réussites")