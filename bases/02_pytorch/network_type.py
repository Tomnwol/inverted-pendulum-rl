import numpy as np
import torch
import torch.nn as nn

def spirales(n, seed):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.25, 1, n) * 4 * np.pi
    r = t / (4 * np.pi)
    x1 = np.c_[r * np.cos(t), r * np.sin(t)]
    x2 = -x1
    X = np.vstack([x1, x2]) + rng.normal(0, 0.03, (2 * n, 2))
    y = np.r_[np.zeros(n), np.ones(n)]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

X_train, y_train = spirales(300, seed=0)
X_test, y_test = spirales(300, seed=1)

def construire(largeurs):
    couches, entree = [], 2
    for l in largeurs:
        couches += [nn.Linear(entree, l), nn.ReLU()]
        entree = l
    couches.append(nn.Linear(entree, 2))
    return nn.Sequential(*couches)

def entrainer(largeurs, seed, epochs=3000):
    torch.manual_seed(seed)
    model = construire(largeurs)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        loss = loss_fn(model(X_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_tr = (model(X_train).argmax(1) == y_train).float().mean().item()
        acc_te = (model(X_test).argmax(1) == y_test).float().mean().item()
    nb = sum(p.numel() for p in model.parameters())
    return nb, acc_tr, acc_te

for largeurs in ([125], [16, 16, 16], [10, 10, 10, 10, 10]):
    for seed in range(3):
        nb, tr, te = entrainer(largeurs, seed)
        print(f"{str(largeurs):24s} | {nb:4d} params | seed {seed} | train {tr:.3f} | test {te:.3f}")