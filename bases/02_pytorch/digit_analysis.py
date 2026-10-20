import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# --- Données : 1797 images 8x8, 10 classes ---
digits = load_digits()
X = digits.data / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)   # classes = entiers 0..9
y_test = torch.tensor(y_test, dtype=torch.long)

# --- Mini-batchs ---
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# --- Modèle : 64 -> 32 -> 10 (pas de softmax ici, voir plus bas) ---
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def evaluer(Xs, ys):
    with torch.no_grad():
        logits = model(Xs)
        loss = loss_fn(logits, ys).item()
        acc = (logits.argmax(dim=1) == ys).float().mean().item()
    return loss, acc

# --- Entraînement ---
for epoch in range(1, 201):
    for xb, yb in loader:                 # une époque = tous les mini-batchs
        logits = model(xb)                # 1) Forward
        loss = loss_fn(logits, yb)        # 2) Loss
        optimizer.zero_grad()
        loss.backward()                   # 3) Gradients
        optimizer.step()                  # 4) Mise à jour

    if epoch % 5 == 0:
        tr_loss, tr_acc = evaluer(X_train, y_train)
        te_loss, te_acc = evaluer(X_test, y_test)
        print(f"Époque {epoch:2d} | train: loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| test: loss {te_loss:.3f} acc {te_acc:.3f}")