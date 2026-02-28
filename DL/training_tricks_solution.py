"""
Solution: production-quality training loop with:
  - gradient clipping
  - cosine-annealing LR scheduler
  - correct eval (model.eval() + torch.no_grad())
  - per-epoch metric logging
"""

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset


# ── tiny model for testing ────────────────────────────────────────────────────
def make_model(in_dim: int = 16, hidden: int = 64, num_classes: int = 4) -> nn.Module:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, num_classes),
    )


def make_loaders(n: int = 512, in_dim: int = 16, num_classes: int = 4, batch_size: int = 64):
    X = torch.randn(n, in_dim)
    y = torch.randint(0, num_classes, (n,))
    split = int(0.8 * n)
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds   = TensorDataset(X[split:], y[split:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size),
    )


# ── implementations ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        # clip_grad_norm_ rescales all gradients so their global L2-norm <= max_grad_norm
        # must come AFTER .backward() and BEFORE .step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    # model.eval()    → batchnorm uses running stats, dropout is disabled
    # torch.no_grad() → no gradient tape; saves memory and is faster
    # They are INDEPENDENT: eval() changes forward behaviour,
    # no_grad() just stops autograd tracking.
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # CosineAnnealingLR decays lr from `lr` down to ~0 over T_max epochs
    # and optionally restarts.  scheduler.step() is called once per epoch.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm)

        # scheduler.step() must come AFTER optimizer.step() (already inside train_one_epoch)
        scheduler.step()

        val_acc = evaluate(model, val_loader, device)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | val_acc={val_acc:.2%} | lr={current_lr:.6f}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders()
    model = make_model().to(device)
    train(model, train_loader, val_loader, device, epochs=10)
