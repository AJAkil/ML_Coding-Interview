"""
Practice scaffold: production-quality training loop with:
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


# ── implement these ───────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Run one training epoch and return average loss.

    Steps:
      1. Set model to training mode.
      2. For each batch:
         a. Zero the gradients.
         b. Forward pass.
         c. Compute loss.
         d. Backward pass.
         e. Clip gradients with torch.nn.utils.clip_grad_norm_.
         f. Optimizer step.
      3. Return mean loss over all batches.

    Key gotcha: clip_grad_norm_ must be called AFTER loss.backward()
                and BEFORE optimizer.step().
    """
    # TODO: implement
    raise NotImplementedError


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Compute accuracy over loader.

    Key gotcha: you need BOTH model.eval() AND torch.no_grad().
      - model.eval()      → disables dropout, uses running stats in batchnorm
      - torch.no_grad()   → stops gradient tape (saves memory/compute)
    They are independent — one does NOT imply the other.
    """
    # TODO: implement
    raise NotImplementedError


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    lr: float = 1e-3,
    max_grad_norm: float = 1.0,
) -> None:
    """
    Full training loop with CosineAnnealingLR scheduler.

    Steps:
      1. Build Adam optimizer and CosineAnnealingLR(optimizer, T_max=epochs).
      2. Each epoch:
         a. Train one epoch (get loss).
         b. Step the scheduler (scheduler.step() comes AFTER optimizer.step()).
         c. Evaluate on val_loader.
         d. Print epoch, loss, val accuracy, current lr.

    Key gotcha: scheduler.step() is called once per EPOCH (not per batch here).
                Use scheduler.get_last_lr()[0] to read the current lr.
    """
    criterion = nn.CrossEntropyLoss()
    # TODO: implement
    raise NotImplementedError


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_loaders()
    model = make_model().to(device)
    train(model, train_loader, val_loader, device, epochs=10)
