"""Reference implementation for the MLP practice scaffold."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyTabularDataset(Dataset):
    def __init__(self, num_samples: int = 512, num_features: int = 20, num_classes: int = 4) -> None:
        self.features = torch.randn(num_samples, num_features)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


class SimpleMLP(nn.Module):
    def __init__(self, in_dim: int = 20, hidden_dim: int = 64, num_classes: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device, epochs: int = 5) -> None:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            total += features.size(0)
        avg_loss = running_loss / total
        acc = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.2%}")


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += features.size(0)
    return correct / max(total, 1)


def run_experiment() -> None:
    device = make_device()
    dataset = DummyTabularDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    train(model, dataloader, criterion, optimizer, device)
    acc = evaluate(model, dataloader, device)
    print(f"Final accuracy: {acc:.2%}")


if __name__ == "__main__":
    run_experiment()
