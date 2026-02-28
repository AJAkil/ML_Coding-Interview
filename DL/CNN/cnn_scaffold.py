"""Practice scaffold: fill in the TODO blocks to build and train a simple CNN on dummy data."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyImageDataset(Dataset):
    """Generates random image-like tensors and integer labels."""

    def __init__(self, num_samples: int = 256, num_classes: int = 10, image_shape=(1, 28, 28)) -> None:
        self.data = torch.randn(num_samples, *image_shape)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]

# class CNN_(nn.Module):
#

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, hidden_channels, k, p, s):
        self.feature = nn.Sequential(
            nn.Conv2d(1, out_channels=hidden_channels, kernel_size=k, padding=p, stride=s),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        return self.classifier(x)


def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device) -> float:
    """Runs one training epoch and returns average loss."""
    # TODO: implement standard training loop (forward -> loss -> backward -> step)
    model.train()
    total = 0
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logit = model(inputs)
        loss = criterion(logit, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)

    return running_loss / total


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Returns accuracy on the provided dataloader."""
    # TODO: switch to eval mode, disable grad, compute accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad:
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

        return correct / max(1, total)


def run_training(epochs: int = 3, batch_size: int = 32) -> None:
    device = make_device()
    dataset = DummyImageDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        acc = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: loss={loss:.4f}, acc={acc:.2%}")

def run_training(epochs, batch):
    device = torch.device("cpu")
    dataset = Dataset()
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.03)

    for epoch in range(epochs):
        # do training first
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        acc = evaluate(model, dataloader, device)
        print()


if __name__ == "__main__":
    run_training()
