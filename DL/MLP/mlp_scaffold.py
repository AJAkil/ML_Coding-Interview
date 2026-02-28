"""Practice scaffold for building an MLP classifier with PyTorch."""

from math import fabs
from pdb import run
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyTabularDataset(Dataset):
    """Randomly generated tabular features and labels."""

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
        # TODO: build a stack of Linear -> activation layers
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward pass through the MLP
        return self.net(x)
        

def train(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device, epochs: int = 5) -> None:
    """Standard supervised training loop."""
    # TODO: implement epoch loop, backprop, and logging
    for epoch in epochs:
        model.train()
        running_loss = 0.0
        total_data = 0

        for feature, label in dataloader:
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            logit = model(feature)
            loss = criterion(logit, label)
            loss.backward()
            running_loss += loss.item() * feature.size(0)
            total_data += feature.size(0)
        
        avg_loss = running_loss / total_data 
        acc = evaluate(model, dataloader, device)


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Computes accuracy on dummy data."""
    # TODO: eval mode + argmax accuracy
    model.eval()
    correct = 0 
    total = 0
    with torch.no_grad():
        for feature, label in dataloader:
            feature, label = feature.to(device), label.to(device)
            logits = model(feature) #(n, num_classes)
            
            # get the predction now, no need for loss
            preds = logits.argmax(dim=1) #(n,1)
            correct = (preds == label).sum().item()
            total += feature.size(0)
        
    return correct / max(total, 1)


# def run_experiment():
#     device = make_device()
#     dataset = DummyTabularDataset()
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     model = SimpleMLP().to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.05)

#     train(model, dataloader, criterion=criterion, optimizer=optim, device=device)
#     acc = evaluate(model, dataloader, device)
#     print()

def run_experiment():
    device = make_device()
    dataset = DummyTabularDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)





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
