"""Reference implementation for the RNN practice scaffold."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummySequenceDataset(Dataset):
    def __init__(self, num_samples: int = 400, seq_len: int = 15, feature_dim: int = 8, num_classes: int = 5) -> None:
        self.data = torch.randn(num_samples, seq_len, feature_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


class VanillaRNNClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.rnn(x)
        final_state = outputs[:, -1, :]
        return self.head(final_state)


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.gru(x)
        final_state = outputs[:, -1, :]
        return self.head(final_state)


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(x)
        final_state = outputs[:, -1, :]
        return self.head(final_state)

def train_sequence_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device, epochs: int = 4) -> None:
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total = 0
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)
            total += sequences.size(0)
        avg_loss = running_loss / total
        acc = evaluate(model, dataloader, device)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.2%}")


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)
            logits = model(sequences)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += sequences.size(0)
    return correct / max(total, 1)


def run_all_models() -> None:
    device = make_device()
    dataset = DummySequenceDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    configs = [
        ("RNN", VanillaRNNClassifier),
        ("GRU", GRUClassifier),
        ("LSTM", LSTMClassifier),
    ]

    for name, cls in configs:
        print(f"\nTraining {name} model")
        model = cls().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-3)
        train_sequence_model(model, dataloader, criterion, optimizer, device)
        acc = evaluate(model, dataloader, device)
        print(f"{name} accuracy: {acc:.2%}")


if __name__ == "__main__":
    run_all_models()
