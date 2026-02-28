"""Practice scaffold for building vanilla RNN, GRU, and LSTM classifiers in PyTorch."""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def make_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummySequenceDataset(Dataset):
    """Random float sequences with categorical labels."""

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
        # TODO: define an nn.RNN layer and a linear classifier head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: run the sequence through the RNN and map final hidden state to logits
        raise NotImplementedError


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        # TODO: define an nn.GRU + classifier head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward computation
        raise NotImplementedError


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 32, num_classes: int = 5) -> None:
        super().__init__()
        # TODO: define an nn.LSTM + classifier head
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward computation
        raise NotImplementedError


def train_sequence_model(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device, epochs: int = 4) -> None:
    """Training loop shared by all recurrent models."""
    # TODO: implement epoch loop with forward/backward updates
    raise NotImplementedError


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Computes accuracy for a sequence model."""
    # TODO: eval mode + argmax accuracy
    raise NotImplementedError


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
