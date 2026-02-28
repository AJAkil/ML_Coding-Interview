import torch
from typing import List, Tuple, Union


def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels:   Union[List[float],      torch.Tensor],
    initial_weights: Union[List[float], torch.Tensor],
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> Tuple[List[float], float, List[float]]:
    """
    Train a single neuron (sigmoid activation) with mean-squared-error loss.

    Returns (updated_weights, updated_bias, mse_per_epoch)
    Ã¢ÂÂ weights & bias are rounded to 4 decimals; each MSE value is rounded too.
    """
    # Your implementation here
    X = torch.as_tensor(features, dtype=torch.float) #(n,m)
    y = torch.as_tensor(labels, dtype=torch.float) # n
    w = torch.as_tensor(initial_weights, dtype=torch.float) #(m,1)
    b = torch.as_tensor(initial_bias, dtype=torch.float) #(1)

    n = X.shape[0]
    mse_values = []

    for _ in range(epochs):
        # forward pass
        z = X @ w + b #(n,m) * (m,1) + (1) # n
        a = torch.sigmoid(z) # n
        error = a - y # n
        grad_w = (1/n) * X.T @ error #(m,n) * (n,1) = (m,1)
        grad_b = (1/n) * torch.sum(error) #(1)
        w = w - learning_rate * grad_w #(m,1)
        b = b - learning_rate * grad_b #(1)
        mse = torch.mean(error**2) #(1)
        mse_values.append(mse.item())

    return w.round(4).tolist(), b.round(4).item(), [round(mse, 4) for mse in mse_values]
       