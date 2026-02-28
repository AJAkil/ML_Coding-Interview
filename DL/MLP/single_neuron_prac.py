import torch


def train(
        features, labels, initial_weights, initial_bias, learning_rate, epochs
):
    X = torch.as_tensor(features, dtype=torch.float) # (n,m)
    y = torch.as_tensor(labels, dtype=torch.float) # (n)
    W = torch.as_tensor(initial_weights, dtype=torch.float) #(m,1)
    b = torch.as_tensor(initial_bias, dtype=torch.float) #(1)

    n = X.shape[0]
    mse_values = []

    for _ in range(epochs):
        Z = X @ W + b # (n,m) @ (m,1) + 1 = n
        y_hat = torch.sigmoid(Z) # (n)
        error = y_hat - y # (n)
        grad_w = (1/n) * X.T @ error #(m,n) @(n) = (m,1)
        grad_b = (1/n) * torch.sum(error) #(1)

        mse = torch.mean(error**2) # (1)
        mse_values.append(mse)
