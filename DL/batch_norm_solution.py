"""
Solution: batch normalization from scratch using PyTorch tensors.

Key insight:
  - Training   → use batch mean/var, update running stats.
  - Inference  → use running mean/var, no update.
  - Running stats use an exponential moving average with momentum.
  - For 4-D (N,C,H,W), average over N, H, W — each channel is normalised
    independently, which is why the stats have shape (C,).
"""

import torch
from torch import nn


def batch_norm_2d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Batch norm for (N, D) input."""
    if training:
        mean = x.mean(dim=0)                        # (D,)
        var  = x.var(dim=0, unbiased=False)         # (D,)

        # EMA update of running stats — detach so no gradient flows into them
        running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
        running_var .mul_(1 - momentum).add_(momentum * var .detach())
    else:
        mean = running_mean
        var  = running_var

    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta


def batch_norm_4d(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    training: bool = True,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Batch norm for (N, C, H, W) input."""
    N, C, H, W = x.shape

    if training:
        # average over batch + spatial dims, keep channel dim
        mean = x.mean(dim=(0, 2, 3))                    # (C,)
        var  = x.var(dim=(0, 2, 3), unbiased=False)     # (C,)

        running_mean.mul_(1 - momentum).add_(momentum * mean.detach())
        running_var .mul_(1 - momentum).add_(momentum * var .detach())
    else:
        mean = running_mean
        var  = running_var

    # reshape for broadcasting over (N, C, H, W)
    mean = mean.view(1, C, 1, 1)
    var  = var .view(1, C, 1, 1)

    x_norm = (x - mean) / torch.sqrt(var + eps)

    # gamma/beta are (C,) — reshape to (1, C, 1, 1) for broadcasting
    return gamma.view(1, C, 1, 1) * x_norm + beta.view(1, C, 1, 1)


# ── tiny integration test ─────────────────────────────────────────────────────

def test_against_pytorch():
    torch.manual_seed(0)
    N, C, H, W = 8, 4, 6, 6
    x = torch.randn(N, C, H, W)

    ref_bn  = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
    ref_out = ref_bn(x)

    gamma = ref_bn.weight.detach().clone()
    beta  = ref_bn.bias.detach().clone()
    running_mean = torch.zeros(C)
    running_var  = torch.ones(C)
    my_out = batch_norm_4d(x, gamma, beta, running_mean, running_var, training=True)

    max_diff = (ref_out - my_out).abs().max().item()
    print(f"Max difference vs nn.BatchNorm2d: {max_diff:.2e}")   # < 1e-5


if __name__ == "__main__":
    test_against_pytorch()
