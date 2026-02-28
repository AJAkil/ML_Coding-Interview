"""
Practice scaffold: batch normalization from scratch using PyTorch tensors.

Batch norm has two distinct behaviours:
  Training   → normalize using the CURRENT BATCH mean/var,
               then update running statistics with momentum.
  Inference  → normalize using the RUNNING mean/var (no update).

Supports:
  - 2-D input (N, D)       → normalize over the batch dim N
  - 4-D input (N, C, H, W) → normalize over N, H, W (keep C distinct)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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
    """
    Batch norm for 2-D input x of shape (N, D).

    Steps when training=True:
      1. Compute batch mean  over dim=0  → shape (D,)
      2. Compute batch var   over dim=0  (unbiased=False)  → shape (D,)
      3. Update running stats (in-place):
             running_mean = (1 - momentum) * running_mean + momentum * batch_mean
             running_var  = (1 - momentum) * running_var  + momentum * batch_var
         Hint: use .mul_() and .add_() or direct assignment.
         Hint: detach the batch stats before storing (no grad through running stats).
      4. Normalise: x_norm = (x - mean) / sqrt(var + eps)
      5. Scale and shift: return gamma * x_norm + beta

    Steps when training=False:
      1. Use running_mean and running_var (do NOT update them).
      2. Normalise and scale/shift as above.
    """
    # TODO: implement
    raise NotImplementedError


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
    """
    Batch norm for 4-D input x of shape (N, C, H, W).

    Same as batch_norm_2d but statistics are computed over dims (0, 2, 3),
    giving mean/var of shape (C,).

    Key difference from 2-D: after computing mean/var you need to reshape
    them to (1, C, 1, 1) so they broadcast correctly over (N, C, H, W).
    """
    # TODO: implement
    raise NotImplementedError


# ── tiny integration test ─────────────────────────────────────────────────────

def test_against_pytorch():
    """Compare your implementation against nn.BatchNorm2d."""
    torch.manual_seed(0)
    N, C, H, W = 8, 4, 6, 6
    x = torch.randn(N, C, H, W)

    # reference
    ref_bn = nn.BatchNorm2d(C, eps=1e-5, momentum=0.1)
    ref_out = ref_bn(x)

    # yours
    gamma = ref_bn.weight.detach().clone()
    beta  = ref_bn.bias.detach().clone()
    running_mean = torch.zeros(C)
    running_var  = torch.ones(C)
    my_out = batch_norm_4d(x, gamma, beta, running_mean, running_var, training=True)

    max_diff = (ref_out - my_out).abs().max().item()
    print(f"Max difference vs nn.BatchNorm2d: {max_diff:.2e}")   # should be < 1e-5


if __name__ == "__main__":
    test_against_pytorch()
