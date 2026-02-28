"""Solution: sinusoidal positional encoding."""

import math
import torch


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding of shape (seq_len, d_model).

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Why log-space for div_term?
        10000^(2i/d) = exp(2i/d * log(10000))
        So 1 / 10000^(2i/d) = exp(-2i/d * log(10000))
        This avoids computing very large intermediate numbers.
    """
    pe = torch.zeros(seq_len, d_model)

    position = torch.arange(seq_len).unsqueeze(1).float()          # (T, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )                                                               # (d_model/2,)

    pe[:, 0::2] = torch.sin(position * div_term)   # even dims: sin
    pe[:, 1::2] = torch.cos(position * div_term)   # odd  dims: cos

    return pe   # (T, d_model)


def add_positional_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    Add sinusoidal PE to input tensor x of shape (B, T, d_model).
    PE does not participate in gradient computation.
    """
    _, seq_len, d_model = x.shape
    pe = sinusoidal_positional_encoding(seq_len, d_model).to(x.device)
    return x + pe   # broadcasts over the batch dim


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    print("PE shape:", pe.shape)          # (10, 16)
    print("PE[:3, :4]:\n", pe[:3, :4])

    x = torch.randn(2, 10, 16)
    out = add_positional_encoding(x)
    print("Output shape:", out.shape)    # (2, 10, 16)
