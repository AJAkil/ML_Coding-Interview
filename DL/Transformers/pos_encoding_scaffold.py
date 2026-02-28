"""Practice scaffold: sinusoidal positional encoding."""

import math
import torch


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding of shape (seq_len, d_model).

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Steps:
      1. Create a zero tensor pe of shape (seq_len, d_model).
      2. Create a position column vector of shape (seq_len, 1).
      3. Compute div_term of shape (d_model/2,) using log-space for stability:
             div_term = exp(arange(0, d_model, 2) * (-log(10000) / d_model))
      4. Fill even columns (0::2) with sin(position * div_term).
      5. Fill odd  columns (1::2) with cos(position * div_term).
      6. Return pe.
    """
    # TODO: implement sinusoidal positional encoding
    pe = torch.zeros((seq_len, d_model), dtype=torch.float32) # (T, d)
    position = torch.arange(seq_len).unsqueeze(1).float() # (T, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


def add_positional_encoding(x: torch.Tensor) -> torch.Tensor:
    """
    Add sinusoidal PE to input tensor x of shape (B, T, d_model).
    The PE should NOT require gradients.
    """
    # TODO: call sinusoidal_positional_encoding and add to x
    # Hint: pe lives on CPU by default — move it to x.device
    raise NotImplementedError


# ── quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    pe = sinusoidal_positional_encoding(seq_len=10, d_model=16)
    print("PE shape:", pe.shape)          # expect (10, 16)
    print("PE[:3, :4]:\n", pe[:3, :4])   # sin/cos pattern visible

    x = torch.randn(2, 10, 16)
    out = add_positional_encoding(x)
    print("Output shape:", out.shape)    # expect (2, 10, 16)
