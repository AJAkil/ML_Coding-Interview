import math
import torch
import torch.nn.functional as F
from typing import Tuple


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Provided: Softmax function."""
    return torch.softmax(x, dim=dim)


def layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Apply layer normalization using PyTorch tensors.
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - mean) / torch.sqrt(var + eps)
    return gamma * normalized + beta


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Returns a (seq_len, seq_len) boolean mask.
    True  → position should be masked out (future token, set to -inf).
    False → position is visible (past or current token).
    torch.triu(diagonal=1) gives the strict upper triangle = future positions.
    """
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Reshape (B, S, D) -> (B, H, S, D/H)."""
    batch_size, seq_len, d_model = x.shape
    head_dim = d_model // num_heads
    x = x.view(batch_size, seq_len, num_heads, head_dim)
    return x.permute(0, 2, 1, 3)


def _combine_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape (B, H, S, d) -> (B, S, H*d)."""
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(batch_size, seq_len, num_heads * head_dim)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Scaled dot-product attention.
    """
    q_proj = Q @ W_q
    k_proj = K @ W_k
    v_proj = V @ W_v
    scores = q_proj @ k_proj.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    return attn @ v_proj

def multi_head_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Multi-head attention forward pass (scaled dot-product).
    Q, K, V: (batch, seq, d_model)
    W_*: (d_model, d_model)
    causal: if True, applies a causal mask (used in decoder self-attention).
    """
    d_model = Q.size(-1)
    head_dim = d_model // num_heads

    q_proj = Q @ W_q
    k_proj = K @ W_k
    v_proj = V @ W_v

    q_heads = _split_heads(q_proj, num_heads)
    k_heads = _split_heads(k_proj, num_heads)
    v_heads = _split_heads(v_proj, num_heads)

    scores = q_heads @ k_heads.transpose(-2, -1)   # (B, h, T, T)
    scores = scores / math.sqrt(head_dim)

    if causal:
        T = Q.shape[1]
        mask = _causal_mask(T, Q.device)            # (T, T), broadcasts over (B, h, T, T)
        scores = scores.masked_fill(mask, float("-inf"))

    attn = torch.softmax(scores, dim=-1)

    context = attn @ v_heads
    combined = _combine_heads(context)
    return combined @ W_o


def feed_forward(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """
    Position-wise feed-forward network with ReLU.
    """
    hidden = F.relu(x @ W1 + b1)
    return hidden @ W2 + b2


def encoder_block(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    gamma1: torch.Tensor,
    beta1: torch.Tensor,
    gamma2: torch.Tensor,
    beta2: torch.Tensor,
    num_heads: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Encoder block: MHA + FFN with residual connections and layer norms.
    """
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1, eps)
    ff_out = feed_forward(x, W1, b1, W2, b2)
    return layer_norm(x + ff_out, gamma2, beta2, eps)
