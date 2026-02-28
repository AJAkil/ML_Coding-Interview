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
    # TODO: implement layer normalization
    # get the mean from the last dimension
    x_mean = x.mean(dim=-1, keepdim=True)
    x_var = x.var(dim=-1, unbiased=False, keepdim=True)
    normalized = (x - x_mean) / (torch.sqrt(x_var + eps))

    return gamma * normalized + beta 


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_dim, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    # then reshape the x tensor from (B, N, d) to (B, h, N, d_k)
    x = x.view(batch_dim, seq_len, num_heads, head_dim)
    x = x.transpose(1, 2).contiguous()
    return x

def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Build a causal (look-ahead) mask of shape (seq_len, seq_len).
    Positions where mask is True should be set to -inf before softmax.
    Hint: torch.triu with diagonal=1 gives you the upper triangle.
    """
    # TODO: return a boolean tensor of shape (seq_len, seq_len)
    # True  → this position should be masked (future tokens)
    # False → this position is allowed (current and past tokens)
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1
    )


def _combine_heads(x: torch.Tensor) -> torch.Tensor:
    """Reshape (B, h, N, d_k) to (B, N, h*d_k)"""
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x.transpose(1, 2).contiguous()  # (B, N, h, d_k)
    x = x.view(batch_size, seq_len, num_heads * head_dim)
    return x


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
    Multi-head attention forward pass.
    If causal=True, applies a causal mask (decoder self-attention).
    """
    # Q (batch, seq_len, d_model)
    # same for K and V, W_q and others (d_model, d_model)
    d_model = Q.shape[-1]
    head_dim = d_model // num_heads

    # first get the projections
    q_proj = Q @ W_q
    k_proj = K @ W_k
    v_proj = V @ W_v

    # then we do the head splitting parts here
    q_heads = _split_heads(q_proj, num_heads=num_heads)
    k_heads = _split_heads(k_proj, num_heads=num_heads)
    v_heads = _split_heads(v_proj, num_heads=num_heads)

    # we then compute the scores here
    scores = q_heads @ k_heads.transpose(-2, -1)
    scores = scores / math.sqrt(head_dim)  # (B, h, N, N)

    # TODO: if causal=True, apply the causal mask using _causal_mask
    # Hint: scores.masked_fill(mask, float('-inf'))
    # The mask (T, T) will broadcast automatically over (B, h, T, T)
    if causal:
        T = Q.shape[1]
        mask = _causal_mask(T, Q.device)
        scores = scores.masked_fill(mask, float('-inf'))
        
    attn = torch.softmax(scores, dim=-1)  # (B, h, N, N)

    context = attn @ v_heads  # (B,h,N,N) @ (B,h,N,d_k) = (B,h,N,d_k)

    # now we combine the heads
    combined = _combine_heads(context)  # (B, h, N, d_k) -> (B, N, d)
    return combined @ W_o  # (B, N, d) * (d,d) = (B, N, d)

    

def feed_forward(
    x: torch.Tensor,
    W1: torch.Tensor,
    b1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """
    Position-wise feed-forward network.
    """
    # TODO: implement 2-layer MLP with activation
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
    # TODO: wire together MHA, residuals, feed-forward, and norms
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x = layer_norm(x + attn_out, gamma1, beta1, eps)
    ff_out = feed_forward(x, W1, b1, W2, b2)
    x = layer_norm(x + ff_out, gamma2, beta2, eps)
    return x
