import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
import math

def layer_norm(x, gamma, beta, eps):
    # x (B, T, d)
    x_mean = x.mean(dim=-1, keepdim=True)
    x_var = x.var(dim=1, keepdim=True, unbiased=False)
    x_ = (x - x_mean) / torch.sqrt(x_var + eps)

    return gamma * x_ + beta

def _split_heads(X, head_dim, heads):
    # (B,T,d) -> (B, h, T, d//h)
    B, T, d = X.shape
    X = X.view(B, T, heads, head_dim) # (B, T, h, d//h)
    return X.permute(0,2,1,3) # (B,h,T,d//h)


def _combine_heads(X):
    # Reshape (B, h, T, d//h) -> (B, T, d)
    B, h, T, head_dim = X.shape
    X = X.permute(0, 2, 1, 3).contigous() # without it we cant use view
    return X.view(B, T, h * head_dim)

def scaled_dot_product_attention(
        Q, K, V, W_q, W_k, W_v, head_dim
):
    # Q (B, T, d) W_q (d,d)
    q_proj = Q @ W_q # (B,T,d)@(d,d) = (B,T,d)
    k_proj = K @ W_k
    v_proj = V @ W_v

    scores = q_proj @ k_proj.transpose(-2, -1) # (B,T,d) @ (B, d, T) = (B, T, T)
    scores = scores / math.sqrt(head_dim) # (B, T, T)
    attn = torch.softmax(scores, dim=-1) # (B, T, T)
    return attn @ v_proj # (B, T, T) @ (B, T, d) = B, T, d

def _causal_mask(T, device):
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1) # (T,T)

def multi_head_attention(
        Q, K, V, W_q, W_k, W_v, W_o, heads, causal
):
    d = Q.size(-1)
    head_dim = d // heads

    q_proj = Q @ W_q # (B,T,d)@(d,d) = (B,T,d)
    k_proj = K @ W_k
    v_proj = V @ W_v

    q_heads = _split_heads(q_proj, head_dim, heads) # (B, h, T, d//h)
    k_heads = _split_heads(k_proj, head_dim, heads)
    v_heads = _split_heads(v_proj, head_dim, heads)

    scores = q_heads @ k_heads.transpose(-2, -1) # (B,h,T, d//h) @ (B, h, d//h, T) = (B,h,T,T)
    scores = scores / math.sqrt(head_dim) # cause we divided the feature dimension to heads

    if causal:
        T = Q.shape[1]
        mask = _causal_mask(T, Q.device) # (T,T) broadcasted to (B,h,T,T)
        scores = scores.masked_fill(mask, float('-inf')) # (B,h,T,T)

    attn = torch.softmax(scores, dim=-1)

    context = attn @ v_heads # (B,h,T,d//h)

    combined = _combine_heads(context) # (B, h, T, d//h) -> (B, T, d)

    return combined @ W_o # (B,T,d) @ (d,d) = (B,T,d)

def feed_forward(x, W1, b1, W2, b2):
    hidden = F.relu(x @ W1 + b1)
    return hidden @ W2 + b2

def encoder_block(x, W_q, W_k, W_v, W_o, W1, W2, b1, b2, causal, heads, beta1, gamma1, beta2, gamma2, eps):
    attn_out =multi_head_attention(x,x,x,W_q, W_k, W_v, W_o, heads, causal=causal)
    x = layer_norm(x + attn_out, gamma1, beta1, eps)
    ff_out = feed_forward(x, W1, b1, W2, b2)
    return layer_norm(x + ff_out, gamma2, beta2, eps)