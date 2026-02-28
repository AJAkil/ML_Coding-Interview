## PyTorch tensor ops: quick gotchas & examples

Shapes below use `b` (batch), `t` (time/seq), `d` (feature), `h` (heads).

---

### `view` vs `reshape` vs `contiguous`

**Key idea**:  
- `view` only changes shape when the tensor is **contiguous in memory**.  
- `reshape` is safer: it will copy if needed.  
- After `permute` / `transpose`, call `.contiguous()` before `view`.

```python
import torch

x = torch.randn(2, 3, 4)          # (2, 3, 4), contiguous
y = x.view(2, 12)                 # OK: (2, 12)

z = x.permute(1, 0, 2)            # (3, 2, 4), non-contiguous view
# z.view(3, 8)  # <-- RuntimeError if z is not contiguous

z_contig = z.contiguous().view(3, 8)  # OK

# reshape handles both cases, may copy internally
z2 = z.reshape(3, 8)
```

**Gotcha**: If you see a `.view(...)` error complaining about contiguity, add `.contiguous()` or use `.reshape(...)`.

---

### `permute` vs `transpose`

**Key idea**:  
- `permute(*dims)` can reorder **any** number of dimensions.  
- `transpose(dim0, dim1)` just swaps **two** dimensions.  
Both return **views** (usually non-contiguous).

```python
x = torch.randn(2, 3, 4)      # (b, t, d)

# permute: arbitrary order
y = x.permute(0, 2, 1)        # (b, d, t)

# transpose: swap two dims
z = x.transpose(1, 2)         # (b, d, t), same as permute(0, 2, 1)
```

**Gotcha**: Always pair `permute` / `transpose` with `.contiguous()` before `view`:

```python
x = torch.randn(2, 4, 8)          # (b, h, d)
x = x.transpose(1, 2)             # (b, d, h), non-contiguous
x = x.contiguous().view(2, 8 * 4) # (b, d*h)
```

---

### Typical multi-head attention reshapes

From `(b, t, d_model)` to `(b, h, t, d_k)` and back:

```python
b, t, d_model, h = 2, 5, 8, 2
d_k = d_model // h
x = torch.randn(b, t, d_model)

# split heads
x_heads = x.view(b, t, h, d_k).transpose(1, 2)     # (b, h, t, d_k)

# combine heads
x_back = x_heads.transpose(1, 2).contiguous()      # (b, t, h, d_k)
x_back = x_back.view(b, t, h * d_k)                # (b, t, d_model)
```

---

### `unsqueeze` / `squeeze`

**Key idea**: add or remove singleton dimensions.

```python
x = torch.randn(3, 4)       # (3, 4)
x1 = x.unsqueeze(0)         # (1, 3, 4)
x2 = x.unsqueeze(-1)        # (3, 4, 1)

y = x2.squeeze(-1)          # (3, 4)
```

**Gotcha**: `squeeze()` with no dim removes **all** size-1 dimensions.

---

### `gather`

**Key idea**: select values along a dimension using an index tensor of the **same shape** as the output.

Common use: pick logits at target indices, or select token embeddings.

```python
# Example: pick token embeddings for specific positions
emb = torch.randn(2, 5, 4)          # (b=2, t=5, d=4)
idx = torch.tensor([[1, 3],        # positions per batch
                    [0, 4]])       # (2, 2)

# we want (b, k, d) where k = 2
# need idx to have a feature dim of size 1 for broadcasting
idx_expanded = idx.unsqueeze(-1).expand(-1, -1, emb.size(-1))  # (2, 2, 4)

selected = torch.gather(emb, dim=1, index=idx_expanded)       # (2, 2, 4)
```

**Gotcha**:  
- `index` must have the same shape as the output.  
- The values in `index` are positions along `dim`.

---

### `scatter_`

**Key idea**: write into a tensor at positions specified by an index tensor.
Great for building one-hot or probability distributions from indices.

```python
batch_size, num_classes = 3, 5
target = torch.tensor([1, 3, 0])        # class indices, (3,)

one_hot = torch.zeros(batch_size, num_classes)
one_hot.scatter_(1, target.unsqueeze(1), 1.0)
# one_hot[i, target[i]] = 1.0
```

**Gotcha**:  
- The `_` suffix means **in-place**.  
- As with `gather`, `index` must match the output shape along non-scatter dims.

---

### Broadcasting gotchas

**Key idea**: right-align shapes; dimensions of size 1 are broadcast.

```python
x = torch.randn(2, 3, 4)      # (b, t, d)
bias = torch.randn(4)         # (d,)

y = x + bias                  # (2, 3, 4), bias broadcast over (b, t)
```

**Gotcha**: Misaligned dimensions can silently broadcast in unexpected ways;
print shapes before operations when debugging.

---

### `masked_fill` + `tril` / `triu` (causal masking)

**Key idea**: build a boolean mask, then stamp `-inf` into those positions before softmax. Used in every decoder / causal attention block.

```python
T = 4
# tril: keep lower triangle (past + current); triu: upper triangle (future)
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# tensor([[False,  True,  True,  True],
#         [False, False,  True,  True],
#         [False, False, False,  True],
#         [False, False, False, False]])

scores = torch.randn(2, 4, T, T)  # (B, h, T, T)
scores = scores.masked_fill(causal_mask, float("-inf"))  # broadcast over B, h
attn = torch.softmax(scores, dim=-1)
```

**Gotcha**:  
- `masked_fill(mask, value)` fills where mask is **True** → put True at positions to *block*.  
- `triu(diagonal=1)` = strict upper triangle (future), `triu(diagonal=0)` includes the diagonal.  
- The mask broadcasts automatically from `(T, T)` onto `(B, h, T, T)`.

---

### `cat` vs `stack`

**Key idea**:  
- `cat` joins along an **existing** dimension (no new dim created).  
- `stack` joins along a **new** dimension (inserts a new axis).

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

torch.cat([a, b], dim=0)    # (4, 3) — concat along existing dim 0
torch.cat([a, b], dim=1)    # (2, 6) — concat along existing dim 1

torch.stack([a, b], dim=0)  # (2, 2, 3) — new dim at position 0
torch.stack([a, b], dim=1)  # (2, 2, 3) — new dim at position 1
```

**Gotcha**: `stack` requires all tensors to have the **same shape**; `cat` only requires matching on non-concat dims.

---

### `expand` vs `repeat`

**Key idea**:  
- `expand` creates a **view** (no memory copy); only works on size-1 dims.  
- `repeat` actually **copies data** into new memory.

```python
x = torch.randn(1, 3)       # (1, 3)

y = x.expand(4, 3)          # (4, 3), no copy, just new strides
z = x.repeat(4, 1)          # (4, 3), actually copies data 4 times

# expand also accepts -1 to mean "keep this dim as-is"
w = x.expand(-1, 3)         # (1, 3), no change
```

**Gotcha**: `expand` output shares memory → writing to it writes to the original. Use `.clone()` if you need an independent copy.

---

### `einsum`

**Key idea**: expressive way to do matmul, batch-matmul, and other contractions. Often clearer than chaining `transpose` + `@`.

```python
# Batch matmul: (B, i, k) @ (B, k, j) -> (B, i, j)
A = torch.randn(2, 3, 4)
B = torch.randn(2, 4, 5)
C = torch.einsum("bik,bkj->bij", A, B)   # (2, 3, 5)

# Scaled dot-product attention scores: (B, h, T, d_k) x (B, h, d_k, T)
# equivalent to q @ k.transpose(-2, -1)
Q = torch.randn(2, 4, 5, 8)  # (B, h, T, d_k)
K = torch.randn(2, 4, 5, 8)
scores = torch.einsum("bhid,bhjd->bhij", Q, K)  # (2, 4, 5, 5)

# Outer product: (n,) x (m,) -> (n, m)
a = torch.randn(3)
b = torch.randn(4)
outer = torch.einsum("i,j->ij", a, b)   # (3, 4)
```

**Gotcha**: Repeated indices that don't appear in the output are summed over (contracted). Any index that appears in the output is kept.

---

### `torch.where`

**Key idea**: element-wise conditional selection. Like `np.where`.

```python
x = torch.tensor([1.0, -2.0, 3.0, -4.0])

# where(condition, if_true, if_false)
y = torch.where(x > 0, x, torch.zeros_like(x))  # ReLU
# tensor([1., 0., 3., 0.])

# Also works on multi-dim tensors with broadcasting
mask = torch.tensor([[True, False], [False, True]])
a = torch.ones(2, 2)
b = torch.zeros(2, 2)
torch.where(mask, a, b)
# tensor([[1., 0.], [0., 1.]])
```

---

### `topk` and `argsort`

**Key idea**: frequently used in sampling, beam search, and evaluation.

```python
x = torch.tensor([3.0, 1.0, 4.0, 1.5, 9.0, 2.6])

values, indices = torch.topk(x, k=3)
# values:  tensor([9., 4., 3.])
# indices: tensor([4, 2, 0])

# argsort: indices that would sort the tensor (ascending by default)
sorted_idx = torch.argsort(x)              # ascending
sorted_idx_desc = torch.argsort(x, descending=True)
```

---

### `clamp` / `clip`

**Key idea**: cap values to a range. Useful for gradient clipping, numerical stability.

```python
x = torch.tensor([-2.0, 0.5, 1.5, 3.0])

torch.clamp(x, min=0.0, max=1.0)   # tensor([0., 0.5, 1., 1.])
x.clamp_(0.0, 1.0)                 # in-place version
```

---

### `arange` / `linspace` (index generation)

**Key idea**: very commonly used to generate position indices or meshes.

```python
# arange: evenly spaced integers
pos = torch.arange(10)           # tensor([0, 1, ..., 9])
pos = torch.arange(0, 10, 2)     # tensor([0, 2, 4, 6, 8])

# linspace: evenly spaced floats between two values
x = torch.linspace(0, 1, steps=5)  # tensor([0.00, 0.25, 0.50, 0.75, 1.00])
```

---

### Quick mental checklist

- **Changing layout only?** → `permute` / `transpose`.
- **Merging/splitting dims?** → `view` / `reshape` (use `.contiguous()` after permute).
- **Index-based selection?** → `gather`.
- **Index-based writing?** → `scatter_`.
- **Need safer reshape (may copy)?** → `reshape` instead of `view`.
- **Causal/attention masking?** → `triu` + `masked_fill(..., float("-inf"))`.
- **Joining tensors along existing dim?** → `cat`.
- **Creating new axis while joining?** → `stack`.
- **Broadcast a size-1 dim (no copy)?** → `expand`.
- **Actually replicate data?** → `repeat`.
- **Readable multi-dim contractions?** → `einsum`.
- **Conditional selection?** → `torch.where`.

