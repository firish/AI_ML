# Tensors, Einsum, and Broadcasting

Everything so far has been vectors (1D) and matrices (2D). In practice, transformers compute with **tensors** — multi-dimensional arrays. This file covers how to read, manipulate, and reason about them.

---

## 1. What Is a Tensor?

A tensor is just an array with any number of dimensions (called **axes** or **ranks**).

```
Scalar:     0 dimensions     just a number              42
Vector:     1 dimension      a list of numbers           [1, 2, 3]           shape: (3,)
Matrix:     2 dimensions     a grid of numbers           [[1,2],[3,4]]       shape: (2, 2)
3D tensor:  3 dimensions     a stack of matrices         shape: (batch, seq, d_model)
4D tensor:  4 dimensions     a stack of stacks           shape: (batch, heads, seq, d_k)
```

The word "tensor" in ML usually just means "multi-dimensional array." Mathematicians have a more precise definition involving coordinate transformations, but in PyTorch/NumPy, tensor = ndarray with a shape.

---

## 2. Shapes You'll See in Transformers

Every tensor's shape tells you exactly what it contains. Learn to read shapes and you can follow any architecture.

### The core shapes

```
Token embeddings:       (batch, seq, d_model)
                         e.g., (32, 512, 768)
                         32 sentences, each 512 tokens, each 768-dim

Q, K, V after projection: (batch, seq, d_model)
                         same shape, different values

Multi-head split:       (batch, num_heads, seq, d_k)
                         e.g., (32, 12, 512, 64)
                         32 sentences, 12 heads, 512 tokens, 64-dim per head

Attention scores:       (batch, num_heads, seq, seq)
                         e.g., (32, 12, 512, 512)
                         per head: a 512×512 matrix of pairwise scores

Weight matrix:          (d_in, d_out)
                         e.g., (768, 3072)
                         shared across all tokens and all batches
```

### Reading shapes left to right

```
(32, 12, 512, 64)

Think of it as nested containers:
    32 batches, each containing
        12 heads, each containing
            512 tokens, each containing
                64 numbers

The rightmost dimension is the "innermost" — the actual data values.
Moving left, each dimension is a higher level of organization.
```

---

## 3. Reshaping and Transposing — Rearranging Without Changing Data

### Reshape

Changes the shape without changing the underlying data or its order.

```
x: shape (32, 512, 768)

Reshape to (32, 512, 12, 64):
    768 values per token → split into 12 groups of 64
    No data moves. Just a different way to index the same numbers.

This is how multi-head attention splits d_model into num_heads × d_k:
    768 = 12 heads × 64 dims per head
```

### Transpose (permute axes)

Swaps the order of dimensions. Data stays the same, but the axes are reordered.

```
x: shape (32, 512, 12, 64)

Transpose axes 1 and 2:
    (32, 512, 12, 64) → (32, 12, 512, 64)

    Before: batch, seq, heads, d_k
    After:  batch, heads, seq, d_k

Now each head's data is contiguous — all 512 tokens for head 0,
then all 512 for head 1, etc. This makes the per-head
attention computation efficient.
```

### The multi-head attention reshape dance

```
1. Start:     (batch, seq, d_model)           (32, 512, 768)
2. Reshape:   (batch, seq, heads, d_k)        (32, 512, 12, 64)
3. Transpose: (batch, heads, seq, d_k)        (32, 12, 512, 64)
4. Attention:  per-head Q×Kᵀ, softmax, ×V    (32, 12, 512, 64)
5. Transpose: (batch, seq, heads, d_k)        (32, 512, 12, 64)
6. Reshape:   (batch, seq, d_model)           (32, 512, 768)

Steps 2-3: split into heads
Steps 5-6: merge heads back
The actual attention (step 4) operates on each head independently.
```

---

## 4. Batched Matrix Multiplication

In file 02, matrix multiplication was (m × n) × (n × p) = (m × p). With tensors, we add **batch dimensions** that ride along without participating in the multiply.

```
A: (32, 12, 512, 64)     — 32 batches × 12 heads × (512 × 64) matrices
B: (32, 12, 64, 512)     — 32 batches × 12 heads × (64 × 512) matrices

A @ B: (32, 12, 512, 512)

The last two dimensions do the matrix multiply: (512×64) × (64×512) = (512×512)
The first two dimensions (32, 12) are batch dims — 32×12 = 384 independent
matrix multiplies happening in parallel.
```

This is how attention scores are computed: Q × Kᵀ for all batches and all heads in one operation. No loops. The GPU does all 384 matrix multiplies simultaneously.

### The rule

```
Batch dimensions:  must match (or be broadcastable — see section 7)
Matrix dimensions: last two axes follow the (m×n) × (n×p) = (m×p) rule

(32, 12, 512, 64) @ (32, 12, 64, 512) = (32, 12, 512, 512)
 batch  batch  ↑matrix multiply↑         batch  batch  result
```

---

## 5. Einsum — A Universal Language for Tensor Operations

Einsum (Einstein summation) is a notation that expresses any tensor contraction in one line. It looks cryptic at first but becomes the clearest way to write tensor operations once you learn it.

### The pattern

```
einsum("input_indices -> output_indices", tensor_a, tensor_b)

Each letter = one axis.
If a letter appears in the input but NOT the output → sum over it.
If a letter appears in both inputs → those axes are "contracted" (dot product).
```

### Dot product

```
a: (d,)     b: (d,)

einsum("i, i ->", a, b)        result: scalar

"i" appears in both inputs → contract (multiply and sum).
"i" absent from output → summed away.
This is a₁b₁ + a₂b₂ + ... = the dot product.
```

### Matrix multiply

```
A: (m, n)     B: (n, p)

einsum("ij, jk -> ik", A, B)     result: (m, p)

"j" appears in both inputs but not output → summed over.
"i" stays from A, "k" stays from B.
Entry [i,k] = Σⱼ A[i,j] × B[j,k]    — the dot product definition.
```

### Batched matrix multiply

```
Q: (batch, heads, seq, d_k)     K: (batch, heads, d_k, seq)

einsum("bhsd, bhds -> bhss", Q, K_transposed)

Wait — but K isn't transposed in storage. Einsum can handle that:

einsum("bhsd, bhqd -> bhsq", Q, K)

"d" appears in both inputs, not in output → summed over (the dot product).
"b" and "h" match up → batch dimensions.
"s" from Q and "q" from K → the two seq dimensions in the output.

Result: (batch, heads, seq, seq) = attention scores.
```

### Transpose

```
A: (m, n)

einsum("ij -> ji", A)     result: (n, m)

Just relabel the axes. No computation — just a view.
```

### Outer product

```
a: (m,)     b: (n,)

einsum("i, j -> ij", a, b)     result: (m, n)

No index is summed over. Every pair (i,j) multiplied.
Result[i,j] = a[i] × b[j].
```

### Why einsum matters

```
Without einsum:
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

With einsum:
    scores = torch.einsum("bhsd, bhqd -> bhsq", Q, K) / math.sqrt(d_k)

The einsum version explicitly shows:
    - which axes are batch dims (b, h)
    - which axis is contracted (d — the dot product)
    - which axes form the output (s, q — the seq×seq score matrix)

For complex operations (multi-head attention, tensor contractions
in custom architectures), einsum is often more readable than
chains of reshape/transpose/matmul.
```

---

## 6. Common Einsum Patterns

```
Operation                Einsum                      Shapes
─────────────────────────────────────────────────────────────
Dot product              "i, i ->"                   (d,) × (d,) → scalar
Matrix-vector            "ij, j -> i"                (m,n) × (n,) → (m,)
Matrix multiply          "ij, jk -> ik"              (m,n) × (n,p) → (m,p)
Batched matmul           "bij, bjk -> bik"           (b,m,n) × (b,n,p) → (b,m,p)
Attention scores         "bhsd, bhqd -> bhsq"        (b,h,s,d) × (b,h,q,d) → (b,h,s,q)
Transpose                "ij -> ji"                  (m,n) → (n,m)
Outer product            "i, j -> ij"                (m,) × (n,) → (m,n)
Trace                    "ii ->"                     (n,n) → scalar
Diagonal                 "ii -> i"                   (n,n) → (n,)
Element-wise × sum       "ij, ij ->"                 (m,n) × (m,n) → scalar  (Frobenius inner product)
Batch dot product        "bi, bi -> b"               (b,d) × (b,d) → (b,)
```

The rule is always the same: repeated index in inputs = multiply along that axis. Missing from output = sum over it.

---

## 7. Broadcasting — When Shapes Don't Match (But It's Fine)

Broadcasting is how NumPy/PyTorch handle operations between tensors of different shapes by automatically expanding the smaller one.

### The rules

```
Align shapes from the RIGHT. Compare each dimension:

    1. If they match → fine
    2. If one is 1 → stretch it to match the other
    3. If neither matches and neither is 1 → error

Examples:
    (32, 512, 768) + (768,)         → (32, 512, 768)    bias added to every token
    (32, 512, 768) + (1, 1, 768)    → (32, 512, 768)    same thing, explicit
    (32, 512, 768) + (512, 768)     → (32, 512, 768)    bias per position
    (32, 512, 768) + (32, 512, 1)   → (32, 512, 768)    one scalar per token, stretched
    (32, 512, 768) + (32, 100, 768) → ERROR              512 ≠ 100, neither is 1
```

### Why broadcasting matters

**Adding bias:** A layer computes output = input × W + b. The bias b has shape (d_out,), but the output has shape (batch, seq, d_out). Broadcasting stretches b across the batch and seq dimensions automatically.

```
output: (32, 512, 768)
bias:   (768,)

The bias is added to EVERY token in EVERY sentence.
Broadcasting makes this one line instead of a nested loop.
```

**Masking in attention:** The causal mask is (seq, seq) but scores are (batch, heads, seq, seq). Broadcasting stretches the mask across batch and heads.

```
scores: (32, 12, 512, 512)
mask:   (1,  1,  512, 512)    or just (512, 512)

Broadcasting applies the same mask to all batches and all heads.
```

**Scaling:** Dividing attention scores by √d_k. The scalar is shape () (zero dimensions), broadcast to (batch, heads, seq, seq).

### The danger

Broadcasting is silent. If shapes accidentally align in a way you didn't intend, you get wrong results with no error.

```
a: (3, 4)     — 3 rows, 4 columns
b: (4,)       — meant to add to each row

a + b works fine. b is broadcast across rows. ✓

c: (3,)       — meant to add to each row, but wrong shape

a + c → ERROR. Good — caught the bug.

d: (3, 1)     — reshaped c to (3,1)

a + d works — but now it broadcasts across COLUMNS, not rows.
Silently wrong. d gets stretched to (3, 4) by repeating each value 4 times.
```

Rule of thumb: when you get unexpected results, check the shapes. Broadcasting bugs are shape bugs.

---

## 8. Putting It Together — Reading Transformer Code

With tensors, einsum, and broadcasting, you can follow real transformer code.

### Attention in one block

```python
# Input: x of shape (batch, seq, d_model)

# Project to Q, K, V
Q = x @ W_q                        # (batch, seq, d_model) @ (d_model, d_model) = (batch, seq, d_model)
K = x @ W_k                        # same
V = x @ W_v                        # same

# Reshape for multi-head
Q = Q.reshape(batch, seq, heads, d_k).transpose(1, 2)   # (batch, heads, seq, d_k)
K = K.reshape(batch, seq, heads, d_k).transpose(1, 2)   # (batch, heads, seq, d_k)
V = V.reshape(batch, seq, heads, d_k).transpose(1, 2)   # (batch, heads, seq, d_k)

# Attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)   # (batch, heads, seq, seq)

# Causal mask (broadcasting: mask is (1, 1, seq, seq), applied to all batches/heads)
scores = scores + mask                           # broadcast!

# Softmax per row
weights = softmax(scores, dim=-1)                # (batch, heads, seq, seq)

# Weighted sum of values
out = weights @ V                                # (batch, heads, seq, d_k)

# Merge heads back
out = out.transpose(1, 2).reshape(batch, seq, d_model)   # (batch, seq, d_model)
```

Every line is a tensor operation you now understand:
- `@` = batched matrix multiply (section 4)
- `.reshape` = rearrange without moving data (section 3)
- `.transpose` = swap axes (section 3)
- `+ mask` = broadcasting (section 7)
- `softmax` = per-row normalization
- The shapes tell you exactly what's happening at each step

---

## 9. Summary

```
Tensors         multi-dimensional arrays — the actual data structure of ML
Shapes          (batch, heads, seq, d_k) — read left to right, innermost on right
Reshape         split/merge dimensions without moving data
Transpose       swap axis order — needed for multi-head attention split/merge
Batched matmul  last two dims do the multiply, leading dims are independent batches
Einsum          universal notation: repeated index = contract, missing from output = sum
Broadcasting    stretch smaller tensors to match larger ones, silent and powerful
```

The linear algebra from files 01-06 (dot products, matrix multiply, eigenvalues, SVD) is the math. Tensors are the **data structure** that lets you do that math efficiently at scale — batching across sentences, parallelizing across heads, applying masks, all without explicit loops.

---

This completes Phase 1 (Linear Algebra) of the Math Roadmap. Next phase: **Calculus and Optimization** — derivatives, chain rule, gradient descent, and how models learn.
