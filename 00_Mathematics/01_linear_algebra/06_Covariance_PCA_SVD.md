# Covariance, PCA, SVD, and Low-Rank Approximation

These four concepts form a chain: covariance measures your data's structure → PCA uses eigenvalues of covariance to find optimal coordinates → SVD generalizes this to any matrix → low-rank approximation falls out of SVD and is the foundation of LoRA.

---

## 1. Variance — How Much One Dimension Moves

Before covariance, start with variance. It measures how spread out values are along a single dimension.

```
Data points along one dimension: [2, 4, 6, 8, 10]

    Mean = 6
    Deviations from mean: [-4, -2, 0, 2, 4]
    Squared deviations: [16, 4, 0, 4, 16]
    Variance = average of squared deviations = 40/5 = 8

High variance (like 8): values are spread out — this dimension carries information.
Low variance (like 0.01): values are bunched together — this dimension is mostly constant.
```

Variance answers: "does this dimension DO anything across my data, or is it basically the same for every point?"

---

## 2. Covariance — How Two Dimensions Move Together

Covariance measures whether two dimensions are related — when one goes up, does the other go up too?

```
Dim 0:  [1, 2, 3, 4, 5]     mean = 3
Dim 1:  [2, 4, 6, 8, 10]    mean = 6

Deviations:
    Dim 0: [-2, -1, 0, 1, 2]
    Dim 1: [-4, -2, 0, 2, 4]

Covariance = average of (deviation_0 × deviation_1)
           = ((-2)(-4) + (-1)(-2) + 0×0 + 1×2 + 2×4) / 5
           = (8 + 2 + 0 + 2 + 8) / 5
           = 4
```

### Reading the sign

```
cov > 0:   when dim 0 is high, dim 1 tends to be high too (move together)
cov < 0:   when dim 0 is high, dim 1 tends to be low (move opposite)
cov ≈ 0:   dim 0 and dim 1 are unrelated (independent)
```

### Why we care

If two dimensions have high covariance, they're partly saying the same thing — they carry redundant information. If you're compressing data (PQ, dimensionality reduction), correlated dimensions are wasteful. If two dimensions are independent (cov ≈ 0), they carry non-redundant information — both worth keeping.

---

## 3. The Covariance Matrix — All Pairs at Once

For d-dimensional data with N points (mean-subtracted), the covariance matrix captures every pair of dimensions simultaneously.

```
X: (N × d) data matrix, mean subtracted

C = (1/N) × Xᵀ × X       shape: (d × d)

    C[i,i] = variance of dimension i       (the diagonal)
    C[i,j] = covariance of dims i and j    (the off-diagonal)
```

### Reading it

```
The diagonal:      how much each dimension varies on its own
The off-diagonal:  how much each pair of dimensions co-varies

C = [var₀     cov₀₁   cov₀₂]
    [cov₁₀   var₁     cov₁₂]
    [cov₂₀   cov₂₁   var₂  ]

C is always symmetric (cov(i,j) = cov(j,i)) and positive semi-definite
(all eigenvalues ≥ 0, because variance can't be negative).
```

### The ideal vs the reality

```
IDEAL for PQ (independent subvectors):

    C = [████  ·    ·    · ]      block-diagonal
        [ ·  ████   ·    · ]      each block's dimensions correlated internally
        [ ·   ·   ████  · ]       but independent across blocks
        [ ·   ·    ·  ████]

REALITY (typical embeddings):

    C = [████  ██  ██  ██]        correlations everywhere
        [██  ████  ██  ██]        dimensions talk to each other
        [██  ██  ████  ██]        across all block boundaries
        [██  ██  ██  ████]

OPQ's rotation pushes reality toward the ideal by rotating the
coordinate system until the cross-block correlations are minimized.
```

---

## 4. PCA — Finding the Best Coordinate System

Your data lives in 768 dimensions, but it might not USE all 768. If the data forms a tilted ellipse in a 2D subspace, 766 dimensions are wasted. PCA finds the directions your data actually uses, sorted by importance.

### The algorithm

```
1. Center the data (subtract the mean from each dimension)
2. Compute covariance matrix: C = (1/N) × Xᵀ × X
3. Find eigenvalues and eigenvectors of C
4. Sort by eigenvalue, largest first
5. The eigenvectors are the "principal components" — the new axes
```

### What each eigenvector/eigenvalue means

```
Eigenvector 1 (largest eigenvalue):
    Direction of MOST variance. The data is most spread out here.
    This is the most informative direction — keep it.

Eigenvector 2 (second largest):
    Direction of most REMAINING variance, perpendicular to #1.

...

Eigenvector 768 (smallest eigenvalue):
    Direction of least variance. Data barely moves here. Noise.
```

### 2D example

```
1000 points forming a tilted ellipse:

        • • •
      • • • • •
    • • • • • • •         Long axis:  eigenvector 1, λ₁ = 10
      • • • • •           Short axis: eigenvector 2, λ₂ = 0.5
        • • •

PCA rotates the coordinate system to align with the ellipse:
    New dim 0 = long axis (high variance, λ=10)
    New dim 1 = short axis (low variance, λ=0.5)

Now if you drop dim 1, you lose only the short-axis variation
(5% of total variance). The shape is mostly preserved.
```

### Dimensionality reduction

```
768 eigenvalues, sorted:
    λ₁ = 45.2, λ₂ = 38.1, ... λ₅₀ = 2.3, λ₅₁ = 0.4, ... λ₇₆₈ = 0.001

Sum of top 50 = 95% of total sum of all eigenvalues.

Project: X_new = X × V₅₀     (768-dim → 50-dim)

    V₅₀: (768 × 50) — columns are the top 50 eigenvectors

95% of information in 50 dims instead of 768.
The eigenvalues told you exactly how much you lose at each cut point.
```

---

## 5. PCA vs OPQ — Same Idea, Different Goal

Both PCA and OPQ rotate coordinates. They differ in what they optimize for.

```
PCA:
    Goal: order dimensions by variance (dim 0 = most, dim 767 = least)
    Use:  drop the bottom dimensions for compression
    Result: all the "good stuff" is packed into the first few dims

OPQ:
    Goal: BALANCE variance across PQ blocks (each 96-dim chunk equally rich)
    Use:  make each PQ codebook effective
    Result: variance is spread evenly, not concentrated at the top

PCA would pack all high-variance stuff into block 0
and leave blocks 1-7 with scraps. That's bad for PQ —
codebooks 1-7 would have nothing useful to quantize.

OPQ distributes variance so every codebook has rich structure.
Same underlying math (eigenvalues, rotation), different objective.
```

---

## 6. SVD — Decomposing Any Matrix

PCA works on covariance matrices (square, symmetric). SVD works on **any** matrix — any shape, any type.

```
A = U × Σ × Vᵀ

    A:  (m × n)  the original matrix
    U:  (m × m)  orthogonal matrix (left singular vectors)
    Σ:  (m × n)  diagonal matrix (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
    Vᵀ: (n × n)  orthogonal matrix (right singular vectors)
```

### What this means geometrically

Every matrix, no matter how complicated, is really three simple steps:

```
    Vᵀ:  rotate the input (align with the matrix's natural axes)
    Σ:   scale each axis independently (stretch some, shrink others)
    U:   rotate the output (orient the result)

"Rotate → Scale → Rotate"
```

The singular values in Σ tell you how much each direction gets stretched. Large σ = important direction. Small σ = negligible. Zero σ = collapsed.

### Worked example (2×2)

```
A = [3, 1]
    [1, 3]

SVD gives:
    U  = [0.707, -0.707]     (rotation by 45°)
         [0.707,  0.707]

    Σ  = [4, 0]              (stretch by 4 in one direction, 2 in the other)
         [0, 2]

    Vᵀ = [0.707,  0.707]     (rotation by 45°)
         [-0.707, 0.707]

Reading: rotate input 45° → stretch one axis by 4, other by 2 → rotate output 45°

Verify: U × Σ × Vᵀ = [3, 1; 1, 3] ✓
```

---

## 7. SVD Reveals Rank

The rank of a matrix = the number of non-zero singular values.

```
Σ = [4, 0, 0]      3 singular values, all non-zero → rank 3
    [0, 2, 0]
    [0, 0, 1]

Σ = [4, 0, 0]      3 singular values, one is zero → rank 2
    [0, 2, 0]      the third direction is collapsed
    [0, 0, 0]

In practice, "non-zero" means "above some threshold."
Σ = [100, 50, 0.001] is effectively rank 2 — the third
singular value is so small it's noise.
```

A 768×768 matrix might have singular values:

```
σ₁ = 45, σ₂ = 38, ... σ₅₀ = 2, σ₅₁ = 0.01, ... σ₇₆₈ = 0.0001

Effective rank ≈ 50. The matrix LOOKS like it uses 768 dimensions,
but 718 of those directions contribute almost nothing.
```

---

## 8. Low-Rank Approximation — The Best Possible Compression

Keep the top-r singular values, zero out the rest:

```
A ≈ U_r × Σ_r × V_rᵀ

    U_r:  (m × r)   — first r columns of U
    Σ_r:  (r × r)   — top-r singular values
    V_rᵀ: (r × n)   — first r rows of Vᵀ
```

This is **provably the best** rank-r approximation of A. No other matrix with rank r is closer (by Frobenius norm). This isn't a heuristic — it's a mathematical theorem.

### How much information is preserved?

```
Energy preserved = (σ₁² + σ₂² + ... + σᵣ²) / (σ₁² + σ₂² + ... + σₙ²)

If the top 50 singular values capture 99% of the total squared energy,
the rank-50 approximation is nearly perfect.
The error comes only from the discarded directions.
```

### Storage savings

```
Original:     m × n values
Low-rank:     m × r + r + r × n values

Example: 768 × 768 matrix
    Full:     590K values
    Rank 8:   768×8 + 8 + 8×768 = 12.3K values

    48× compression, and if the true rank is ~8, almost no loss.
```

---

## 9. Why Low-Rank Matters — LoRA

The insight behind LoRA: when you fine-tune a pretrained model, the weight **change** ΔW is approximately low-rank. The model doesn't need to change everything — just a few directions.

```
Full fine-tuning:
    W_new = W_old + ΔW      ΔW is (768 × 768) = 590K parameters to learn

LoRA:
    W_new = W_old + A × B   A: (768 × r), B: (r × 768), r = 8

    A × B is a (768 × 768) matrix, but rank r.
    Parameters: 768×8 + 8×768 = 12K instead of 590K.
```

Why rank 8 is enough: fine-tuning typically adjusts how the model handles a specific domain or task. That adjustment lives in a low-dimensional subspace — maybe 8-64 directions out of 768. LoRA forces this structure by construction (the bottleneck at r), and empirically it works almost as well as full fine-tuning.

The SVD connection: if you did full fine-tuning and then ran SVD on ΔW, you'd find that only a few singular values are large. The rest are near zero. LoRA just skips computing the full matrix and directly parametrizes the low-rank structure.

---

## 10. SVD Solves the Procrustes Problem

"Given two sets of points, find the orthogonal matrix R that best aligns them."

```
Problem: minimize ||R × V - X̂||²   subject to R orthogonal

Solution:
    1. Compute SVD of V × X̂ᵀ = U × Σ × Wᵀ
    2. R = W × Uᵀ

One SVD. Closed-form. No iteration needed.
```

This is step 2 of OPQ training. Given fixed PQ codebooks, find the rotation that minimizes total reconstruction error. SVD gives the exact answer.

OPQ's full training loop:

```
Repeat until convergence:
    Step 1: Fix R, train codebooks (run k-means on rotated data)
    Step 2: Fix codebooks, find best R (solve Procrustes via SVD)

Typically 5-10 iterations. Each step has a clean solution.
```

---

## 11. PCA via SVD

PCA is actually a special case of SVD. For mean-centered data X:

```
X = U × Σ × Vᵀ

The principal components = columns of V (right singular vectors)
The eigenvalues of covariance = σ²/N (squared singular values, scaled)
The projected data = U × Σ (or equivalently X × V)
```

Why use SVD instead of computing eigenvectors of the covariance matrix?

```
Direct PCA:
    1. Compute C = (1/N) Xᵀ × X     (768 × 768 matrix)
    2. Find eigenvectors of C

SVD approach:
    1. Compute SVD of X directly

SVD is numerically more stable. The covariance matrix XᵀX can
amplify rounding errors (you're multiplying X with itself).
SVD on X avoids this. Every practical PCA implementation uses SVD.
```

---

## 12. How It All Connects

```
Variance         "how much does this dimension move?"
    ↓
Covariance       "how much do these two dimensions move TOGETHER?"
    ↓
Covariance matrix  all pairs at once — the complete correlation structure
    ↓
Eigenvalues of C   how much variance in each principal direction
    ↓
PCA              sort by eigenvalue, keep the top-k → dimensionality reduction
    ↓
SVD              generalizes eigendecomposition to any matrix
    ↓
Singular values    how much the matrix stretches each direction
    ↓
Rank             number of non-zero singular values = true dimensionality
    ↓
Low-rank approx  keep top-r singular values → best possible compression
    ↓
LoRA             fine-tuning ΔW is low-rank → parametrize directly as A×B
    ↓
Procrustes       SVD solves "find best rotation" → step 2 of OPQ training
```

The unifying theme: **most data and most transformations are simpler than they look.** A 768-dim dataset might really live in 50 dimensions. A 768×768 weight update might only change 8 directions. SVD reveals this hidden simplicity, and everything else (PCA, LoRA, OPQ) exploits it.

---

**Next:** `07_Tensors_Einsum_Broadcasting.md` — going beyond 2D matrices to the multi-dimensional arrays that transformers actually compute with.
