# Linear Algebra for ML — Notes

Every technique in these notes — attention, embeddings, quantization, search — is matrix operations under the hood. This file covers what each operation is, why we care about it, and what breaks if you don't understand it.

---

## 1. Vectors and the Dot Product

A vector is a list of numbers. In ML it represents a point in some space — an embedding, a gradient, a row of weights.

### The dot product

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ     (multiply elementwise, sum)

Example:
    a = [1, 2, 3]
    b = [4, 5, 6]
    a · b = 1×4 + 2×5 + 3×6 = 32
```

What it measures geometrically:

```
a · b = ||a|| × ||b|| × cos(θ)

where θ is the angle between the vectors.

    cos(0°) = 1      → parallel, same direction   → large positive dot product
    cos(90°) = 0     → perpendicular              → dot product is zero
    cos(180°) = -1   → opposite directions         → large negative dot product
```

### Why we care

The dot product is the fundamental operation of "how similar are these two things?" Two vectors pointing in roughly the same direction give a large dot product. Perpendicular vectors give zero. Opposite vectors give negative.

This is why the entire attention mechanism runs on dot products — when a query and key have a high dot product, it means that token is relevant, so it gets more attention weight. When you search a vector database for the closest match to your query, you're computing dot products (or distances, which are dot products in disguise). When a neuron fires strongly, it's because the input had a high dot product with that neuron's weight vector — the input matched the pattern the neuron learned to detect.

If dot products get corrupted (by bad quantization, by numerical error, by lossy compression), the model makes wrong decisions about what's similar to what. This is why so much of quantization research (QJL, PolarQuant) is obsessed with preserving dot products specifically.

### The norm (length)

```
||a|| = √(a · a) = √(a₁² + a₂² + ... + aₙ²)

This is the L2 norm (Euclidean distance from origin).
```

Why it matters: the dot product conflates two things — direction and magnitude. Two vectors can have a high dot product either because they point in similar directions OR because one is very long. Cosine similarity fixes this by dividing out the norms:

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||)
```

Now you're measuring pure directional similarity. This is also why L2 distance and dot product are related:

```
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

If vectors are normalized (length 1), maximizing dot product = minimizing L2 distance. They're the same search.

---

## 2. Matrix Multiplication

A matrix is a grid of numbers. Multiplying two matrices is just doing many dot products at once.

```
If A is (m × n) and B is (n × p), then C = A × B is (m × p).

Each entry C[i,j] = dot product of row i of A with column j of B.

    [1, 2]   [5, 6]     [1×5+2×7, 1×6+2×8]     [19, 22]
    [3, 4] × [7, 8]  =  [3×5+4×7, 3×6+4×8]  =  [43, 50]
```

### Why we care

Every layer in a neural network is a matrix multiply. The forward pass is `output = input × W + b`. Each column of W is one neuron's weight vector, and the matrix multiply computes the dot product of the input with every neuron simultaneously. So matrix multiplication is really "check the input against all patterns at once."

The dimension rule — inner dimensions must match, outer dimensions give the result shape — is worth internalizing because it tells you what a matrix multiply is doing:

```
(batch × input_dim) × (input_dim × output_dim) = (batch × output_dim)

Reading this: "take batch-many inputs of size input_dim,
and produce batch-many outputs of size output_dim."
The input_dim disappears — it gets summed over (the dot product).
```

This is why attention computes Q × Kᵀ as `(seq × d_k) × (d_k × seq) = (seq × seq)`. The result is a seq-by-seq matrix where entry [i,j] is the dot product of query i with key j. One matrix multiply gives you all pairwise similarity scores.

---

## 3. Transpose

Flip rows and columns. Row i becomes column i.

```
A = [1, 2, 3]      Aᵀ = [1, 4]
    [4, 5, 6]            [2, 5]
                          [3, 6]

Shape (m × n) → (n × m)
```

### Why we care

Transpose is the operation that reverses the direction of a mapping. If a matrix W maps from space A to space B, then Wᵀ maps from B back toward A. Not perfectly (that's the inverse), but it "points back."

**Practical consequence 1: making dot products work.** Q and K are both shaped (seq × d_k). You can't multiply them — inner dimensions don't match. Transposing K to (d_k × seq) makes it work: `Q × Kᵀ = (seq × d_k) × (d_k × seq) = (seq × seq)`. Without transpose, there's no way to compute all-pairs dot products in one operation.

**Practical consequence 2: reusing matrices in both directions.** The embedding table maps token IDs → vectors (vocab × d_model). The prediction head needs the reverse: vectors → vocabulary scores. GPT-2 just uses Eᵀ for the prediction head. Same learned relationships, reversed direction, zero extra parameters. This works because if token "cat" has high dot product with a certain embedding direction going in, it should have high dot product coming out too.

**Key identity:** `(A × B)ᵀ = Bᵀ × Aᵀ` — order reverses. This comes up when deriving backpropagation gradients.

---

## 4. Projection Matrices — What "Project" Means

A projection takes a vector and maps it to a different space, usually lower-dimensional.

```
W_q: shape (d_model × d_k), e.g., (768 × 64)

    q = x × W_q

    768-dim input → 64-dim query vector
```

### Why we care

This is NOT just dropping dimensions. Each of the 64 output values is a learned weighted sum of all 768 input dimensions. The network learns which combinations of input features matter for each purpose.

This is how attention creates three different "views" of the same token:

```
Same input x, three different projections:

    q = x × W_q     "What am I looking for?"
    k = x × W_k     "What do I contain that others might want?"
    v = x × W_v     "What information should I pass along?"

The token "bank" needs a different query representation (looking for
financial context vs river context) than its key representation
(advertising what it contains to other tokens). Separate projections
let the model learn these different roles.
```

### Low-rank projection

A key insight: you don't always need a full-sized matrix. If most of the "action" happens in a small number of directions, a low-rank matrix suffices.

```
Full matrix:  ΔW of shape (768 × 768) = 590K parameters
Low-rank:     ΔW = A × B, where A is (768 × 8) and B is (8 × 768) = 12K parameters

The product A × B is a (768 × 768) matrix, but it can only change
8 directions in the space. Everything else stays fixed.
```

This is the core idea behind LoRA — fine-tuning usually only needs to adjust a few directions in weight space, not all 590K. The rank r controls how expressive the update is vs how many parameters you train.

More generally, "rank" tells you the true dimensionality of what a matrix can do. A rank-8 matrix, no matter how large, can only map inputs into an 8-dimensional subspace. This is also why SVD matters — it reveals the effective rank.

---

## 5. Identity Matrix and Inverse

### Identity matrix (I)

The "do nothing" matrix. Every vector passes through unchanged.

```
I = [1, 0, 0]
    [0, 1, 0]       A × I = A     I × A = A
    [0, 0, 1]
```

Why we care: the identity is the reference point for "no transformation." When we say Rᵀ × R = I, we mean "applying R then Rᵀ gets you back to exactly where you started." When LoRA initializes with ΔW = 0, the model behaves as if no fine-tuning happened, because W + 0 = W (identity-like behavior).

### Inverse (A⁻¹)

The matrix that perfectly undoes A.

```
A × A⁻¹ = I      A⁻¹ × A = I

If y = A × x,  then x = A⁻¹ × y     (you can recover x exactly)
```

Not all matrices have inverses. If a matrix crushes 3D space down to a 2D plane (a projection), there's no way to recover the lost dimension — information was destroyed. A matrix is invertible only when it preserves all dimensions (full rank, square, non-zero determinant).

### Why we care

In practice you almost never compute inverses explicitly (it's slow and numerically unstable). But the concept matters because it tells you whether a transformation is **reversible**:

- Orthogonal matrices: inverse = transpose (free to compute, perfectly reversible)
- Projection to lower dimension: no inverse (information lost)
- Singular matrix: no inverse (some directions collapsed)

The question "can I undo this?" comes up constantly. Can I recover the original vector after quantization? (No — information lost.) Can I recover it after rotation? (Yes — orthogonal, perfectly reversible.) Can I recover the original weights after LoRA? (Yes — just subtract ΔW.)

---

## 6. Determinant

The determinant is a single number computed from a matrix that tells you **how much the matrix scales area (or volume)**.

### 2D intuition

```
Take a unit square (1×1) and multiply its corners by matrix A.
The square deforms into a parallelogram.

    det(A) = signed area of that parallelogram.

For a 2×2 matrix:
    det([a, b]) = a×d - b×c
       ([c, d])
```

### What the value tells you

```
det = 1:     area preserved exactly (rotation, shear)
det = 2:     area doubled (stretching)
det = 0.5:   area halved (shrinking)
det = 0:     area collapsed to zero — the matrix squashes space
             into a lower dimension (a plane to a line, a volume to a plane).
             The matrix is SINGULAR. No inverse exists.
det < 0:     area scaled AND orientation flipped (reflection)
```

### Why we care

The determinant answers two questions that come up constantly:

**"Is this matrix invertible?"** If det = 0, no. Some dimension got collapsed — information is destroyed and you can't get it back. If det ≠ 0, yes. This is the fast check for whether a transformation is reversible.

**"Does this matrix preserve, expand, or crush space?"**

```
det(I) = 1               identity preserves everything
det(rotation) = 1         rotation preserves area (just reorients)
det(reflection) = -1      reflection preserves area but flips orientation
det(scaling by k) = kⁿ    scaling in n dimensions
det(singular) = 0         space collapsed — not invertible
```

The connection to eigenvalues: `det(A) = product of all eigenvalues`. So if any eigenvalue is zero, the determinant is zero and the matrix is singular. This ties the two concepts together — a zero eigenvalue means a direction got collapsed, which means area got crushed to zero, which means the determinant is zero, which means no inverse exists.

### In higher dimensions

For 3D, det = volume scale factor. For 768-dim, det = "hypervolume" scale factor. The interpretation is the same — it tells you whether the transformation preserves, expands, or crushes the space, and whether it flips orientation.

---

## 7. Orthogonal Matrices

An orthogonal matrix R satisfies:

```
Rᵀ × R = I       (the transpose IS the inverse)
```

R's columns are all unit length and mutually perpendicular.

### What it does

R is a pure rotation (and possibly reflection). No stretching, squishing, or skewing.

```
For ANY two vectors a and b:

    ||R × a|| = ||a||                  lengths preserved
    ||R×a - R×b|| = ||a - b||          distances preserved
    (R×a) · (R×b) = a · b             dot products preserved
    angle(R×a, R×b) = angle(a, b)     angles preserved
```

### Why we care

Orthogonality is the mathematical guarantee that a transformation is **completely safe**. "Safe" meaning: no information is created, destroyed, or distorted. Every geometric relationship in your data survives perfectly.

This is critical because most of ML is built on distances and similarities. If you transform your data and distances change, your search results change, your attention scores change, your model behaves differently. An orthogonal transformation lets you reorganize your data — shuffle which axis carries which information — without any of those downstream effects.

Concretely: OPQ rotates 768-dim embeddings so that correlated dimensions land in the same PQ subvector block. This dramatically reduces quantization error. But we'd never do this if rotation distorted distances — that would corrupt every search result. Because R is orthogonal, the rotation is guaranteed to preserve all nearest-neighbor relationships. The data cloud is identical; we've just rotated the coordinate grid underneath it.

The other key property: reversibility is free. Since the inverse is just the transpose, undoing the rotation costs nothing extra. At query time, you rotate the query with R and search against rotated codes. After search, you could rotate results back with Rᵀ if needed. No expensive inverse computation.

### 2D rotation matrix (the building block)

```
R(θ) = [cos θ, -sin θ]
       [sin θ,  cos θ]

Rotates any 2D vector by angle θ counterclockwise.

Example: rotate [1, 0] by 90°:
    [0, -1]   [1]     [0]
    [1,  0] × [0]  =  [1]

    (1,0) → (0,1). 90° counterclockwise. ✓

You can verify it's orthogonal:
    Rᵀ × R = [cos θ,  sin θ] × [cos θ, -sin θ] = [1, 0] = I  ✓
              [-sin θ, cos θ]   [sin θ,  cos θ]   [0, 1]
```

Higher-dimensional rotations are built from stacking 2D rotations in independent planes. RoPE does exactly this — it rotates pairs of dimensions (0,1), (2,3), (4,5), etc., each by a different angle depending on sequence position. The full rotation matrix is block-diagonal: many independent 2D rotation blocks stacked.

### Rotation vs reflection — both orthogonal, but fundamentally different

Both rotations and reflections are orthogonal — they both preserve all distances, angles, and dot products. But they're not the same operation.

**Rotation** is continuous. You can smoothly rotate from 0° to 180°, passing through every angle in between. Think of the letter "R" printed on a sheet — rotate it any amount and it's still a normal "R", just tilted.

**Reflection** jumps. It flips the letter "R" into a backwards "Я". No amount of rotation can turn "Я" back into "R". You have to flip the sheet over.

The mathematical tell is the determinant:

```
det(rotation)   = +1    preserves handedness (clockwise stays clockwise)
det(reflection) = -1    reverses handedness (clockwise becomes counterclockwise)
```

Both are orthogonal (Rᵀ × R = I). Both preserve geometry. But rotation is "proper" orthogonal (det = +1) and reflection is "improper" (det = -1). When ML papers say "orthogonal matrix" in the context of OPQ or PolarQuant, they typically mean rotations — the data gets reoriented, not mirrored.

---

## 8. Eigenvalues and Eigenvectors

An eigenvector is a direction where a matrix does the simplest possible thing — just scales, no rotation. The eigenvalue is the scale factor. Together they reveal what a matrix actually does: which directions it amplifies, which it suppresses, which it destroys.

They explain why RNN gradients vanish (eigenvalue < 1 compounds to zero over time steps), how PCA knows which dimensions to keep (eigenvalues = variance per direction), and whether optimization will converge smoothly (condition number = ratio of largest to smallest eigenvalue).

**Full treatment in `09_Eigenvalues_and_Eigenvectors.md`** — concrete 2D walkthrough, repeated application, connection to vanishing gradients, PCA, and optimization.

---

## 9. Covariance Matrix

The covariance matrix captures how dimensions move together across your data.

```
For data X with N samples of d dimensions (mean-subtracted):

    C = (1/N) × Xᵀ × X        shape: (d × d)

    C[i,i] = variance of dimension i          (how much it varies)
    C[i,j] = covariance of dims i and j       (how much they co-move)
```

### Reading it

```
C[i,j] > 0:  when dim i is high, dim j tends to be high too
C[i,j] < 0:  when dim i is high, dim j tends to be low
C[i,j] ≈ 0:  dim i and dim j move independently

A diagonal covariance matrix (all off-diagonal ≈ 0) means
every dimension is independent of every other.
```

### Why we care

The covariance matrix is a complete summary of the linear relationships in your data. It answers: "which dimensions carry redundant information?"

If dims 95 and 96 have high covariance, they're partly saying the same thing. Storing them separately wastes capacity. This is exactly the problem that hits PQ — if those two correlated dimensions land in different subvector blocks, each codebook models half a pattern that only makes sense as a pair.

The covariance matrix makes this problem precise and measurable. OPQ's goal, stated in covariance terms: find a rotation R that makes the covariance matrix **block-diagonal** — high values within each subvector block (rich internal structure for the codebook to model), near-zero values between blocks (no cross-block dependencies for PQ to miss).

```
Before rotation:                    After rotation:

C = [████ ██ ██ ██]                 C' = [████  ·  ·  · ]
    [██ ████ ██ ██]                      [ · ████  ·  · ]
    [██ ██ ████ ██]                      [ · ·  ████ · ]
    [██ ██ ██ ████]                      [ · ·   · ████]

    Lots of cross-block correlation      Each block is self-contained
```

PCA also uses the covariance matrix — its eigenvectors are the principal components. But PCA orders globally (dim 0 = most variance), while OPQ balances across blocks. Different goals, same underlying structure.

---

## 10. PCA (Principal Component Analysis)

PCA rotates your coordinate system so that the new axes align with the directions of maximum variance in the data.

### The algorithm

```
1. Center the data (subtract mean from each dimension)
2. Compute covariance matrix C = (1/N) × Xᵀ × X
3. Find eigenvectors of C, sorted by eigenvalue (largest first)
4. The top-k eigenvectors are the k "principal components"
5. Project: X_new = X × V_k
```

### What this does geometrically

```
Your 768-dim data lives in a cloud with some shape.
PCA finds the "spine" of that cloud:

    Eigenvector 1: direction the cloud stretches MOST (largest eigenvalue)
    Eigenvector 2: direction of most remaining stretch, perpendicular to #1
    Eigenvector 3: most remaining, perpendicular to #1 and #2
    ...

The eigenvalue tells you HOW MUCH spread each direction has.

If the top 50 eigenvalues sum to 95% of the total,
your 768-dim data is effectively 50-dimensional.
The other 718 directions are noise.
Projecting to 50 dims loses 5% of the information.
```

### Why we care

PCA answers "what's the true dimensionality of my data?" If your 768-dim embeddings only use 100 meaningful directions, you're wasting storage and compute on the other 668. PCA lets you compress to 100 dims with mathematically quantified loss.

But the deeper reason PCA matters is that it introduces the core idea behind OPQ, SVD, and half of quantization: **choosing the right coordinate system changes everything**. The data doesn't change — you're just describing it using axes that match its natural structure instead of arbitrary ones. In the right coordinates, compression is easy. In the wrong coordinates, it's wasteful.

PCA picks coordinates that are globally optimal for variance. OPQ picks coordinates that are optimal for PQ's block structure. The principle is the same: rotate first, then do the cheap operation.

---

## 11. SVD (Singular Value Decomposition)

Any matrix A — any size, any shape — can be decomposed into three matrices:

```
A = U × Σ × Vᵀ

    U:  (m × m) orthogonal matrix
    Σ:  (m × n) diagonal matrix with singular values σ₁ ≥ σ₂ ≥ ... ≥ 0
    Vᵀ: (n × n) orthogonal matrix
```

### What this means

Every linear transformation, no matter how complicated, is really just three simple steps: rotate, scale each axis independently, rotate again.

```
    Vᵀ: rotate the input to align with the matrix's "natural axes"
    Σ:  stretch or shrink along each axis (the singular values)
    U:  rotate the output into the final orientation

The singular values tell you HOW MUCH the matrix stretches each direction.
Large σ = important direction. Small σ = negligible direction. Zero σ = destroyed.
```

### Why we care

**1. It reveals the true rank of a matrix.** The number of non-zero singular values = the rank = the number of independent directions the matrix actually uses. A 768×768 matrix might have rank 50 — it looks huge but only does 50 meaningful things. Everything else is noise or redundancy.

**2. It gives the best possible compression.** Keep the top-r singular values, zero out the rest:

```
A ≈ U_r × Σ_r × V_rᵀ

This is provably the best rank-r approximation of A.
No other rank-r matrix is closer (by Frobenius norm).
```

This is the mathematical backbone of LoRA. The claim: when you fine-tune a model, the weight change ΔW is approximately low-rank — it only modifies a few directions. LoRA parametrizes ΔW = A × B with A (d × r) and B (r × d), which forces it to be rank-r. If the true change is indeed low-rank, you get nearly the same result with 50× fewer parameters.

**3. It solves the Procrustes problem.** "Given two datasets, find the rotation that best aligns them" has a closed-form answer via SVD:

```
Compute SVD of:  V × X̂ᵀ = U × Σ × Wᵀ
Optimal rotation: R = W × Uᵀ
```

This is one step of OPQ training — given fixed codebooks, find the rotation R that minimizes total reconstruction error. One SVD, exact answer, no iteration needed for this step.

**4. PCA is a special case of SVD.** The principal components of a dataset are the right singular vectors of the centered data matrix. Computing PCA via SVD is more numerically stable than computing eigenvectors of the covariance matrix, which is why most implementations use SVD under the hood.

---

## 12. Random Projection and the JL Lemma

The Johnson-Lindenstrauss (JL) lemma:

```
You can project N points from d dimensions down to m = O(log N / ε²) dimensions
while preserving ALL pairwise distances to within (1 ± ε) multiplicative error.

The projection matrix can be RANDOM. No training on the data needed.
```

### Why we care

This result is counterintuitive. Compressing 4096 dimensions to 128 sounds like it should destroy structure. JL says it doesn't — if you only care about pairwise distances (which is what search, nearest neighbors, and dot products are about), the geometry survives random projection with high probability.

The key insight: high-dimensional spaces have far more room than you'd expect. Points are "spread out enough" that even a random low-dimensional snapshot captures their relative positions. The required target dimension scales as log(N), not N or d. For a billion points, you need only ~60-90 dimensions to preserve distances within 10%.

This is why random matrices appear in quantization. QJL multiplies Key vectors by a random Gaussian matrix S, then keeps only the sign bits (+1/-1). The random projection preserves the dot product structure, and the sign quantization adds bounded noise on top. No calibration data, no training, no per-model tuning — the randomness itself provides the guarantee.

More broadly, JL tells you that dimensionality reduction doesn't require careful learning (PCA, autoencoders). Random projection is a valid, cheap, universal baseline. When it's good enough, you skip the training cost entirely.

---

## 13. Putting It Together

These concepts build on each other:

```
Dot product is the core operation — similarity, attention, neuron activation.

Matrix multiply does many dot products at once — an entire layer in one operation.

Transpose lets you reverse directions — enabling all-pairs dot products (Q×Kᵀ)
    and weight reuse (embedding ↔ prediction head).

Projection maps between spaces — the basis of attention heads, LoRA,
    and dimensionality reduction.

Orthogonality guarantees safety — if a transform is orthogonal, it preserves
    all geometry, so you can reorganize data without corrupting distances.

Eigenvalues reveal true structure — which directions matter, which are noise,
    and whether a system is stable.

Covariance measures relationships — which dimensions carry redundant info,
    guiding where compression is cheap vs expensive.

PCA finds optimal coordinates — align axes with data structure so
    compression/truncation loses minimal information.

SVD decomposes any matrix — reveals rank, enables compression (LoRA),
    solves alignment problems (Procrustes/OPQ).

Random projection works because high-dimensional geometry is forgiving —
    you don't always need the optimal coordinates, random ones are often fine.
```

The recurring theme: choosing the right coordinate system makes hard problems easy. Rotation before quantization (OPQ), projection before attention (W_q, W_k, W_v), random projection before compression (QJL) — they're all the same idea. Transform the data so the next step works better.
