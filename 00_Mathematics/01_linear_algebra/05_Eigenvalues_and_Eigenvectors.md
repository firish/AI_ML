# Eigenvalues and Eigenvectors

---

## 1. A Matrix Is a Machine

From files 02 and 03, you know a matrix transforms vectors — it takes a vector in and produces a different vector out. What does "different" mean? The output can have a different direction, a different length, or both.

```
A × v = v'

The matrix can rotate, stretch, shrink, shear, flip, or collapse v.
Most matrices do several of these at once, differently in
different directions. A vector pointing north might get stretched.
One pointing east might get squished. One pointing northeast
might get rotated AND stretched.
```

Every direction gets its own treatment. It feels chaotic. Eigenvalues and eigenvectors cut through this chaos by finding the **simple directions** — the ones where the matrix does the simplest possible thing.

---

## 2. A Catalog of What Matrices Can Do

Before eigenvalues make sense, see each transformation type in isolation. All examples use input [1, 1] so you can compare.

### Uniform scaling

```
[3, 0]   [1]     [3]
[0, 3] × [1]  =  [3]

Both dims multiplied by 3. Same direction, 3× longer.
Like zooming in on a photo.

The equal diagonal values are the tell:
    [k, 0; 0, k]  = scale everything by k.
```

### Non-uniform scaling

```
[3, 0]   [1]     [3]
[0, 1] × [1]  =  [1]

x stretched 3×, y untouched. Direction changed — pulled toward x-axis.
Different diagonal values = different treatment per axis.
```

### Shrinking

```
[0.5, 0  ]   [1]     [0.5]
[0,   0.5] × [1]  =  [0.5]

Half the length. Applied 10 times: 0.5¹⁰ = 0.001 — nearly vanished.
This is vanishing gradients in miniature.
```

### Reflection (flip)

```
[1,  0]   [1]     [ 1]
[0, -1] × [1]  =  [-1]

y-axis negated. Mirrored across x-axis.
The -1 on the diagonal means "reverse this dimension."
```

### Collapse

```
[1, 0]   [1]     [1]
[0, 0] × [1]  =  [0]

y-dimension destroyed. [1,1], [1,2], [1,999] all map to [1,0].
Information lost. Not invertible (det = 0).
```

### Rotation

```
[0.707, -0.707]   [1]     [0   ]
[0.707,  0.707] × [1]  =  [1.41]

Rotated 45° counterclockwise. Length unchanged (1.41 ≈ √2 both times).
No stretching — just rotation. Orthogonal matrix (from file 04).
```

### Shear

```
[1, 1]   [1]     [2]
[0, 1] × [1]  =  [1]

x became x+y = 2. y unchanged. Dimensions MIXED.
Like a deck of cards sliding — rectangles become parallelograms.

Shear ≠ non-uniform scaling. Scaling reads each axis independently.
Shear reads ACROSS axes — output in one dim depends on input from another.
Diagonal values = independent scaling. Off-diagonal values = mixing.
```

### Reading a matrix at a glance

```
Diagonal only [a,0; 0,b]:    pure scaling, each axis independently
Off-diagonal values:          mixing between dimensions
Symmetric (A = Aᵀ):          scaling along perpendicular axes (no shear)
Orthogonal (Rᵀ×R = I):       pure rotation/reflection, no stretching
```

---

## 3. The Definition

An eigenvector is a direction where the matrix does the **simplest possible thing** — just scales. No rotation, no shearing, no mixing. The vector comes out pointing the exact same direction, just longer or shorter.

```
A × v = λ × v

    v = eigenvector    (the special direction)
    λ = eigenvalue     (the scale factor — a single number)
```

λ × v is just "the vector v, multiplied by the number λ." A number can only scale — it can't rotate or shear. So when A × v = λ × v, the matrix's entire complicated behavior collapses to simple multiplication by a number along that direction.

An n×n matrix always has exactly n eigenvalues (counting repeats). A 2×2 matrix has 2. A 768×768 matrix has 768.

---

## 4. Concrete Example

```
A = [2, 1]
    [1, 2]
```

Feed it some vectors and watch what happens.

### [1, 0] — gets rotated AND scaled

```
A × [1, 0] = [2×1 + 1×0] = [2]
              [1×1 + 2×0]   [1]

Input:  [1, 0]  — points east
Output: [2, 1]  — points northeast-ish

Direction changed. NOT an eigenvector.
```

### [0, 1] — same thing

```
A × [0, 1] = [1]
              [2]

Input:  [0, 1]  — points north
Output: [1, 2]  — points northeast-ish

Direction changed. Not an eigenvector.
```

### [1, 1] — only scales!

```
A × [1, 1] = [2+1] = [3]
              [1+2]   [3]

Input:  [1, 1]  — points at 45°
Output: [3, 3]  — points at 45°

SAME DIRECTION. Just 3× longer.
Eigenvector [1,1] with eigenvalue λ = 3.
```

### [1, -1] — only scales!

```
A × [1, -1] = [2-1] = [ 1]
               [1-2]   [-1]

Input:  [1, -1]  — points at -45°
Output: [1, -1]  — points at -45°

SAME DIRECTION. Same length (λ = 1).
Eigenvector [1,-1] with eigenvalue λ = 1.
```

### Summary

```
A = [2, 1]
    [1, 2]

Eigenvector [1, 1]   eigenvalue 3  → stretched 3×
Eigenvector [1, -1]  eigenvalue 1  → unchanged

Every other direction (infinitely many) gets rotated AND scaled.
These two directions only get scaled. That's what makes them special.
```

---

## 5. How to Find Them

### Step 1: Find eigenvalues

Start from A × v = λ × v. Rearrange: (A - λI) × v = 0.

For a non-zero v to get mapped to zero, (A - λI) must be singular (from file 03 — singular means det = 0, some dimension collapses). So:

```
det(A - λI) = 0
```

This gives a polynomial in λ. The roots are the eigenvalues.

```
A = [2, 1]
    [1, 2]

A - λI = [2-λ,  1 ]
         [1,   2-λ]

det = (2-λ)(2-λ) - (1)(1) = λ² - 4λ + 3 = (λ-3)(λ-1) = 0

Eigenvalues: λ = 3 and λ = 1    ✓  matches section 4
```

For a 2×2 matrix, it's always a quadratic — solvable by factoring or the quadratic formula. For 768×768, it's a degree-768 polynomial — you call `np.linalg.eig()`.

### Step 2: Find eigenvectors

For each eigenvalue, solve (A - λI) × v = 0.

```
λ = 3:
    (A - 3I) × v = 0
    [-1, 1] × [v₁] = [0]      row 1: -v₁ + v₂ = 0  →  v₁ = v₂
    [1, -1]   [v₂]   [0]      row 2:  v₁ - v₂ = 0  →  same thing

    Both rows say v₁ = v₂. Pick simplest: v = [1, 1]  ✓

λ = 1:
    (A - I) × v = 0
    [1, 1] × [v₁] = [0]       row 1: v₁ + v₂ = 0  →  v₁ = -v₂
    [1, 1]   [v₂]   [0]       row 2: same thing

    Both rows say v₁ = -v₂. Pick: v = [1, -1]  ✓
```

The rows are always redundant (that's guaranteed — we chose λ to make the matrix singular). So you get one relationship between the variables and pick the simplest values.

---

## 6. What the Eigenvalue Tells You

```
λ > 1:       stretches (amplifies) that direction
λ = 1:       leaves it completely unchanged
0 < λ < 1:   shrinks (dampens) that direction
λ = 0:       collapses it to zero (dimension destroyed)
λ < 0:       flips AND scales (reflection along that direction)
```

### Repeated application — where eigenvalues become powerful

```
Apply A once:    A × v = λ × v
Apply A twice:   A × (λv) = λ(A × v) = λ² × v
Apply A T times: Aᵀ × v = λᵀ × v
```

The eigenvalue **compounds**:

```
λ = 3,   T = 50:    3⁵⁰ = 7 × 10²³      EXPLOSION
λ = 0.9, T = 50:    0.9⁵⁰ = 0.005        vanishes to ~zero
λ = 1,   T = 50:    1⁵⁰ = 1              perfectly stable
```

This isn't abstract. This is literally what happens in RNNs.

---

## 7. Why This Matters — Vanishing and Exploding Gradients

An RNN applies the same weight matrix W at every time step:

```
h₁ = W × h₀
h₂ = W × h₁ = W² × h₀
h₃ = W × h₂ = W³ × h₀
...
hₜ = Wᵀ × h₀
```

Gradients flow backwards through those same multiplications. Decompose into eigenvector directions:

```
Say W has eigenvalues λ₁ = 1.1 and λ₂ = 0.8

After 50 steps:
    Component along eigenvector 1: 1.1⁵⁰ = 117       → exploding
    Component along eigenvector 2: 0.8⁵⁰ = 0.00001   → vanished

The gradient for the λ₂ direction is essentially zero.
The network CANNOT learn anything that depends on that direction
from 50 steps ago. The information is gone.
```

**LSTMs fix this:** The cell state flows through an element-wise multiply (the forget gate), not a full matrix multiply. The gate learns values near 1 for information worth remembering — keeping the effective eigenvalue at 1. The gates give explicit control: eigenvalue ≈ 1 = remember, eigenvalue ≈ 0 = forget.

**Transformers bypass this entirely:** No repeated matrix multiplication. Each token attends directly to any other via attention — the gradient flows directly from output to any input token. No compounding eigenvalues. No vanishing.

---

## 8. Why This Matters — PCA and Compression

The covariance matrix of your data has eigenvectors and eigenvalues that tell you the complete story of its shape.

```
Covariance matrix C: 768 × 768, symmetric

Eigenvectors:  768 perpendicular directions
Eigenvalues:   λ₁ ≥ λ₂ ≥ ... ≥ λ₇₆₈ ≥ 0

    λ₁ = variance along the most spread-out direction
    λ₇₆₈ = variance along the least spread-out direction
```

### 2D example

```
1000 points forming a tilted ellipse:

    •   • •
      • • • •
    • • • • • •       Long axis:  eigenvector 1, λ₁ = 10
      • • • •         Short axis: eigenvector 2, λ₂ = 0.5
        • •
```

Big eigenvalue = data is spread out (carries information). Small eigenvalue = data is bunched up (mostly noise).

### This IS PCA

```
768 eigenvalues, sorted:
    λ₁ = 45.2    (tons of variance — keep)
    λ₂ = 38.1    (keep)
    ...
    λ₅₀ = 2.3    (still meaningful — keep)
    λ₅₁ = 0.4    (getting tiny)
    ...
    λ₇₆₈ = 0.001 (noise)

If top 50 eigenvalues capture 95% of total variance:
    768 dims → 50 dims with 95% of information retained.
    The eigenvalues told you exactly where to cut.
```

Without eigenvalues, there's no principled way to decide what's signal vs noise.

---

## 9. Why This Matters — Optimization

### The condition number

```
condition number = λ_max / λ_min    (of the Hessian — second derivatives of loss)

    = 1:      perfect — all directions equally steep
    = 10:     mild
    = 10000:  terrible — optimization will zigzag
```

Imagine the loss landscape as a long, narrow valley:

```
                 ← narrow direction (small eigenvalue)
    ____________________________________
   /                                    \
  |  ←— gradient descent zigzags →     |      ← wide direction
   \____________________________________/       (large eigenvalue)

One direction is 1000× steeper than the other.
Gradient descent takes huge steps across the narrow dimension
and tiny steps along the long dimension. It zigzags.

High condition number = elongated valley = slow convergence.
```

This is why Adam works better than vanilla SGD — it normalizes each direction by its curvature (related to eigenvalues), so all directions converge at similar rates.

---

## 10. Key Properties

### Symmetric matrices are nice

```
If A = Aᵀ (matrix equals its transpose):
    1. All eigenvalues are REAL (no complex numbers)
    2. Eigenvectors are mutually PERPENDICULAR
    3. A = V × Λ × Vᵀ  (V orthogonal, Λ diagonal)

Why "perpendicular eigenvectors" matters:
    The standard axes (dim 0, dim 1) are always perpendicular —
    that's just how coordinate systems work. But eigenvector
    directions are NOT always perpendicular for general matrices.
    They can be at 30°, 72°, anything.

    Symmetric matrices guarantee eigenvectors are exactly 90° apart.
    They form a clean coordinate system you can rotate into,
    where the matrix becomes pure diagonal scaling.

    Covariance matrices are always symmetric → their eigenvectors
    (principal components) are always perpendicular. This is why
    PCA gives you a clean, non-redundant coordinate system.
```

### Trace and determinant (from file 03)

```
trace(A) = sum of diagonal = sum of eigenvalues
det(A)   = product of eigenvalues

    det = 0   → some eigenvalue is 0 → singular, not invertible
    trace = n → eigenvalues average 1 (for n×n matrix)
```

### Eigenvalues of special matrices

```
Identity I:           all λ = 1         (nothing changes)
Diagonal D:           λ = diagonal entries
Orthogonal R:         all |λ| = 1      (no stretching — from file 04)
Singular:             some λ = 0       (dimension collapsed)
Positive definite:    all λ > 0        (no collapse, no flip)
```

---

## 11. Eigenvalues vs Singular Values

These are related but answer different questions.

```
Eigenvalues:      A × v = λ × v        (only for square matrices)
Singular values:  A = U × Σ × Vᵀ       (for ANY matrix, any shape)
```

When they're the same:

```
For a symmetric positive semi-definite matrix (like a covariance matrix):
    singular values = eigenvalues

For a general matrix:
    singular values = √(eigenvalues of AᵀA)
    singular values are always ≥ 0
    eigenvalues can be negative or complex
```

When to use which:

```
Eigenvalues:       "What happens if I keep multiplying by this matrix?"
                   Stability, convergence, vanishing gradients.

Singular values:   "What's this matrix's one-time effect?"
                   Rank, best low-rank approximation, compression.
```

Both come up in ML. Eigenvalues when analyzing stability and covariance. Singular values when doing compression (LoRA, PCA via SVD, Procrustes). More on SVD in a later file.

---

## 12. The Full Picture

```
A matrix is a transformation. Most directions get rotated AND scaled.
Eigenvectors are the special directions that ONLY get scaled.
Eigenvalues are the scale factors.

    Repeated application:
        λ > 1   → EXPLODES         (exploding gradients)
        λ < 1   → VANISHES         (vanishing gradients)
        λ = 1   → STABLE           (what LSTMs aim for)

    Data analysis:
        Large λ → high variance → important → keep it
        Small λ → low variance  → noise     → discard it
        This IS PCA.

    Optimization:
        λ_max / λ_min = condition number
        High → elongated loss landscape → slow zigzag convergence
        Adam compensates by adapting per-direction learning rates.

    Matrix structure:
        All |λ| = 1  → orthogonal (pure rotation)
        Some λ = 0   → singular (dimension collapsed, not invertible)
        All λ > 0    → positive definite (well-behaved)

An n×n matrix has exactly n eigenvalues.
They are the matrix's DNA — they tell you everything about
what it does, how stable it is, what it preserves,
and what it destroys.
```

---

**Next:** `06_Covariance_PCA_SVD.md` — measuring data structure, finding optimal coordinates, and decomposing any matrix into rotate × scale × rotate.
