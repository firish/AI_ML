# Eigenvalues and Eigenvectors — Notes

---

## 1. Start Here: What Does a Matrix DO?

Before eigenvalues make sense, you need to think of a matrix not as a grid of numbers but as a **machine that transforms vectors**.

```
You feed a vector in. A different vector comes out.

    A × v = v'

    Input:  v  (some vector)
    Output: v' (a DIFFERENT vector — different direction, different length)
```

A 2×2 matrix transforms 2D vectors. A 768×768 matrix transforms 768-dim vectors. But in every case, the matrix takes in a vector and spits out a new one.

What can this transformation do? It can rotate, stretch, squish, shear, flip, or collapse vectors. Most matrices do several of these at once, differently in different directions. A vector pointing north might get stretched. A vector pointing east might get squished. A vector pointing northeast might get rotated AND stretched.

It feels chaotic — every direction gets a different treatment. Eigenvalues and eigenvectors are the tool that cuts through this chaos.

But first — let's see each type of transformation in isolation so you can recognize them.

---

## 2. A Catalog of What Matrices Can Do

Every example below is a 2×2 matrix acting on 2D vectors. The matrix multiplies on the left: output = A × input. All examples use the same input vector [1, 1] so you can compare.

### Uniform scaling (zoom in or out)

```
A = [3, 0]       A × [1] = [3]
    [0, 3]           [1]   [3]

Both dimensions multiplied by 3. The vector gets longer but keeps
its direction. Like zooming in on a photo — everything gets bigger,
nothing rotates.

The diagonal values are equal. That's the tell:
    [k, 0]
    [0, k]  = scale everything by k.
```

### Non-uniform scaling (stretch one axis more than the other)

```
A = [3, 0]       A × [1] = [3]
    [0, 1]           [1]   [1]

Dim 0 (x-axis) stretched 3×. Dim 1 (y-axis) unchanged.

Input [1, 1] pointed at 45°.
Output [3, 1] points much more toward the x-axis.
Direction changed — the vector got "pulled" toward the stretched axis.

    [a, 0]
    [0, b]  = stretch x by a, stretch y by b. Different values = different axes
              get different treatment. This is where eigenvectors start to matter.
```

### Shrinking (values between 0 and 1)

```
A = [0.5, 0  ]       A × [1] = [0.5]
    [0,   0.5]           [1]   [0.5]

Both dimensions multiplied by 0.5. Vector shrinks to half length,
same direction. This is what "eigenvalue < 1" looks like —
repeated application shrinks toward zero.

    Applied 10 times: 0.5¹⁰ = 0.001 — nearly vanished.
    This is vanishing gradients in miniature.
```

### Reflection (flip across an axis)

```
Flip across x-axis:

A = [1,  0]       A × [1] = [ 1]
    [0, -1]           [1]   [-1]

Dim 0 untouched. Dim 1 negated. The vector [1,1] becomes [1,-1] —
mirrored across the x-axis. Like flipping a photo upside down.

Flip across y-axis:

A = [-1, 0]       A × [1] = [-1]
    [ 0, 1]           [1]   [ 1]

The -1 on the diagonal is the tell. It means "reverse this dimension."
Eigenvalue = -1: the eigenvector direction gets flipped, not rotated.
```

### Collapse (projection — loses a dimension)

```
A = [1, 0]       A × [1] = [1]
    [0, 0]           [1]   [0]

Dim 0 kept. Dim 1 zeroed out. Every 2D point gets smashed onto
the x-axis. The vector [1,1] becomes [1,0].

Information is destroyed — you can't recover the original y-value.
This is what "eigenvalue = 0" means: that direction is annihilated.
The matrix is singular (not invertible) because you can't undo the collapse.

All of [1,1], [1,2], [1,999] map to [1,0]. One-way trip.
```

### Rotation (orthogonal matrix)

```
Rotate 45° counterclockwise:

A = [0.707, -0.707]       A × [1] = [0  ]
    [0.707,  0.707]           [1]   [1.41]

    (0.707 ≈ cos 45°, sin 45°)

Input [1,1] pointed at 45°. Output [0, 1.41] points straight up.
The vector rotated 45° counterclockwise. Length unchanged (1.41 ≈ √2 both times).

Key property: NO stretching. The matrix only rotates.
This is what "orthogonal" means — all eigenvalues have magnitude 1.
The general pattern:

    [cos θ, -sin θ]
    [sin θ,  cos θ]  = rotate by angle θ.

Important: individual values CAN change sign after rotation.
    [1, 0] rotated 180° becomes [-1, 0].
But the LENGTH is always preserved: ||input|| = ||output||, always.
And the ANGLE between any two vectors is preserved too.
Rotation changes coordinates, not geometry.
```

### Rotation vs reflection — both orthogonal, NOT the same

```
Both preserve distances, lengths, angles, dot products.
Both are orthogonal (Rᵀ × R = I).
But they differ in one way: the DETERMINANT.

    det(rotation)   = +1     preserves handedness
    det(reflection) = -1     reverses handedness

The determinant of a matrix is a single number that tells you
how the matrix scales area (2D) or volume (higher-D):

    det = 1:   area preserved exactly
    det = 2:   area doubled
    det = 0:   area collapsed to zero (singular, no inverse)
    det < 0:   area preserved but ORIENTATION FLIPPED

For a 2×2 matrix [a,b; c,d]:   det = ad - bc.

    Rotation 90°:   [0, -1; 1, 0]   det = 0×0 - (-1)×1 = +1   ✓ rotation
    Flip x-axis:    [1, 0; 0, -1]   det = 1×(-1) - 0×0 = -1   ✓ reflection

Think of the letter "R" printed on a sheet:
    Rotate it any amount → still "R", just tilted.
    Reflect it           → becomes "Я". No rotation can undo this.

Rotation is continuous (you can smoothly go from 0° to 180°).
Reflection is a jump (you flip the sheet over — no smooth path).
```

### Shear (tilt without rotating or scaling the axes)

```
A = [1, 1]       A × [1] = [2]
    [0, 1]           [1]   [1]

Dim 1 (y) is unchanged. But dim 0 (x) gets the y-value ADDED to it.
The vector [1,1] becomes [2,1] — it slid sideways.

Think of a deck of cards: the bottom card stays put, each card above
slides a bit more to the right. Rectangles become parallelograms.
Nothing gets longer or shorter along the axes, but angles change.

The 1 in the off-diagonal is the tell:
    [1, k]
    [0, 1]  = shear by k in the x-direction. Every point's x-coordinate
              gets k × its y-coordinate added.
```

### Shear vs non-uniform scaling — they look similar, they're not

```
Non-uniform scaling:     each axis reads ONLY its own value.

    [3, 0]   [x]     [3x]
    [0, 1] × [y]  =  [y]      x got tripled. y untouched. No mixing.

Shear:                   one axis reads ANOTHER axis's value.

    [1, 1]   [x]     [x+y]
    [0, 1] × [y]  =  [y]      x became x+y. Dimensions MIXED.

The difference is the off-diagonal. Diagonal entries scale each
axis independently. Off-diagonal entries mix axes together —
your output in one dimension depends on input from a different one.

Quick rule: diagonal-only = pure scaling. Off-diagonal = mixing.
```

### Combining transformations

Real matrices in ML do several of these at once. Every matrix can be decomposed into "rotate, then scale, then rotate again" (this is SVD — see `08_Linear_Algebra_for_ML.md`).

```
A = [2, 1]       What does this do?
    [1, 2]

It's not obvious from looking at it. But its eigenvalues are 3 and 1,
and its eigenvectors are [1,1] and [1,-1]. So:

    Along the [1,1] direction:   stretches by 3
    Along the [1,-1] direction:  unchanged (scale 1)

It stretches diagonally. In every other direction, it's a mix of
stretch and rotation. The eigenvalues and eigenvectors decompose
the complicated behavior into simple pieces.
```

### Reading a matrix at a glance

```
Diagonal-only matrix:          pure scaling (each axis independently)
    [a, 0]
    [0, b]

Diagonal with negative values: scaling + flipping
    [a,  0]
    [0, -b]

Off-diagonal values:           mixing between dimensions (shear, rotation)
    [a, c]
    [b, d]

Symmetric (A = Aᵀ):           no shear — only scaling along perpendicular axes
    [a, b]                     (the eigenvectors are perpendicular)
    [b, d]

    What "perpendicular eigenvectors" means:
    The standard axes (dim 0, dim 1) are always perpendicular —
    that's just how coordinate systems work. But EIGENVECTOR axes
    are different. They're the special directions where the matrix
    only scales. For a general non-symmetric matrix, eigenvectors
    can point in any directions — 30° apart, 72° apart, whatever.

    Symmetric matrices guarantee eigenvectors are exactly 90° apart.
    This means they form a clean coordinate system — you can rotate
    into those axes and the matrix becomes pure diagonal scaling.

        Non-symmetric:  eigenvectors at 40° and 70° — messy
        Symmetric:      eigenvectors at 45° and 135° — clean, perpendicular

    This is why PCA works cleanly — the covariance matrix is always
    symmetric, so its eigenvectors (principal components) are always
    perpendicular. They form a proper coordinate system to rotate into.

Orthogonal (Rᵀ×R = I):        pure rotation (or reflection)
    [cos θ, -sin θ]           no stretching at all
    [sin θ,  cos θ]
```

---

## 3. The One Sentence Definition

An eigenvector is a direction where the matrix does the simplest possible thing: it just **scales**. No rotation, no shearing. The vector comes out pointing in the exact same direction, just longer or shorter.

```
A × v = λ × v

    v = eigenvector    (the special direction)
    λ = eigenvalue     (how much it scales)
```

λ × v means "the same vector v, multiplied by the number λ." So the output is just a scaled copy of the input. The matrix, which normally does complicated things to vectors, does nothing but stretch or shrink along this one direction.

---

## 4. Concrete Example — See It With Numbers

Take this matrix:

```
A = [2, 1]
    [1, 2]
```

Let's feed it some vectors and see what comes out.

### A random vector — gets rotated AND scaled

```
v = [1, 0]

A × [1, 0] = [2×1 + 1×0] = [2]
              [1×1 + 2×0]   [1]

Input:  [1, 0]  — points east
Output: [2, 1]  — points northeast-ish

Direction changed. This is NOT an eigenvector.
```

### Try another — same thing

```
v = [0, 1]

A × [0, 1] = [2×0 + 1×1] = [1]
              [1×0 + 2×1]   [2]

Input:  [0, 1]  — points north
Output: [1, 2]  — points northeast-ish

Direction changed again. Not an eigenvector.
```

### Now try [1, 1]

```
v = [1, 1]

A × [1, 1] = [2×1 + 1×1] = [3]
              [1×1 + 2×1]   [3]

Input:  [1, 1]  — points at 45°
Output: [3, 3]  — points at 45°

SAME DIRECTION. Just 3× longer.

This IS an eigenvector, with eigenvalue λ = 3.
```

### And try [1, -1]

```
v = [1, -1]

A × [1, -1] = [2×1 + 1×(-1)] = [1]
               [1×1 + 2×(-1)]   [-1]

Input:  [1, -1]   — points at -45°
Output: [1, -1]   — points at -45°

SAME DIRECTION. Same length (λ = 1 — unchanged).

Eigenvector with eigenvalue λ = 1.
```

### Summary for this matrix

```
A = [2, 1]
    [1, 2]

Eigenvector [1, 1]  with eigenvalue 3   → stretched 3×
Eigenvector [1, -1] with eigenvalue 1   → unchanged

Every other direction gets rotated AND scaled.
But these two directions only get scaled.
```

---

## 5. What the Eigenvalue Tells You

The eigenvalue λ is the scale factor along that eigenvector direction.

```
λ > 1:      stretches (amplifies) that direction
λ = 1:      leaves it completely unchanged
0 < λ < 1:  shrinks (dampens) that direction
λ = 0:      collapses it to zero (that dimension is destroyed)
λ < 0:      flips it AND scales it
```

### What does applying the matrix REPEATEDLY do?

This is where eigenvalues become powerful. If you apply A twice:

```
A × A × v = A × (λv) = λ × (A × v) = λ × λ × v = λ² × v
```

After T applications:

```
Aᵀ × v = λᵀ × v      (T-th power of the eigenvalue)
```

Now the eigenvalue's size matters enormously:

```
λ = 3, applied 50 times:   3⁵⁰ = 7 × 10²³       → EXPLOSION
λ = 0.9, applied 50 times: 0.9⁵⁰ = 0.005          → vanishes to ~zero
λ = 1, applied 50 times:   1⁵⁰ = 1                 → perfectly stable
```

This is not abstract. This is literally what happens in a recurrent neural network, where the weight matrix is applied at every time step.

---

## 6. Why This Matters — Vanishing and Exploding Gradients

An RNN processes a sequence by applying the same weight matrix W at every step:

```
h₁ = W × h₀
h₂ = W × h₁ = W² × h₀
h₃ = W × h₂ = W³ × h₀
...
hₜ = Wᵀ × h₀
```

During backpropagation, the gradient flows backwards through those same multiplications. After T time steps, the gradient is proportional to Wᵀ.

Now decompose the hidden state h₀ into eigenvector directions:

```
Say W has eigenvalues λ₁ = 1.1 and λ₂ = 0.8

After T = 50 steps:
    Component along eigenvector 1: scaled by 1.1⁵⁰ = 117       → exploding
    Component along eigenvector 2: scaled by 0.8⁵⁰ = 0.00001   → vanished

The gradient for the λ₂ direction is essentially zero.
The network CAN'T LEARN anything that depends on that direction
from 50 time steps ago. The information is gone.
```

This is the vanishing gradient problem, made precise. It's not vague — the eigenvalues of the weight matrix determine exactly which directions survive and which die over time.

**Why LSTMs fix this:** The cell state in an LSTM flows through a simple element-wise multiply (the forget gate), not a full matrix multiply. The forget gate learns values near 1 for information that should be remembered, which is equivalent to keeping the effective eigenvalue at 1 for that component. The gates give the network explicit control over which eigenvalues are near 1 (remember) vs near 0 (forget).

**Why transformers bypass this entirely:** Transformers don't apply the same matrix repeatedly. Each token can attend directly to any other token via attention — there's no chain of matrix multiplies where eigenvalues compound. The gradient flows directly from output to any input token. No vanishing, no exploding.

---

## 7. Why This Matters — PCA and Compression

The covariance matrix of your data describes how much variance exists in each direction. Its eigenvectors and eigenvalues tell you the complete story:

```
Data: 1000 points in 768 dimensions

Covariance matrix C: 768 × 768, symmetric

Eigenvectors of C: 768 directions, mutually perpendicular
Eigenvalues of C:  λ₁ ≥ λ₂ ≥ ... ≥ λ₇₆₈ ≥ 0

    λ₁ = the amount of variance in direction 1 (the most spread-out direction)
    λ₂ = the amount of variance in direction 2 (perpendicular to direction 1)
    ...
    λ₇₆₈ = the amount of variance in the LEAST spread-out direction
```

### Toy example: 2D data

```
Imagine 1000 points forming a tilted ellipse:

    •   • •
      • • • •
    • • • • • •         Long axis: most variance
      • • • •           Short axis: least variance
        • •

Covariance matrix eigenvectors:
    v₁ = direction of the LONG axis       eigenvalue λ₁ = 10  (lots of spread)
    v₂ = direction of the SHORT axis      eigenvalue λ₂ = 0.5 (little spread)
```

The eigenvalues quantify the shape of your data cloud. Big eigenvalue = the data is spread out in that direction (that direction carries information). Small eigenvalue = the data is bunched up (that direction is mostly noise).

### This IS PCA

PCA just sorts the eigenvectors by eigenvalue and keeps the top-k:

```
768 eigenvalues, sorted:
    λ₁ = 45.2    (this direction has tons of variance — keep it)
    λ₂ = 38.1    (keep)
    ...
    λ₅₀ = 2.3    (still meaningful — keep)
    λ₅₁ = 0.4    (getting tiny)
    ...
    λ₇₆₈ = 0.001 (basically zero — noise)

If λ₁ + λ₂ + ... + λ₅₀ = 95% of total variance,
you can throw away dims 51-768 and lose only 5%.

768 dims → 50 dims with 95% of information retained.
The eigenvalues told you exactly where to cut.
```

Without eigenvalues, you'd have no principled way to decide which dimensions are signal vs noise.

---

## 8. Why This Matters — Understanding What a Matrix Amplifies

Every matrix has its own "personality" — directions it likes (amplifies) and directions it suppresses. Eigenvalues are the X-ray that reveals this.

### The condition number

```
condition number = λ_max / λ_min

    = 1:      perfect — all directions treated equally
    = 10:     mild — some directions 10× more sensitive
    = 10000:  terrible — optimization is nearly impossible
```

Why this matters for gradient descent: imagine a loss landscape shaped like a long, narrow valley.

```
                     ← narrow direction (small eigenvalue of Hessian)
    ____________________________________
   /                                    \
  |  ←— gradient descent path zigzags → |      ← wide direction
   \____________________________________/       (large eigenvalue)

The gradient points toward the minimum, but because one direction
is 1000× steeper than the other, gradient descent oscillates —
taking huge steps across the narrow dimension and tiny steps
along the long dimension.

High condition number = elongated valley = slow, zigzag convergence.
```

This is why optimizers like Adam work better than vanilla gradient descent — they effectively normalize each direction by its curvature (related to eigenvalues of the Hessian), so all directions converge at similar rates.

---

## 9. How to Find Them (The Mechanics)

You don't compute eigenvalues by hand in practice. But understanding the procedure builds intuition.

### The characteristic equation

Start from the definition: A × v = λ × v

Rearrange: A × v - λ × v = 0 → (A - λI) × v = 0

This says: the matrix (A - λI) maps v to zero. For a non-zero v to get mapped to zero, the matrix (A - λI) must be "singular" (not invertible, some direction gets collapsed). That happens when:

```
det(A - λI) = 0
```

This gives you a polynomial in λ. The roots are the eigenvalues.

### Worked example

```
A = [2, 1]
    [1, 2]

A - λI = [2-λ, 1  ]
         [1,   2-λ]

det = (2-λ)(2-λ) - (1)(1) = λ² - 4λ + 3 = (λ-3)(λ-1)

Eigenvalues: λ = 3 and λ = 1    ← matches what we found in section 4!
```

For each eigenvalue, find the eigenvector by solving (A - λI) × v = 0:

```
λ = 3:  (A - 3I) × v = 0
        [-1, 1] × [v₁] = [0]
        [1, -1]   [v₂]   [0]

        → v₁ = v₂ → eigenvector = [1, 1]  ✓

λ = 1:  (A - I) × v = 0
        [1, 1] × [v₁] = [0]
        [1, 1]   [v₂]   [0]

        → v₁ = -v₂ → eigenvector = [1, -1]  ✓
```

### In practice

For a 768×768 matrix, solving this polynomial (degree 768) is not feasible by hand. Libraries use iterative algorithms (QR decomposition, power iteration, Lanczos) that converge to eigenvalues numerically. You call `np.linalg.eig(A)` and get them. But the insight — "eigenvalues are the roots of the characteristic polynomial" — explains why an n×n matrix has exactly n eigenvalues (counting multiplicity), why symmetric matrices always have real eigenvalues, and why eigenvalues can be complex for non-symmetric matrices.

---

## 10. Key Properties Worth Knowing

### Symmetric matrices are nice

```
If A = Aᵀ (the matrix equals its transpose):
    1. All eigenvalues are REAL (no complex numbers)
    2. Eigenvectors are mutually PERPENDICULAR (orthogonal)
    3. A = V × Λ × Vᵀ  where V is orthogonal, Λ is diagonal

This is called the "spectral theorem" and it's why PCA works cleanly.
Covariance matrices are always symmetric, so their eigenvectors
are always perpendicular directions — a clean coordinate system.
```

### Trace and determinant

```
trace(A) = sum of diagonal entries = sum of eigenvalues
det(A) = product of eigenvalues

These are useful sanity checks:
    det = 0  → at least one eigenvalue is 0 → matrix is singular
    trace = 768 and matrix is 768×768 → eigenvalues average 1
```

### Eigenvalues of special matrices

```
Identity matrix I:        all eigenvalues = 1  (nothing changes)
Diagonal matrix D:        eigenvalues = the diagonal entries
Orthogonal matrix R:      eigenvalues have |λ| = 1 (no stretching)
Singular (rank-deficient): some eigenvalues = 0 (dimensions collapsed)
Positive definite:        all eigenvalues > 0 (no direction collapsed or flipped)
```

Covariance matrices are always positive semi-definite (eigenvalues ≥ 0), because variance can't be negative.

---

## 11. Eigenvalues vs Singular Values

These are related but different, and it's easy to confuse them.

```
Eigenvalues:     A × v = λ × v         (only for square matrices)
Singular values: A = U × Σ × Vᵀ        (for ANY matrix, any shape)

For a symmetric positive semi-definite matrix (like a covariance matrix):
    singular values = eigenvalues       (they're the same!)

For a general matrix:
    singular values = square roots of eigenvalues of AᵀA
    singular values are always ≥ 0
    eigenvalues can be negative or complex
```

Why the distinction matters:

- **Eigenvalues** tell you about the matrix's behavior when applied repeatedly (stability, convergence, vanishing gradients). They answer: "what happens if I keep multiplying by this matrix?"

- **Singular values** tell you about the matrix's one-time action — how much it stretches in each direction. They answer: "what's the effective rank?" and "what's the best low-rank approximation?"

SVD works on any matrix. Eigendecomposition only works on square matrices and is cleanest for symmetric ones. In ML, you'll hit both — eigenvalues when analyzing stability and covariance, singular values when doing compression (LoRA, PCA via SVD, Procrustes).

---

## 12. The Full Picture

```
A matrix is a transformation. Most directions get rotated AND scaled.
Eigenvectors are the special directions that only get scaled.
Eigenvalues are the scale factors.

This matters because:

    Repeated application:
        eigenvalue > 1  → that direction EXPLODES     (exploding gradients)
        eigenvalue < 1  → that direction VANISHES      (vanishing gradients)
        eigenvalue = 1  → that direction is STABLE     (what LSTMs aim for)

    Data analysis:
        Large eigenvalue → high variance → important direction → keep it
        Small eigenvalue → low variance  → noise direction    → discard it
        This IS PCA. Eigenvalues tell you where to cut.

    Optimization:
        λ_max / λ_min = condition number
        High condition number → elongated loss landscape → slow convergence
        Adam-style optimizers compensate by adapting per-direction learning rates.

    Matrix structure:
        All eigenvalues |λ| = 1 → orthogonal matrix → pure rotation, no distortion
        Some eigenvalues = 0   → singular matrix → some dimensions collapsed
        All eigenvalues > 0    → positive definite → invertible, well-behaved

A matrix's eigenvalues are its DNA. They tell you everything about
what it does, how stable it is, what it preserves, and what it destroys.
```
