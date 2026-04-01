# Orthogonal Matrices and Rotation

---

## 1. The Definition

An orthogonal matrix R satisfies:

```
Rᵀ × R = I       (the transpose IS the inverse)
```

This means two things about R's columns:
- Every column has length 1 (unit vectors)
- Every pair of columns is perpendicular (dot product = 0)

```
R = [0.6, -0.8]       column 0 = [0.6, 0.8]    length = √(0.36+0.64) = 1  ✓
    [0.8,  0.6]       column 1 = [-0.8, 0.6]    length = √(0.64+0.36) = 1  ✓

    col0 · col1 = 0.6×(-0.8) + 0.8×0.6 = -0.48 + 0.48 = 0  ✓  perpendicular

Verify: Rᵀ × R = [0.6,  0.8] × [0.6, -0.8] = [1, 0] = I  ✓
                  [-0.8, 0.6]   [0.8,  0.6]   [0, 1]
```

---

## 2. What Orthogonal Matrices Do

An orthogonal matrix is a pure **rotation** (and possibly reflection). Nothing else. No stretching, no squishing, no shearing, no collapsing.

```
For ANY two vectors a and b:

    ||R × a|| = ||a||                  lengths preserved
    ||R×a - R×b|| = ||a - b||          distances preserved
    (R×a) · (R×b) = a · b             dot products preserved
    angle(R×a, R×b) = angle(a, b)     angles preserved
```

Everything about the geometry is identical after the transformation. The data cloud is the same shape — you've just spun the coordinate grid underneath it.

### Why this is a strong guarantee

Most matrices distort geometry. A scaling matrix changes distances. A shear changes angles. A projection destroys dimensions entirely. An orthogonal matrix is the ONLY type of matrix that preserves everything.

```
Scaling [3,0; 0,1]:     distances change, angles change
Shear [1,1; 0,1]:       angles change, areas preserved
Projection [1,0; 0,0]:  dimension collapsed, distances change
Orthogonal:              NOTHING changes except orientation
```

---

## 3. Why This Matters in AI

The guarantee "nothing changes except orientation" is extremely powerful because most of AI is built on distances and similarities.

### Nearest neighbors survive rotation

```
Before rotation:
    "king" is closest to "queen"
    distance("king", "queen") = 2.3

After rotation:
    "king" is STILL closest to "queen"
    distance("king", "queen") = 2.3      (exactly the same)

Every search result, every ranking, every similarity score
is identical before and after. The rotation is invisible
to any distance-based algorithm.
```

This is why OPQ can rotate your entire embedding database before quantization. The rotation reorganizes which dimensions carry which information (to make PQ's subvector splits better), without corrupting a single search result.

### Attention scores survive rotation

```
score = Q · K

After rotating both Q and K by the same R:
    score = (R×Q) · (R×K) = Q · K      (dot product preserved)

Attention scores are identical. The model's behavior doesn't change.
```

This is why RoPE works — it rotates Q and K by different amounts based on position, and the dot product Q·K depends only on the RELATIVE rotation (relative position), not the absolute orientations.

### The inverse is free

```
Undoing a rotation: R⁻¹ = Rᵀ

Computing a transpose is essentially free — just read the matrix
in the other order. No expensive inverse computation needed.

At query time in OPQ:
    Rotate the query:   q' = R × q          (one matrix multiply)
    Search normally against rotated database
    If needed, rotate results back: v = Rᵀ × v'   (one transpose multiply)
```

---

## 4. The 2D Rotation Matrix

Every 2D rotation is:

```
R(θ) = [cos θ, -sin θ]
       [sin θ,  cos θ]

Rotates any vector by angle θ counterclockwise.
```

### Worked examples

```
Rotate [1, 0] by 90°:

    [cos 90°, -sin 90°]   [1]     [ 0]
    [sin 90°,  cos 90°] × [0]  =  [ 1]

    (1,0) → (0,1). East → North. 90° counterclockwise. ✓


Rotate [1, 1] by 45°:

    cos 45° ≈ 0.707,  sin 45° ≈ 0.707

    [0.707, -0.707]   [1]     [0   ]
    [0.707,  0.707] × [1]  =  [1.41]

    (1,1) → (0, 1.41). Was at 45°, now at 90°. Rotated 45° more. ✓


Rotate [3, 4] by 0° (no rotation):

    [1, 0]   [3]     [3]
    [0, 1] × [4]  =  [4]

    R(0°) = the identity matrix. Does nothing. ✓
```

### Verify it's orthogonal

```
R(θ)ᵀ × R(θ) = [cos θ,  sin θ] × [cos θ, -sin θ]
                [-sin θ, cos θ]   [sin θ,  cos θ]

              = [cos²θ + sin²θ,            0        ]
                [       0,         cos²θ + sin²θ     ]

              = [1, 0] = I  ✓
                [0, 1]

Uses the identity cos²θ + sin²θ = 1.
```

### Verify lengths are preserved

```
Input: [a, b],  length = √(a² + b²)

Output: [a×cosθ - b×sinθ,  a×sinθ + b×cosθ]

Length² = (a×cosθ - b×sinθ)² + (a×sinθ + b×cosθ)²
        = a²cos²θ - 2ab×cosθsinθ + b²sin²θ
        + a²sin²θ + 2ab×sinθcosθ + b²cos²θ
        = a²(cos²θ + sin²θ) + b²(sin²θ + cos²θ)
        = a² + b²

Length = √(a² + b²)    same as input. ✓
```

---

## 5. Higher-Dimensional Rotations

In 2D, there's only one plane to rotate in. In higher dimensions, there are many independent planes.

### Block-diagonal structure

A rotation in 768 dimensions is typically built from many independent 2D rotations stacked:

```
R = [R₁  0   0   0  ...]
    [0   R₂  0   0  ...]
    [0   0   R₃  0  ...]
    [0   0   0   R₄ ...]
    [...                ]

Each R_i is a 2×2 rotation matrix acting on a pair of dimensions.
R₁ rotates dims (0,1), R₂ rotates dims (2,3), R₃ rotates dims (4,5), etc.

Each pair is rotated independently. The rotations don't interfere.
```

This is exactly how RoPE works:

```
RoPE for a token at position p:

    Dims (0,1):  rotated by angle p × θ₁
    Dims (2,3):  rotated by angle p × θ₂
    Dims (4,5):  rotated by angle p × θ₃
    ...

    θ_i = 1 / 10000^(2i/d)     (different frequency per pair)

Token at position 5 gets a different rotation than position 10.
When you compute Q·K, the dot product depends on the DIFFERENCE
in rotations = the DIFFERENCE in positions = relative position.
```

### General rotations (not block-diagonal)

OPQ's rotation matrix is a full 768×768 orthogonal matrix, not block-diagonal. It can rotate in any plane, not just axis-aligned pairs. This is more expressive — it can mix any dimensions with any other dimensions.

```
Block-diagonal:  each pair of dimensions rotates independently
                 dim 0 only mixes with dim 1, dim 2 only with dim 3, etc.

Full orthogonal: any dimension can mix with any other
                 dim 0 can blend information from dims 47, 300, 512, etc.

OPQ needs full rotation to push correlated dimensions
(which might be anywhere) into the same subvector block.
RoPE only needs pair-wise rotation because position encoding
works naturally in 2D planes.
```

---

## 6. Rotation vs Reflection

Both are orthogonal. Both preserve all geometry. But they're fundamentally different.

```
Rotation:     det(R) = +1     continuous, smooth
Reflection:   det(R) = -1     a jump, a flip
```

### The "R" test

```
Rotate the letter "R" any amount → still "R", just tilted.
Reflect the letter "R"           → becomes "Я". No rotation undoes this.

Rotation preserves handedness (clockwise stays clockwise).
Reflection reverses it (clockwise becomes counterclockwise).
```

### Combining reflections

```
Two reflections = one rotation.

Reflect across x-axis, then across y-axis:
    [1,0; 0,-1] × [-1,0; 0,1] = [-1,0; 0,-1]

    det = (-1)×(-1) = +1.  It's a rotation (180°). ✓

An odd number of reflections = reflection.
An even number = rotation.
```

### In ML

When papers say "orthogonal matrix" in the context of OPQ or PolarQuant, they typically mean rotation (det = +1). The distinction rarely matters in practice — both preserve geometry equally well. But if you're reading proofs, det = +1 vs -1 can come up.

---

## 7. Orthogonal Matrices and Eigenvalues

The eigenvalues of an orthogonal matrix always have magnitude 1.

```
|λ| = 1 for every eigenvalue of an orthogonal matrix.

This means: λ = +1, λ = -1, or λ is complex with |λ| = 1.
```

### Why this makes sense

Eigenvalues are the scale factors along eigenvector directions. Orthogonal matrices don't scale anything — they only rotate and reflect. So the scale factor must be exactly 1 (in magnitude):

```
λ = +1:    this direction is completely unchanged
λ = -1:    this direction is flipped (reflected)
λ = e^(iθ): this direction is rotated (complex eigenvalue)
```

### Contrast with other matrices

```
Scaling matrix [3,0; 0,2]:      eigenvalues 3 and 2 (stretches)
Orthogonal matrix:               eigenvalues all |λ|=1 (no stretching)
Singular matrix [1,0; 0,0]:     eigenvalue 0 (collapses)

Orthogonal matrices never have λ=0 (nothing collapses)
and never have |λ|>1 or |λ|<1 (nothing stretches or shrinks).
```

This is why repeated application of an orthogonal matrix doesn't explode or vanish:

```
λ = 1 applied 1000 times: 1¹⁰⁰⁰ = 1      (stable)
λ = 1.1 applied 1000 times: 1.1¹⁰⁰⁰ → ∞   (explosion)
λ = 0.9 applied 1000 times: 0.9¹⁰⁰⁰ → 0    (vanishes)
```

---

## 8. Building Intuition: What Rotation Does to a Point Cloud

Imagine 1000 embedding vectors in 768-dim space. They form a cloud with some shape — clusters, corridors, dense regions, sparse regions.

### Before rotation

```
The cloud has some shape relative to the standard axes.
Dim 0 might capture "topic", dim 47 might capture "sentiment",
dim 300 might capture "formality."

PQ chops at every 96 dims: [0-95 | 96-191 | 192-287 | ...]
If "topic" (dim 0) and "sentiment" (dim 47) are correlated,
they're in the same block — fine.
If "topic" (dim 0) and some feature at dim 100 are correlated,
they're split across blocks — bad for PQ.
```

### After rotation

```
The cloud has the EXACT SAME SHAPE. Same clusters, same corridors,
same distances between every pair of points.

But the AXES have been reoriented. Now the PQ block boundaries
[0-95 | 96-191 | ...] happen to align with the cloud's natural
structure. Correlated dimensions land in the same block.

Nothing about the data changed. The coordinate labels changed.
PQ's job got easier because the arbitrary cuts now match the data.
```

### The key insight

```
The data's structure is fixed — it's determined by the embeddings.
The axis labels are arbitrary — they were assigned by the training process.
Rotation lets you CHOOSE better axis labels for a specific purpose
(like PQ quantization) without disturbing any property of the data.
```

---

## 9. Practical Properties

### Composing rotations

```
R₁ × R₂ = another orthogonal matrix (another rotation/reflection)

Rotating then rotating again is just one combined rotation.
The set of orthogonal matrices is "closed" under multiplication.
```

### Orthogonal matrices are always square

```
An (m × n) matrix with m ≠ n cannot be orthogonal.
Orthogonality requires Rᵀ × R = I, which needs a square matrix.

Rectangular matrices CAN have orthonormal columns
(each column unit length, all pairs perpendicular),
but they're not called "orthogonal matrices."
They're called matrices with "orthonormal columns."
```

### Determinant is always ±1

```
det(Rᵀ × R) = det(I) = 1
det(Rᵀ) × det(R) = 1
det(R)² = 1
det(R) = +1 or -1

No other determinant is possible for an orthogonal matrix.
+1 = rotation. -1 = reflection.
```

---

## 10. Summary

```
Orthogonal:  Rᵀ × R = I
    What it does:       rotates (and possibly reflects) — nothing else
    What it preserves:  lengths, distances, dot products, angles — everything
    Inverse:            Rᵀ (free — just read the matrix transposed)
    Eigenvalues:        all |λ| = 1 (no stretching, no collapse)
    Determinant:        +1 (rotation) or -1 (reflection)
    Why it's safe:      nearest neighbors, attention scores, all similarity
                        measures are identical before and after

Where it appears:
    OPQ         full 768×768 rotation to align data with PQ block structure
    RoPE        block-diagonal 2D rotations encoding token position
    PolarQuant  random orthogonal rotation to concentrate angle distributions
    Proofs      whenever you need "this transform doesn't change anything important"
```

The core reason orthogonal matrices are everywhere in AI: they let you **reorganize data without corrupting it**. Any time you need to rearrange, reorient, or remap — but you can't afford to change distances — an orthogonal matrix is the tool.

---

**Next:** `05_Eigenvalues_and_Eigenvectors.md` — what a matrix actually does to each direction, why gradients vanish, and how to find them.
