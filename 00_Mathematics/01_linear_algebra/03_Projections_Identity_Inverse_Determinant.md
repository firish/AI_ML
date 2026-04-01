# Projections, Identity, Inverse, and Determinant

---

## 1. What Is a Projection?

A projection takes a vector in one space and maps it to a different space, usually lower-dimensional.

```
W: (768 × 64)

    input × W = output
    (1 × 768) × (768 × 64) = (1 × 64)

    768 dimensions in, 64 dimensions out.
```

This is NOT just dropping 704 dimensions. Each of the 64 output values is a **learned weighted sum** of all 768 input dimensions. The matrix learns which combinations of input features matter for this particular purpose.

### The simplest projection: dropping a dimension

```
W = [1, 0]
    [0, 1]       (3 × 2) — projects from 3D to 2D
    [0, 0]

    [x, y, z] × W = [x, y]

This literally drops the z-coordinate. But it's the dumbest possible
projection — it ignores all information in dim 2.
```

### A learned projection: mixing before dropping

```
W = [0.7, 0.3]
    [0.2, 0.8]     (3 × 2)
    [0.5, 0.1]

    [x, y, z] × W = [0.7x + 0.2y + 0.5z,  0.3x + 0.8y + 0.1z]

Each output dimension is a BLEND of all three input dimensions.
The network learned these weights to keep the most useful information
while compressing from 3 to 2 dimensions.
```

---

## 2. Projections in Transformers

### Attention projections (the most important example)

The same input token gets projected three different ways:

```
x: (1 × 768)    — one token's embedding

q = x × W_q     (1×768) × (768×64) = (1×64)    "What am I looking for?"
k = x × W_k     (1×768) × (768×64) = (1×64)    "What do I contain?"
v = x × W_v     (1×768) × (768×64) = (1×64)    "What should I pass along?"
```

Three different 768→64 projections. Each one extracts a different "view" of the same input. The word "bank" needs a different query (looking for financial vs river context) than its key (what it advertises to other tokens). Separate projection matrices let the model learn these different roles.

### FFN expansion and compression

```
Transformer FFN (two layers):

    hidden = input × W₁        (768 × 3072)   — expand
    hidden = activation(hidden)                 — nonlinearity
    output = hidden × W₂        (3072 × 768)   — compress back

W₁ projects UP to a high-dimensional space where the activation
function can selectively switch features on and off.
W₂ projects back DOWN to the original dimension.
```

### LoRA: low-rank projection for cheap fine-tuning

```
Instead of learning a full (768 × 768) weight update:

    ΔW = A × B     where A: (768 × 8), B: (8 × 768)

The input gets projected DOWN to 8 dimensions (through A),
then back UP to 768 (through B).

    768 → 8 → 768

The bottleneck at 8 forces the update to only modify 8 directions
in the space. Full update: 590K parameters. LoRA: 12K parameters.
This works because fine-tuning typically only needs to adjust
a handful of directions, not all 590K.
```

---

## 3. Subspaces — Where Projections Land

When a (768 × 64) matrix projects from 768-dim to 64-dim, the output lives in a **64-dimensional subspace** of the original 768-dim space.

```
Think of it geometrically:

    768-dim space = a vast room with 768 independent directions
    64-dim subspace = a flat "hyperplane" inside that room

    The projection maps every 768-dim point onto that hyperplane.
    Points not on the hyperplane lose their off-plane components.
```

### Intuition: projecting a shadow

```
3D → 2D projection = casting a shadow:

    A 3D object (cube, sphere) gets squished onto a 2D wall.
    The shadow keeps the shape's outline but loses depth.

    Similarly, a 768→64 projection keeps the "outline" in 64 directions
    but loses all structure in the other 704 directions.
```

### The columns of the matrix define the subspace

```
W: (768 × 64)

    Column 0 of W: a 768-dim vector — one direction in the output subspace
    Column 1 of W: another 768-dim vector — another direction
    ...
    Column 63 of W: the 64th direction

    These 64 column vectors SPAN the output subspace.
    The output is always a linear combination of these 64 directions.
    No matter what input you feed in, the output can only land in
    the space defined by these 64 columns.
```

This connects to rank: if some of those 64 columns are redundant (one is a linear combination of others), the effective subspace is smaller than 64. The **rank** of the matrix = the true number of independent directions it uses.

---

## 4. Rank — The True Dimensionality of a Matrix

The rank of a matrix is the number of independent columns (or equivalently, independent rows).

```
Full rank:
    W = [1, 0]
        [0, 1]       rank 2 — both columns are independent
        [0, 0]

Rank deficient:
    W = [1, 2]
        [2, 4]       rank 1 — column 2 is just 2× column 1
        [3, 6]

    This matrix looks like it maps to 2D, but column [2,4,6] = 2 × [1,2,3].
    Both columns point in the same direction. The output is actually 1D —
    everything lands on a line, not a plane.
```

### Why rank matters

```
A (768 × 768) matrix with rank 768: full rank, uses all dimensions.
A (768 × 768) matrix with rank 50:  only uses 50 directions.
    The other 718 dimensions are collapsed or redundant.
    It LOOKS like a full-size transformation but most of it is wasted.

This is why LoRA works:
    The weight UPDATE during fine-tuning is approximately rank 8-64.
    You don't need 590K parameters to represent a rank-8 change.
    LoRA parametrizes ΔW = A × B as rank-r by construction.
```

---

## 5. The Identity Matrix — "Do Nothing"

The identity matrix I has 1s on the diagonal and 0s everywhere else.

```
I = [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]

A × I = A       I × A = A       I × v = v

Multiplying by I changes nothing. It's the matrix equivalent of × 1.
```

### Why we care

The identity is the **reference point** for "no transformation." This comes up in:

```
Orthogonal matrices:   Rᵀ × R = I
    "Applying R then undoing it gets you back to the start."

Residual connections:  output = layer(x) + x
    If the layer outputs zeros, this reduces to output = x.
    The skip connection acts like the identity — a "do nothing" path.

LoRA initialization:   ΔW starts at 0, so W + ΔW = W
    The model starts behaving identically to the original.
    Fine-tuning gradually moves away from identity.
```

---

## 6. The Inverse — "Undo"

The inverse A⁻¹ is the matrix that perfectly undoes A.

```
A × A⁻¹ = I       A⁻¹ × A = I

If:   y = A × x        (apply A)
Then: x = A⁻¹ × y      (undo A — recover original x)
```

### When does an inverse exist?

Only when the matrix doesn't destroy information:

```
A = [2, 0]       A⁻¹ = [0.5, 0  ]
    [0, 3]              [0,   0.33]

    A scales x by 2 and y by 3.
    A⁻¹ scales x by 1/2 and y by 1/3. Perfectly undone.

A = [1, 0]       A⁻¹ = ??? DOES NOT EXIST
    [0, 0]

    A zeroes out the y-dimension. Given output [5, 0],
    was the input [5, 0]? [5, 1]? [5, 999]? No way to know.
    Information was destroyed. Can't undo.
```

A matrix is **invertible** when:
- It's square (same input and output dimension)
- Full rank (no dimension gets collapsed)
- Determinant ≠ 0 (see next section)

A matrix is **singular** (NOT invertible) when:
- Some dimension gets crushed to zero
- The columns are not all independent (rank < n)
- Determinant = 0

### Inverses in practice

You rarely compute inverses explicitly in ML (it's slow and numerically unstable). But the concept tells you whether a transformation is **reversible**:

```
Rotation (orthogonal):       reversible — inverse is just the transpose (free!)
Projection 768→64:           NOT reversible — 704 dimensions lost forever
Quantization FP32→INT4:      NOT reversible — precision lost
Full-rank square matrix:     reversible — inverse exists, just expensive to compute
```

---

## 7. The Determinant — "How Much Does Space Get Scaled?"

The determinant is a single number computed from a matrix that tells you how the matrix scales area (2D) or volume (higher dimensions).

### 2D computation

```
A = [a, b]
    [c, d]

det(A) = ad - bc

Example:
    A = [3, 1]
        [0, 2]
    det = 3×2 - 1×0 = 6

    This matrix scales area by 6×.
    A unit square (area 1) becomes a parallelogram with area 6.
```

### What the value tells you

```
det = 1:     area preserved exactly (rotations, some shears)
det = 6:     area multiplied by 6 (stretching)
det = 0.5:   area halved
det = 0:     area COLLAPSED to zero — flat, singular, no inverse
det = -1:    area preserved, but ORIENTATION FLIPPED (reflection)
```

### Why we care

The determinant answers two critical questions:

**1. "Is this matrix invertible?"**

```
det ≠ 0  →  YES, inverse exists. No information destroyed.
det = 0  →  NO. Some dimension collapsed. Information lost. Singular matrix.

This is the fastest way to check invertibility.
```

**2. "Rotation or reflection?"**

```
Both orthogonal matrices have |det| = 1 (area preserved).
But:
    det = +1:  rotation     (handedness preserved, "R" stays "R")
    det = -1:  reflection   (handedness flipped, "R" becomes "Я")
```

### Connection to eigenvalues

```
det(A) = product of all eigenvalues

    Eigenvalues: 3 and 2   →  det = 6  (space stretched 6×)
    Eigenvalues: 3 and 0   →  det = 0  (singular — one dimension collapsed)
    Eigenvalues: 1 and -1  →  det = -1 (reflection)
```

If ANY eigenvalue is zero, the determinant is zero, the matrix is singular, and no inverse exists. These three facts are always linked.

### Computing determinants of larger matrices

```
For 2×2: ad - bc (memorize this)
For 3×3: a formula exists but it's messy
For 768×768: nobody computes this by hand — use np.linalg.det()

The concept matters more than the computation.
```

---

## 8. How These Concepts Connect

They form a chain of questions about any matrix:

```
Start with matrix A (n × n):

    What's its rank?
    ├── rank = n (full rank)
    │   ├── det ≠ 0
    │   ├── inverse exists
    │   ├── no eigenvalue is zero
    │   └── no dimension is destroyed — transformation is reversible
    │
    └── rank < n (rank deficient)
        ├── det = 0
        ├── no inverse
        ├── at least one eigenvalue is zero
        └── some dimensions collapsed — information lost, not reversible
```

For rectangular matrices (m × n, m ≠ n):
```
    768 → 64 projection: not invertible by definition.
        You went from 768 dims to 64. The other 704 dims are gone.
        No square matrix, no determinant, no inverse.
        (Pseudoinverse exists but doesn't perfectly recover the original.)
```

---

## 9. Worked Example: Following a Vector Through Transformations

Start with v = [2, 3].

### Projection that preserves both dimensions

```
A = [2, 0]       det = 4 (area scaled 4×, invertible)
    [0, 2]

A × v = [4, 6]

A⁻¹ = [0.5, 0]
      [0, 0.5]

A⁻¹ × [4, 6] = [2, 3]  ✓  recovered original
```

### Projection that collapses one dimension

```
B = [1, 0]       det = 0 (singular!)
    [0, 0]

B × v = [2, 0]      — y-component destroyed

Trying to invert: was the original [2, 0]? [2, 3]? [2, -100]?
All map to [2, 0]. The y-information is gone forever.
```

### Rotation (orthogonal, always safe)

```
R = [0, -1]       det = +1 (rotation, area preserved)
    [1,  0]

R × v = [-3, 2]     — rotated 90° counterclockwise

Rᵀ = [0,  1]        — transpose = inverse for orthogonal matrices
     [-1, 0]

Rᵀ × [-3, 2] = [2, 3]  ✓  recovered original (just the transpose, free!)
```

### Reflection

```
F = [1,  0]       det = -1 (reflection, area preserved but flipped)
    [0, -1]

F × v = [2, -3]     — flipped across x-axis

F⁻¹ = F itself      — reflecting twice gets you back

F × [2, -3] = [2, 3]  ✓  recovered original
```

---

## 10. Putting It Together

```
Projection          maps between spaces (768→64), learned combinations, not just dropping dims
Subspace            the "hyperplane" a projection maps onto, defined by the matrix's columns
Rank                true number of independent directions — can be less than the matrix size
Identity            the "do nothing" matrix, reference point for all other transformations
Inverse             the "undo" matrix — only exists when no information was destroyed
Determinant         single number: how much area/volume is scaled, zero = singular = no inverse
Invertible          full rank, det ≠ 0, no zero eigenvalues — all say the same thing
Singular            rank deficient, det = 0, some eigenvalue is zero — also all the same thing
```

The big takeaway: every matrix either preserves all information (invertible) or destroys some (singular). The determinant, rank, and eigenvalues are three different lenses on the same question: **did anything get lost?**

---

**Next:** `04_Orthogonal_Matrices_and_Rotation.md` — transformations that are guaranteed to preserve all geometry, and why they're central to OPQ, RoPE, and safe data manipulation.
