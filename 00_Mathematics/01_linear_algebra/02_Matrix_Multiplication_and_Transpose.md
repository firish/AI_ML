# Matrix Multiplication and Transpose

---

## 1. What Is a Matrix?

A matrix is a grid of numbers with rows and columns. Its **shape** is written as (rows × columns).

```
A = [1, 2, 3]       shape: (2 × 3) — 2 rows, 3 columns
    [4, 5, 6]
```

A vector is just a matrix with one column (or one row):

```
column vector: [3]     shape: (3 × 1)
               [7]
               [2]

row vector:    [3, 7, 2]    shape: (1 × 3)
```

### Why matrices are the language of AI

A single neuron takes a dot product of its weights with the input. A **layer** of neurons does many dot products at once — one per neuron. That's exactly what a matrix multiply does. So:

```
One neuron:  output = w · x              (dot product)
One layer:   output = W × x             (matrix multiply = many dot products)
Full model:  output = f(W₃ × f(W₂ × f(W₁ × x)))   (chain of matrix multiplies with activations)
```

Every weight matrix W stores one neuron's weight vector per row (or column, depending on convention). The matrix multiply fires all neurons simultaneously.

---

## 2. Matrix Multiplication — The Mechanic

To multiply A × B:

```
Rule: (m × n) × (n × p) = (m × p)
              ↑   ↑
              must match (the "inner dimensions")

The result has: rows from A, columns from B.
```

Each entry in the result is a **dot product** — row i of A dotted with column j of B.

```
A = [1, 2]     B = [5, 6]
    [3, 4]         [7, 8]

(2×2) × (2×2) = (2×2)  ✓  inner dimensions match (2 = 2)

C[0,0] = row 0 of A · col 0 of B = [1,2] · [5,7] = 1×5 + 2×7 = 19
C[0,1] = row 0 of A · col 1 of B = [1,2] · [6,8] = 1×6 + 2×8 = 22
C[1,0] = row 1 of A · col 0 of B = [3,4] · [5,7] = 3×5 + 4×7 = 43
C[1,1] = row 1 of A · col 1 of B = [3,4] · [6,8] = 3×6 + 4×8 = 50

C = [19, 22]
    [43, 50]
```

### The dimension rule — when multiplication is possible

```
(m × n) × (n × p) = (m × p)
         ↑   ↑
         These MUST be equal. If they're not, multiplication is undefined.

(3 × 2) × (2 × 5) = (3 × 5)    ✓  inner dims match (2 = 2)
(3 × 2) × (5 × 2) = ???         ✗  inner dims don't match (2 ≠ 5)
(1 × 768) × (768 × 64) = (1 × 64)  ✓  this is a projection from 768-d to 64-d
```

Read the shapes out loud: "take m things of size n, and produce m things of size p." The inner dimension n disappears — it gets summed over in the dot products.

---

## 3. What Matrix Multiplication MEANS

### View 1: Every output is a dot product

```
output[i,j] = row i of A · column j of B

So the entire result matrix is: "compute the dot product of
every row of A with every column of B."
```

### View 2: Every output row is a linear combination

```
Each row of the output is a linear combination of the rows of B,
with weights given by the corresponding row of A.

    A = [2, 3]     B = [10, 20]        row 0 of B = [10, 20]
        [1, 4]         [30, 40]        row 1 of B = [30, 40]

    output row 0 = 2 × [10,20] + 3 × [30,40] = [20,40] + [90,120] = [110, 160]
    output row 1 = 1 × [10,20] + 4 × [30,40] = [10,20] + [120,160] = [130, 180]
```

This view is how attention works: the output for each token is a linear combination of value vectors, weighted by attention scores.

### View 3: A matrix is a transformation

```
A × v = v'

The matrix A takes a vector v and produces a new vector v'.
It transforms the input — rotating, stretching, projecting, or some
combination of these depending on the matrix.

Different matrices = different transformations.
Same matrix applied to different vectors = same transformation,
different inputs.
```

All three views describe the same operation. Use whichever one helps you reason about the specific situation.

---

## 4. Matrix Multiply in Neural Networks

### A single layer (the most common pattern)

```
output = input × W + bias

    input:  (batch × d_in)        e.g., (32 × 768)
    W:      (d_in × d_out)        e.g., (768 × 3072)
    bias:   (d_out)               e.g., (3072)
    output: (batch × d_out)       e.g., (32 × 3072)
```

What this does: each of the 32 input vectors (768-dim) gets transformed into a 3072-dim vector. Each output dimension is a dot product of the full input with one column of W.

```
Column 0 of W:  the "pattern" that output dimension 0 looks for
Column 1 of W:  the "pattern" that output dimension 1 looks for
...
Column 3071 of W: the "pattern" that output dimension 3071 looks for

output[i, j] = input[i] · W[:, j]
"How much does input i match the pattern for output dimension j?"
```

### Attention: Q × Kᵀ (we'll need transpose for this — see section 6)

```
scores = Q × Kᵀ       (seq × d_k) × (d_k × seq) = (seq × seq)

scores[i, j] = query_i · key_j
"How relevant is token j to what token i is looking for?"
```

This one matrix multiply computes ALL pairwise similarity scores at once.

---

## 5. Important: Order Matters

Matrix multiplication is **NOT commutative**. A × B ≠ B × A in general.

```
A = [1, 2]     B = [5, 6]
    [3, 4]         [7, 8]

A × B = [19, 22]       B × A = [23, 34]
        [43, 50]               [31, 46]

Different results.
```

In fact, sometimes A × B is valid but B × A isn't even defined:

```
A: (3 × 2)    B: (2 × 5)

A × B: (3×2) × (2×5) = (3×5)    ✓
B × A: (2×5) × (3×2) = ???      ✗  inner dims 5 ≠ 3
```

In neural networks, the order is always: **input first, then weight matrix**. Swapping them gives wrong shapes and wrong results.

### But it IS associative

```
(A × B) × C = A × (B × C)

You can group multiplications however you want, as long as
you don't reorder them.

This matters for efficiency:
    Matrix sizes: A(10×1000), B(1000×1000), C(1000×5)

    (A × B) × C:  first multiply gives (10×1000), then × C gives (10×5)
                   Total: 10M + 50K operations

    A × (B × C):  first multiply gives (1000×5), then A × that gives (10×5)
                   Total: 5M + 50K operations

    Same result, half the computation. Order of operations matters.
```

---

## 6. Transpose

The transpose flips rows and columns. Row i becomes column i.

```
A = [1, 2, 3]       Aᵀ = [1, 4]
    [4, 5, 6]             [2, 5]
                           [3, 6]

Shape (m × n) → (n × m)
    (2 × 3) → (3 × 2)
```

For a vector: a column becomes a row and vice versa.

```
v = [3]       vᵀ = [3, 7, 2]
    [7]
    [2]
```

### What transpose does

It mirrors the matrix across its diagonal. Entry A[i,j] becomes Aᵀ[j,i].

```
A = [a, b]       Aᵀ = [a, c]
    [c, d]             [b, d]

The diagonal (a, d) stays put. The off-diagonal (b, c) swaps.
```

---

## 7. Why Transpose Matters

### Making dimensions match for dot products

Q and K are both shaped (seq × d_k). You want the dot product of every query with every key. But you can't multiply (seq × d_k) × (seq × d_k) — inner dimensions don't match.

Transpose K:

```
Q:  (seq × d_k)
Kᵀ: (d_k × seq)      ← transposed

Q × Kᵀ = (seq × d_k) × (d_k × seq) = (seq × seq)
                ↑   ↑
                match!

Result[i, j] = query i · key j

One matrix multiply gives you ALL pairwise dot products.
Without transpose, this isn't possible.
```

### Reversing the direction of a mapping

The embedding matrix E maps token IDs → vectors:

```
E: (vocab × d_model)     e.g., (50257 × 768)

"cat" = token 9246
embedding = E[9246]  →  a 768-dim vector
```

The prediction head needs the reverse: vectors → vocabulary scores.

```
logits = hidden × Eᵀ

    hidden: (batch × 768)
    Eᵀ:    (768 × 50257)
    logits: (batch × 50257)     ← one score per vocab word

logits[i, j] = hidden[i] · E[j]
"How much does this hidden state match the embedding for word j?"
```

GPT-2 uses the same matrix E for both directions (called weight tying). The transpose reverses the mapping direction without any extra learned parameters.

### Key identity

```
(A × B)ᵀ = Bᵀ × Aᵀ

Transpose of a product = product of transposes in REVERSE order.

This comes up in backpropagation derivations. If the forward pass is
output = input × W, the gradient with respect to the input involves Wᵀ.
```

---

## 8. The Dot Product as a Matrix Multiply

A dot product is actually a special case of matrix multiplication:

```
a · b = aᵀ × b

    a = [1]     aᵀ = [1, 2, 3]     b = [4]
        [2]                              [5]
        [3]                              [6]

    aᵀ × b = [1, 2, 3] × [4]  = [1×4 + 2×5 + 3×6] = [32]
                           [5]
                           [6]

    (1×3) × (3×1) = (1×1)   — a single number. That's the dot product.
```

This is why matrix multiplication and dot products are the same thing at different scales:

```
Dot product:        one row × one column  = one number
Matrix-vector:      many rows × one column = one vector (many dot products)
Matrix-matrix:      many rows × many columns = one matrix (many many dot products)
```

---

## 9. Rectangular Matrices — Changing Dimensions

Square matrices (n × n) transform vectors within the same space. Rectangular matrices change the dimension.

### Going smaller (projection / compression)

```
W: (768 × 64)

    input:  768-dim vector
    output: 64-dim vector

    output = input × W

Each of the 64 output values is a dot product of the 768-dim input
with one column of W. The matrix compresses from 768 to 64 dimensions.
This is NOT just dropping 704 dimensions — it's 64 LEARNED combinations
of all 768 input dimensions.
```

This is what W_q, W_k, W_v do in attention: project from d_model (768) down to d_k (64) per attention head.

### Going bigger (expansion)

```
W: (768 × 3072)

    input:  768-dim vector
    output: 3072-dim vector

    output = input × W

This is the first half of the FFN in a transformer.
Expand to a higher dimension → apply activation → project back down.
The expansion creates room for the activation function to
selectively switch features on and off.
```

### Shape tells you the story

```
(768 × 64):    compression / projection down
(768 × 768):   same-size transformation (rotation, mixing, etc.)
(768 × 3072):  expansion up
(768 × 50257): mapping to vocabulary scores (prediction head)
```

Just reading the matrix shape tells you what a layer does.

---

## 10. Worked Example: Full Attention in Matrix Form

Putting it all together. One attention head, 4 tokens, d_model=6, d_k=3.

### Step 1: Project to Q, K, V

```
Input X: (4 × 6)     — 4 tokens, each 6-dim

W_q: (6 × 3)    W_k: (6 × 3)    W_v: (6 × 3)

Q = X × W_q     (4×6) × (6×3) = (4×3)    — 4 queries, each 3-dim
K = X × W_k     (4×6) × (6×3) = (4×3)    — 4 keys, each 3-dim
V = X × W_v     (4×6) × (6×3) = (4×3)    — 4 values, each 3-dim

Three matrix multiplies. Same input, three different projections.
```

### Step 2: Compute attention scores

```
scores = Q × Kᵀ / √d_k

    Q:  (4 × 3)
    Kᵀ: (3 × 4)        ← transpose to make dims match
    Q × Kᵀ: (4 × 3) × (3 × 4) = (4 × 4)

    scores[i,j] = (query i · key j) / √3

A 4×4 matrix where entry [i,j] = how much token i attends to token j.
One matrix multiply computed all 16 pairwise scores.
```

### Step 3: Apply softmax (per row)

```
weights = softmax(scores, dim=-1)

Each row sums to 1. Row i = the attention distribution for token i.

    weights[i] = [0.1, 0.6, 0.2, 0.1]
    means: token i pays 60% attention to token 1, 10% to token 0, etc.
```

### Step 4: Weighted sum of values

```
output = weights × V

    weights: (4 × 4)
    V:       (4 × 3)
    output:  (4 × 4) × (4 × 3) = (4 × 3)

    output[i] = weights[i,0]×V[0] + weights[i,1]×V[1] + weights[i,2]×V[2] + weights[i,3]×V[3]

Each output token is a linear combination of ALL value vectors,
weighted by how much attention it pays to each.
```

### Summary of shapes

```
X         (4 × 6)     input tokens
Q, K, V   (4 × 3)     projected (3 matrix multiplies)
scores    (4 × 4)     all-pairs similarity (1 matrix multiply + transpose)
weights   (4 × 4)     softmax of scores
output    (4 × 3)     weighted blend of values (1 matrix multiply)

Total: 5 matrix multiplies + 1 softmax = one attention head.
All of attention is matrix operations.
```

---

## 11. Putting It Together

```
Matrix multiplication    = many dot products at once = one layer
Transpose                = flip rows ↔ columns = reverse mapping direction
Inner dimensions match   = multiplication is valid
Outer dimensions         = result shape
Order matters            = A×B ≠ B×A
Rectangular matrices     = change dimension (project, expand, compress)
Dot product              = special case: (1×n) × (n×1) = scalar
```

The entire forward pass of a transformer is a chain of matrix multiplies with nonlinearities (activation functions, softmax) in between. Understanding the shapes and what each multiply does is enough to read any transformer architecture.

---

**Next:** `03_Projections_and_Subspaces.md` — what "project" really means, why W_q/W_k/W_v are projections, rank, and subspaces.
