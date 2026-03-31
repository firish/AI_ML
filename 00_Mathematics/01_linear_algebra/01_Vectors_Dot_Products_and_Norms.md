# Vectors, Dot Products, and Norms

---

## 1. What Is a Vector?

A vector is an ordered list of numbers. That's it.

```
v = [3, 7, 2]          — a 3-dimensional vector
v = [0.12, -0.84, 0.5, 0.33]  — a 4-dimensional vector
```

Each number is a **component** (also called a coordinate or dimension). The number of components is the vector's **dimension**.

### Why vectors are the language of AI

Everything in ML is a vector:

```
A word embedding:         768 numbers representing meaning
An image patch:           3072 numbers (32×32 pixels × 3 colors)
A model's parameters:     millions of numbers (all the weights)
A gradient:               one number per parameter (direction to improve)
A single neuron's weights: one number per input feature
```

When you hear "the model learned a representation," that representation is a vector. When you hear "compute similarity between two items," you're comparing their vectors.

---

## 2. Vectors as Points vs Vectors as Arrows

Two ways to think about the same thing:

```
v = [3, 2]

As a POINT:  a location in 2D space at coordinates (3, 2)

           y
           |       • (3,2)
           |
           +--------x

As an ARROW: a direction + magnitude from the origin to (3, 2)

           y
           |      ↗ (3,2)
           |    /
           |  /
           +--------x
```

Both views are useful:

- **Point view:** when you think about embeddings as positions in a space. "King" and "queen" are nearby points. "King" and "banana" are far apart.
- **Arrow view:** when you think about direction and magnitude. A gradient points in the direction of steepest ascent. A weight vector points in the "pattern" a neuron detects.

---

## 3. Vector Addition

Add component by component.

```
a = [1, 3]
b = [4, 1]

a + b = [1+4, 3+1] = [5, 4]
```

Geometrically: place arrow b at the tip of arrow a. The result points from the origin to where you end up.

```
     y
     |         • (5,4)
     |       ↗
     | • (1,3)
     |   ↗
     +--------x
```

### Why addition matters in AI

**Residual connections** are vector addition:

```
output = layer(x) + x

The layer's correction is ADDED to the original input.
Each layer contributes a small adjustment, not a full replacement.
```

**Word analogies** are vector addition:

```
king - man + woman ≈ queen

Subtracting "man" and adding "woman" moves the vector
from male-royalty to female-royalty in the embedding space.
```

---

## 4. Scalar Multiplication

Multiply every component by the same number (the "scalar").

```
v = [2, 3]
3 × v = [6, 9]      — same direction, 3× longer

-1 × v = [-2, -3]   — same line, opposite direction
0.5 × v = [1, 1.5]  — same direction, half the length
```

The scalar stretches or shrinks the vector without changing its direction (unless the scalar is negative, which flips it).

### Why scalar multiplication matters in AI

**Learning rate:**

```
new_weights = old_weights - learning_rate × gradient

The learning rate is a scalar that controls how big a step you take.
Too large → overshoot. Too small → barely move.
```

**Attention weights:**

```
output = 0.7 × v₁ + 0.2 × v₂ + 0.1 × v₃

Each value vector is SCALED by its attention weight,
then summed. High weight = that token contributes more.
```

**Eigenvalues:**

```
A × v = λ × v

The eigenvalue λ is a scalar that stretches the eigenvector.
This is literally "the matrix just does scalar multiplication
along this direction."
```

---

## 5. The Dot Product

The single most important operation in AI.

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ

Multiply corresponding components, sum the results.

Example:
    a = [1, 2, 3]
    b = [4, 5, 6]
    a · b = 1×4 + 2×5 + 3×6 = 4 + 10 + 18 = 32
```

### What the dot product measures

```
a · b = ||a|| × ||b|| × cos(θ)

where θ is the angle between the two vectors.

    cos(0°)   =  1    → same direction       → large positive
    cos(90°)  =  0    → perpendicular         → zero
    cos(180°) = -1    → opposite directions   → large negative
```

The dot product combines two things: how long the vectors are, and how much they point in the same direction. Large dot product means the vectors are both long AND aligned.

### The dot product is a similarity measure

```
High positive dot product:  vectors point similarly     → similar
Near zero:                  vectors are perpendicular   → unrelated
Large negative:             vectors point opposite      → dissimilar
```

This is why attention works:

```
score(query, key) = query · key

High score = this key is relevant to this query = pay attention here.
Low score  = irrelevant = ignore.
```

And why every neuron is a dot product:

```
neuron output = weights · input + bias

The neuron computes a dot product of its weight vector with the input.
High output = the input matches the pattern this neuron detects.
Low output  = no match.
```

### Worked example: why similar vectors have high dot products

```
a = [1, 0, 1, 0]       — "activates" in dims 0 and 2
b = [1, 0, 1, 0]       — exact same pattern
a · b = 1+0+1+0 = 2    — high (perfect match)

c = [0, 1, 0, 1]       — "activates" in dims 1 and 3
a · c = 0+0+0+0 = 0    — zero (completely different patterns)

d = [-1, 0, -1, 0]     — opposite pattern to a
a · d = -1+0-1+0 = -2  — negative (anti-correlated)
```

Each multiplication a_i × b_i asks: "do these two vectors agree on dimension i?" The sum adds up all the agreements. More agreement = bigger dot product.

---

## 6. The Norm (Length of a Vector)

The norm measures how long a vector is.

### L2 norm (Euclidean norm) — the default

```
||a|| = √(a₁² + a₂² + ... + aₙ²)

      = √(a · a)      — the dot product of a vector with itself, square rooted

Example:
    a = [3, 4]
    ||a|| = √(9 + 16) = √25 = 5

    a = [1, 1, 1, 1]
    ||a|| = √(1 + 1 + 1 + 1) = √4 = 2
```

This is just the Pythagorean theorem generalized to any number of dimensions. In 2D, a vector [3, 4] forms a right triangle with legs 3 and 4, hypotenuse 5.

### Why norms matter

**The dot product conflates direction and magnitude.** Two vectors can have a large dot product either because they point the same way OR because one is very long:

```
a = [1, 0]
b = [1, 0]         a · b = 1   (similar, both short)
c = [1000, 0]      a · c = 1000 (same direction, but c is huge)
```

Sometimes you want pure directional similarity, ignoring length. That's where normalization and cosine similarity come in.

### L1 norm (Manhattan distance)

```
||a||₁ = |a₁| + |a₂| + ... + |aₙ|

Sum of absolute values. Called "Manhattan" because it's the
distance you walk on a grid (like NYC blocks — no diagonals).

    a = [3, -4]
    ||a||₁ = 3 + 4 = 7
    ||a||₂ = √(9 + 16) = 5     — L2 is shorter (diagonal)
```

L1 shows up in regularization (L1 regularization / Lasso pushes weights to exactly zero, creating sparsity).

---

## 7. Unit Vectors and Normalization

A **unit vector** has norm = 1. It encodes pure direction, no magnitude.

```
To normalize a vector (make it unit length):

    â = a / ||a||      — divide each component by the norm

Example:
    a = [3, 4],  ||a|| = 5
    â = [3/5, 4/5] = [0.6, 0.8]
    ||â|| = √(0.36 + 0.64) = √1 = 1  ✓
```

### Why normalization matters in AI

**Cosine similarity** is the dot product of normalized vectors:

```
cosine_sim(a, b) = (a · b) / (||a|| × ||b||) = â · b̂

This gives pure directional similarity, ranging from -1 to +1:
    +1:  identical direction
     0:  perpendicular
    -1:  opposite direction
```

Many embedding models normalize their output vectors. Once normalized, dot product = cosine similarity, and maximizing dot product = minimizing L2 distance. The three similarity measures become equivalent:

```
If ||a|| = ||b|| = 1:
    maximize  a · b
    maximize  cosine_sim(a, b)       — same thing
    minimize  ||a - b||²             — also equivalent

Proof: ||a - b||² = ||a||² + ||b||² - 2(a·b) = 1 + 1 - 2(a·b) = 2 - 2(a·b)
    Minimizing this ↔ maximizing a·b.
```

This is why vector databases can offer "dot product," "cosine," or "L2" distance and they often give the same ranking for normalized vectors.

---

## 8. L2 Distance (Euclidean Distance Between Two Vectors)

The straight-line distance between two points.

```
||a - b|| = √((a₁-b₁)² + (a₂-b₂)² + ... + (aₙ-bₙ)²)

Example:
    a = [1, 2]
    b = [4, 6]
    ||a - b|| = √((1-4)² + (2-6)²) = √(9 + 16) = √25 = 5
```

### The relationship between L2 distance and dot product

```
||a - b||² = ||a||² + ||b||² - 2(a · b)
```

This says: distance depends on both lengths and the dot product. If you know the lengths, the dot product tells you the distance and vice versa. They're not independent measures — they're the same information rearranged.

This identity is why PQ (Product Quantization) works. PQ approximates distances by decomposing them into partial dot products across subvectors. The math works because distance decomposes cleanly:

```
||q - x||² = ||q_sub0 - x_sub0||² + ||q_sub1 - x_sub1||² + ... + ||q_sub7 - x_sub7||²

Each term is an independent sub-distance. Sum them up = total distance.
```

---

## 9. Orthogonality (Perpendicular Vectors)

Two vectors are **orthogonal** (perpendicular) when their dot product is zero.

```
a · b = 0    →    a and b are orthogonal

Example:
    a = [1, 0]     (points east)
    b = [0, 1]     (points north)
    a · b = 0      (perpendicular — 90° apart)

    c = [1, 1]     (points northeast)
    d = [1, -1]    (points southeast)
    c · d = 1×1 + 1×(-1) = 0     (also perpendicular!)
```

### Why orthogonality matters

**Orthogonal = independent = no redundancy.** If two vectors are orthogonal, knowing one tells you nothing about the other. They carry completely independent information.

This is why:

- **Eigenvectors of symmetric matrices are orthogonal** — each one captures an independent direction of variation. No redundancy between principal components.
- **The standard basis [1,0,0], [0,1,0], [0,0,1] is orthogonal** — each dimension is independent of the others.
- **Orthogonal matrices preserve geometry** — the columns are orthogonal unit vectors, forming a new coordinate system that's just as clean as the original.
- **PQ wants subvectors to be independent** — OPQ rotates data so that subvector blocks are as orthogonal (uncorrelated) as possible.

---

## 10. Linear Combinations — The Building Block of Everything

A linear combination is: take some vectors, scale each one, add them up.

```
result = c₁v₁ + c₂v₂ + c₃v₃

where c₁, c₂, c₃ are scalars (numbers) and v₁, v₂, v₃ are vectors.
```

This sounds simple, but it's the atomic operation of neural networks.

### Every layer output is a linear combination

```
A neuron with 3 inputs:

    output = w₁x₁ + w₂x₂ + w₃x₃ + bias

This is a linear combination of the inputs, weighted by the learned weights.
(The bias shifts it, the activation function makes it nonlinear.)
```

### Attention output is a linear combination

```
output = α₁v₁ + α₂v₂ + ... + αₙvₙ

Each value vector vᵢ is scaled by its attention weight αᵢ (from softmax),
then summed. The output is a weighted blend of all value vectors.
A token that gets attention weight 0.8 contributes 80% of the result.
```

### Span — what a set of vectors can "reach"

The **span** of a set of vectors is every point you can reach by taking linear combinations of them.

```
v₁ = [1, 0]   v₂ = [0, 1]

span(v₁, v₂) = all of 2D space.
Any point [a, b] = a×[1,0] + b×[0,1].

v₁ = [1, 0]   v₂ = [2, 0]

span(v₁, v₂) = just the x-axis.
Both vectors point the same direction. Adding multiples of them
can only move along that line. You can never reach [0, 1].
v₂ is REDUNDANT — it doesn't add any new reachable directions.
```

This connects to rank: a matrix's rank = how many independent directions its columns span. A 768×768 matrix with rank 50 can only map inputs into a 50-dimensional subspace, even though it looks like it has 768 dimensions to work with.

---

## 11. Putting It Together

```
Vectors               the universal data format of ML
Addition              residual connections, word analogies
Scalar multiplication learning rate, attention weights, eigenvalues
Dot product           similarity, attention scores, neuron activation
Norm                  vector length, regularization
Normalization         cosine similarity, stable comparisons
L2 distance           nearest neighbor search, loss computation
Orthogonality         independence, clean coordinate systems, PCA
Linear combination    layer outputs, attention blending, span and rank
```

Everything in ML — from a single neuron to a full transformer — is built from these operations applied at different scales. A forward pass through GPT is millions of dot products organized into matrix multiplies organized into attention blocks organized into layers. But the atomic unit is always: take vectors, combine them with dot products and linear combinations, measure distances.

---

**Next:** `02_Matrix_Multiplication.md` — doing many dot products at once, and why every layer is a matrix multiply.
