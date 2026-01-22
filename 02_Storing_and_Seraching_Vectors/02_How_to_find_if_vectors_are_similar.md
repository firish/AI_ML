## Vector Geometry & Similarity Cheatsheet

## 1 · Direction vs. Magnitude  
*Think of each vector as an arrow drawn from the origin.*

| Term | Meaning | Tiny example |
|------|---------|--------------|
| **Magnitude** (length) | How long the arrow is. | `(3, 4)` → `√(3² + 4²) = 5`. Scale to `(6, 8)` and length doubles to 10. |
| **Direction** | Where the arrow points, ignoring length. | `(3, 4)` and `(6, 8)` share the same direction; `(-4, 3)` points left-up (different). |

---

## 2 · Why Normalise (L2-normalisation)?  

Normalise each vector \( v \) to unit length
```math
\hat v \;=\;\frac{v}{\lVert v\rVert}
```

* Cancels magnitudes so cosine becomes just a dot-product.  
* Stops “long” vectors looking similar to everything.  
* Distance `1 − cosine` now obeys the triangle inequality.

**Intuition that actually holds**

Assume a **well-trained encoder**, where *meaning is encoded in direction*, not just length.

#### Example vectors (toy but realistic)
```text
dog                  = (1.0, 1.0)
friendly dog in park = (1.1, 1.0)     # very similar direction
finance & computers  = (-1.0, 1.0)    # very different direction
```

---

#### The Problem: WITHOUT Normalization

Computing **dot product similarity** directly:
```text
dog · (friendly dog)  = 1.0×1.1 + 1.0×1.0 = 2.1
dog · (finance)       = 1.0×(-1.0) + 1.0×1.0 = 0.0
```

Now imagine a **longer version** of the finance text:
```text
finance & computers (verbose) = (-2.0, 2.0)  # same direction, just 2× longer

dog · (finance verbose) = 1.0×(-2.0) + 1.0×2.0 = 0.0
```

This looks fine, but now consider:
```text
friendly dog (verbose) = (2.2, 2.0)  # same direction, 2× longer

dog · (friendly dog verbose) = 1.0×2.2 + 1.0×2.0 = 4.2  ← much higher!
```

**Problem:** The similarity score depends on **vector length**, not just meaning. Longer documents artificially get higher scores.

#### Computing Cosine Similarity (without pre-normalization)
```text
cos(dog, friendly dog) = 2.1 / (√2 × √2.21) = 2.1 / 2.10 = 1.00  ✓
cos(dog, finance)      = 0.0 / (√2 × √2)    = 0.0 / 2.00 = 0.00  ✓

cos(dog, friendly verbose) = 4.2 / (√2 × √12.84) = 4.2 / 5.07 = 0.83  ✓
```

Cosine handles length properly, **but requires division per comparison** (expensive at scale).


#### The Solution: WITH L2-Normalization

After L2-normalization (divide by ‖v‖):
```text
dog                  → (0.707, 0.707)   # ‖v‖ = √2
friendly dog in park → (0.740, 0.672)   # ‖v‖ = √2.21
finance & computers  → (-0.707, 0.707)  # ‖v‖ = √2
```

Now compute **dot product** (which equals cosine for normalized vectors):
```text
dog · (friendly dog)  = 0.707×0.740 + 0.707×0.672 = 0.998  ← very high similarity ✓
dog · (finance)       = 0.707×(-0.707) + 0.707×0.707 = 0.000  ← no similarity ✓
```

Even with verbose versions:
```text
friendly dog (verbose) = (2.2, 2.0) → normalized: (0.740, 0.672)  # same as before!

dog · (friendly verbose) = 0.707×0.740 + 0.707×0.672 = 0.998  ✓
```

**Key insight:** After normalization, length no longer matters—only direction.

#### Side-by-Side Comparison

| Query: `dog` | Vector | Without Norm (dot) | With Norm (dot = cos) |
|--------------|--------|-------------------:|----------------------:|
| friendly dog | (1.1, 1.0) | 2.1 | **0.998** |
| finance | (-1.0, 1.0) | 0.0 | **0.000** |
| friendly (2× longer) | (2.2, 2.0) | 4.2 ❌ | **0.998** ✓ |

Without normalization, the verbose version scores **twice as high** (4.2 vs 2.1) despite having the same meaning!

With normalization, both friendly dog variants score **identically** (0.998), as they should.

---

#### 1. Cancels magnitude so cosine becomes a dot product

Cosine similarity is defined as:
```
cos(u,v) = (u·v) / (‖u‖‖v‖)
```

If all vectors are normalized beforehand (‖u‖ = ‖v‖ = 1):
```
cos(u,v) = u·v
```

**Why this matters:**
- Simplifies computation: cosine ≡ dot product
- Faster on GPUs and vector databases
- Most ANN indexes assume this geometry

#### 2. Prevents "long" vectors from looking similar to everything

Without normalization, longer vectors (often from longer text) produce larger dot products even when meaning differs.

Normalization enforces:
- Similarity depends on **direction only**, not how "loud" or verbose the vector is

This prevents:
- Long documents dominating search results
- Spurious matches caused by vector length instead of meaning

#### 3. Makes distance geometry stable for search

After normalization, all vectors lie on the **unit sphere**.

This has practical benefits:
- `1 − cosine` behaves like a proper distance
- Nearest-neighbor graphs are more reliable
- ANN search (HNSW, IVF) has predictable recall and latency

**Normalization makes similarity search stable, fair, and index-friendly.**

---

## 3 · Similarity / Distance Measures  

| Measure | Quick formula* | Intuition (easy) | What it really checks (medium) | Keeps length? |
|---------|----------------|------------------|--------------------------------|---------------|
| **Cosine similarity** | `cos(u,v) = (u·v)/(‖u‖‖v‖)` | “Do the arrows point the same way?” | Direction only (after normalise). | ✗ |
| **Euclidean (L2) distance** | `√Σ(uᵢ − vᵢ)²` | “How far apart are the tips?” | Straight-line gap; direction **and** length. | ✓ |
| **Dot product** | `Σ uᵢ vᵢ` | “Long aligned arrows score huge.” | Blends angle and magnitude; unbounded. | ✓ |
| **Angular distance** | `θ = arccos(cos(u,v))` | Actual angle (0–π). | Pure angle; monotonic to cosine. | ✗ |

\*Vectors are real-valued; `‖u‖` is the L2 norm.

### Tiny numeric demo (2-D)

```text
A = (3, 4)   # len 5
B = (6, 8)   # len 10  (same dir)
C = (-4, 3)  # len 5   (~90°)
```

| Pair | Cosine | Euclidean | Dot |
|------|-------:|----------:|----:|
| A ↔ B | **1.00** |  5.0 |  60 |
| A ↔ C |   ≈0    |  7.1 |  ≈0 |
| B ↔ C |   ≈0    | 12.0 |  –8 |

---

## 4 · Pros & Cons in Plain Language

| Measure | Good for… | Watch out for… |
|---------|-----------|----------------|
| **Cosine** | Text or image embeddings where only meaning (direction) matters and vector lengths vary. | Throws away length info: a confident vs. weak embedding look identical after normalising. |
| **Euclidean** | Low-dimensional data (dozens of numbers) where absolute scale matters—e.g., a robot arm’s XYZ position. | In hundreds of dimensions every distance converges; KD-trees & similar indexes slow down. |
| **Dot product** | Recommendation models where longer vectors mean “more confident”; super-fast GPU math (one fused multiply-add). | Scores explode with length—one giant-norm vector can dominate even if its direction is off. |
| **Angular** | Situations that literally care about the angle (directional sensors) or need an interpretable 0-to-π metric. | Requires a slow `arccos` unless you convert via cosine first; same limits as cosine otherwise. |

---


## 5 · Best-Fit Use-Cases  

| Choose this… | When… | Why… |
|--------------|-------|------|
| **Cosine** | High-dim (≥256 d) embeddings for text/images. | Direction captures semantics; length irrelevant. |
| **Euclidean** | Low-dim (<30 d) numeric data where scale matters (robot XYZ, sensors). | Geometric distance intuitive; KD-trees work. |
| **Dot product** | Need similarity **and** confidence (recommender logits, energy models). | Norm encodes “strength”; GPU dot is fast. |
| **Angular** | You literally need the angle (directional sensors, viz thresholds). | Bounded 0–π; directly interpretable. |

---

## Rule of Thumb  

* **Embeddings / high-dim:** normalise → **Cosine**.  
* **Physical coords / small-dim:** **Euclidean**.  
* **Want length as confidence:** raw **Dot**.  
* **Need literal angle:** **Angular** (or cosine + arccos).
