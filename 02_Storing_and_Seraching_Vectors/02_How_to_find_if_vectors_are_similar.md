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
