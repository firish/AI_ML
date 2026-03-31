# OPQ (Optimized Product Quantization) — Notes

## 0. The Problem PQ Has

PQ chops vectors at fixed boundaries: dims 0-95 go to codebook 0, dims 96-191 to codebook 1, etc. Each codebook trains independently — it never sees what the other codebooks are doing.

This is fine if dimensions within each block are correlated with each other and independent from other blocks. But real embeddings aren't that tidy. If dims 95 and 96 move together (they're correlated), PQ splits them across two codebooks. Neither codebook can model the joint pattern. Quantization error goes up.

---

## 1. The 2D Intuition

Start simple. Your data is a cloud of points shaped like a tilted ellipse — stretching diagonally from bottom-left to top-right.

```
PQ with M=2 means: dim 0 → codebook 1, dim 1 → codebook 2.
Each codebook quantizes its axis independently.

    dim 1
      |       . . .
      |     . . . . .
      |   . . . . . . .        ← data stretches DIAGONALLY
      |     . . . . .
      |       . . .
      +------------------dim 0

The data's real structure runs diagonally.
Neither axis alone captures the pattern well.
You get high quantization error because you're approximating
a diagonal spread with axis-aligned clusters.
```

Now rotate the ellipse so its long axis aligns with dim 0 and its short axis aligns with dim 1:

```
    dim 1
      |       . . .
      |       . . .
      | . . . . . . . . .      ← long axis now along dim 0
      |       . . .
      |       . . .
      +------------------dim 0

Codebook 1 handles the high-variance direction (dim 0).
Codebook 2 handles the low-variance direction (dim 1).
They don't need to "communicate" across the split boundary.
Quantization error drops because each codebook works with
the natural structure of the data.
```

---

## 2. Scaling to 768 Dimensions

Same idea, bigger space. Your 768 dimensions have complex correlation patterns. When you naively chop at every 96 dims, some correlated dimensions get separated.

```
Original vector:  v          (768-dim)
Rotated vector:   R × v      (768-dim, same size, just rotated)
Then apply PQ:    PQ_ENCODE(R × v)

R is a 768 × 768 orthogonal matrix, learned during training.
```

R remixes the dimensions so that within each 96-dim block, you capture as much structure as possible, and across blocks, the correlations are minimized. Think of it as PCA-like, but optimized specifically for the PQ objective.

---

## 3. Why Distances Are Preserved — What "Orthogonal" Means

A matrix R is **orthogonal** when its columns are all unit length and mutually perpendicular — meaning `Rᵀ × R = Identity`.

Practically: R only rotates (and possibly reflects) vectors without any stretching, squishing, or skewing. Every distance, every angle, every dot product between any pair of vectors stays identical after multiplication by R.

```
||R × v₁ - R × v₂|| = ||v₁ - v₂||     exactly, always.
```

It's like spinning a rigid object — the shape doesn't change, just its orientation relative to the coordinate axes. You're not adding or destroying any information. You're relabeling which direction is called "dim 0" vs "dim 47" vs "dim 300." The data cloud is identical; you've just rotated your coordinate grid underneath it.

This is why rotation is "safe" for search — nearest neighbors don't change.

---

## 4. What About Semantic Dimensions?

Embedding dimensions often encode meaningful things — maybe dim 23 relates to "formality" and dim 150 to "topic." Does rotating them destroy meaning?

No. Those axis assignments are already **arbitrary**. When BERT was trained, nothing forced "formality" into dim 23. A different random seed would have produced an equally valid model where formality is spread across dims 23, 87, and 412. The semantic information lives in the **geometry** of the vector space — the distances and angles between vectors — not in which specific axis carries which meaning.

Since orthogonal rotation preserves all distances and angles, it preserves all semantic relationships. The vector for "king" is still the same distance from "queen" after rotation. Nearest neighbors of any vector remain exactly the same.

The only place approximation enters is the PQ quantization step after rotation — and that's exactly what OPQ is minimizing.

---

## 5. What OPQ Actually Optimizes

The goal is: find R such that **total reconstruction error under PQ is minimized**.

```
minimize  Σ ||R × vᵢ - PQ_DECODE(PQ_ENCODE(R × vᵢ))||²
   R       i

subject to: R is orthogonal (Rᵀ × R = I)
```

This naturally pushes toward two properties:

### Maximize variance within each subvector

If a 96-dim block has high internal variance, the k-means codebook for that block has rich structure to work with — the 256 centroids spread out to cover the data well, and quantization error per point is low.

If a block has low variance (all points are nearly the same in those dims), then the 256 centroids are wasted — they cluster around a single region. Low variance = easy to quantize but the block carries little information.

OPQ balances variance across blocks so no block is starved and no block is overwhelmed.

### Minimize correlation across subvectors

If two dimensions in different blocks are correlated, that's a joint pattern that neither block can capture. Each block sees only its half of the story.

```
Example without rotation:
    dim 95 and dim 96 are correlated (they move together).
    dim 95 → codebook 0, dim 96 → codebook 1.
    Codebook 0 sees dim 95 changing but doesn't know dim 96 is tracking it.
    Codebook 1 sees dim 96 changing but doesn't know dim 95 is tracking it.
    Both codebooks place centroids suboptimally.

After rotation:
    R remixes so that correlated information lands in the SAME block.
    One codebook sees the full pattern, quantizes it well.
```

The ideal scenario for PQ: each 96-dim block is statistically independent from every other block. If that's true, quantizing each block separately loses nothing compared to quantizing them jointly. OPQ's rotation pushes the data toward this ideal.

---

## 6. The Two-Step Training (Alternating Optimization)

OPQ can't solve for R and the codebooks simultaneously — they depend on each other. So it alternates:

### Step 1: Fix R, train codebooks

```
Rotate all training vectors:  v' = R × v  for each v in training set.
Split each v' into M subvectors.
Run k-means on each subspace independently → get M codebooks.

This is exactly normal PQ training, just on rotated data.
```

### Step 2: Fix codebooks, optimize R

```
Given the current codebooks, each training vector has:
    - Its original form:      v
    - Its PQ reconstruction:  x̂  (decode the PQ codes back to a vector)

We want R such that R × v ≈ x̂ (the rotated original is close to
what PQ reconstructs).

    minimize  ||R × V - X̂||²
    subject to: R is orthogonal

This is the Procrustes problem — it has a closed-form solution via SVD:

    Compute:  V × X̂ᵀ = U × Σ × Wᵀ       (SVD)
    Solution: R = W × Uᵀ

No iterative search needed for this step — one SVD gives the optimal R.
```

### Repeat until convergence

```
Iteration 1: Start with R = Identity (or random orthogonal)
             → train codebooks on unrotated data
             → solve for better R via Procrustes

Iteration 2: Rotate data with new R
             → retrain codebooks (they shift to match rotated structure)
             → solve for even better R

...

Typically converges in 5-10 iterations.
Each iteration: one round of k-means + one SVD.
```

The R that falls out naturally pushes correlated dimensions into the same subvector block and balances variance across blocks, because that's what minimizes reconstruction error under the PQ structure.

---

## 7. PCA vs OPQ

They're related but not identical.

```
PCA:  finds a global rotation that orders dimensions by variance.
      Dim 0 = highest variance, dim 1 = second highest, etc.
      Global optimum for a different objective (total variance explained).

OPQ:  finds a rotation that makes the BLOCK structure work well.
      It wants each 96-dim chunk to be internally rich and
      externally independent from other chunks.
      Optimized specifically for the PQ reconstruction objective.
```

PCA would put all the high-variance stuff in the first block and leave later blocks with scraps. OPQ distributes variance more evenly across blocks, because PQ needs every codebook to work well, not just the first one.

---

## 8. In Practice

**In FAISS:** `OPQ8_96,PQ8` means "apply OPQ rotation (8 subspaces, 96 dims each), then PQ with 8 codebooks."

**Query time:** Rotate the query too — compute `R × q`, then run normal PQ search on the rotated query against the rotated codes. One extra matrix multiply per query (768 × 768) — negligible cost.

**When to use:**
- Almost always. Strictly better than PQ with negligible extra cost at search time.
- Especially helps when data has strong cross-dimension correlations (which embeddings typically do).
- Typical improvement: 2-5% recall gain for free.
- The only cost is training time (the alternating optimization), which is offline.

---

## Key Takeaways

1. **PQ's weakness:** arbitrary subvector boundaries split correlated dimensions
2. **OPQ's fix:** learn an orthogonal rotation R that aligns data with the block structure
3. **Orthogonal = safe:** distances, angles, nearest neighbors all preserved exactly
4. **Two-step training:** fix R → train codebooks (k-means), fix codebooks → optimize R (Procrustes/SVD), repeat
5. **Maximizes within-block variance, minimizes across-block correlation** — the ideal conditions for PQ
6. **Semantic dimensions are arbitrary anyway** — rotating them preserves all meaningful geometry
7. **Related to PCA but different** — PCA orders by variance globally, OPQ balances variance across blocks
8. **Always use it** — almost free improvement over plain PQ

---

**Next:** `16_IVF_PQ.md` — how clustering (which vectors to search) and compression (how to store and compare them) combine for production-scale vector search.
