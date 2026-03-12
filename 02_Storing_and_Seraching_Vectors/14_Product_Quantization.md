# Product Quantization (PQ) — Notes

## 0. What PQ Fixes from SQ

**SQ's limitation:** Compresses each value independently. A 768-dim int8 vector is still 768 bytes. For 1B vectors, that's ~768 GB.

**PQ solution:**
> Don't compress individual values. Replace **entire subvectors** with short codes. A 768-dim vector becomes ~8–64 bytes.

That's 50–400x compression. This is what makes billion-scale search possible in RAM.

---

## 1. The Core Idea

Three steps:

1. **Split** each vector into M subvectors
2. **Learn** a codebook (dictionary) for each subvector position using k-means
3. **Encode** each subvector as the ID of its closest codebook entry

A full vector becomes M short code IDs instead of D floats.

**Analogy:** Instead of writing a full sentence, you have a phrasebook for each part of the sentence. You replace each phrase with its phrasebook entry number. The sentence becomes a list of numbers.

---

## 2. Structure: Subvectors and Codebooks

### Splitting into subvectors

A 768-dim vector split into M=8 subvectors:
```
[----sub1----][----sub2----]...[----sub8----]
  96 dims       96 dims          96 dims
```

Each subvector has `D/M = 768/8 = 96` dimensions.

### Codebook per subvector position

For each of the M positions, train a separate k-means with K* centroids (typically K*=256).

```
Codebook 1: 256 centroids, each 96-dim    (for subvector position 1)
Codebook 2: 256 centroids, each 96-dim    (for subvector position 2)
...
Codebook 8: 256 centroids, each 96-dim    (for subvector position 8)
```

**Why 256?** Because 256 = 2^8, so each code ID fits in 1 byte (uint8).

### Encoding a vector

For each subvector, find the closest centroid in its codebook:
```
subvector 1 → closest centroid in codebook 1 → code ID 42
subvector 2 → closest centroid in codebook 2 → code ID 187
...
subvector 8 → closest centroid in codebook 8 → code ID 91
```

**Encoded vector:** `[42, 187, ..., 91]` — just 8 bytes instead of 3,072.

---

## 3. Training the Codebooks

```python
function PQ_TRAIN(vectors, M, Kstar):
    """
    vectors : all N training vectors, each D-dim
    M       : number of subvector splits
    Kstar   : number of centroids per codebook (typically 256)
    d       : subvector dimensionality = D / M

    codebooks : list of M codebooks, each with Kstar centroids
    """
    d = D / M
    codebooks = []

    for m in range(M):
        # extract subvector at position m from all vectors
        sub_vectors = [v[m*d : (m+1)*d] for v in vectors]

        # run k-means on these subvectors
        centroids = K_MEANS(sub_vectors, Kstar)
        codebooks.append(centroids)

    return codebooks
```

**Key detail:** Each codebook is trained independently on its own subspace. The codebooks don't know about each other.

---

## 4. Encoding Vectors

```python
function PQ_ENCODE(v, codebooks, M):
    """
    v          : original D-dim vector
    codebooks  : M codebooks from training
    M          : number of subvectors
    d          : subvector dimensionality

    codes      : M code IDs (the compressed representation)
    """
    d = D / M
    codes = []

    for m in range(M):
        sub = v[m*d : (m+1)*d]
        # find closest centroid in codebook m
        code_id = argmin(dist(sub, c) for c in codebooks[m])
        codes.append(code_id)

    return codes    # M bytes (if Kstar=256)
```

---

## 5. Memory Math

| Parameter | Value |
|---|---|
| Original vector | 768 dims × 4 bytes = 3,072 bytes |
| M (subvectors) | 8 |
| K* (centroids) | 256 (1 byte per code) |
| **PQ code** | **8 bytes** |
| **Compression ratio** | **384x** |

**1 billion vectors:**
```
Original:  1B × 3,072 bytes = ~3 TB
PQ codes:  1B × 8 bytes     = ~8 GB
```

8 GB fits in RAM on a single machine.

**Codebook overhead:** 8 codebooks × 256 centroids × 96 dims × 4 bytes = ~768 KB. Negligible.

---

## 6. Distance Computation: ADC (Asymmetric Distance Computation)

This is the clever part. You need to compute `dist(query, stored_vector)`, but stored vectors are compressed to codes. How?

### The naive approach (decode then compare)

Reconstruct the approximate vector from codes, compute distance. Works but wasteful.

### ADC: Precompute a distance table

**Key insight:** The query is NOT quantized. Only the stored vectors are. That's why it's "asymmetric."

```python
function PQ_SEARCH(q, codes_db, codebooks, M, k):
    """
    q          : query vector (full precision, NOT quantized)
    codes_db   : list of PQ codes for all N stored vectors
    codebooks  : M codebooks
    M          : number of subvectors
    k          : number of nearest neighbors to return

    dist_table : precomputed distances from query subvectors to all centroids
    """
    d = D / M

    # Step 1: build distance table (M × Kstar)
    dist_table = []
    for m in range(M):
        q_sub = q[m*d : (m+1)*d]
        # distance from query's subvector m to every centroid in codebook m
        table_m = [dist(q_sub, c) for c in codebooks[m]]
        dist_table.append(table_m)

    # Step 2: compute approximate distance to every stored vector
    results = []
    for i, codes in enumerate(codes_db):
        # sum up partial distances using table lookups
        approx_dist = 0
        for m in range(M):
            approx_dist += dist_table[m][codes[m]]    # just a table lookup!
        results.append((approx_dist, i))

    # Step 3: return top-k
    return top_k_closest(results, k)
```

### Why ADC is fast

**Without ADC:** For each stored vector, compute M subvector distances (each is a 96-dim distance). That's `N × M × 96` floating point operations.

**With ADC:** Precompute `M × K*` distances once (the table). Then for each stored vector, do `M` table lookups (integer indexing). That's `N × M` lookups.

```
Table build:     M × K* distances    = 8 × 256 = 2,048 distance computations
Per vector:      M table lookups     = 8 additions
1B vectors:      8B additions vs ~600B float ops
```

The table lookup replaces expensive distance math with cheap array indexing.

---

## 7. Walk-Through Example

**Setup:** D=8, M=4 (4 subvectors of 2 dims each), K*=4 (2-bit codes for simplicity).

**Codebooks (trained):**
```
Codebook 0 (sub dims 0-1): c0=[1,0], c1=[0,1], c2=[-1,0], c3=[0,-1]
Codebook 1 (sub dims 2-3): c0=[2,2], c1=[2,-2], c2=[-2,2], c3=[-2,-2]
Codebook 2 (sub dims 4-5): c0=[1,1], c1=[1,-1], c2=[-1,1], c3=[-1,-1]
Codebook 3 (sub dims 6-7): c0=[0,3], c1=[3,0], c2=[0,-3], c3=[-3,0]
```

**Encoding vector v = [0.9, 0.1, 1.8, 2.1, -0.9, 1.1, 0.2, 2.8]:**
```
sub0 = [0.9, 0.1]  → closest to c0=[1,0]   → code 0
sub1 = [1.8, 2.1]  → closest to c0=[2,2]   → code 0
sub2 = [-0.9, 1.1] → closest to c2=[-1,1]  → code 2
sub3 = [0.2, 2.8]  → closest to c0=[0,3]   → code 0
```

**PQ code for v:** `[0, 0, 2, 0]` — 4 bytes instead of 32.

**Querying with q = [0, 1, -2, 2, 1, 1, 3, 0]:**

Build distance table (L2 squared to each centroid):
```
dist_table[0]: dist([0,1], c0)=2, dist([0,1], c1)=0, dist([0,1], c2)=2, dist([0,1], c3)=4
dist_table[1]: dist([-2,2], c0)=16, dist([-2,2], c1)=32, dist([-2,2], c2)=0, dist([-2,2], c3)=36
dist_table[2]: dist([1,1], c0)=0, dist([1,1], c1)=4, dist([1,1], c2)=4, dist([1,1], c3)=8
dist_table[3]: dist([3,0], c0)=9, dist([3,0], c1)=0, dist([3,0], c2)=9, dist([3,0], c3)=36
```

**Distance to v (code [0, 0, 2, 0]):**
```
dist_table[0][0] + dist_table[1][0] + dist_table[2][2] + dist_table[3][0]
= 2 + 16 + 4 + 9
= 31
```

Just 4 table lookups and 3 additions. No vector math at query time.

---

## 8. What PQ Gets Wrong (The Tradeoff)

### Quantization error

Each subvector is replaced by its closest centroid. The difference is lost.

```
Original subvector:  [0.9, 0.1]
Centroid:            [1.0, 0.0]
Error:               [0.1, 0.1]
```

These errors accumulate across M subvectors. The approximate distance ≠ true distance.

### Independence assumption

PQ assumes subvector positions are independent — it trains each codebook separately. If dimensions are correlated across subvector boundaries, PQ misses that structure.

**Example:** If dims 95 and 96 are highly correlated but land in different subvectors, PQ can't exploit that.

**Fix:** OPQ (see section 8B below).

### More subvectors = less error per sub, but less expressiveness per code

| M (subvectors) | Sub-dim (D/M) | Code size | Quantization error |
|---|---|---|---|
| 4 | 192 | 4 bytes | Higher per sub (192 dims approximated by 1 of 256 centroids) |
| 8 | 96 | 8 bytes | Moderate |
| 16 | 48 | 16 bytes | Lower per sub |
| 32 | 24 | 32 bytes | Lowest |

More M = more bytes but better accuracy. It's a compression vs accuracy knob.

---

### 8B. OPQ (Optimized Product Quantization)

**The problem PQ has:** Subvector boundaries are arbitrary. You just chop the vector at every D/M dimensions. If correlated dimensions land in different subvectors, each codebook misses that structure and quantization error increases.

```
Dims: [0...95 | 96...191 | ...]
             ↑
       If dim 95 and dim 96 are correlated,
       they're split across codebooks 1 and 2.
       Neither codebook captures their relationship.
```

**OPQ's fix:** Learn a rotation matrix R that transforms vectors before splitting, so that within each subvector the variance is maximized and across subvectors the correlation is minimized.

```
Original vector:  v          (768-dim)
Rotated vector:   R × v      (768-dim, same size, just rotated)
Then apply PQ:    PQ_ENCODE(R × v)
```

**The rotation doesn't change distances** (it's orthogonal), but it redistributes information so the subvector splits are better aligned with the data's structure.

**Training:** Alternates between:
1. Fix rotation R → train PQ codebooks
2. Fix codebooks → optimize R to minimize total quantization error
3. Repeat until convergence

**In FAISS:** `OPQ8_96,PQ8` means "apply OPQ rotation (8 subspaces, 96 dims each), then PQ with 8 codebooks."

**When to use OPQ:**
- Almost always. It's strictly better than PQ with negligible extra cost
- Especially helps when data has strong cross-dimension correlations (which embeddings typically do)
- Typical improvement: 2-5% recall gain for free

---

### 8C. SDC vs ADC (Symmetric vs Asymmetric Distance)

**ADC (what we covered in section 6):** Query stays at full precision, only stored vectors are quantized. Distance = `dist(q_sub, centroid)` — asymmetric because one side is exact, one is approximate.

**SDC (Symmetric Distance Computation):** Quantize the query too. Distance = `dist(centroid_q, centroid_v)` — both sides are approximate.

**SDC advantage:** Can precompute all centroid-to-centroid distances (K* × K* table per subvector). No per-query table build needed.

**SDC disadvantage:** Two sources of quantization error instead of one. Worse recall.

**In practice:** ADC is almost always used. The per-query table build is cheap (M × K* = ~2048 distances), and the recall gain over SDC is significant. SDC only makes sense in extreme throughput scenarios where even building the table is too slow.

---

## 9. PQ Parameters

### M (number of subvectors)

- Controls code size (M bytes per vector)
- Must evenly divide D
- **Typical:** 8, 16, 32

### K* (centroids per codebook)

- Almost always **256** (fits in 1 byte)
- Some systems use 2^4=16 or 2^12=4096 for different tradeoffs
- 256 is the universal default

### Training set

- Codebooks are trained on a sample of vectors (doesn't need all N)
- Sample should be representative of the full dataset
- Larger sample = better codebooks, slower training

---

## 10. SQ vs PQ

| Property | SQ (int8) | PQ (M=8, K*=256) |
|---|---|---|
| Compression | 4x | 384x |
| Bytes per 768-dim vector | 768 | 8 |
| Accuracy loss | <1% recall | 5-15% recall (depends on data) |
| Distance computation | On quantized values | Table lookup |
| Training needed | Just min/max | K-means per subspace |
| Complexity | Trivial | Moderate |
| Use case | Moderate scale | Billion scale |

**SQ** is "good enough" compression.
**PQ** is "extreme but usable" compression.

---

## Key Takeaways

1. **PQ = split vector into subvectors, replace each with a codebook entry ID**
2. **Compression is massive** — 768-dim float32 → 8 bytes (384x)
3. **ADC makes search fast** — precompute distance table, then just do table lookups per vector
4. **Quantization error is the cost** — approximate distances, not exact
5. **K*=256 is universal** — one byte per subvector code
6. **M is the accuracy knob** — more subvectors = more bytes = better accuracy
7. **Codebooks are trained offline** via k-means on each subspace independently
8. **OPQ rotates before splitting** — reduces cross-subvector correlation, almost always worth using
9. **ADC > SDC** — keep query at full precision, only quantize stored vectors

---

**Next:** IVF + PQ — how clustering (which vectors to search) and compression (how to store and compare them) combine for production-scale vector search.