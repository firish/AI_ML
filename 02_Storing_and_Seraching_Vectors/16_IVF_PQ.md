# IVF + PQ: The Production Combo — Notes

## 0. Why Neither IVF Nor PQ Is Enough Alone

**IVF alone (file 12):**
- Tells you *which* vectors to search (cluster pruning)
- But vectors inside each cluster are still full-precision float32
- Memory: still too much for billion-scale

**PQ alone (file 14):**
- Compresses vectors to tiny codes (384x compression)
- But you still scan *all* N codes to find nearest neighbors
- Speed: still O(N) comparisons

**IVF + PQ together:**
> IVF narrows *which* vectors to check. PQ makes each check cheap and the vectors tiny.

This is the standard combination for billion-scale vector search. FAISS calls it `IndexIVFPQ`.

---

## 1. The Key Insight: Encode Residuals, Not Raw Vectors

This is the most important idea in IVF+PQ. Get this right and everything else follows.

### Why not just PQ-encode the raw vectors?

If you assign vector `v` to cluster centroid `c`, then `v` is close to `c`. All vectors in the same cluster are close to `c`. They look similar.

PQ-encoding these similar vectors wastes codebook capacity on the shared component (proximity to `c`). The codebook spends its 256 entries representing "near c" in many slightly different ways, instead of capturing the fine differences between vectors.

### The fix: encode the residual

```
residual = v - c
```

The residual is the **difference** between the vector and its cluster centroid. It captures only what's unique about `v` within its cluster.

**Properties of residuals:**
- Centered near zero (the shared component is subtracted out)
- Smaller magnitude than raw vectors
- Capture fine-grained differences between cluster members
- PQ codebooks can focus entirely on these differences

```
Raw vectors in cluster 5:    [0.81, 0.79, 0.83, ...]    (all similar)
Centroid of cluster 5:       [0.80, 0.80, 0.80, ...]
Residuals:                   [0.01, -0.01, 0.03, ...]   (diverse, small)
```

PQ is much more effective on the residuals — the codebook entries represent meaningful variations instead of redundant "near the centroid" patterns.

---

## 2. The Full Pipeline

### 2A. Training (Offline)

```python
function IVFPQ_TRAIN(vectors, K, M, Kstar):
    """
    vectors : training vectors (sample or full dataset)
    K       : number of IVF clusters
    M       : number of PQ subvectors
    Kstar   : PQ centroids per codebook (typically 256)

    centroids    : K cluster centroids from IVF
    pq_codebooks : M codebooks trained on residuals
    """
    # Step 1: Train IVF centroids
    centroids = K_MEANS(vectors, K)

    # Step 2: Compute residuals
    residuals = []
    for v in vectors:
        c = closest_centroid(v, centroids)
        residuals.append(v - c)

    # Step 3: Train PQ codebooks on the residuals
    pq_codebooks = PQ_TRAIN(residuals, M, Kstar)

    return centroids, pq_codebooks
```

**Two k-means runs:**
1. First k-means: learn the coarse partition (IVF centroids)
2. Second k-means: learn the fine compression (PQ codebooks on residuals)

### 2B. Adding Vectors (Index Build)

```python
function IVFPQ_ADD(v, centroids, pq_codebooks, inverted_lists):
    """
    v              : vector to add
    centroids      : IVF centroids
    pq_codebooks   : PQ codebooks (trained on residuals)
    inverted_lists : one list per centroid, stores (vector_id, pq_code) pairs

    residual       : v minus its assigned centroid
    pq_code        : compressed residual (M bytes)
    """
    # Step 1: Find closest centroid
    c_idx = argmin(dist(v, c) for c in centroids)

    # Step 2: Compute residual
    residual = v - centroids[c_idx]

    # Step 3: PQ-encode the residual
    pq_code = PQ_ENCODE(residual, pq_codebooks, M)

    # Step 4: Store in inverted list
    inverted_lists[c_idx].append((vector_id, pq_code))
```

**What's stored per vector:** just the cluster assignment (implicit from list) + M bytes of PQ code. The original float32 vector is thrown away.

### 2C. Query (Online)

```python
function IVFPQ_QUERY(q, k, nprobe, centroids, pq_codebooks, inverted_lists):
    """
    q              : query vector (full precision)
    k              : number of nearest neighbors to return
    nprobe         : number of clusters to search
    centroids      : IVF centroids
    pq_codebooks   : PQ codebooks
    inverted_lists : stored (vector_id, pq_code) pairs per cluster

    q_residual     : query's residual relative to each searched centroid
    dist_table     : precomputed distances for ADC lookup
    """
    # Step 1: Find nprobe closest centroids
    closest_clusters = top_nprobe(
        [(dist(q, c), i) for i, c in enumerate(centroids)],
        nprobe
    )

    results = []

    for c_idx in closest_clusters:
        # Step 2: Compute query residual relative to THIS centroid
        q_residual = q - centroids[c_idx]

        # Step 3: Build ADC distance table for this cluster
        dist_table = ADC_TABLE(q_residual, pq_codebooks, M)

        # Step 4: Scan inverted list using table lookups
        for (vec_id, pq_code) in inverted_lists[c_idx]:
            approx_dist = 0
            for m in range(M):
                approx_dist += dist_table[m][pq_code[m]]
            results.append((approx_dist, vec_id))

    # Step 5: Return top-k
    return top_k_closest(results, k)
```

**Critical detail in Step 2:** The query residual is recomputed for each cluster. Why? Because stored residuals are relative to their own cluster centroid. To compare `dist(q, v)` using residuals:

```
dist(q, v) ≈ dist(q - c, v - c) = dist(q_residual, v_residual)
```

Both sides must use the same centroid `c`. So for each cluster you probe, you recompute `q_residual = q - c` and rebuild the ADC table.

---

## 3. Why Residuals Matter: A Concrete Example

**Without residuals (PQ on raw vectors):**

Cluster centroid: `[0.5, 0.5, 0.5, 0.5]`

Vectors in cluster:
```
v1 = [0.51, 0.49, 0.52, 0.48]
v2 = [0.50, 0.51, 0.49, 0.50]
v3 = [0.52, 0.48, 0.51, 0.49]
```

PQ codebook sees: "these all look like [0.5, 0.5, ...]." Most centroids cluster around 0.5. The fine differences (0.01, 0.02) are lost in quantization noise.

**With residuals (PQ on v - c):**

```
r1 = [0.01, -0.01, 0.02, -0.02]
r2 = [0.00, 0.01, -0.01, 0.00]
r3 = [0.02, -0.02, 0.01, -0.01]
```

PQ codebook sees: centered-around-zero vectors with meaningful variation. Codebook entries capture the actual differences between vectors. Much better use of the 256 entries.

---

## 4. The Cost Breakdown

For N=1B vectors, D=768, K=4096, nprobe=16, M=8:

### Memory

```
Centroids:         4096 × 768 × 4 bytes          = ~12 MB
PQ codebooks:      8 × 256 × 96 × 4 bytes        = ~768 KB
Inverted lists:    1B × 8 bytes (PQ codes)        = ~8 GB
Vector IDs:        1B × 8 bytes (int64)           = ~8 GB
                                            Total ≈ 16 GB
```

Compare: raw float32 storage = 3 TB. **~200x reduction.**

### Query cost

```
Centroid search:   4096 × 768 dims               = ~3M float ops
Per cluster:
  Residual:        768 dims                       = ~768 float ops
  ADC table:       8 × 256 × 96 dims             = ~196K float ops
  List scan:       (1B/4096) × 8 lookups          = ~2M lookups
                                                  ≈ 244K vectors scanned

Total across 16 probes:                           ≈ 3.9M vectors scanned
```

Compare: brute force = 1B distance computations. **~256x speedup.**

---

## 5. How the ADC Table Rebuild Works Per Cluster

This is the part that confused many people, so let's be explicit.

**Stored residual in cluster 3:** `r_v = v - centroid_3`, then PQ-encoded.

**Query residual for cluster 3:** `r_q = q - centroid_3`.

**ADC table for cluster 3:** distances from `r_q`'s subvectors to each PQ centroid.

```
For subvector position m:
  r_q_sub = r_q[m*d : (m+1)*d]
  dist_table[m][j] = dist(r_q_sub, pq_codebooks[m][j])   for j in 0..255
```

**When scanning cluster 5 next:** recompute `r_q = q - centroid_5`, rebuild the table. Different centroid → different residual → different table.

**Cost of table rebuild:** M × K* × (D/M) = 8 × 256 × 96 ≈ 196K float ops. Tiny compared to scanning the list.

---

## 6. Parameters and Tuning

### IVF parameters

| Parameter | Controls | Typical |
|---|---|---|
| K (clusters) | Coarseness of partition | sqrt(N) to 4*sqrt(N) |
| nprobe | Clusters searched per query | 1–64, tune for recall target |

### PQ parameters

| Parameter | Controls | Typical |
|---|---|---|
| M (subvectors) | Code size, accuracy | 8, 16, 32 |
| K* (codebook size) | Expressiveness per subvector | 256 (always) |

### Interaction between them

- **Higher nprobe** compensates for PQ quantization error (searching more clusters catches what PQ distances get wrong)
- **Higher M** reduces quantization error but increases code size and scan time
- **Higher K** means smaller clusters → fewer vectors per probe → faster scan, but more centroids to compare

**Practical tuning order:**
1. Fix M based on memory budget
2. Fix K based on dataset size
3. Tune nprobe to hit your recall target

---

## 7. FAISS String Format

FAISS uses a compact string to describe index configurations:

```
"IVF4096,PQ8"
```
- IVF with 4096 clusters
- PQ with 8 subvectors (8 bytes per code)

```
"OPQ8_96,IVF4096,PQ8"
```
- OPQ preprocessing: rotate into 8 subspaces of 96 dims
- IVF with 4096 clusters
- PQ with 8 subvectors

```
"IVF16384,PQ16"
```
- More clusters (finer partition), more PQ bytes (better accuracy)
- Good for larger datasets where you can afford 16 bytes per vector

---

## 8. What Can Go Wrong

### Bad centroids (poor clustering)

If k-means doesn't converge well:
- Clusters are uneven (some huge, some empty)
- Large clusters slow down search
- Small clusters waste centroid budget

**Fix:** Train on a larger sample, use more k-means iterations, or use k-means++ initialization.

### Training data doesn't match query distribution

If you train on dataset A but query with distribution B:
- Centroids don't represent the query-relevant regions
- PQ codebooks are calibrated for wrong residual distribution

**Fix:** Train on data representative of actual usage.

### nprobe too low

Most common production issue. User sets nprobe=1, gets bad recall, blames the index.

**Fix:** Always benchmark recall@k vs nprobe and pick the right tradeoff.

---

## 9. The Full Picture: Where Everything Fits

```
Query arrives
    │
    ▼
[Compare q to K centroids]          ← IVF: coarse search (O(K))
    │
    ▼
[Pick nprobe closest clusters]      ← IVF: cluster selection
    │
    ▼
For each cluster:
    │
    ├─ [Compute q_residual = q - centroid]     ← Residual: recentering
    │
    ├─ [Build ADC table from q_residual]       ← PQ: precompute distances
    │
    └─ [Scan PQ codes with table lookups]      ← PQ: fast approximate distances
         │
         ▼
[Merge results across clusters]
    │
    ▼
[Return top-k]
```

---

## Key Takeaways

1. **IVF+PQ = cluster pruning + compressed storage + fast distance** — the production trifecta
2. **Encode residuals, not raw vectors** — PQ is far more effective on (v - centroid) than on v
3. **ADC table is rebuilt per cluster** — because residuals are relative to each cluster's centroid
4. **Memory: ~16 GB for 1B vectors** (vs 3 TB raw) — fits on a single machine
5. **Speed: ~256x over brute force** — search a fraction of vectors, each comparison is a table lookup
6. **nprobe is the main query-time knob** — trade speed for recall without rebuilding
7. **This is what production vector DBs actually run** — FAISS `IndexIVFPQ`, Milvus, Weaviate all build on this

---

**The indexing stack is now complete.** You understand flat search → graph indexes (NSW, HNSW) → space partitioning (KD/Ball/VP trees) → IVF → SQ → PQ → IVF+PQ. This covers the core of how modern vector databases store and search vectors at scale.