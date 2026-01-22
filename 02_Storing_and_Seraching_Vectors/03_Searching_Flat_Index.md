# Exact (Brute-Force) Vector Search — The Baseline

Before we talk about clever indexing (HNSW, IVF, PQ), we must understand the **ground truth**:
what happens if we simply compare a query vector against *every* stored vector.

This gives us:
- **Perfect recall** (nothing is missed)
- **Predictable correctness**
- A latency baseline to beat

Everything else is a speed–accuracy tradeoff relative to this.

---

## 1. Naïve brute-force search (conceptual)

### Problem setup

- Database: `N` vectors, each of dimension `d`
- Query: one vector `q`
- Goal: find top-k closest vectors

### Naïve algorithm

```text
for each vector v_i in database:
    compute distance(q, v_i)
return k smallest distances
````

### Cost

```text
Time:  O(N × d)
Space: O(N × d)
```

Example:

* 1 million vectors
* 768 dimensions
  → ~768 million multiplications per query

This is **too slow in Python**, but very fast in **compiled linear algebra**.

---

## 2. Matrix multiplication view (the key optimisation)

Instead of looping, stack all vectors into a matrix:

```text
V ∈ R^(N × d)
q ∈ R^d
```

Then compute:

```math
scores = V · q
```

This gives **all dot products at once**.

Why this is powerful:

* CPUs & GPUs are insanely good at matrix multiplication
* Cache-aware
* Vectorised
* Parallelised

This is the foundation of **BLAS**, **Faiss Flat**, and GPU search.

---

## 3. BLAS / Faiss Flat scanners

### What “Flat” means

**Flat index = no index at all**

* Store vectors exactly as-is
* Scan everything every time
* Return exact nearest neighbours

### Faiss Flat variants

| Index             | Distance                   |
| ----------------- | -------------------------- |
| `IndexFlatL2`     | Euclidean                  |
| `IndexFlatIP`     | Inner product              |
| `IndexFlatCosine` | (cosine via normalisation) |

Internally:

* Uses BLAS (CPU) or CUDA (GPU)
* Highly optimised C++
* Often faster than “indexed” methods below ~100k vectors

### Why Flat is important

* **Ground truth** for recall evaluation
* Baseline for benchmarking ANN methods
* Often used as final re-ranker

---

## 4. SIMD & GPU batching (why brute force is still viable)

### SIMD (CPU)

SIMD = Single Instruction, Multiple Data

* CPU computes 8–32 multiplications per clock cycle
* Dot products vectorised automatically
* Memory prefetching + cache locality

### GPU batching

On GPUs:

* Thousands of dot products computed in parallel
* Memory bandwidth dominates, not compute
* Flat search over millions of vectors can be **<10 ms**

This is why:

> brute-force is still used in production at moderate scale

---

## 5. Early-exit optimisation

Early-exit is a *small but powerful trick*.

### Idea

When computing a distance dimension by dimension:

```text
partial_sum += (q[i] - v[i])²
```

If `partial_sum` already exceeds the current worst top-k distance:
→ **stop computing this vector**

This works because:

* Many candidates are obviously bad
* Saves wasted multiplications

Used heavily in:

* L2 distance
* CPU-based flat search

---

## 6. Top-k heap (selection strategy)

After computing distances, we need the **k smallest**.

Naïve approach:

* Sort all N distances → O(N log N) ❌

Correct approach:

* Maintain a **max-heap of size k**

Algorithm:

```text
for each distance d:
    if heap.size < k:
        push d
    else if d < heap.max:
        pop heap.max
        push d
```

Cost:

```text
O(N log k)
```

Since `k` is small (10–100), this is cheap.

---

## 7. Why exact search matters (even if you won’t use it)

Exact search establishes:

### 1. Ground-truth recall

ANN methods are judged by:

```text
recall@k = ANN_results ∩ exact_results
```

### 2. Latency floor

If brute-force takes:

* 40 ms on CPU
* 5 ms on GPU

Then ANN must beat this **meaningfully**, or it’s pointless.

### 3. Re-ranking strategy

Most real systems:

1. ANN retrieves 100–1000 candidates
2. Exact distance is computed on those
3. Final top-k returned

Exact search never disappears — it just moves downstream.

---

## 8. When brute-force is actually enough

Use **Flat search** when:

* N ≤ 100k (CPU)
* N ≤ few million (GPU)
* You need perfect recall
* You batch queries
* Latency budget allows 10–50 ms

Avoid indexing until:

* You *measure* brute-force and it’s too slow
* You know your recall target

---

## Key mental model (important)

> **ANN indexes don’t replace brute force.
> They approximate it faster.**

If you don’t understand exact search deeply, you will:

* Tune ANN parameters blindly
* Misinterpret recall numbers
* Choose the wrong index

---

## TL;DR

* Exact search = compare against everything
* Implemented efficiently via matrix multiply
* Faiss Flat = gold standard baseline
* SIMD & GPUs make brute-force surprisingly fast
* Used for benchmarking, re-ranking, and small-scale production
