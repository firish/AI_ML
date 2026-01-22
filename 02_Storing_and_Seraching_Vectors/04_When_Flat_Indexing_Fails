# Why Brute-Force Breaks at Scale — Notes

## Two Independent Failure Modes

### (A) Data Scale Explodes
- 1M vectors → OK
- 100M vectors → borderline
- 1B vectors → not OK (even on GPU)

**Bottleneck shifts:** compute → **memory bandwidth**
- Must load all vectors from memory
- Must touch every dimension
- Per query, every time

---

### (B) High Dimensionality Kills Geometry

## The Curse of Dimensionality

### Low vs High Dimensions

| Dimension | Behavior |
|-----------|----------|
| 2D-3D | Points cluster, clear neighbors |
| 10D | Fewer obvious neighbors |
| 100D | Most distances similar |
| 768D | Nearest ≈ average ≈ farthest |

**Mathematical collapse:**
```text
(max_dist − min_dist) / min_dist → 0  as d → ∞
```

> **Relative distance contrast collapses**

---

## Why Curse Breaks Optimizations

### 1. Early-Exit Becomes Useless

Early-exit requires: "bad candidates get bad quickly"

**In high dimensions:**
- Partial distances grow slowly
- Must compute almost all dimensions
- Early termination rarely triggers
- Pay full O(d) cost anyway

---

### 2. Tree-Based Indexes Stop Pruning

KD-trees, Ball trees, VP-trees work by:
> "Split space, prune far regions"

**In high dimensions:**
- Almost every region overlaps query
- Visit most branches anyway
- Performance → brute-force

**KD-trees die beyond ~30 dimensions**

---

## The Breaking Point

Brute-force fails when:
- N ≥ 10M
- d ≥ 256
- Frequent queries
- Memory doesn't fit in cache

**Results:**
- Latency spikes
- Cost explodes
- Throughput tanks

→ **Must avoid touching most vectors**

---

## What Indexes Fundamentally Do

**One thing:**
> **Avoid comparing query against most vectors**

**How:**
- Group vectors
- Navigate structure
- Prune candidates aggressively
- Accept small accuracy loss


## Two Index Philosophies

### 1. Partition-Based (IVF, ScaNN)
> "Only search few clusters near query"

- Pre-cluster space
- Probe 1-5% of vectors
- Skip rest entirely

### 2. Graph-Based (HNSW)
> "Walk neighborhood graph greedily"

- Start from random node
- Jump closer and closer
- Never scan everything

**Both exploit:** Semantic embeddings have **local structure**, even if global geometry collapses

---

## Core Tradeoff

| Exact Search | ANN Search |
|--------------|------------|
| Perfect recall | Approximate |
| Touch every vector | Touch 0.1-5% |
| Predictable | Tunable accuracy |
| Poor scalability | Scales to billions |

> **Exact search does not degrade gracefully**


## Key Takeaway

- High-D space destroys distance contrast
- Brute-force must touch everything
- Memory bandwidth = real bottleneck
- **Indexes exist to avoid comparisons, not speed them up**
