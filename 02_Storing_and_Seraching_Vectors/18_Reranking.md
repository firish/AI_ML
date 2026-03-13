# Re-ranking: Cheap Retrieval → Expensive Precision — Notes

## 0. The Core Tradeoff That Motivates Re-ranking

Every search system faces the same tension:

| Method | Speed | Quality | Can run on N vectors |
|---|---|---|---|
| PQ / HNSW / IVF | Fast | Approximate | Yes (millions–billions) |
| Exact L2 / cosine | Moderate | Exact distances | Only small sets |
| Cross-encoder | Slow | Best relevance | Only tiny sets (50–500) |

**The insight:** You don't need the best method on all N vectors. You need:
1. A **cheap method** to narrow N down to a small candidate set
2. An **expensive method** to precisely rank that small set

That's re-ranking. Two passes. Cheap then expensive.

---

## 1. The General Pattern

```
All vectors (N = 1 billion)
    │
    ▼
[Stage 1: Retrieval — fast, approximate]
    │   HNSW, IVF+PQ, hybrid search
    │   Returns top-C candidates (C = 100–1000)
    ▼
Candidate set (C = 100–1000)
    │
    ▼
[Stage 2: Re-ranking — slow, precise]
    │   Exact distances, cross-encoder, or both
    │   Returns top-k (k = 10–50)
    ▼
Final results (k = 10)
```

Stage 1 optimizes for **recall** — don't miss good results.
Stage 2 optimizes for **precision** — get the ordering right.

---

## 2. Three Levels of Re-ranking

### Level 1: Exact Distance Re-ranking

**Stage 1:** PQ approximate distances → top-C candidates
**Stage 2:** Recompute exact float32 distances for those C candidates

```python
function EXACT_RERANK(q, candidates, raw_vectors, k):
    """
    q            : query vector (full precision)
    candidates   : top-C results from approximate search (e.g., PQ/IVF)
    raw_vectors  : original float32 vectors (stored on disk or in memory)
    k            : final number of results to return

    C            : size of candidate set (e.g., 100-1000)
    """
    reranked = []
    for (approx_dist, vec_id) in candidates:
        # fetch original vector and compute exact distance
        exact_dist = dist(q, raw_vectors[vec_id])
        reranked.append((exact_dist, vec_id))

    return top_k_closest(reranked, k)
```

**Why this helps:** PQ distances have quantization error. Two vectors with PQ distances 0.45 and 0.47 might have exact distances 0.42 and 0.51. The ranking changes.

**Cost:** C exact distance computations. For C=500 and D=768, that's 500 × 768 float ops. Trivial.

**Where raw vectors live:**
- In RAM (if you can afford it): fastest
- On SSD: still fast enough (500 random reads ≈ 1-2ms on NVMe)
- This is the "disk-based ANN" pattern — PQ codes in RAM, raw vectors on disk

### Level 2: Cross-Encoder Re-ranking

**Stage 1:** Dense + sparse retrieval → top-C candidates
**Stage 2:** Cross-encoder scores each (query, document) pair

```python
function CROSS_ENCODER_RERANK(query_text, candidates, documents, k):
    """
    query_text  : original query string
    candidates  : top-C results from retrieval
    documents   : original document texts
    k           : final number of results

    cross_encoder : a model that takes (query, document) and outputs a relevance score
    """
    reranked = []
    for (_, doc_id) in candidates:
        # cross-encoder sees BOTH query and document together
        relevance = cross_encoder.score(query_text, documents[doc_id])
        reranked.append((relevance, doc_id))

    return top_k_by_score(reranked, k)
```

**Why this is better than exact distances:**

Bi-encoders (what produces embeddings) encode query and document **independently**:
```
query_vec  = encoder("what causes headaches")
doc_vec    = encoder("Migraines are triggered by stress and dehydration")
similarity = cosine(query_vec, doc_vec)
```

Cross-encoders process query and document **together**:
```
relevance = cross_encoder("what causes headaches", "Migraines are triggered by stress and dehydration")
```

The cross-encoder sees the full interaction — it can attend across both texts, catch negations, understand paraphrasing, resolve ambiguity. This is fundamentally more powerful than comparing two independently computed vectors.

**Cost:** One model forward pass per candidate. For C=100 with a small cross-encoder, ~50-200ms total. Too slow for millions, fine for 100.

### Level 3: Multi-Stage Pipeline

```
All vectors (1B)
    │
    ▼
[Stage 1: IVF+PQ — approximate]     → top 1000
    │
    ▼
[Stage 2: Exact distances]           → top 100
    │
    ▼
[Stage 3: Cross-encoder]            → top 10
```

Each stage is more expensive but runs on a smaller set. The pipeline gets progressively more precise.

---

## 3. Why Re-ranking Works (The Math Intuition)

### Recall vs precision at each stage

**Stage 1 goal:** High recall — make sure the true top-10 are somewhere in your top-1000.

If PQ search has 95% recall@1000, that means 95% of the true top-1000 are in your candidate set. Good enough.

**Stage 2 goal:** Correct ordering — among those 1000, find the actual top-10.

PQ distances might scramble the ordering. Exact distances fix it. Cross-encoder makes it even better.

### Why not just use the expensive method?

```
Cross-encoder on 1B documents:  1B × 2ms = 23 days
Cross-encoder on 100 documents: 100 × 2ms = 200ms
```

The cheap first pass makes the expensive second pass feasible.

---

## 4. The Candidate Set Size (C) Tradeoff

C is how many candidates you pass from stage 1 to stage 2.

| C (candidates) | Recall of stage 1 | Stage 2 cost | Overall quality |
|---|---|---|---|
| 10 | Low — might miss true top-k | Cheapest | Risky |
| 100 | Good | Fast | Sweet spot for cross-encoder |
| 500 | Very good | Moderate | Sweet spot for exact distance |
| 1000+ | Excellent | Expensive for cross-encoder | Diminishing returns |

**Rule of thumb:**
- Exact distance re-rank: C = 500–1000 (distance computation is cheap)
- Cross-encoder re-rank: C = 50–200 (model inference is expensive)

**The critical invariant:** C must be large enough that the true top-k results are almost certainly in the candidate set. If true result #3 isn't retrieved in stage 1, no amount of re-ranking will find it.

---

## 5. Bi-Encoder vs Cross-Encoder (The Key Distinction)

This matters for understanding why re-ranking improves quality.

### Bi-encoder (used in retrieval)

```
query  ──→ [Encoder] ──→ query_vec   ─┐
                                       ├──→ cosine similarity
doc    ──→ [Encoder] ──→ doc_vec     ─┘
```

- Query and document encoded **independently**
- Document vectors can be precomputed and indexed
- Fast: one query encoding + ANN lookup
- But: can't model fine-grained query-document interaction

### Cross-encoder (used in re-ranking)

```
(query, doc) ──→ [Encoder] ──→ relevance score
```

- Query and document processed **together** as one input
- Full attention between query and document tokens
- Slow: must run the model for each (query, doc) pair
- But: much higher quality — sees interactions, negations, paraphrases

**Why bi-encoders can't be as good:** They compress all information about a document into a single fixed vector before ever seeing the query. Some query-specific nuance is inevitably lost.

**Why cross-encoders can't be used for retrieval:** You'd need to run the model on every (query, document) pair in the corpus. For 1M documents, that's 1M forward passes per query.

**Re-ranking is the bridge:** Use bi-encoder for fast retrieval, cross-encoder for precise ranking on the small candidate set.

---

## 6. Practical Re-ranking Models

### For exact distance re-ranking

No model needed — just store raw vectors and recompute distances. Common in FAISS workflows:

```python
# FAISS pattern: PQ search + exact re-rank
index = faiss.IndexIVFPQ(...)     # fast, approximate
index.search(query, k=1000)       # stage 1

# re-rank with flat index or raw vectors
exact_distances = compute_exact(query, candidate_vectors)  # stage 2
```

### For cross-encoder re-ranking

Popular models (as of 2025):
- **Cohere Rerank** — API-based, easy to use
- **BGE Reranker** — open source, good quality
- **cross-encoder/ms-marco-MiniLM** — small, fast, decent
- **Jina Reranker** — open source, multilingual

**Typical latencies:**
- Small cross-encoder (MiniLM): ~1-2ms per pair → 100 pairs ≈ 100-200ms
- Large cross-encoder: ~5-10ms per pair → 100 pairs ≈ 500ms-1s

---

## 7. Re-ranking in RAG Pipelines

In Retrieval-Augmented Generation, re-ranking sits between retrieval and generation:

```
User query
    │
    ▼
[Retrieval: dense + sparse]          → 100 chunks
    │
    ▼
[Re-rank: cross-encoder]            → top 5-10 chunks
    │
    ▼
[LLM generation with top chunks]    → answer
```

**Why this matters for RAG:**
- LLMs have limited context windows
- Stuffing 100 chunks into the prompt is wasteful and noisy
- Re-ranking selects the 5-10 most relevant chunks
- Higher relevance in context → better generated answers

**The quality chain:** Better retrieval → better re-ranking → better context → better LLM output. Re-ranking is often the highest-leverage improvement in a RAG pipeline.

---

## 8. When to Use Each Level

| Scenario | Re-ranking level | Why |
|---|---|---|
| PQ/IVF search, need better accuracy | Exact distance | Fixes quantization error, cheap |
| Semantic search for RAG | Cross-encoder | Best relevance for LLM context |
| Latency-critical (< 50ms) | Exact distance or none | Cross-encoder too slow |
| High-stakes search (legal, medical) | Full pipeline (exact + cross-encoder) | Maximum quality |
| Prototype / MVP | None | Add re-ranking when recall is the bottleneck |

---

## Key Takeaways

1. **Re-ranking = cheap retrieval on N vectors, expensive precision on C candidates**
2. **Two main types:** exact distance (fixes PQ error, cheap) and cross-encoder (best relevance, slower)
3. **Bi-encoder encodes independently** (fast, indexable) — **cross-encoder encodes together** (slow, precise)
4. **Candidate set size C is critical** — too small risks missing true results, too large wastes compute
5. **Multi-stage pipelines** progressively narrow and refine: IVF+PQ → exact distances → cross-encoder
6. **In RAG:** re-ranking is often the highest-leverage improvement — better chunks → better LLM output
7. **Re-ranking can't fix bad retrieval** — if stage 1 misses a relevant document entirely, stage 2 can't recover it