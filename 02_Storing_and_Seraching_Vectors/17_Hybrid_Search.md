# Hybrid Search: Dense + Sparse — Notes

## 0. Why Pure Vector Search Isn't Enough

Dense vector search (what we've covered so far) captures **semantic similarity** — "dog" matches "puppy", "canine", "pet."

But it fails at things exact keyword matching handles trivially:

- **Proper nouns:** "Dr. Patel's clinic" — vector search might return any doctor's clinic
- **Product IDs / codes:** "SKU-7829X" — no semantic meaning, must match exactly
- **Rare terms:** "CRISPR-Cas9" — if the embedding model saw it rarely, the vector is unreliable
- **Negation / precision:** "Python NOT snake" — vectors don't negate well

**Keyword/sparse search** (BM25, TF-IDF) handles these perfectly but misses semantic relationships.

**Hybrid search:** Run both, combine the results. Get the best of both worlds.

---

## 1. The Two Types of Representation

### Dense vectors (semantic)

What we've been studying. Every document/chunk gets a dense embedding:

```
"The dog sat on the mat" → [0.12, -0.45, 0.78, ..., 0.33]    (768 floats)
```

- Fixed size (768, 1024, 1536 dims)
- Every dimension has a value
- Captures meaning, not exact words
- Produced by embedding models (BERT, OpenAI, etc.)

### Sparse vectors (lexical)

Each document gets a vector where most values are zero. Non-zero entries correspond to specific terms:

```
"The dog sat on the mat" → {dog: 2.1, sat: 1.8, mat: 1.5, ...}    (mostly zeros)
```

- Dimensionality = vocabulary size (can be 30K–100K+)
- Only a few dozen non-zero entries per document
- Captures exact term presence and importance
- Produced by BM25, TF-IDF, or learned sparse models (SPLADE)

---

## 2. BM25 in 30 Seconds

BM25 is the standard keyword relevance scoring function. You don't need to memorize the formula, just the intuition:

**A term scores higher when:**
- It appears in the document (term frequency — TF)
- It's rare across all documents (inverse document frequency — IDF)
- The document is short (length normalization)

```
score(query, doc) = sum over query terms t:
    IDF(t) × (TF(t, doc) × (k1 + 1)) / (TF(t, doc) + k1 × (1 - b + b × |doc|/avgdl))
```

**Key insight for hybrid search:** BM25 gives a **relevance score**, not a distance. Higher = more relevant. Dense search gives a **distance** (or similarity). These are on completely different scales.

---

## 3. Why You Can't Just Average the Scores

Dense search returns: similarity scores (e.g., cosine: 0.0 to 1.0)
BM25 returns: relevance scores (e.g., 0 to 25+, unbounded)

```
Dense result:  doc_A, score = 0.82
BM25 result:   doc_B, score = 14.7
```

You can't add 0.82 + 14.7 meaningfully. The scales, distributions, and semantics are different.

This is the core challenge of hybrid search: **how to combine two ranked lists from incompatible scoring systems.**

---

## 4. Fusion Strategy 1: Reciprocal Rank Fusion (RRF)

The simplest and most popular approach. **Ignores scores entirely, uses only rank positions.**

```python
function RRF(ranked_lists, k_constant=60):
    """
    ranked_lists : list of ranked result lists (e.g., [dense_results, sparse_results])
    k_constant   : smoothing parameter (default 60, from the original paper)

    rrf_score    : fused score for each document
    rank         : position in a ranked list (1 = best)
    """
    scores = {}

    for result_list in ranked_lists:
        for rank, doc in enumerate(result_list, start=1):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1.0 / (k_constant + rank)

    return sorted(scores.items(), by=score, descending)
```

### How it works

```
Dense results:  [doc_A (rank 1), doc_C (rank 2), doc_B (rank 3)]
BM25 results:   [doc_B (rank 1), doc_A (rank 2), doc_D (rank 3)]

k = 60

doc_A: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
doc_B: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
doc_C: 1/(60+2) + 0         = 0.0161
doc_D: 0         + 1/(60+3) = 0.0159

Fused ranking: doc_A > doc_B > doc_C > doc_D
```

**Pros:**
- Dead simple
- No normalization needed — ranks are already on the same scale
- Works surprisingly well in practice
- No tuning required (k=60 is robust)

**Cons:**
- Throws away score magnitudes — a document at rank 2 with score 0.99 is treated the same as rank 2 with score 0.50
- Equal weight to both retrieval systems (can be modified with weighting)

**This is what most production systems use.** Weaviate, Qdrant, and many RAG pipelines default to RRF.

---

## 5. Fusion Strategy 2: Weighted Score Fusion

Normalize scores to the same scale, then combine with weights.

```python
function WEIGHTED_FUSION(dense_results, sparse_results, alpha=0.5):
    """
    dense_results  : [(doc, score), ...] from dense search
    sparse_results : [(doc, score), ...] from sparse search
    alpha          : weight for dense (1-alpha for sparse)
    """
    # Step 1: normalize each list to [0, 1]
    dense_norm = min_max_normalize(dense_results)
    sparse_norm = min_max_normalize(sparse_results)

    # Step 2: combine
    scores = {}
    for doc, score in dense_norm:
        scores[doc] = alpha * score
    for doc, score in sparse_norm:
        scores[doc] = scores.get(doc, 0) + (1 - alpha) * score

    return sorted(scores.items(), by=score, descending)
```

### Normalization methods

**Min-max:** Scale to [0, 1] using min and max of each result list.
```
normalized = (score - min) / (max - min)
```

**Z-score:** Normalize by mean and standard deviation.
```
normalized = (score - mean) / std
```

**Pros:**
- Uses score magnitudes (not just ranks)
- `alpha` lets you tune the dense/sparse balance

**Cons:**
- Normalization is fragile — depends on score distribution of each query
- Requires tuning alpha (0.5 is okay, but optimal varies by use case)
- Different queries may need different alpha values

---

## 6. Fusion Strategy 3: Learned Fusion (Re-ranking)

Use a **cross-encoder** or learned model to re-rank the merged candidate set.

```
Step 1: Dense search → top 100
Step 2: BM25 search → top 100
Step 3: Merge → ~150-200 unique candidates
Step 4: Cross-encoder scores each (query, candidate) pair
Step 5: Sort by cross-encoder score → final ranking
```

**Pros:**
- Best quality — cross-encoder sees the full query-document interaction
- No manual weight tuning

**Cons:**
- Slowest — cross-encoder runs on each candidate (not scalable to millions)
- Only works as a re-ranking step on a small candidate set

**This is the gold standard for RAG pipelines:** cheap retrieval (dense + sparse) → expensive re-ranking (cross-encoder) on top candidates.

---

## 7. SPLADE: Learned Sparse Representations

Traditional BM25 is a fixed formula. **SPLADE** is a learned model that produces sparse vectors:

```
"The dog sat on the mat" → {dog: 2.1, puppy: 0.8, canine: 0.5, sat: 1.2, mat: 1.5, rug: 0.3, ...}
```

Notice: it added "puppy", "canine", "rug" — terms not in the original text. The model learned to **expand** terms with related words.

**Why SPLADE matters:**
- Bridges the gap between dense and sparse — it's sparse but has some semantic awareness
- Still fast (sparse dot products on inverted indexes)
- Can sometimes replace hybrid search entirely

**In hybrid search:** Some systems use SPLADE instead of BM25 as the sparse component. It's a stronger sparse retriever.

---

## 8. Architecture: How Systems Run Both Searches

### Option A: Two separate indexes

```
                 Query
                /     \
    Dense Index         Sparse Index
    (HNSW/IVF)         (Inverted Index / BM25)
         |                    |
    dense results       sparse results
         \                /
          Fusion (RRF)
              |
         Final results
```

Most common. Elasticsearch + vector plugin, Weaviate, Qdrant all do this.

### Option B: Single index with both representations

Store both dense and sparse vectors per document. Index handles both search types internally.

Vespa, Pinecone (with sparse-dense), and some custom systems do this.

**Tradeoff:** Option A is simpler, Option B avoids the overhead of maintaining two indexes.

---

## 9. When to Use What

| Scenario | Best approach |
|---|---|
| General semantic search | Dense only |
| Exact term matching matters | Sparse (BM25) only |
| Production RAG pipeline | Hybrid (dense + sparse + re-rank) |
| Proper nouns / codes / rare terms | Must include sparse |
| Quick prototype | Dense only, add sparse later if recall is low |

### Alpha tuning guidance

| alpha (dense weight) | When |
|---|---|
| 0.7–0.9 | Queries are mostly semantic ("explain how X works") |
| 0.3–0.5 | Queries mix semantic + keyword ("CRISPR-Cas9 gene editing mechanism") |
| 0.1–0.3 | Queries are mostly keyword / lookup ("error code 0x80070005") |

---

## Key Takeaways

1. **Dense captures meaning, sparse captures exact terms** — neither is sufficient alone for production search
2. **RRF is the default fusion** — rank-based, no normalization needed, works well out of the box
3. **Scores are incompatible** — you can't directly add cosine similarity and BM25 scores
4. **Re-ranking with cross-encoders** is the quality ceiling — expensive but best results
5. **SPLADE** bridges dense and sparse by learning to expand terms
6. **Most RAG pipelines** use: dense + sparse retrieval → RRF fusion → cross-encoder re-rank on top candidates
7. **Hybrid search is about recall** — cast a wider net, then precision-filter with re-ranking