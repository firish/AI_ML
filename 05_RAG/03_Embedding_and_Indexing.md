## Embedding and Indexing — Turning Chunks into Searchable Vectors

---

## 0. Where This Fits in the Pipeline

```text
Full RAG pipeline:

    Documents
        │
        ▼
    Chunking          ← file 02
        │
        ▼
    Embedding         ← this file (first half)
        │
        ▼
    Indexing          ← this file (second half)
        │
        ▼
    Retrieval         ← file 04
        │
        ▼
    Augmentation + Generation  ← file 05

After chunking, you have a list of text strings.
Embedding converts each string into a vector.
Indexing stores those vectors so you can search them fast.
```

---

## 1. What an Embedding Model Does

An embedding model maps a piece of text to a fixed-length vector of floats.

```text
Input:  "Refunds are accepted within 30 days of purchase."
Output: [0.021, -0.847, 0.334, 0.119, -0.562, ..., 0.088]
         ↑ 768 or 1536 or 3072 numbers, depending on the model

Input:  "How do I return an item?"
Output: [0.019, -0.831, 0.341, 0.127, -0.571, ..., 0.092]
         ↑ very similar values — these texts have similar meaning

Input:  "The French Revolution began in 1789."
Output: [0.712, 0.234, -0.891, -0.445, 0.023, ..., -0.341]
         ↑ very different values — unrelated meaning
```

The model is trained so that texts with similar meaning produce vectors that are close together in the vector space (high cosine similarity), and texts with different meaning produce vectors that are far apart.

```text
Cosine similarity:
    sim("Refunds accepted within 30 days", "How do I return an item?") = 0.91  ← close
    sim("Refunds accepted within 30 days", "French Revolution 1789")    = 0.04  ← far
```

This is the same technology covered in Phase 1 (encoder models, [CLS] pooling, contrastive training). The difference here is that the models are specifically trained on retrieval tasks — matching queries to relevant passages.

---

## 2. Bi-Encoder vs Cross-Encoder

There are two types of models used in the retrieval context. Understanding the difference is important.

### Bi-encoder (what you use for embedding + search)

```text
Process query and document SEPARATELY, each gets its own vector.

    Query:    "How do I return an item?"     → q_vec = [0.019, -0.831, ...]
    Document: "Refunds within 30 days..."   → d_vec = [0.021, -0.847, ...]

    Similarity = cosine(q_vec, d_vec) = 0.91

Speed: FAST
    You pre-compute and store all document vectors offline.
    At query time: embed the query (one model call), then do nearest-neighbour
    search against the stored vectors. No need to re-run the document through the model.

Weakness: the query and document never "see" each other during encoding.
    The model can't resolve subtle relevance that depends on comparing the two.
    "bank" → financial institution or river bank?
    The bi-encoder doesn't know which meaning fits until it sees the document.
    This limits precision for ambiguous queries.
```

### Cross-encoder (what you use for re-ranking)

```text
Process query and document TOGETHER. The model sees both at once.

    Input: "[query] How do I return an item? [sep] Refunds within 30 days..."
    Output: relevance score = 0.94  (a single number, not a vector)

Speed: SLOW
    Can't pre-compute. Every (query, document) pair requires a full model pass.
    For 10,000 candidate documents: 10,000 model calls at query time. Unusable.

Strength: much higher precision.
    The model can attend across query and document simultaneously.
    Resolves ambiguity, catches nuance, understands the relationship.

How it's used in practice:
    Step 1: Bi-encoder retrieves top-100 candidates quickly
    Step 2: Cross-encoder re-ranks those 100 to find the best 5

This two-stage approach gets near-cross-encoder quality at near-bi-encoder speed.
Re-ranking is covered in file 07.
```

---

## 3. Embedding Models — What's Available

### OpenAI (API, closed)

```text
Model                       Dimensions    Max tokens    Notes
──────────────────────────────────────────────────────────────────────────
text-embedding-3-small      1536          8191          Best price/performance
text-embedding-3-large      3072          8191          Highest quality OpenAI model
text-embedding-ada-002      1536          8191          Legacy, worse than v3-small

Cost (as of 2024):
    text-embedding-3-small: $0.02 / 1M tokens
    text-embedding-3-large: $0.13 / 1M tokens

When to use:
    Easy integration (one API call), no GPU needed, good quality.
    Risk: vendor lock-in, ongoing API cost, data leaves your infra.
```

### Cohere (API, closed)

```text
Model                       Dimensions    Notes
──────────────────────────────────────────────────────────────────────────
embed-english-v3.0          1024          Best for English retrieval
embed-multilingual-v3.0     1024          Strong multilingual support

Standout feature: Cohere models accept an input_type parameter.
    input_type = "search_document"  → use when embedding chunks
    input_type = "search_query"     → use when embedding the query

This asymmetric encoding produces meaningfully better retrieval results
because query and document have different linguistic patterns.
```

### Open-source / self-hosted

```text
Model                       Dimensions    Max tokens    Notes
──────────────────────────────────────────────────────────────────────────
BAAI/bge-large-en-v1.5      1024          512           Top OSS English model
BAAI/bge-m3                 1024          8192          Multilingual + long context
intfloat/e5-large-v2        1024          512           Strong general retrieval
sentence-transformers/       384           128           Tiny, fast, surprisingly good
    all-MiniLM-L6-v2                                     for speed-constrained cases
Nomic/nomic-embed-text-v1   768           8192          Apache 2.0, long context

When to use:
    Private data (chunks never leave your infra).
    High volume (no per-token cost — just GPU cost).
    Custom fine-tuning on your domain (possible with OSS models).

Tradeoff: you manage the model (GPU server, updates, scaling).
```

---

## 4. How to Pick an Embedding Model

### The MTEB Benchmark

```text
MTEB (Massive Text Embedding Benchmark) is the standard leaderboard
for comparing embedding models across tasks.

    https://huggingface.co/spaces/mteb/leaderboard

Task categories:
    Retrieval     ← most relevant for RAG (this is what you care about)
    Classification
    Clustering
    Reranking
    STS (semantic textual similarity)

Key metric for RAG: NDCG@10 on retrieval tasks.
    NDCG@10 = "how good are the top-10 results?"
    Higher = better retrieval quality.

Don't just pick the #1 overall model — check the retrieval subtask score,
and check models in the size/cost range you can actually run.
```

### Practical decision matrix

```text
Situation                          Recommendation
────────────────────────────────────────────────────────────────────────
Prototype / hackathon              text-embedding-3-small (cheap, easy)
Production, English only           bge-large-en-v1.5 (OSS) or embed-english-v3.0
Production, multilingual           bge-m3 or embed-multilingual-v3.0
Private data, no external API      Any OSS model on your own GPU
Long documents (>512 tokens)       bge-m3 or nomic-embed-text (8K window)
Latency-critical                   all-MiniLM-L6-v2 (384 dims, tiny and fast)
Domain-specific (medical, legal)   Fine-tune bge or e5 on your domain data
```

### Dimensions and their trade-offs

```text
Higher dimensions → richer representation → better accuracy
Lower dimensions  → faster search → less memory → cheaper storage

Typical impact:
    3072-dim vs 1536-dim: ~2-5% better retrieval on benchmarks
    1536-dim vs 768-dim:  ~3-8% better
    768-dim vs 384-dim:   ~5-10% better

Storage impact:
    1M chunks × 1536 dims × 4 bytes (float32) = 6 GB of raw vectors
    1M chunks × 384 dims × 4 bytes             = 1.5 GB

Matryoshka Representation Learning (MRL):
    Some newer models (text-embedding-3, nomic-embed) support truncating
    their vectors to a smaller dimension without retraining.

    text-embedding-3-large (3072 dims) truncated to 256 dims:
        Storage: 12× smaller
        Latency: much faster ANN search
        Quality: only ~10-15% worse than full 3072 dims

    This lets you tune the accuracy/cost tradeoff at query time,
    with no re-embedding needed.
```

---

## 5. Asymmetric vs Symmetric Retrieval

Not all embedding models treat query and document the same way.

```text
Symmetric retrieval:
    Query and document are embedded the same way.
    Designed for: finding similar texts (duplicate detection, clustering).

    Example: "How do I return an item?" → finds other similar questions.
    Bad for RAG: the stored chunks are ANSWERS, not questions.
                 You want semantic matching across different text styles.

Asymmetric retrieval:
    The model is trained to match SHORT queries against LONG passages.
    This is what RAG needs.

    Query: "refund policy international orders" (short, keyword-like)
    Document: "For international customers, our return policy allows refunds
               within 14 days of delivery. Customs duties paid by..." (long, prose)

    Asymmetric models handle this mismatch well.
    Symmetric models struggle — the query and passage look very different.

In practice:
    Most modern retrieval models (bge, e5, Cohere embed v3) are trained
    for asymmetric retrieval and handle this correctly by default.
    Cohere makes it explicit with the input_type parameter.
```

---

## 6. Embedding in Batch

For large document sets, embedding one chunk at a time is slow. Batch it.

```text
Naive (slow):
    for chunk in chunks:
        vector = embed(chunk)    ← one API call per chunk
        store(vector)

    For 100,000 chunks: 100,000 sequential API calls.

Batched (fast):
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vectors = embed_batch(batch)    ← one call returns 100 vectors
        store_batch(vectors)

    For 100,000 chunks: 1,000 API calls.

OpenAI: accepts up to 2048 inputs per call.
Cohere: 96 texts per call.
OSS (local): limited by GPU memory, typically 32–512 texts per batch.

Estimated time for 100K chunks at text-embedding-3-small:
    Sequential: ~5-10 minutes (rate limits)
    Batched:    ~30-60 seconds
```

---

## 7. What Gets Stored in the Index

Each chunk produces three things that get stored together:

```text
┌──────────────────────────────────────────────────────────────────┐
│  chunk_id: "doc_42_chunk_7"                                      │
│                                                                  │
│  vector: [0.021, -0.847, 0.334, ..., 0.088]   ← 1536 floats     │
│          ↑ used for ANN search                                   │
│                                                                  │
│  metadata: {                                                     │
│      "text":        "Refunds within 30 days of purchase...",     │
│      "source":      "return_policy.pdf",                         │
│      "page":        3,                                           │
│      "section":     "International Returns",                     │
│      "created_at":  "2024-09-15",                                │
│      "doc_id":      "doc_42",                                    │
│      "parent_id":   "doc_42_section_2"   ← for hierarchical RAG │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘

The vector is what gets searched.
The metadata is what gets returned and shown to the LLM.

Metadata also enables filtered search:
    "Find the most relevant chunks, but only from documents
     created after 2024-01-01 and from source = 'legal_docs'"
    → vector search + metadata filter simultaneously
    → covered more in Phase 2 (file 16: filtered vector search)
```

---

## 8. How Indexing Works

This is where Phase 2 (vector databases) connects directly to RAG.

```text
After embedding all chunks, you load the vectors into a vector database.
The DB builds an index structure for fast approximate nearest-neighbour (ANN) search.

Two common index types (covered in depth in Phase 2):

HNSW (Hierarchical Navigable Small World):
    Builds a multi-layer graph over the vectors.
    Search traverses the graph, skipping large regions quickly.
    Query time: O(log n) — fast even at millions of vectors.
    Build time: slow (graph construction).
    Memory: high (stores the graph structure).
    Used by: Pinecone, Weaviate, Qdrant, Chroma.

IVF (Inverted File Index):
    Clusters vectors into k groups via k-means.
    At query time: only search the nprobe nearest clusters.
    Query time: fast (only searches a fraction of the index).
    Build time: moderate.
    Memory: lower than HNSW.
    Used by: FAISS, older Milvus configs.

For most RAG use cases: HNSW is the default.
    It has better recall (finds true nearest neighbours more reliably)
    and handles new insertions cleanly (no need to rebuild).
```

### Popular vector databases

```text
Database        Hosting         Best for
────────────────────────────────────────────────────────────────────
Pinecone        Managed cloud   Production RAG, no infra management
Weaviate        Self-host/cloud Hybrid search (dense + BM25 built-in)
Qdrant          Self-host/cloud High performance, rich filtering
Chroma          Local / OSS     Local development, prototyping
FAISS           Library (OSS)   Pure library, no server, research use
pgvector        PostgreSQL ext. Already using Postgres, want to add vector search

The right choice usually depends on what you're already running.
If you're on Postgres, pgvector avoids adding a new infra component.
If you need managed and scalable from day one, Pinecone.
```

---

## 9. Keeping the Index Fresh

Documents change. New documents arrive. Old ones get deleted.

```text
Three update patterns:

1. Full re-index (simplest, most expensive):
   Delete everything, re-embed all documents, rebuild the index.
   When to use: small corpus, infrequent updates (weekly/monthly).

2. Incremental add (most common):
   When a new document arrives: chunk → embed → upsert into index.
   Vector DBs support upserts (insert if new, update if exists by chunk_id).
   When to use: append-only document sets.

3. Delete + re-add on update:
   When a document changes: delete all old chunks by doc_id, re-embed new version.
   Requires storing doc_id as metadata so you can find and delete all chunks.
   When to use: documents that get edited (wikis, policies, contracts).

   Implementation:
       # When doc_42 is updated:
       vector_db.delete(filter={"doc_id": "doc_42"})
       new_chunks = chunk(updated_document)
       new_vectors = embed(new_chunks)
       vector_db.upsert(new_vectors, metadata={"doc_id": "doc_42", ...})
```

---

## 10. Common Mistakes

```text
Mistake 1: Embedding the query and chunks with different models
    The vector spaces are incompatible. Similarity scores are meaningless.
    Always use the SAME model for both query and document embedding.

Mistake 2: Not normalising vectors before cosine similarity
    Most libraries handle this, but some FAISS configs use raw dot product.
    If not normalised: dot product ≠ cosine similarity.
    Fix: use cosine distance metric explicitly, or L2-normalise your vectors.

Mistake 3: Embedding chunks that include boilerplate
    Header text, footers, navigation menus, "Page 1 of 12" repeated 50 times.
    These pollute your embeddings. Strip boilerplate before embedding.

Mistake 4: Ignoring the embedding model's max token length
    Chunks longer than the model's max are silently truncated.
    The tail of a long chunk is ignored during embedding.
    Fix: ensure chunk_size < model max tokens (not just roughly — exactly).

Mistake 5: Using one embedding model to build the index, switching later
    Every query must use the same model as the indexed chunks.
    Switching models = full re-index of the entire corpus.
    Choose carefully before indexing at scale.
```

---

## Summary

```text
1. Embedding model: converts text → dense vector (768–3072 dims)
   Bi-encoders for retrieval (fast, pre-compute docs offline)
   Cross-encoders for re-ranking (slow, high precision, used on shortlist only)

2. Picking a model:
   Check MTEB retrieval leaderboard.
   Default: text-embedding-3-small (API) or bge-large-en-v1.5 (OSS)
   Private data → OSS self-hosted. Multilingual → bge-m3 or Cohere multilingual.

3. Each chunk stores: vector + original text + metadata (source, page, section, etc.)
   Metadata enables filtered search at query time.

4. Index type: HNSW for most RAG use cases (fast ANN, good recall, easy updates)

5. Keep index fresh: upsert on add, delete-by-doc-id + re-add on update

6. Biggest mistake: using different models for query and document embedding.
```
