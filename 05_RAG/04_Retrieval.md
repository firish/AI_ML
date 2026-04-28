## Retrieval — Finding the Right Chunks at Query Time

---

## 0. Where This Fits

```text
Full RAG pipeline:

    Chunking → Embedding → Indexing      ← files 02, 03 (offline)
        │
        ▼
    Retrieval                            ← this file (online, per query)
        │
        ▼
    Augmentation + Generation            ← file 05
```

Retrieval is the online half. A user asks a question. You have milliseconds
to find the most relevant chunks from a corpus of potentially millions.
Everything in this file happens at query time.

---

## 1. Dense Retrieval (Vector Search)

The core retrieval mechanism in RAG.

### How it works

```text
Step 1: Embed the query
    User query: "What is the refund policy for international orders?"
    Run through the SAME embedding model used to index your chunks.
    → query_vector = [0.019, -0.831, 0.341, ...]

Step 2: Nearest-neighbour search
    Compare query_vector against every stored chunk vector.
    Return the k chunks with highest cosine similarity.

Step 3: Return top-k chunks
    Rank 1 (sim=0.91): "International customers may return items within 14 days..."
    Rank 2 (sim=0.87): "For refunds on international shipments, customs duties..."
    Rank 3 (sim=0.82): "Our standard return policy applies globally with exceptions..."
    Rank 4 (sim=0.71): "Shipping costs are non-refundable on all orders..."
    Rank 5 (sim=0.68): "Contact support@example.com for international return labels..."
```

### Approximate Nearest Neighbour (ANN)

```text
Exact nearest-neighbour search = compare query against EVERY vector.
    1M chunks × 1536 dims = 1.536B multiplications per query.
    At 100 GFLOPS: ~15ms — feasible but slow.
    At 100M chunks: ~1.5 seconds — too slow.

ANN (HNSW, IVF) sacrifices a tiny bit of accuracy for massive speed gains.
    HNSW: traverses a graph, explores ~1% of vectors, returns in <5ms.
    Recall@10 (fraction of true top-10 found): typically 95–99%.

For RAG, ANN is always used in production. Exact search only for tiny corpora.
```

---

## 2. Similarity Metrics

Three ways to measure how close two vectors are.

### Cosine similarity

```text
cosine(a, b) = (a · b) / (|a| × |b|)

Range: -1 to 1. Higher = more similar.
    1.0  = identical direction (semantically equivalent)
    0.0  = orthogonal (unrelated)
   -1.0  = opposite direction (rare in practice for text)

Properties:
    - Ignores vector magnitude — only cares about direction
    - Best for text: embedding magnitude carries little meaning
    - Standard choice for most RAG setups

Example:
    query = [0.6, 0.8]
    doc_A = [0.6, 0.8]   → cosine = 1.00  (same direction, different magnitude)
    doc_B = [0.3, 0.4]   → cosine = 1.00  (same direction, same result!)
    doc_C = [-0.6, 0.8]  → cosine = 0.28  (different direction)
```

### Dot product

```text
dot(a, b) = a · b = Σ(aᵢ × bᵢ)

Range: unbounded. Higher = more similar.
    Unlike cosine, magnitude DOES matter.

When to use dot product:
    Your embedding model was trained with dot product as the similarity metric.
    (Some models, e.g. OpenAI text-embedding-3, are trained this way.)
    Using cosine on a dot-product-trained model won't break things, but
    using the intended metric gives marginally better results.

    If vectors are L2-normalised (|v| = 1 for all vectors):
        dot(a, b) = cosine(a, b)   ← they are identical
    Most retrieval pipelines normalise vectors, so it often doesn't matter.
```

### L2 distance (Euclidean)

```text
L2(a, b) = √(Σ(aᵢ - bᵢ)²)

Range: 0 to ∞. LOWER = more similar (it's a distance, not similarity).

Rarely used for semantic retrieval.
    Penalises magnitude differences as well as direction differences.
    Works fine when vectors are normalised (then L2 ∝ 1 - cosine).
    More common in image search, not text.

Rule of thumb:
    Text retrieval: cosine or dot product.
    Always check your embedding model's recommended metric.
```

---

## 3. Sparse Retrieval (BM25)

Dense retrieval finds semantically similar chunks. But it can miss exact keyword matches.

### The problem dense retrieval has with keywords

```text
User query: "GPT-4o API rate limit"

Dense retrieval finds chunks about:
    "how to handle API throttling in OpenAI applications"    ← semantically similar
    "managing request limits in language model APIs"          ← semantically similar

But might miss:
    "GPT-4o has a rate limit of 10,000 TPM on Tier 1 accounts."  ← exact match
    (If the embedding of "rate limit" drifts from "GPT-4o", this chunk ranks lower)

Dense retrieval is great at semantic similarity.
It's worse at exact keyword matching, especially for proper nouns, product names,
model identifiers, version numbers, error codes.
```

### BM25 — the standard sparse retrieval algorithm

```text
BM25 scores how relevant a document is to a query based on:
    1. Term frequency (TF): how often the query words appear in the chunk
    2. Inverse document frequency (IDF): how rare those words are across ALL chunks
    3. Document length normalisation: penalises very long documents

Score(query, chunk) = Σ IDF(term) × [TF(term, chunk) × (k₁+1)] / [TF(term, chunk) + k₁ × (1 - b + b × len(chunk)/avglen)]

    k₁ = 1.5 (term frequency saturation — more occurrences help, but with diminishing returns)
    b   = 0.75 (length normalisation — longer docs penalised)

In plain terms:
    BM25 asks: "Does this chunk contain the query words?
                How many times? Are these rare words (more signal) or common ones (less)?"

Example:
    Query: "GPT-4o rate limit"

    Chunk A: "GPT-4o has a rate limit of 10,000 TPM on Tier 1 accounts."
        "GPT-4o" → rare term, appears once      ← high score
        "rate"   → common term, appears once    ← moderate score
        "limit"  → common term, appears once    ← moderate score
        BM25 score: high

    Chunk B: "Managing API throttling for language model applications."
        "GPT-4o" → not present                  ← 0 contribution
        "rate"   → not present                  ← 0 contribution
        "limit"  → not present                  ← 0 contribution
        BM25 score: 0

Dense retrieval would rank Chunk B highly (semantically related).
BM25 gives Chunk B a 0 and Chunk A a high score.
```

### BM25 vs dense retrieval

```text
                BM25                    Dense (vector search)
────────────────────────────────────────────────────────────────────
Strengths       Exact keyword matching  Semantic matching
                Rare terms, proper      Synonyms, paraphrases
                nouns, codes, IDs       Cross-lingual
                Fast, no GPU needed     Handles vague queries
                Interpretable           Better for conversational Q&A

Weaknesses      Misses synonyms         Misses exact keywords
                "car" ≠ "automobile"    Diluted by common words
                No meaning, just stats  Requires GPU or API
```

---

## 4. Hybrid Search — Combining Dense and Sparse

The best production RAG systems use both. This is called hybrid search.

### Reciprocal Rank Fusion (RRF)

```text
The standard method for merging two ranked lists without needing score calibration.

BM25 results (ranked):
    Rank 1: Chunk A  (BM25 score: 14.2)
    Rank 2: Chunk C  (BM25 score: 11.8)
    Rank 3: Chunk B  (BM25 score:  9.1)
    Rank 4: Chunk D  (BM25 score:  7.3)

Dense results (ranked):
    Rank 1: Chunk B  (cosine: 0.91)
    Rank 2: Chunk D  (cosine: 0.88)
    Rank 3: Chunk A  (cosine: 0.82)
    Rank 4: Chunk E  (cosine: 0.71)

RRF score for each chunk:
    RRF(chunk, k=60) = Σ  1 / (k + rank_in_list)

    Chunk A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    Chunk B: 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226   ← same as A!
    Chunk C: 1/(60+2) + 0        = 0.01613             = 0.01613
    Chunk D: 1/(60+4) + 1/(60+2) = 0.01563 + 0.01613  = 0.03176
    Chunk E: 0        + 1/(60+4) = 0.01563             = 0.01563

Final merged ranking:
    Rank 1: Chunk A (0.03226) — consistently near top in both lists
    Rank 1: Chunk B (0.03226) — same
    Rank 3: Chunk D (0.03176)
    Rank 4: Chunk C (0.01613)
    Rank 5: Chunk E (0.01563)

Why k=60?
    The constant k dampens the effect of rank differences at the top.
    Without it, rank 1 vs rank 2 is a huge gap. With k=60, it's small.
    k=60 is the empirically validated default from the original RRF paper.
```

### Why RRF works well

```text
Problem with just averaging scores:
    BM25 score range:  0–30 (varies by corpus, hard to calibrate)
    Cosine score range: 0–1 (bounded)
    Averaging raw scores gives BM25 10× more weight just because its range is larger.

RRF avoids this entirely — it only uses RANK, not raw score.
Ranks are always 1, 2, 3, 4... regardless of the scoring method.
No calibration needed. Works out of the box.
```

---

## 5. Choosing k — How Many Chunks to Retrieve

```text
k = the number of chunks returned by the retriever and passed to the LLM.

Too few (k=1):
    High precision (only the single best chunk).
    High miss rate — if that one chunk is wrong, the answer fails.
    Use only for very simple, single-fact queries.

Too many (k=20):
    Low miss rate — the answer is almost certainly in there somewhere.
    Low precision — the LLM's context is full of irrelevant chunks.
    "Lost in the middle" problem: LLMs pay less attention to content
    in the middle of a long context. Relevant chunks buried at position 10
    contribute less to the answer than chunks at position 1 or 20.
    Slower + more expensive per query.

Practical guidance:
    k = 3–5    for focused, factual queries
    k = 5–10   for complex questions that may need multiple chunks
    k = 10–20  before re-ranking (retrieve many, then re-rank to top 5)

The most common production setup:
    Retrieve k=20 (high recall), re-rank to top 5 (high precision).
    Re-ranking covered in file 07.
```

### The "lost in the middle" effect

```text
Experiment (Liu et al., 2023): give GPT-3.5 a context with 20 chunks.
The answer is always in one of them. Which position does the model find it from?

    Position 1 (first):  correct 80% of the time
    Position 10 (middle): correct 52% of the time
    Position 20 (last):  correct 77% of the time

Performance DROPS for chunks in the middle of a long context.

Implication: when passing multiple chunks to the LLM:
    Put the most relevant chunks FIRST (or first + last).
    Don't just preserve retrieval rank order blindly.
    After re-ranking, put rank 1 first and rank 2 last — not rank 2 in the middle.
```

---

## 6. Query Preprocessing

The raw user query is often not the best input for retrieval.

### Query cleaning

```text
Common transformations:
    Strip punctuation, lowercase                   → "What's the REFUND POLICY?!" → "what's the refund policy"
    Expand contractions                            → "what's" → "what is"
    Remove stop words (for BM25 only)              → "what is the refund policy" → "refund policy"

Stop word removal for BM25:
    "The", "is", "a", "of" have near-zero IDF (appear in every doc → low signal).
    Removing them makes BM25 focus on the meaningful terms.
    Do NOT remove stop words from the dense query — the embedding model
    uses them for semantic context.
```

### Hypothetical Document Embedding (HyDE)

```text
Instead of embedding the raw query, ask an LLM to generate a
HYPOTHETICAL document that would answer the query, then embed that.

    Query: "What is the refund policy for international orders?"

    HyDE step: ask LLM to write a fake answer:
        "For international orders, our refund policy allows returns
         within 14 days. Customs duties are non-refundable. Customers
         must initiate the return via our online portal."

    Embed this hypothetical answer (not the query).

Why this helps:
    The query and the stored chunks are stylistically different.
    Query: short, interrogative.
    Chunks: long, declarative, similar to the hypothetical answer.
    Embedding a document-style text finds document-style chunks better.

Trade-off:
    One extra LLM call per query (adds ~200-500ms latency).
    Works well for knowledge-base Q&A; less clearly beneficial for conversational queries.
    Covered more in file 07 (Advanced RAG).
```

### Multi-query retrieval

```text
Run the same query multiple times with different phrasings,
merge the results.

    User query: "How long do I have to return something?"

    Generated queries:
        Q1: "How long do I have to return something?"     ← original
        Q2: "What is the return window for purchases?"
        Q3: "Refund deadline policy"
        Q4: "How many days to request a refund?"

    Retrieve top-5 for each query → 20 candidates total
    Deduplicate by chunk_id → ~10-15 unique chunks
    Re-rank or score by how many queries retrieved each chunk

Why: different phrasings activate different parts of the embedding space.
     A chunk relevant to Q3 but not Q1 would be missed by single-query retrieval.
     Multi-query trades latency (4× embedding calls) for recall.
```

---

## 7. Metadata Filtering

Retrieval doesn't have to be global. You can restrict the search space.

```text
Example: "What did the Q3 2024 earnings report say about APAC?"

Without filtering:
    Search ALL chunks across all documents.
    May retrieve chunks from Q1 2023 report (semantically similar, but wrong date).

With filtering:
    Filter: {"source": "earnings_report", "quarter": "Q3", "year": "2024"}
    Search ONLY chunks matching this metadata.
    Much more precise.

How it works in vector DBs:

    Pre-filter (filter first, then search):
        Find all chunk_ids where metadata matches.
        Run ANN search restricted to those IDs.
        Fast if the filter is selective (small subset), slow if broad.

    Post-filter (search first, then filter):
        Run ANN search to get top-k×10 candidates.
        Filter by metadata.
        Risk: if the true answer is in a filtered-out chunk, you miss it.

    In-filter (interleaved):
        Some DBs (Weaviate, Qdrant) integrate filtering into the graph traversal.
        Best recall, more complex to implement.

Rule: use filtering whenever you have a structured attribute that narrows
the search (date range, document type, author, category, language).
```

---

## 8. Retrieval Failure Modes

```text
Failure 1: Query too vague
    "Tell me about our policies."
    → retriever gets a broad embedding → retrieves diverse, unrelated chunks
    → no single chunk covers the whole answer
    Fix: force users toward specific queries, or use query decomposition (file 07).

Failure 2: Query uses different terminology than the corpus
    Query: "PTO rules"  ← "PTO" (paid time off)
    Corpus: uses "annual leave" and "holiday entitlement"
    Dense retrieval: may not bridge this gap if the embedding space doesn't align
    BM25: will miss entirely (keyword mismatch)
    Fix: hybrid search + query expansion (add synonyms to the query).

Failure 3: Answer requires combining multiple chunks
    "Compare our Q3 and Q4 revenue."
    → answer is spread across two different documents
    → single-query retrieval returns top-k chunks from either Q3 OR Q4
    → LLM may not receive both
    Fix: multi-query retrieval, query decomposition, agentic RAG (file 07).

Failure 4: Answer is in a table or image
    Dense retrieval works on text. Tables embedded as raw text lose structure.
    "What was revenue in the APAC row of Table 3?"
    → embedding of "APAC row Table 3" may not match the linearised table text
    Fix: specialised table parsing (markdown table → text row by row),
         multimodal embeddings (beyond scope of this file).

Failure 5: k is too small
    The correct chunk is rank 6, but you only retrieve k=5.
    Fix: increase k, then re-rank.

Failure 6: Stale index
    A document was updated but the index wasn't refreshed.
    Retriever returns old information.
    Fix: proper incremental indexing pipeline (covered in file 03).
```

---

## 9. Putting It Together — Retrieval in a Production System

```text
User query
    │
    ▼
Query preprocessing
    ├── Clean / normalise
    ├── (Optional) Expand to multiple phrasings
    └── (Optional) Generate hypothetical document (HyDE)
    │
    ▼
Parallel retrieval
    ├── Dense search: embed query → ANN search → top-k chunks
    └── Sparse search: BM25 → top-k chunks
    │
    ▼
Merge with RRF
    → Deduplicated, ranked list of ~10-20 candidates
    │
    ▼
Metadata filtering
    → Remove chunks that don't match query constraints
    │
    ▼
(Optional) Re-ranking
    → Cross-encoder scores each (query, chunk) pair
    → Reorder, cut to top-3 to top-5
    → Covered in file 07
    │
    ▼
Top chunks passed to augmentation + generation (file 05)
```

---

## Summary

```text
1. Dense retrieval: embed query → ANN search → semantically relevant chunks
   Similarity metric: cosine (default), dot product (if model trained with it)

2. Sparse retrieval (BM25): keyword matching → catches exact terms, proper nouns
   Different strength from dense — not a replacement, a complement

3. Hybrid search (RRF): merge BM25 + dense rankings by rank, not raw score
   Best recall in practice. Standard in production RAG.

4. k matters: too few → miss the answer, too many → LLM gets buried
   Common pattern: retrieve k=20, re-rank to top 5

5. Lost in the middle: LLMs attend less to chunks in the middle of a long context
   Put most relevant chunks first (and last).

6. Query preprocessing: cleaning, multi-query expansion, HyDE — all trade latency for recall

7. Metadata filtering: restrict the search space using structured attributes

8. Main failure modes: vague queries, terminology mismatch, multi-hop questions, stale index
```
