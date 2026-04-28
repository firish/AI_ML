## Advanced RAG — HyDE, Re-ranking, Query Rewriting, and Agentic RAG

---

## 0. Why Naive RAG Falls Short

```text
Naive RAG pipeline:
    embed query → ANN search → top-k chunks → stuff into prompt → generate

This works for simple, well-formed queries against well-chunked documents.
It breaks down in four common scenarios:

    1. The query is vague or uses different vocabulary than the corpus
       → Embedding mismatch. Wrong chunks retrieved.

    2. The answer requires combining information from multiple documents
       → Single-pass retrieval can't gather all the pieces.

    3. The retrieved chunks include noise (irrelevant content near relevant content)
       → Precision is low. LLM's context is diluted.

    4. The query is complex: multi-part, ambiguous, or requires reasoning
       → Single retrieval pass can't handle it.

Advanced RAG techniques each target one of these failure modes.
They layer on top of the naive pipeline — you pick the ones that fix
your specific bottleneck (as identified by RAGAS evaluation, file 06).
```

---

## 1. Query Rewriting

Transform the raw user query before retrieval to improve embedding alignment.

### Step-back prompting

```text
Problem: user queries are often conversational and context-dependent.
    "What about for international orders?"
    ← this query depends on the previous conversational turn.
       Embedded in isolation, it matches nothing useful.

Step-back: rewrite the query into a standalone, search-optimised form.

    Conversation context:
        User: "What is your return policy?"
        Assistant: "You have 30 days to return items..."
        User: "What about for international orders?"

    Rewritten query: "What is the return policy for international orders?"

    Now the query is self-contained and embeds correctly.

Implementation: ask an LLM before retrieval:
    "Given the conversation history, rewrite the user's latest message
     into a standalone search query that captures the full intent."
```

### Decomposition for complex queries

```text
Problem: multi-part questions can't be answered by a single retrieval pass.
    "Compare our Q3 and Q4 APAC revenue and explain the difference."

Single retrieval: returns chunks about Q3 OR Q4, rarely both. Misses the comparison.

Decomposition:
    Split into sub-queries, retrieve for each independently:
        Sub-query 1: "APAC revenue Q3 2024"
        Sub-query 2: "APAC revenue Q4 2024"
        Sub-query 3: "APAC revenue trends or explanations"

    Run retrieval for each → merge results (deduplicate by chunk_id) → generate.

    The LLM now has context for both Q3 and Q4 and can compare them.

Implementation:
    LLM call before retrieval:
        "Break the following question into 2-4 simpler sub-questions
         that together fully address the original question."
    Run retrieval pipeline for each sub-question.
    Merge and deduplicate retrieved chunks.
    Pass all unique chunks to generation.
```

---

## 2. HyDE — Hypothetical Document Embedding

Introduced briefly in file 04. Full treatment here.

### The core problem

```text
At retrieval time, you're comparing two stylistically different texts:

    Query:   "What is the return policy for international orders?"
              ← short, interrogative, conversational

    Chunks:  "For international customers, returns must be initiated
              within 14 days of delivery. Customs duties are the
              customer's responsibility."
              ← declarative, document-style, longer

The embedding model maps similar MEANINGS close together, but the
surface-level style gap creates a representational mismatch.
The query vector and the chunk vector are not as close as they could be.
```

### How HyDE fixes it

```text
Instead of embedding the raw query, generate a HYPOTHETICAL DOCUMENT
that would answer it, then embed that document.

Step 1: LLM call (fast, no retrieval yet):
    Prompt: "Write a short paragraph answering: What is the return policy
             for international orders? Note: this may be inaccurate — just
             write what a policy document might say."
    Output: "For international orders, customers have 14 days from the date
             of delivery to initiate a return. Customs duties paid at import
             are non-refundable. Items must be in original packaging..."

Step 2: Embed the hypothetical answer (not the original query).
    hypothetical_vec = embed("For international orders, customers have 14 days...")

Step 3: ANN search with hypothetical_vec.
    The hypothetical answer is document-style → matches real document chunks better.

Step 4: Pass retrieved chunks to generation as normal.
    The generation step uses the REAL chunks, not the hypothetical answer.
    The hypothetical was only used to improve the search vector.
```

### When HyDE helps and when it doesn't

```text
HyDE helps:
    ✓ Knowledge-base Q&A where query style ≠ document style
    ✓ Long-form documents with dense, declarative text
    ✓ Technical documentation retrieval

HyDE hurts:
    ✗ The LLM generates a confidently wrong hypothetical.
       The wrong vector retrieves wrong chunks. Garbage in, garbage out.
       Especially risky for very specific factual queries.

    ✗ Adds one LLM call per query (~100-300ms extra latency).

Practical advice: test HyDE vs baseline on your eval set (file 06).
It typically improves context recall by 5-15% for knowledge-base RAG.
It sometimes hurts precision. Let the metrics decide.
```

---

## 3. Re-ranking

Retrieve many chunks cheaply, then use a stronger model to reorder them.

### The two-stage architecture

```text
Stage 1: Recall-optimised retrieval (fast, approximate)
    Dense search + sparse search → top-50 candidates
    Goal: high recall. Get all relevant chunks in the candidate set.
    Speed: fast (ANN search + BM25)
    Cost: cheap (no LLM call)

Stage 2: Precision-optimised re-ranking (slow, accurate)
    Cross-encoder scores each (query, chunk) pair jointly.
    Reorders the 50 candidates. Keep top-5.
    Goal: high precision. Put the best chunks first.
    Speed: slower (50 model calls or one batched forward pass)
    Cost: moderate (cross-encoder model, can be small and fast)
```

### Cross-encoder re-ranking

```text
Cross-encoder input: "[query] What is the international return policy? 
                      [SEP] For international customers, returns must be..."
Cross-encoder output: relevance score = 0.94

Unlike the bi-encoder (file 03), the cross-encoder sees BOTH texts simultaneously.
Attention flows between query and document.
It can resolve:
    - Ambiguous terms ("bank" → financial or river, depending on the document)
    - Negation ("what is NOT covered by the return policy")
    - Subtle relevance that requires comparing query intent to document content

Popular cross-encoder models:
    cross-encoder/ms-marco-MiniLM-L-6-v2   ← fast, good, widely used
    BAAI/bge-reranker-large                 ← stronger, slower
    Cohere Rerank API                       ← hosted, strong performance

Typical latency impact:
    Stage 1 (retrieve 50 chunks): ~20ms
    Stage 2 (re-rank 50 chunks with MiniLM): ~50-100ms
    Total: ~70-120ms — acceptable for most RAG applications
```

### Maximal Marginal Relevance (MMR)

```text
Re-ranking for diversity, not just relevance.

Problem: top-5 by relevance may all say the same thing.
    Chunk 1 (sim=0.91): "Returns within 30 days."
    Chunk 2 (sim=0.89): "Items can be returned within 30 days."
    Chunk 3 (sim=0.87): "Our 30-day return window applies to all orders."
    ← all three are nearly identical. Sending all three wastes context tokens.

MMR selects chunks that are relevant to the query AND diverse from each other.

MMR score = λ × sim(chunk, query) - (1-λ) × max sim(chunk, already_selected)

    λ = 1.0: pure relevance (standard retrieval)
    λ = 0.5: balance relevance and diversity (typical setting)
    λ = 0.0: pure diversity (useless for RAG)

Selection process:
    Round 1: pick chunk with highest sim(chunk, query) → Chunk 1
    Round 2: for remaining chunks, compute MMR score.
             Chunk 2: 0.5×0.89 - 0.5×sim(Chunk2, Chunk1) = 0.445 - 0.5×0.97 = -0.04
             Chunk 4: 0.5×0.82 - 0.5×sim(Chunk4, Chunk1) = 0.410 - 0.5×0.30 = 0.26
             → Pick Chunk 4 (more diverse, still relevant)
    Continue until k chunks selected.

Use MMR when: the corpus has lots of near-duplicate chunks, or the query
is broad and you want to cover different facets.
```

---

## 4. Hierarchical / Parent-Child Retrieval

Retrieve with small chunks, generate with large chunks. Covered in chunking (file 02) from the indexing side — here's the retrieval side.

```text
The asymmetry:
    Small chunks → better retrieval precision (focused, specific)
    Large chunks → better generation quality  (more context for the LLM)

At query time:
    1. Embed query → ANN search against the SMALL chunk index
    2. Identify the top-k small chunks (high precision)
    3. For each small chunk, look up its parent chunk by ID
    4. Pass the PARENT chunks to the LLM (not the small chunks)

Example:
    Small chunk (indexed): "Customs duties are the customer's responsibility." (15 tokens)
    Parent chunk (stored): Full "International Returns" section (300 tokens)

    Retrieval finds the small chunk precisely.
    LLM receives the full parent section — enough context to answer completely.

    Without parent lookup: LLM sees "Customs duties are the customer's responsibility."
    → incomplete answer (doesn't know the return window, required packaging, etc.)

    With parent lookup: LLM sees the full section.
    → complete answer drawn from rich context.

This directly improves context recall (file 06):
    The small chunk matches the query well.
    The parent contains all the related information.
```

---

## 5. FLARE — Forward-Looking Active Retrieval

Retrieve on-demand during generation, not just before.

### The limitation of retrieve-then-generate

```text
Standard RAG: retrieve ONCE before generation, then generate the full answer.

Problem: the LLM may start generating correctly, then "run out" of context
         and start hallucinating mid-answer.

    Context retrieved: international return policy (14 days, customs duties).
    Generation:
        "International returns must be initiated within 14 days [correct].
         Customs duties are non-refundable [correct].
         You can track your return using the tracking number sent to [correct].
         The refund will appear in your account within 3-5 business days [??]"
         ← is "3-5 business days" in the retrieved chunks? Maybe not.
```

### FLARE approach

```text
FLARE generates speculatively and retrieves when confidence drops.

Step 1: Generate the next sentence tentatively (low temperature).

Step 2: Check the probability of each generated token.
        If any token has probability < threshold:
            → The model is uncertain here. Retrieve before continuing.

        Uncertain tokens look like: "...refund will appear within [3] [-] [5]..."
        The model gave these tokens low probability → it's guessing.

Step 3: Trigger retrieval using the sentence generated so far as the query.
        "refund processing time after return received"
        → retrieve chunks about refund processing times

Step 4: Continue generation with the newly retrieved chunks added to context.

Step 5: Repeat: generate → check confidence → retrieve if uncertain.

Result: generation is grounded at each step, not just at the start.
```

### When to use FLARE

```text
✓ Long-form answers that require information from many different parts of the corpus
✓ When hallucinations tend to appear mid-answer (not at the start)
✓ Complex summarisation or synthesis tasks

✗ Simple factual Q&A — single retrieval pass is sufficient
✗ Low-latency requirements — FLARE adds multiple retrieval + generation rounds
✗ Complex to implement compared to the alternatives
```

---

## 6. Corrective RAG (CRAG)

Adds a self-correction step after retrieval before generation.

```text
Problem: standard RAG trusts whatever the retriever returns.
         If retrieval quality is low (wrong chunks), generation quality suffers.

CRAG inserts a "retrieval evaluator" between retrieval and generation:

    Retrieval → [Evaluator] → decide action → Generation

Evaluator options:
    Correct:    Retrieved chunks are clearly relevant.
                → Pass directly to generation.

    Ambiguous:  Retrieved chunks are partially relevant or uncertain.
                → Perform web search or additional retrieval to supplement.
                → Combine original chunks + new results → generation.

    Incorrect:  Retrieved chunks are irrelevant.
                → Discard them entirely.
                → Rewrite the query, search the web or a fallback source.
                → Use new results → generation.

The evaluator is a lightweight classifier or a small LLM prompt:
    "Given this query and these retrieved documents, are the documents
     relevant to answering the query? Answer: Correct / Ambiguous / Incorrect"

CRAG diagram:
    Query
      │
      ▼
    Retriever → chunks
      │
      ▼
    Evaluator
      ├── Correct   → Generation
      ├── Ambiguous → Web search → merge → Generation
      └── Incorrect → Query rewrite → Web search → Generation
```

---

## 7. Self-RAG

The LLM decides when to retrieve and reflects on its own outputs.

```text
Standard RAG: always retrieve, regardless of whether retrieval is needed.
    "What is 2 + 2?" → retrieval triggered → finds irrelevant chunks → generation confused.

Self-RAG: the LLM first asks itself "do I need to retrieve?"

Four special tokens trained into the model:
    [Retrieve]:    "I need external information to answer this."
    [No Retrieve]: "I can answer from my own knowledge."
    [Relevant]:    "The retrieved document is useful."
    [Irrelevant]:  "The retrieved document is not useful."

Generation flow:
    Input: "What is 2 + 2?"
    Model generates: [No Retrieve] "The answer is 4."
    → No retrieval triggered.

    Input: "What is our Q3 APAC revenue?"
    Model generates: [Retrieve]
    → Retrieval triggered. Returns chunks.
    Model generates: [Relevant] "According to the Q3 report, APAC revenue was $42M."

    If model generates [Irrelevant]:
    → Retrieval repeated with a different query.

Additionally, Self-RAG trains "critique tokens":
    [Supported]:     "My answer is grounded in the retrieved document."
    [Partially Supported]: "Some claims are grounded, some are not."
    [No Support]:    "My answer is not supported by the retrieved document."

The model critiques its own generation and can revise before returning the answer.

Trade-off:
    Requires a fine-tuned model (can't use standard GPT-4o or Claude as-is).
    Much more complex than standard RAG.
    Significant improvement in faithfulness on benchmarks.
    Academic technique — rarely deployed in production as-is.
```

---

## 8. Agentic RAG

The most powerful and most complex form. RAG as part of a larger reasoning loop.

### The core idea

```text
Instead of a fixed pipeline (retrieve → augment → generate),
the LLM is an agent that decides HOW to retrieve and WHEN to stop.

The agent has access to tools:
    search(query)      → retrieves chunks from the vector DB
    web_search(query)  → retrieves from the web
    calculator(expr)   → runs a calculation
    sql_query(query)   → queries a database

The LLM reasons about which tools to call, in what order,
with what arguments, and when the answer is complete.
```

### Example: multi-hop reasoning

```text
Query: "What was the percentage change in APAC revenue from Q3 to Q4 2024,
        and how does that compare to the EMEA region?"

A fixed RAG pipeline can't handle this — it needs multiple retrievals and arithmetic.

Agentic RAG (ReAct pattern):

    Thought: I need Q3 APAC revenue first.
    Action:  search("APAC revenue Q3 2024")
    Observation: "Q3 2024 APAC revenue: $42M" (from chunk)

    Thought: Now I need Q4 APAC revenue.
    Action:  search("APAC revenue Q4 2024")
    Observation: "Q4 2024 APAC revenue: $48M" (from chunk)

    Thought: APAC change = (48-42)/42 = 14.3%. Now I need EMEA.
    Action:  search("EMEA revenue Q3 Q4 2024")
    Observation: "Q3 EMEA: $31M, Q4 EMEA: $29M" (from chunk)

    Thought: EMEA change = (29-31)/31 = -6.5%. I have everything.
    Action:  calculator("(48-42)/42 * 100")
    Observation: 14.29

    Final answer: "APAC grew 14.3% from Q3 to Q4 ($42M → $48M), while
                  EMEA declined 6.5% ($31M → $29M). APAC outperformed
                  EMEA by 20.8 percentage points."
```

### When agentic RAG is necessary

```text
Use agentic RAG when:
    ✓ Questions require multi-hop retrieval (answer in multiple documents)
    ✓ Questions require calculations or comparisons across retrieved data
    ✓ The number of retrieval steps is not known upfront
    ✓ The corpus includes multiple data sources (docs, database, web)

Keep naive or advanced RAG when:
    ✓ Questions are single-hop (answer in one chunk)
    ✓ Latency is critical (agentic adds multiple round trips)
    ✓ Predictability matters (agents can go off course)
    ✓ Cost matters (multiple LLM calls per query)

Latency comparison (approximate):
    Naive RAG:    1 retrieval + 1 LLM call   → ~500ms
    Advanced RAG: 1 retrieval + rerank + 1 call → ~800ms
    Agentic RAG:  3-5 retrieval + 3-5 LLM calls → ~3-10 seconds
```

---

## 9. Choosing the Right Technique

```text
Problem                                    Technique
──────────────────────────────────────────────────────────────────────────
Vague / conversational queries             Query rewriting (step-back)
Multi-part questions                       Query decomposition
Query style ≠ document style              HyDE
Low retrieval precision (noisy chunks)     Re-ranking (cross-encoder)
Redundant retrieved chunks                 MMR re-ranking
Incomplete answers (missing info)          Larger k + parent-child retrieval
Generation hallucinations mid-answer       FLARE
Retriever returns bad chunks               CRAG (corrective RAG)
Complex multi-hop reasoning                Agentic RAG (ReAct)
All of the above, budget permitting        Multi-query + hybrid + rerank + parent-child
```

---

## 10. Advanced RAG Pipeline (Full)

```text
Query
  │
  ▼
Query rewriting / decomposition
  │  (rewrite vague queries; split multi-part into sub-queries)
  │
  ▼
For each (sub-)query:
  │
  ├── Dense retrieval (bi-encoder ANN, top-50)
  └── Sparse retrieval (BM25, top-50)
        │
        ▼
      RRF merge + deduplication → ~60 candidates
        │
        ▼
      [Optional] CRAG evaluator → discard irrelevant, trigger web search if needed
        │
        ▼
      Cross-encoder re-ranking → top-5 to top-10
        │
        ▼
      [Optional] Parent-child lookup → expand small chunks to parent
        │
        ▼
      MMR selection → final k chunks (diverse + relevant)
  │
  ▼
Merge sub-query results (if decomposed)
  │
  ▼
Augmentation (file 05): format chunks, build prompt
  │
  ▼
Generation: LLM call, temperature 0.1, citations
  │
  ▼
[Optional] FLARE: check token confidence, re-retrieve if uncertain
  │
  ▼
Response + citations
```

Not every system needs every component. Add techniques only when evaluation
(file 06) shows they improve the metric you care about.

---

## Summary

```text
1. Query rewriting: fix vague/conversational queries before retrieval
   Decomposition: split multi-part queries, retrieve for each sub-query

2. HyDE: generate a hypothetical answer, embed it instead of the query
   Bridges the style gap between short queries and long document chunks

3. Re-ranking: retrieve k=50 cheaply, re-rank with cross-encoder to top-5
   Cross-encoders see query + doc jointly → much higher precision than bi-encoders
   MMR: selects for diversity — avoids redundant near-duplicate chunks

4. Parent-child retrieval: retrieve small chunks (precision), pass parent chunks to LLM (context)

5. FLARE: retrieve during generation when confidence drops → reduces mid-answer hallucination

6. CRAG: evaluate retrieved chunks BEFORE generation → discard bad results, trigger fallback

7. Self-RAG: model decides when to retrieve and critiques its own outputs (requires fine-tuning)

8. Agentic RAG: LLM as an agent with tools — handles multi-hop, cross-source, arithmetic queries
   Most powerful, most expensive, most complex — use only when simpler techniques fall short
```
