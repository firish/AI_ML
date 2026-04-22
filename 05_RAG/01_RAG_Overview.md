## RAG — Retrieval-Augmented Generation

---

## 0. The Problem with Pure LLMs

LLMs are powerful, but they have three hard limitations that make them unreliable for real-world applications.

### Limitation 1: Knowledge Cutoff

```text
An LLM's knowledge is frozen at training time.

    GPT-4 trained on data up to April 2023.
    Ask it about something that happened in June 2023 → it doesn't know.
    Ask it about your internal company docs → it's never seen them.
    Ask it about last week's earnings report → hallucination likely.

The model can't update itself. You can't just "tell it" new facts without
retraining (which costs millions of dollars).
```

### Limitation 2: Hallucination

```text
LLMs generate plausible-sounding text, not verified facts.

    Q: "What did our Q3 2024 earnings report say about APAC revenue?"
    A: The model has never seen your earnings report.
       It will make up a number that sounds reasonable.
       It will sound confident. It will be wrong.

This isn't a bug. The model is doing exactly what it was trained to do:
predict the most likely next token. It doesn't have a "don't make things
up" circuit — it just generates fluent text.
```

### Limitation 3: Context Window Limits

```text
Even if you COULD put all your knowledge into the prompt, you can't.

    Claude 3.5 Sonnet: 200K tokens ≈ 150,000 words ≈ ~500 pages
    Your company's internal documentation: likely millions of pages

You have to choose WHAT to give the model. The question is: how?
```

---

## 1. The RAG Idea

RAG solves all three problems with a simple insight:

```text
Instead of storing all knowledge in model weights,
RETRIEVE the relevant knowledge at query time and give it to the model.

User asks: "What were our Q3 2024 APAC revenue numbers?"

Without RAG:
    LLM makes up an answer. ← hallucination

With RAG:
    1. Search your document database for content about "Q3 2024 APAC revenue"
    2. Find the actual Q3 report chunk that contains this data
    3. Stuff it into the prompt: "Here is the relevant document: [actual text]
       Based on this, answer: What were our Q3 2024 APAC revenue numbers?"
    4. LLM reads the real document and answers accurately

The LLM is now a reader and reasoner, not a memoriser.
```

The name breaks down as:
```text
Retrieval  →  find relevant documents from a database
Augmented  →  add them to the prompt
Generation →  let the LLM generate an answer using that context
```

---

## 2. The Full RAG Pipeline

RAG has two distinct phases: an **offline** phase (done once, ahead of time) and an **online** phase (done at query time).

### Offline Phase — Building the Index

```text
Your raw documents (PDFs, Word docs, HTML pages, database records...)
        │
        ▼
┌─────────────────┐
│    Chunking     │  Split documents into smaller pieces
│                 │  A 50-page PDF → ~500 chunks of ~200 words each
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embedding     │  Turn each chunk into a vector
│                 │  chunk text → [0.21, -0.87, 0.43, ...] (768 or 1536 dims)
│                 │  Using a model like text-embedding-3-small or BGE
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Indexing     │  Store vectors in a vector database (Pinecone, Weaviate,
│                 │  Chroma, FAISS) with HNSW or IVF index for fast search
└─────────────────┘

This runs ONCE (or incrementally as new documents arrive).
Result: a searchable index of your entire knowledge base.
```

### Online Phase — Answering a Query

```text
User query: "What is our refund policy for international orders?"
        │
        ▼
┌─────────────────┐
│    Retrieve     │  Embed the query → search the vector index
│                 │  Find the top-k most similar chunks (k = 3-10 typically)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Augment       │  Build a prompt:
│                 │  "Context:\n[chunk 1]\n[chunk 2]\n[chunk 3]\n
│                 │   Based on the above, answer: {user query}"
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Generate     │  Send the augmented prompt to the LLM
│                 │  LLM reads the retrieved context and generates an answer
└─────────────────┘
        │
        ▼
"Our refund policy for international orders allows returns within 30 days..."
```

---

## 3. Why Chunking Is the First Critical Decision

```text
Everything downstream depends on your chunks.

If chunks are too large:
    → Each chunk contains many topics
    → When you retrieve chunk for "refund policy", you also drag in
      unrelated content about "shipping times" and "warranty terms"
    → The LLM's context is polluted with irrelevant text
    → Answers are less focused

If chunks are too small:
    → Each chunk has no context
    → Chunk: "within 30 days."
    → What does "within 30 days" mean without the surrounding sentence?
    → The LLM can't answer from this fragment

If chunk boundaries are wrong:
    → A key sentence gets split across two chunks
    → Neither chunk retrieved in isolation contains the complete answer

Getting chunking right is the most impactful single decision in RAG.
It determines the ceiling on retrieval quality. Covered in file 02.
```

---

## 4. Where Embedding and Vector Search Fit In

```text
The embedding model converts text to a vector such that
semantically similar texts have similar vectors (high cosine similarity).

    "What is your return policy?"   → [0.21, 0.87, ...]
    "How do I get a refund?"        → [0.22, 0.85, ...]   ← very close!
    "Tell me about GPU memory"      → [0.91, -0.34, ...]  ← far away

The vector database (covered in Phase 2) then finds the nearest
neighbours in this vector space — i.e., chunks whose meaning is
closest to the query's meaning.

Key insight: this is SEMANTIC search, not keyword search.
    Keyword search: "refund policy" matches chunks containing those exact words
    Semantic search: finds chunks about returns, money-back guarantees,
                     cancellations — even without the word "refund"

This is why embeddings are the bridge between Phase 1 (making vectors)
and RAG. The embedding model trained in Phase 1 is the same technology
doing the heavy lifting here.
```

---

## 5. Naive RAG vs Production RAG

The simple pipeline above is called "naive RAG." It works as a proof of concept but breaks down in production.

```text
Naive RAG:
    embed query → nearest neighbour search → stuff top-k chunks → generate

Problems in practice:
    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ Problem                     │ What happens                             │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ Bad chunks                  │ Retrieved content is incomplete/noisy    │
    │ Wrong chunks retrieved      │ Answer misses key information            │
    │ Too many chunks / redundant │ Context is bloated, key info diluted     │
    │ Vague query                 │ "Tell me about our product" → everything │
    │ Multi-hop question          │ Answer requires combining facts from 3   │
    │                             │ different documents                      │
    │ No source attribution       │ Can't verify which doc the answer came from│
    └─────────────────────────────┴──────────────────────────────────────────┘

Advanced RAG (covered in file 07) fixes these with:
    - Query rewriting / HyDE
    - Re-ranking retrieved chunks
    - Hierarchical / parent-child chunk retrieval
    - Multi-hop retrieval
    - Agentic RAG
```

---

## 6. RAG vs Fine-Tuning vs Prompting

```text
When should you use RAG?

Approach        | Best for                          | Not great for
────────────────────────────────────────────────────────────────────
Prompting       | Few static facts, short context   | Large / changing knowledge base
Fine-tuning     | Style, format, domain adaptation  | Factual grounding, fresh data
RAG             | Large, dynamic, private knowledge  | Real-time data, low latency apps

Rule of thumb:
    If the answer lives in a document → RAG
    If you want the model to behave differently → fine-tuning
    If the context fits in a prompt → just include it

These aren't mutually exclusive. Production systems often do all three:
    - Fine-tune for domain style
    - RAG for factual grounding
    - System prompt for behaviour/format

Covered in full in file 08.
```

---

## Summary

```text
1. Pure LLMs: frozen knowledge, hallucinate on unknowns, can't access private data
2. RAG: retrieve relevant chunks at query time → augment the prompt → generate
3. Two phases: offline (chunk → embed → index) / online (retrieve → augment → generate)
4. Chunking quality is the #1 lever on RAG performance
5. Vector search is semantic (meaning-based), not keyword-based
6. Naive RAG is a starting point; production RAG adds query rewriting, re-ranking, etc.
```
