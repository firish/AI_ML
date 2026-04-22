## Chunking — How to Split Documents for Retrieval

---

## 0. Why Chunking Is Hard

```text
You have a 200-page PDF. You need to split it into pieces small enough
to fit in context, but large enough to be meaningful.

Two failure modes:

    Chunk too small:
        Chunk: "within 30 days of purchase."
        Query: "What is your return policy?"
        Problem: the retriever finds this chunk. The LLM sees "within 30 days
                 of purchase" and has no idea what it applies to. No context.

    Chunk too large:
        Chunk: [entire 200-page PDF]
        Query: "What is your return policy?"
        Problem: you find the document, but the LLM's context is 200 pages.
                 The relevant 2 sentences are buried. Slow and expensive.

The goal: chunks that are semantically self-contained but small enough
          that your retriever can be precise about WHICH chunks are relevant.
```

The right chunk size and strategy depend entirely on your documents and use case. There is no universal answer. This file covers every major approach.

---

## 1. Fixed-Size Chunking (Character-Level)

The simplest strategy: split every N characters, regardless of content.

### How it works

```text
Document: "The return policy allows refunds within 30 days of purchase.
           Shipping costs are non-refundable. International orders follow
           the same policy. Contact support@example.com for assistance."

chunk_size = 60, overlap = 0:

    Chunk 1: "The return policy allows refunds within 30 days of pur"
    Chunk 2: "chase.\n           Shipping costs are non-refundable. I"
    Chunk 3: "nternational orders follow\n           the same policy. "
    Chunk 4: "Contact support@example.com for assistance."

→ "purchase" is split across chunks 1 and 2. Neither is coherent.
```

### With overlap

```text
chunk_size = 100, overlap = 20:

    Chunk 1: "The return policy allows refunds within 30 days of purchase.
              Shipping costs are non-re"
    Chunk 2: "hipping costs are non-refundable. International orders follow
              the same policy. Contact"
    Chunk 3: "policy. Contact support@example.com for assistance."

Overlap repeats the last 20 chars of each chunk at the start of the next.
Purpose: ensure a sentence broken at the boundary appears in full in at least one chunk.
```

### When to use

```text
✓ Quick prototypes
✓ Documents with no meaningful structure (raw OCR output)
✓ When you want total control over chunk sizes for latency/cost budgeting

✗ Often produces broken sentences and incoherent chunks
✗ Overlap is a hack, not a fix — you can still split mid-sentence
```

---

## 2. Sentence-Based Chunking

Split on sentence boundaries instead of arbitrary character counts.

### How it works

```text
Use a sentence tokenizer (NLTK, spaCy, or simple regex on [.!?]) to
find sentence boundaries, then group sentences into chunks.

Document:
    "Refunds are accepted within 30 days. Shipping is non-refundable.
     International orders are subject to customs delays. Items must be
     in original packaging. Contact us for large orders."

Split into sentences:
    S1: "Refunds are accepted within 30 days."
    S2: "Shipping is non-refundable."
    S3: "International orders are subject to customs delays."
    S4: "Items must be in original packaging."
    S5: "Contact us for large orders."

Group into chunks of ~2 sentences:
    Chunk 1: S1 + S2  →  "Refunds are accepted within 30 days. Shipping is non-refundable."
    Chunk 2: S3 + S4  →  "International orders are subject to customs delays. Items must be in original packaging."
    Chunk 3: S5       →  "Contact us for large orders."
```

### Sliding window variant

```text
Instead of grouping sequentially, slide a window across sentences:

window_size = 3 sentences, step = 1:
    Chunk 1: S1 + S2 + S3
    Chunk 2: S2 + S3 + S4   ← overlaps with chunk 1
    Chunk 3: S3 + S4 + S5

Every sentence appears in multiple chunks.
Pros: key sentences are always surrounded by context.
Cons: more chunks, higher storage and retrieval cost, redundant results.
```

### When to use

```text
✓ Prose documents with clear sentences (articles, books, transcripts)
✓ Q&A use cases where each sentence might independently answer a question

✗ Documents with bullet points, tables, or code — no clear sentence structure
✗ Sentence tokenizers can break on abbreviations ("Dr. Smith" → split at "Dr.")
```

---

## 3. Paragraph / Delimiter-Based Chunking

Split on natural paragraph breaks (`\n\n`) or document-specific delimiters.

### How it works

```text
Split on double newlines:

    "Our return policy is simple. We accept returns within 30 days.\n\n
     Shipping costs are paid by the customer and are non-refundable.\n\n
     For international orders, customs duties apply."

    Chunk 1: "Our return policy is simple. We accept returns within 30 days."
    Chunk 2: "Shipping costs are paid by the customer and are non-refundable."
    Chunk 3: "For international orders, customs duties apply."

Each chunk is a semantically coherent unit — it was a paragraph in the original.
```

### When to use

```text
✓ Well-structured documents where paragraphs are natural units of meaning
✓ Web content, emails, forum posts — all naturally paragraph-structured

✗ Variable paragraph sizes: some paragraphs are 2 lines, others are 30 lines
✗ No control over max chunk size — a very long paragraph becomes a very large chunk
✗ PDFs often lose paragraph structure during extraction (becomes one big blob)
```

---

## 4. Recursive Character Text Splitting

The most widely used general-purpose strategy. LangChain's default.

### The idea

```text
Try to split on "natural" boundaries, in order of preference.
If a chunk is still too large after splitting on the first boundary,
try the next one. Keep going until chunks are small enough.

Priority order:
    1. "\n\n"    (paragraph break)
    2. "\n"      (line break)
    3. ". "      (sentence end)
    4. " "       (word break)
    5. ""        (character — last resort)
```

### Worked example

```text
chunk_size = 100 characters

Input text:
    "Refunds are accepted within 30 days of purchase.\n\n
     Shipping costs are non-refundable. All items must be in original packaging.\n\n
     Contact support for large orders."

Step 1: Try splitting on "\n\n"
    Part 1: "Refunds are accepted within 30 days of purchase."          (50 chars) ✓
    Part 2: "Shipping costs are non-refundable. All items must be in original packaging."  (76 chars) ✓
    Part 3: "Contact support for large orders."                          (33 chars) ✓

All parts fit in 100 chars → done. No need to recurse.

Now imagine Part 2 was 200 chars (too big):
Step 2: Try splitting Part 2 on "\n"
    Sub-part 2a: ...  ← if this fits, keep it
    Sub-part 2b: ...  ← if still too big, try ". "
Step 3: Split on ". " ...
Step 4: Split on " " (word boundary) ...
Step 5: Split on "" (character, absolute last resort)
```

### Why this beats simple fixed-size chunking

```text
Fixed-size (100 chars, no overlap):
    "Refunds are accepted within 30 days of pur" ← cuts word "purchase"

Recursive (100 chars):
    "Refunds are accepted within 30 days of purchase." ← clean sentence boundary

The recursion finds the largest natural boundary that keeps chunk ≤ max_size.
It degrades gracefully — only splits mid-word if there's absolutely no alternative.
```

### When to use

```text
✓ Default choice for mixed/unknown document types
✓ Good balance of simplicity and quality
✓ Works on prose, code, HTML — adapts to what's present

✗ Still doesn't understand document semantics (headings, sections)
✗ May merge unrelated paragraphs if they're both short
```

---

## 5. Document-Structure-Aware Chunking (Markdown / HTML)

Instead of splitting on text patterns, respect the document's actual structure.

### Markdown chunking

```text
Document:
    # Return Policy

    ## Standard Returns
    We accept returns within 30 days. Items must be unused.

    ## International Returns
    Customs duties are the customer's responsibility.
    Returns must be initiated within 14 days of delivery.

    ## Exceptions
    Sale items and gift cards are non-refundable.

Structure-aware chunking:

    Chunk 1:
        header: "Return Policy > Standard Returns"
        content: "We accept returns within 30 days. Items must be unused."

    Chunk 2:
        header: "Return Policy > International Returns"
        content: "Customs duties are the customer's responsibility.
                  Returns must be initiated within 14 days of delivery."

    Chunk 3:
        header: "Return Policy > Exceptions"
        content: "Sale items and gift cards are non-refundable."

Each chunk carries its full heading path as metadata.
When retrieved, the LLM sees: "Return Policy > International Returns: ..."
→ it knows exactly what this chunk is about, even without surrounding context.
```

### HTML chunking

```text
Similar idea — split on <section>, <article>, <div class="content"> tags
instead of markdown headers. Useful for web scraping pipelines.

Metadata to extract per chunk:
    - Page title
    - Section heading
    - URL
    - Publication date
```

### When to use

```text
✓ Documentation sites, wikis, technical manuals — always use this
✓ Markdown/HTML-native content (README files, Notion exports, blog posts)
✓ When sections have clear, titled meanings

✗ Unstructured content (PDFs, scanned docs, raw text)
✗ Short sections may still need to be merged or split further
```

---

## 6. Semantic Chunking

Split where the *meaning* changes, not where a delimiter appears.

### The idea

```text
Embed every sentence. When consecutive sentences have a large drop
in embedding similarity, that's a semantic boundary — split there.

Sentence embeddings (simplified):

    S1: "Refunds are accepted within 30 days."    → v1
    S2: "Items must be in original packaging."    → v2    sim(v1,v2) = 0.85  ← same topic
    S3: "Shipping costs are non-refundable."      → v3    sim(v2,v3) = 0.78  ← same topic
    S4: "International shipping takes 7-14 days." → v4    sim(v3,v4) = 0.72  ← related
    S5: "Our engineering team uses Python."       → v5    sim(v4,v5) = 0.11  ← BIG DROP → split!
    S6: "We run CI on GitHub Actions."            → v6    sim(v5,v6) = 0.88  ← same topic

Split at S4/S5 boundary:
    Chunk 1: S1 + S2 + S3 + S4  (all about shipping/returns)
    Chunk 2: S5 + S6             (all about engineering)
```

### The percentile threshold approach

```text
For a document with N sentences:
    1. Compute embedding for each sentence
    2. Compute cosine similarity between each consecutive pair
    3. Find similarities that fall below the Xth percentile
       (X = 95 means "split at the 5% most dramatic topic shifts")
    4. Those low-similarity points are chunk boundaries

This adapts to the document — a document about one topic will have
high similarity throughout and produce fewer, larger chunks.
A multi-topic document will produce many, smaller chunks.
```

### Cost and speed

```text
Embedding every sentence is expensive:
    A 100-page doc with ~3000 sentences → 3000 embedding API calls
    Or one batch call, but still takes time and costs money.

In practice:
    Use semantic chunking offline during indexing (one-time cost).
    Not viable for real-time document ingestion at scale.
```

### When to use

```text
✓ Documents with abrupt topic changes (meeting transcripts, mixed-topic reports)
✓ When high retrieval precision matters more than speed or cost
✓ Offline indexing pipelines (cost doesn't matter as much)

✗ Homogeneous documents (a single-topic article) — produces arbitrary boundaries
✗ Real-time ingestion — too slow
✗ Adds complexity; often only marginally better than recursive splitting
```

---

## 7. Sliding Window Chunking

Every chunk overlaps significantly with its neighbours. A different approach from sentence-level sliding windows.

### How it works

```text
window_size = 200 tokens, stride = 100 tokens (50% overlap):

    Chunk 1: tokens 0–199
    Chunk 2: tokens 100–299    ← overlaps chunk 1 by 100 tokens
    Chunk 3: tokens 200–399    ← overlaps chunk 2 by 100 tokens
    Chunk 4: tokens 300–499
    ...

Every token appears in at least 2 chunks (usually 2, exactly 2 with 50% overlap).
```

### Why overlap helps

```text
Problem: A key sentence sits at the boundary between two chunks.
    Chunk 1 ends: "...the refund will be processed within"
    Chunk 2 starts: "5 business days of receiving the return."

Neither chunk alone contains the complete fact.
With 50% overlap:
    Chunk 1 ends: "...the refund will be processed within 5 business days of receiving the return."
    Chunk 2 starts: "processed within 5 business days of receiving the return. Contact us..."

Now the complete sentence appears in Chunk 1. Retrieval can find it.
```

### The redundancy cost

```text
50% overlap → ~2× as many chunks as non-overlapping.
25% overlap → ~1.33× as many chunks.

More chunks = more storage, more vectors to search, potentially more
redundant results (retriever returns chunk 3 and chunk 4, which share half their content).

Typical overlap in practice: 10–20% of chunk size.
    chunk_size = 500 tokens, overlap = 50–100 tokens
```

### When to use

```text
✓ Dense, information-rich text where no sentence can be sacrificed
✓ When you can't predict where key facts will fall
✓ Simple to implement, low risk of major retrieval failures

✗ Increases index size and retrieval cost proportionally
✗ Duplicate/redundant content in retrieved results
```

---

## 8. Hierarchical / Parent-Child Chunking

Store two levels of chunks: large "parent" chunks for context, small "child" chunks for precision.

### The core insight

```text
Small chunks are better for retrieval (precise, focused search).
Large chunks are better for generation (give the LLM more context).

The solution: use small chunks to FIND the answer,
              then give the LLM the PARENT chunk for answering.
```

### How it works

```text
Parent chunk (500 tokens):
    "Section 3: Return Policy
     We accept returns within 30 days of purchase. Items must be in
     original, unused condition with all tags attached. Shipping costs
     are non-refundable. International customers must initiate returns
     within 14 days. Customs duties are the customer's responsibility.
     Sale items and gift cards are final sale. For damaged items,
     contact support@example.com with photos within 48 hours."

Child chunks (each ~100 tokens, all linked to parent):
    Child 3a: "We accept returns within 30 days of purchase. Items must
               be in original, unused condition with all tags attached."
    Child 3b: "Shipping costs are non-refundable. International customers
               must initiate returns within 14 days."
    Child 3c: "Customs duties are the customer's responsibility. Sale items
               and gift cards are final sale."
    Child 3d: "For damaged items, contact support@example.com with photos
               within 48 hours."

At query time:
    Query: "What do I do about a damaged item?"
    Retrieval: finds Child 3d (small, precise match)
    Lookup: fetch Parent 3 (full section context)
    LLM sees: the full 500-token parent section, not just the 100-token child
```

### Small-to-big retrieval diagram

```text
                    Query: "damaged item"
                           │
                           ▼
              ┌────────────────────────┐
              │   Vector Search         │ ← searches small child chunks
              │   (child embeddings)    │    precise, focused
              └────────────┬───────────┘
                           │
                    Child 3d found
                           │
                           ▼
              ┌────────────────────────┐
              │   Parent Lookup        │ ← fetch the parent by ID
              │   (by chunk ID)        │    no vector search needed
              └────────────┬───────────┘
                           │
                    Parent 3 fetched
                           │
                           ▼
              ┌────────────────────────┐
              │   LLM generation       │ ← sees full 500-token context
              └────────────────────────┘
```

### Sentence Window variant

```text
A lightweight version of parent-child:
    Index individual sentences (very small = very precise retrieval)
    At retrieval time, expand to include ±k surrounding sentences

    Retrieved: "Customs duties are the customer's responsibility." (sentence 5)
    Expand by ±2:
        Return to LLM: sentences 3, 4, 5, 6, 7 (surrounding context)

No pre-built parent chunks needed — just a sentence store + original text.
```

### When to use

```text
✓ Long, dense documents where precise retrieval AND rich context both matter
✓ Technical documentation, legal documents, financial reports
✓ When naive RAG is missing context ("the answer is in the chunk but incomplete")

✗ More complex to implement (two-level storage, ID linkage)
✗ Parent chunks can still be too large if the document structure is flat
```

---

## 9. Late Chunking (Embedding-First)

A newer approach that flips the order: embed the full document FIRST, then chunk the embeddings.

### The standard problem

```text
Standard approach:
    chunk text → embed each chunk → store chunk embeddings

Problem:
    "it" → embedding has no context. What does "it" refer to?

    Document: "Our product won the 2024 innovation award. It is now available in 50 countries."
    Chunked:
        Chunk 1: "Our product won the 2024 innovation award."
        Chunk 2: "It is now available in 50 countries."    ← "it" is ambiguous in isolation

    The embedding of Chunk 2 must guess what "it" means.
    If retrieved alone, the LLM also can't resolve "it".
```

### The late chunking solution

```text
Late chunking (introduced by Jina AI, 2024):

Step 1: Run the FULL document through the embedding model
        → get a contextualised token embedding for EVERY token
        (Modern LLMs produce contextualised representations via attention)

Step 2: THEN chunk the token embeddings into groups
        → each chunk's embedding is the mean-pooled token embeddings for that span

The key: when we embedded the full document, the token for "it" in
"It is now available in 50 countries" already attended to "Our product"
and has a representation that reflects "the product", not just "it".

Chunk 2's embedding now carries the contextual meaning of the full document.
```

### Trade-offs

```text
Pros:
    Better embeddings for anaphora (pronouns, "the above", "this policy")
    Better for documents with heavy cross-references

Cons:
    Document must fit in the embedding model's context window (up to 8K-128K tokens)
    Can't be used for very long documents without further splitting
    Newer technique — less widely supported in standard RAG libraries
    Marginally better in practice than recursive splitting + overlap for most use cases
```

---

## 10. Agentic / Query-Adaptive Chunking

Let the LLM decide the chunk boundaries.

### How it works

```text
Pass the full document (or large sections) to an LLM with instructions:

    "Below is a document. Identify the distinct topics and split the
     document into coherent sections. Return a JSON list of sections,
     each with a 'title' and 'content' field."

The LLM reads the document and proposes semantically coherent splits,
writes titles for each section, and can even summarise them.

Alternatively: for a specific query, retrieve a large candidate chunk,
then ask the LLM: "Extract the specific passage that answers: {query}"
→ use the extracted passage as the context for generation.
```

### When to use

```text
✓ One-time processing of a small, high-value document corpus
✓ When semantic accuracy matters more than cost
✓ Documents with unusual structure that other strategies mangle

✗ Expensive — requires LLM call per document
✗ Not scalable to millions of documents
✗ Output is non-deterministic (LLM may split the same doc differently each time)
```

---

## 11. Choosing Chunk Size

Chunk size affects retrieval precision, context richness, and cost.

```text
Chunk size      Retrieval      Context quality     Use case
──────────────────────────────────────────────────────────────────────
~50–100 tokens  Very precise   Poor — fragments    Sentence-level Q&A (FAQ matching)
~200–300 tokens Precise        Good                General purpose RAG — best starting point
~500–800 tokens Moderate       Richer              Technical docs, long-form reasoning
~1000+ tokens   Coarse         Very rich           When LLM context is cheap and large

Rule: start at 256–512 tokens, measure retrieval quality, adjust.
```

### Embedding model context window constraint

```text
Most embedding models have a max input length. Chunks longer than this are TRUNCATED.

    text-embedding-ada-002:    max 8,191 tokens (generous)
    text-embedding-3-small:    max 8,191 tokens
    BGE-small:                 max 512 tokens  ← common OSS model
    E5-large:                  max 512 tokens

If your chunk is 800 tokens and your embedding model caps at 512:
    The last 288 tokens of the chunk are IGNORED during embedding.
    The chunk vector represents only the first 512 tokens.
    Retrieval quality degrades for content in the truncated portion.

Match chunk size to embedding model max length.
```

---

## 12. Strategy Selection Guide

```text
Document type                   Recommended strategy
──────────────────────────────────────────────────────────────────────────
Markdown docs / wikis           Document-structure-aware (by heading)
HTML web pages                  HTML-aware + sentence sliding window
PDFs (clean, structured)        Recursive character splitting + overlap
PDFs (scanned/OCR)              Fixed-size + large overlap (structure lost)
Meeting transcripts             Semantic chunking or sentence sliding window
Code files                      Split by function/class, not by character
Q&A pairs / FAQ                 Each Q&A pair = one chunk (preserve unit)
Legal / financial reports       Hierarchical (parent-child)
Mixed / unknown                 Recursive character splitting (safe default)
```

---

## Summary

```text
Strategy                   Key idea                          Best for
────────────────────────────────────────────────────────────────────────
Fixed-size                 Split every N chars               Quick prototypes
Sentence-based             Split on sentence boundaries      Prose, transcripts
Paragraph / delimiter      Split on \n\n                     Well-structured docs
Recursive character        Try \n\n → \n → . → space        General purpose (default)
Document-structure-aware   Split on headers / HTML tags      Markdown, wikis, HTML
Semantic                   Split where meaning changes       Topic-shifting docs
Sliding window             Fixed-size with large overlap     Dense, boundary-sensitive text
Hierarchical (parent-child) Small for retrieval, large for LLM  Long technical docs
Late chunking              Embed full doc, then chunk        Anaphora-heavy documents
Agentic                    LLM decides boundaries            Small, high-value corpora

The single most impactful decision: choose the strategy that respects
your document's natural structure. For most use cases, recursive character
splitting with 10–20% overlap at 256–512 tokens is the right starting point.
```
