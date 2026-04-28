## Augmentation and Generation — Prompting the LLM with Retrieved Context

---

## 0. Where This Fits

```text
Full RAG pipeline:

    Chunking → Embedding → Indexing      ← files 02, 03 (offline)
    Retrieval                            ← file 04
        │
        ▼
    Augmentation                         ← this file (first half)
        │
        ▼
    Generation                           ← this file (second half)
```

Retrieval gives you a ranked list of relevant chunks.
Augmentation is the step where you turn those chunks into a prompt.
Generation is the LLM reading that prompt and producing an answer.

This sounds simple. It has a surprising number of ways to go wrong.

---

## 1. What Augmentation Actually Is

Without RAG, a prompt looks like:

```text
System: You are a helpful assistant.
User:   What is the refund policy for international orders?
```

The LLM has no knowledge of your refund policy. It either hallucinates or says it doesn't know.

With RAG, augmentation injects the retrieved chunks:

```text
System: You are a helpful assistant. Answer the user's question using only
        the context provided below. If the answer is not in the context,
        say "I don't have that information."

        Context:
        [1] Source: return_policy.pdf, Section: International Returns
            "For international customers, returns must be initiated within
             14 days of delivery. Customs duties are the customer's
             responsibility and will not be refunded."

        [2] Source: return_policy.pdf, Section: Standard Returns
            "Items must be returned in original, unused condition.
             Refunds are processed within 5 business days of receiving
             the returned item."

        [3] Source: faq.md, Section: Shipping
            "International shipping takes 7–14 business days.
             Expedited options are not available for all countries."

User:   What is the refund policy for international orders?
```

The LLM now reads the real policy and answers from it. No hallucination.

Augmentation = deciding what context to inject, how to format it, and where in the prompt.

---

## 2. Prompt Structure

The retrieved chunks can be placed in different positions in the prompt.

### Option A: System prompt context (most common)

```text
┌─────────────────────────────────────────────────────┐
│ SYSTEM                                              │
│   Role instructions                                 │
│   Answer grounding instructions                     │
│   Context:                                          │
│       [1] chunk text...                             │
│       [2] chunk text...                             │
│       [3] chunk text...                             │
├─────────────────────────────────────────────────────┤
│ USER                                                │
│   What is the refund policy for international orders│
├─────────────────────────────────────────────────────┤
│ ASSISTANT                                           │
│   (generated here)                                  │
└─────────────────────────────────────────────────────┘

Pros: context is always "above" the query — LLM reads it before the question.
Cons: system prompt is re-sent every turn in multi-turn conversations (token cost).
```

### Option B: User message context

```text
┌─────────────────────────────────────────────────────┐
│ SYSTEM                                              │
│   Role instructions + grounding instructions        │
├─────────────────────────────────────────────────────┤
│ USER                                                │
│   Context:                                          │
│       [1] chunk text...                             │
│       [2] chunk text...                             │
│   Question: What is the refund policy for...?       │
├─────────────────────────────────────────────────────┤
│ ASSISTANT                                           │
│   (generated here)                                  │
└─────────────────────────────────────────────────────┘

Pros: fresh context per turn — better for multi-turn conversations
      where retrieved chunks change each turn.
Cons: context is interleaved with the user query — slightly less clean structure.
```

### Option C: Separate context turns (RAG with conversation history)

```text
For multi-turn conversations, inject context as a "pseudo-assistant" message
or as an explicit context block before the user message each turn.

Turn 1:
    User:      "What is the return window?"
    [retrieved chunks injected here]
    Assistant: "You have 30 days to return items."

Turn 2:
    User:      "What about international orders?"
    [NEW retrieved chunks injected here]
    Assistant: "International orders have a 14-day window."

Key: re-run retrieval on EVERY turn. The relevant chunks change per question.
     Don't reuse the same chunks across the whole conversation.
```

---

## 3. Context Formatting

How you format the chunks inside the prompt affects LLM comprehension.

### Minimal formatting

```text
Context:
Refunds are accepted within 30 days. Items must be in original condition.
International returns must be initiated within 14 days. Customs duties apply.
Our shipping times for international orders are 7–14 business days.
```

Problem: the LLM can't tell where one chunk ends and another begins.
         Can't attribute answers to sources. Chunks bleed into each other.

### Numbered chunks with source metadata

```text
Context:
[1] Source: return_policy.pdf | Section: Standard Returns
Refunds are accepted within 30 days. Items must be in original condition.

[2] Source: return_policy.pdf | Section: International Returns
International returns must be initiated within 14 days. Customs duties apply.

[3] Source: shipping_faq.md | Section: International Shipping
Our shipping times for international orders are 7–14 business days.
```

Better: numbered references allow the LLM to cite sources in its answer.

### XML-tagged chunks (best for instruction-following models)

```text
<context>
  <document index="1" source="return_policy.pdf" section="Standard Returns">
    Refunds are accepted within 30 days. Items must be in original condition.
  </document>
  <document index="2" source="return_policy.pdf" section="International Returns">
    International returns must be initiated within 14 days. Customs duties apply.
  </document>
  <document index="3" source="shipping_faq.md" section="International Shipping">
    Our shipping times for international orders are 7–14 business days.
  </document>
</context>
```

Best for modern models (Claude, GPT-4o): XML tags create explicit structure that
attention heads can parse cleanly. Models trained on XML/HTML parse structure well.

The answer can then cite: "According to [2], international returns must be..."

---

## 4. Chunk Ordering in the Prompt

The order you place chunks in the prompt affects answer quality.

```text
Recall from file 04: "lost in the middle" — LLMs attend more to content
at the START and END of a long context than to content in the MIDDLE.

Experiment results (Liu et al., 2023):
    Correct chunk at position 1:   80% correct answers
    Correct chunk at position 10:  52% correct answers
    Correct chunk at position 20:  77% correct answers

Implication: put the MOST relevant chunk first.

If you retrieved k=5 and re-ranked them [A, B, C, D, E] by relevance:
    Naive order: A, B, C, D, E  ← A is most relevant, in the first position. Good.
    Better order: A, E, D, C, B ← most relevant first, second-most relevant last.
                                   C and D (middling relevance) are in the middle
                                   where attention is weakest anyway.

In practice: most teams just use rank order (most relevant first) and
             call it sufficient. The lost-in-the-middle effect is real
             but its impact depends on query complexity and context length.
```

---

## 5. The Grounding Instruction

The single most important sentence in a RAG prompt.

```text
Without grounding instruction:
    "You are a helpful assistant."

    LLM sees context, but also draws freely from training knowledge.
    It may blend real context with hallucinated additions.
    "International returns take 14 days. Also, you can expedite returns
     by calling our 24/7 support line at 1-800-RETURNS." ← hallucinated

With strong grounding instruction:
    "Answer the user's question using ONLY the information in the context
     below. Do not use prior knowledge. If the answer is not in the
     context, say exactly: 'I don't have that information.'"

    LLM is now constrained to the retrieved context.
    Hallucination drops dramatically for factual Q&A.
```

### Calibrating the grounding instruction

```text
Strict grounding:
    "Use ONLY the context. Do not add anything not present in the context."
    Best for: compliance, legal, medical — where hallucination is dangerous.
    Risk: overly literal. May refuse to paraphrase or synthesise sensibly.

Moderate grounding:
    "Primarily use the context. You may use general knowledge to explain
     terms or provide background, but all specific facts must come from
     the context."
    Best for: most production RAG. Balances accuracy and helpfulness.

Loose grounding (not really RAG):
    "Here is some relevant context that may help. Answer the question."
    The model treats context as hints, not constraints.
    Use only when you want the LLM's general knowledge + context as supplement.
```

---

## 6. Handling the No-Match Case

Retrieval doesn't always find a relevant chunk. The LLM must be told what to do.

### The silent hallucination problem

```text
If you don't handle this, the LLM will often make something up:

    User: "What is the refund policy for cryptocurrency payments?"
    Retrieved chunks: [top-5 semantically similar chunks about general refunds]
    → None mention cryptocurrency.
    → LLM sees context about refunds and generates a plausible-sounding answer
      about crypto refunds that is entirely invented.
```

### Detection approaches

```text
Approach 1: Similarity threshold
    After retrieval, check: is the top chunk's similarity score above a threshold?
    If max(similarity) < 0.75: skip augmentation, answer with "I don't know."

    Problem: threshold is hard to set. A score of 0.70 might be relevant
             for one query and irrelevant for another.

Approach 2: LLM relevance check
    After retrieval, add a pre-generation step:
        "Given the following query and context, does the context contain
         enough information to answer the query? Answer YES or NO."
    If NO: skip generation or answer with fallback.
    More accurate than threshold, but adds one extra LLM call per query.

Approach 3: Instruction-based fallback (most common)
    Include in the grounding instruction:
        "If the answer is not present in the context, respond with:
         'I don't have information about that in my knowledge base.'"
    The LLM itself detects when the context is insufficient.
    Simple, works well in practice. Not 100% reliable — model can miss it.

Approach 4: Combine threshold + instruction
    If similarity < threshold: don't even run the LLM. Return a canned fallback.
    If similarity ≥ threshold: run LLM with grounding instruction as backup.
    Two-layer defence.
```

---

## 7. Context Window Budgeting

Every RAG query consumes tokens across multiple components. You have a fixed budget.

```text
Context window budget (example: Claude Sonnet, 200K token window):

    ┌────────────────────────────────────┬──────────────┐
    │ Component                          │ Tokens       │
    ├────────────────────────────────────┼──────────────┤
    │ System prompt (role + instructions)│ 200–500      │
    │ Retrieved chunks (k=5, ~300 each)  │ 1,500        │
    │ Conversation history (last 5 turns)│ 500–2,000    │
    │ User query                         │ 10–100       │
    │ Reserved for generation (output)   │ 500–2,000    │
    ├────────────────────────────────────┼──────────────┤
    │ Total used                         │ ~5,000       │
    └────────────────────────────────────┴──────────────┘

For Claude or GPT-4o with 128K-200K windows:
    Budget is rarely an issue for typical RAG (5K–20K tokens used).

For smaller models (Llama 3.1 8B with 8K context, or old GPT-3.5 with 4K):
    Budget is tight. Every token counts.
    You may need to truncate chunks or reduce k.
```

### Truncation strategy when over budget

```text
If the assembled prompt exceeds the budget, truncate in this order:

    1. First: trim conversation history (oldest turns dropped first)
    2. Second: reduce k (remove lowest-ranked chunks first)
    3. Third: truncate individual chunk text (cut from the end of long chunks)
    4. Never: truncate the system prompt or the current user query

Why this order:
    Old conversation history is least likely to affect the current answer.
    Low-ranked chunks have the lowest relevance signal.
    Truncating a chunk text loses some content but keeps the chunk present.
    The system prompt and current query are irreplaceable.
```

---

## 8. Source Attribution and Citations

RAG enables a feature that pure LLMs can't provide: traceable answers.

### Basic citation

```text
Prompt the LLM to cite sources:
    "When answering, cite the document index number(s) in square brackets
     like [1] or [2] after each claim."

LLM output:
    "International returns must be initiated within 14 days of delivery [2].
     Customs duties are the customer's responsibility [2]. Standard returns
     are accepted within 30 days [1]."

Each claim is now traceable to a source chunk.
```

### Returning sources to the user

```text
After generation, parse the citations from the response.
Look up the original source metadata for each cited chunk.
Return to the user:

    Answer: "International returns must be initiated within 14 days..."

    Sources:
        [1] return_policy.pdf, Section: Standard Returns, Page 3
        [2] return_policy.pdf, Section: International Returns, Page 5

This is the "show your work" feature that makes RAG trustworthy.
Users can click through to the original document to verify.
```

### Hallucination in citations

```text
A subtle failure mode: the LLM cites [2] for a claim that is NOT in chunk [2].
It fabricated the fact, then fabricated a citation to look grounded.

Detection:
    After generation, verify each cited claim:
    Run a separate LLM call: "Does the following text [chunk 2] support
    the following claim [extracted claim]? YES/NO."
    Flag unsupported citations.

This automated citation verification is part of the RAGAS evaluation
framework, covered in file 06.
```

---

## 9. Generation Settings for RAG

The LLM's sampling parameters matter differently in RAG vs open-ended generation.

```text
Temperature:
    Controls randomness of token sampling.
    0.0  = greedy (always pick the most likely token)
    1.0  = high randomness

    For RAG (factual Q&A):
        Use temperature = 0.0 to 0.3.
        You want the LLM to read the context and extract the answer.
        High temperature introduces variation that serves no purpose here —
        the answer is in the context, not in creative sampling.
        Creativity ≠ accuracy for retrieval-grounded tasks.

    For RAG (summarisation or synthesis):
        Temperature = 0.3 to 0.7 is fine.
        Some variation in phrasing is acceptable when synthesising.

Top-p (nucleus sampling):
    Keep only the top tokens whose cumulative probability ≥ p.
    At p=0.9: if the top 5 tokens cover 90% of probability mass, only sample from those 5.
    For RAG: top-p = 0.9 is reasonable. Top-p = 1.0 (disabled) is also fine at low temperature.

Max output tokens:
    Set explicitly. Don't leave it at the model default (often very large).
    For factual Q&A: 256–512 tokens is usually enough.
    For summarisation: 512–1024 tokens.
    Unbounded output → longer, padded answers → more latency → higher cost.
```

---

## 10. Putting It All Together — The Augmentation + Generation Step

```text
Input: top-k chunks from retrieval (ranked, with metadata)

Step 1: Token budget check
    Count tokens: system prompt + chunks + conversation history + query.
    If over budget: trim history → reduce k → truncate chunks.

Step 2: Order chunks
    Most relevant first (rank 1 at top of context block).

Step 3: Format context block
    Use numbered, source-labelled chunks (or XML tags for Claude/GPT-4o).

Step 4: Assemble full prompt
    System:  [role] + [grounding instruction] + [context block]
    User:    [query]

Step 5: Check similarity threshold (optional)
    If max chunk similarity < threshold → return fallback without calling LLM.

Step 6: Call LLM
    temperature = 0.0–0.3, max_tokens = 256–1024

Step 7: Parse response
    Extract inline citations [1], [2], etc.
    Map citations back to source metadata.

Step 8: Return to user
    Answer text + list of cited sources (document name, section, page).
```

---

## 11. Common Mistakes

```text
Mistake 1: No grounding instruction
    The LLM blends context with hallucinated training knowledge.
    Fix: always include explicit "use only this context" instruction.

Mistake 2: Pasting chunks without labels or separators
    LLM can't tell where one chunk ends and another begins.
    Fix: number each chunk and include source metadata.

Mistake 3: Not handling the no-match case
    LLM silently hallucinates when context doesn't contain the answer.
    Fix: similarity threshold + explicit fallback instruction.

Mistake 4: Using high temperature for factual RAG
    Introduces unnecessary randomness into grounded answers.
    Fix: temperature 0.0–0.3 for factual queries.

Mistake 5: Returning all k chunks regardless of relevance
    If chunk 5 has similarity 0.3, it's probably noise — don't include it.
    Fix: apply a minimum similarity cutoff before augmentation (e.g. sim > 0.6).

Mistake 6: Not re-retrieving in multi-turn conversations
    Reusing chunks from turn 1 for turn 3's question.
    Fix: re-run retrieval on every user turn.

Mistake 7: Overloading the context with too many chunks
    15 chunks × 300 tokens = 4,500 tokens of context before the question.
    LLM attention is diluted. Relevant content gets lost in the middle.
    Fix: retrieve more (k=15), re-rank, pass only top 3–5 to the LLM.
```

---

## Summary

```text
1. Augmentation = inserting retrieved chunks into the prompt before the query

2. Prompt structure:
   - Format: numbered + source metadata or XML tags
   - Order: most relevant chunk first
   - Position: system prompt (simple) or user message (better for multi-turn)

3. Grounding instruction: the most important line in a RAG prompt
   "Answer using ONLY the context. If not present, say I don't know."

4. No-match handling: similarity threshold + fallback instruction (two-layer defence)

5. Context window budget: trim history first, then reduce k, then truncate chunks

6. Citations: prompt the LLM to cite [1] [2] → map back to source metadata → return to user

7. Generation settings: temperature 0.0–0.3 for factual RAG, explicit max_tokens

8. Biggest mistake: no grounding instruction + no no-match handling = hallucinations
```
