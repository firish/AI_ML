## Key Concepts: PagedAttention

---

## 0. Recap — The KV Cache Memory Problem at Scale

### Serving isn't one user — it's thousands

```text
A single user generating tokens:
    LLaMA 2 13B, 4K context, FP16:
        KV cache = 2 × 40 layers × 5120 dims × 4096 tokens × 2 bytes
                 = 3.2 GB per request

Serving 100 concurrent users:
    KV cache = 3.2 GB × 100 = 320 GB  ← just for caches

A100 80GB GPU holds the model (~26 GB in FP16) + maybe 15-20 concurrent users.
The bottleneck isn't compute — it's memory for KV caches.

Anything that reduces KV cache memory waste directly increases
how many users you can serve on the same hardware.
```

### The problem with how KV caches are allocated

```text
Standard approach: pre-allocate a contiguous block of memory
for each request's KV cache, sized for the MAXIMUM possible length.

    Request A: "What is 2+2?" → might generate 10 tokens
        Pre-allocated: 4096 tokens × 3.2 GB = 3.2 GB reserved
        Actually used: 20 tokens × ~0.016 GB = 0.016 GB
        Wasted: 99.5%

    Request B: "Write me a 2000-word essay on..." → might generate 2000 tokens
        Pre-allocated: 4096 tokens × 3.2 GB = 3.2 GB reserved
        Actually used: 2000 tokens × ~1.6 GB = 1.6 GB
        Wasted: 50%

You don't know how long a response will be before generation starts.
So you reserve the maximum — and most of it goes unused.

Studies show 60-80% of KV cache memory is wasted in typical serving.
```

---

## 1. The Analogy: Contiguous vs Paged Memory

### This is the same problem operating systems solved decades ago

```text
Early computers: each program got one CONTIGUOUS block of memory.

    RAM: [████ Program A ████][░░░░ free ░░░░][████ Prog B ████][░ free ░]

    Problems:
        1. Fragmentation: 4 GB free total, but split into 2 GB + 2 GB chunks.
           A program needing 3 GB can't run — no single block is big enough.
        2. Over-allocation: don't know how much a program needs upfront,
           so reserve the max → most is wasted.

    The fix: virtual memory with PAGING.
        Split memory into small fixed-size pages (4 KB).
        A program doesn't need contiguous memory.
        Its "contiguous" address space maps to scattered physical pages.
        Pages allocated on demand — only use what you need.

PagedAttention applies this exact idea to KV cache memory on GPUs.
```

---

## 2. PagedAttention — How It Works

### Blocks instead of contiguous allocation

```text
Instead of one big contiguous buffer per request, divide KV cache memory
into small fixed-size BLOCKS (typically 16 tokens per block).

    Each block stores K and V vectors for 16 tokens.
    Blocks can be anywhere in GPU memory — non-contiguous.
    A block table (like a page table) maps logical positions to physical blocks.

Request A: "What is 2+2?" (prompt=6 tokens, generates 5 tokens = 11 total)

    Logical KV cache: [token 0-15]
    Physical: 1 block allocated (block #47 somewhere in GPU memory)
    Block table: {0 → block #47}

Request B: long response, 50 tokens so far

    Logical KV cache: [token 0-15][token 16-31][token 32-47][token 48-50...]
    Physical: 4 blocks allocated (blocks #12, #103, #7, #89)
    Block table: {0 → #12, 1 → #103, 2 → #7, 3 → #89}

    Blocks are scattered in GPU memory — that's fine.
    The block table handles the mapping.
```

### Allocation on demand

```text
Blocks are allocated as the sequence GROWS, not upfront:

    Token 0-15 generated:  allocate block 1    (from free pool)
    Token 16-31 generated: allocate block 2    (from free pool)
    Token 32-47 generated: allocate block 3    (from free pool)
    Request finishes:      all 3 blocks returned to free pool

    No pre-allocation of max length.
    No wasted memory for tokens that were never generated.

Memory waste:
    Old way: reserve 4096 tokens upfront, use 50 → 98.8% waste
    PagedAttention: allocate 4 blocks of 16 = 64 token slots, use 50 → 21.9% waste
    (Only the last block can have internal waste — at most 15 tokens.)

    Internal fragmentation: < 4% on average (from partially filled last blocks).
    vs 60-80% waste with contiguous allocation.
```

### How attention reads from paged KV cache

```text
During attention, the query needs to read K, V from scattered blocks:

    Q for new token at position 50:
        Need K, V from positions 0-50.
        Block table says:
            positions 0-15  → physical block #12
            positions 16-31 → physical block #103
            positions 32-47 → physical block #7
            positions 48-50 → physical block #89

    The attention kernel gathers K, V from these blocks using the table.
    Slightly more complex than reading one contiguous buffer,
    but the memory savings far outweigh the gather overhead.

    vLLM implements custom CUDA kernels that handle this efficiently.
```

---

## 3. Continuous Batching

### The problem with static batching

```text
Naive serving: collect a batch of requests, process them together,
wait for ALL to finish before starting new ones.

    Batch of 4 requests:
        Request A: generates 10 tokens   → done at step 10
        Request B: generates 200 tokens  → done at step 200
        Request C: generates 50 tokens   → done at step 50
        Request D: generates 300 tokens  → done at step 300

    Static batching: all 4 run together until step 300.
        Request A finishes at step 10 but its slot is occupied until step 300.
        290 steps of wasted GPU compute for that slot.
        No new requests can enter until the entire batch completes.

    Timeline:
        Step 0─────10────50─────200────300
        Req A: [████████]░░░░░░░░░░░░░░░░░  ← idle for 290 steps
        Req B: [████████████████████████]░░░░░░
        Req C: [████████████████]░░░░░░░░░░░░░
        Req D: [██████████████████████████████]
                                              ↑ batch ends, new batch starts
```

### Continuous batching: swap requests in and out

```text
As soon as a request finishes, immediately replace it with a new one.
No waiting for the entire batch to complete.

    Timeline:
        Step 0─────10────50─────200────300
        Req A: [████████]
        Req E:           [████████████████████]     ← enters at step 10
        Req C: [████████████████]
        Req F:                   [████████████████]  ← enters at step 50
        Req B: [████████████████████████]
        Req G:                           [██████████] ← enters at step 200
        Req D: [██████████████████████████████]

    The GPU is always fully utilized. No idle slots.
    New requests don't wait for the slowest request in the batch.

This requires:
    1. Per-request KV cache management (PagedAttention provides this)
    2. Ability to add/remove requests from the batch at each decode step
    3. A scheduler that decides which requests to run

vLLM, TGI (Text Generation Inference), and TensorRT-LLM all do this.
```

### Why PagedAttention enables continuous batching

```text
With contiguous KV cache allocation:
    Each request owns a fixed block of GPU memory.
    Adding a new request means finding a new contiguous block.
    Memory is fragmented from finished requests → may not fit.
    Compaction (defragmentation) is expensive.

With PagedAttention:
    Finished request → its blocks return to the free pool immediately.
    New request → allocate blocks from the pool one at a time.
    No fragmentation problem. No contiguous block needed.

    PagedAttention makes continuous batching practical.
```

---

## 4. Prefix Caching

### The observation: many requests share the same prefix

```text
In real serving, many requests share identical prefixes:

    System prompt (shared by ALL requests to the same API):
        "You are a helpful assistant. You are precise and concise..."
        → 200 tokens of identical KV cache across every request

    Few-shot examples (shared across similar requests):
        "Here are some examples of good summaries: ..."
        → 500 tokens of identical KV cache

    RAG context (shared when multiple users query the same document):
        "[document content]"
        → 1000+ tokens

Without prefix caching:
    User A: system prompt (200 tokens) + "What is Python?" (5 tokens)
    User B: system prompt (200 tokens) + "Explain gravity" (4 tokens)
    User C: system prompt (200 tokens) + "Write a poem" (4 tokens)

    KV cache stores the 200-token system prompt THREE times.
    3 × 200 = 600 token slots used for identical data.
```

### How prefix caching works with PagedAttention

```text
Since KV cache is split into blocks, identical prefixes produce
identical blocks. These can be SHARED across requests.

    System prompt "You are a helpful..." = 200 tokens = 13 blocks (of 16)

    Without prefix caching:
        User A: [13 blocks for prefix] + [1 block for query]
        User B: [13 blocks for prefix] + [1 block for query]
        User C: [13 blocks for prefix] + [1 block for query]
        Total: 39 + 3 = 42 blocks

    With prefix caching:
        Shared: [13 blocks for prefix] ← computed once, reused
        User A: [pointer to shared] + [1 block for query]
        User B: [pointer to shared] + [1 block for query]
        User C: [pointer to shared] + [1 block for query]
        Total: 13 + 3 = 16 blocks  ← 2.6× less memory

    The block tables for A, B, C all point to the SAME physical blocks
    for the prefix portion. Only the unique suffixes get their own blocks.

This is copy-on-write from OS virtual memory:
    Multiple processes share the same physical memory pages.
    Only when a process writes (diverges) does it get its own copy.
    Same idea: requests share prefix blocks until their tokens diverge.
```

### Automatic prefix detection

```text
vLLM uses a hash-based approach:

    For each block of 16 tokens, compute a hash of the token IDs.
    If a block with the same hash already exists in the cache → reuse it.
    If not → compute it and store it for future reuse.

    This works automatically — no manual prefix specification needed.
    Even partial prefix matches are exploited (block-level granularity).

    Request 1: "Summarize: [article A]" → computes all blocks
    Request 2: "Summarize: [article A] in 3 bullet points"
               ↑ prefix matches! → reuse those blocks, only compute new ones
```

### Prefix caching saves compute too

```text
Shared prefix blocks are already computed (prefill already ran).
New requests that match the prefix skip prefill for those tokens entirely.

    Without: each request prefills the 200-token system prompt → 200 tokens of compute
    With:    system prompt blocks are cached → prefill starts at token 201

    For a 2000-token RAG context shared across 50 users:
        Without: 50 × 2000 = 100K tokens of redundant prefill compute
        With:    1 × 2000 = 2K tokens of prefill + 50 small unique suffixes

    This directly reduces time-to-first-token for users.
```

---

## 5. Practical Numbers

```text
vLLM benchmarks (LLaMA 13B, A100 80GB):

    Throughput improvement over static batching:
        PagedAttention + continuous batching: 2-4× higher throughput
        + prefix caching (with shared system prompts): additional 1.5-2×

    Memory utilization:
        Static allocation:    20-40% of GPU memory effectively used
        PagedAttention:       >95% of GPU memory effectively used

    Concurrent users (same GPU):
        Static allocation:    ~15 concurrent requests
        PagedAttention:       ~50+ concurrent requests (depending on lengths)

Serving frameworks that implement these:
    vLLM:             PagedAttention (invented it), continuous batching, prefix caching
    TGI (HuggingFace): continuous batching, paged KV cache
    TensorRT-LLM:     similar paging, in-flight batching (their name for continuous batching)
    SGLang:           RadixAttention (tree-based prefix caching, more aggressive reuse)
```

---

## Summary

```text
The problem: serving LLMs to many users wastes 60-80% of KV cache memory
because caches are pre-allocated at max length in contiguous blocks.

PagedAttention (Kwon et al., 2023 — vLLM):
    Split KV cache into small blocks (16 tokens each).
    Allocate blocks on demand as the sequence grows.
    Blocks can be non-contiguous — a block table maps logical → physical.
    Waste drops from 60-80% to <4%.
    More users per GPU → directly reduces serving cost.

Continuous batching:
    Replace finished requests immediately, don't wait for the batch.
    PagedAttention makes this practical (no fragmentation from churn).
    GPU is always fully utilized → 2-4× throughput over static batching.

Prefix caching:
    Shared prefixes (system prompts, RAG context) → shared physical blocks.
    Copy-on-write: requests share blocks until they diverge.
    Saves both memory (don't store duplicates) and compute (skip redundant prefill).

All three ideas work together. PagedAttention is the foundation that
enables the other two. This is why vLLM became the standard for LLM serving.
```
