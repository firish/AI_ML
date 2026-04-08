## Key Concepts: MQA and GQA — Shrinking the KV Cache

---

## 0. Recap — Where We Are

From `01_KV_cache`: every decode step computes Q, K, V for one new token,
appends K and V to a per-layer cache, and uses that cache for attention.
The cache grows by one row per token, per layer, and must live in GPU memory
for the entire generation.

From `02_RoPE`: Q and K are rotated before attention; V is never touched.
The cache stores rotated K vectors (position baked in) and plain V vectors.

What neither file examined closely: the model doesn't run ONE attention
computation per layer. It runs MANY — one per head.

---

## 1. Recap — Multi-Head Attention

### Why multiple heads?

A single attention pass answers one kind of question: "which tokens are
relevant to this one?" But relevance means different things simultaneously:

```text
"The bank by the river was steep."

Head 0 might ask: syntactic subject → "bank" attends to "was"
Head 1 might ask: modifier → "steep" attends to "bank"
Head 2 might ask: disambiguation → "bank" attends to "river" (not financial bank)
Head 3 might ask: coreference → nothing interesting here

All of these are happening at the same token, at the same layer.
A single attention head can only capture one of them at a time.
Multi-head attention runs them in parallel.
```

### How it works mechanically

The model has `d_model` dimensions and `n_heads` heads. Each head operates on
a slice of size `d_k = d_model / n_heads`:

```text
Example: d_model = 4096, n_heads = 32, d_k = 128

Each head h has its own projection matrices:
    W_q[h]: (4096, 128)   ← project token → query for head h
    W_k[h]: (4096, 128)   ← project token → key for head h
    W_v[h]: (4096, 128)   ← project token → value for head h

For one token x (1, 4096):
    Q[h] = x × W_q[h]   → (1, 128)
    K[h] = x × W_k[h]   → (1, 128)
    V[h] = x × W_v[h]   → (1, 128)

    score[h] = Q[h] · K_cache[h]ᵀ    (attends to cached keys for head h)
    output[h] = softmax(score[h]) × V_cache[h]   → (1, 128)

Concatenate all heads: [output[0], output[1], ..., output[31]] → (1, 4096)
Project back: × W_o (4096, 4096) → (1, 4096)
```

In practice the projections are run as one big matrix multiply (all heads at
once), then split. But logically: each head has independent W_k and W_v.

---

## 2. The Problem — KV Cache Scales with Heads

The KV cache stores K and V for every head, at every layer.

```text
One token added to the cache:
    Per head:  K vector (128-d) + V vector (128-d) = 256 dims
    32 heads:  256 × 32 = 8,192 dims per layer
    80 layers: 8,192 × 80 = 655,360 values
    FP16:      655,360 × 2 bytes = 1.28 MB per token

At 4,096 tokens (LLaMA 2 70B, full context):
    1.28 MB × 4,096 = 5.2 GB just for KV cache

At 128K tokens:
    1.28 MB × 128,000 = 163 GB
    The model weights themselves are 140 GB.
    The cache has overtaken the model.
```

The root cause: each of the 32 heads stores its own K,V independently.
32 sets of K,V per layer per token. If we can reduce the number of K,V sets
without hurting quality much, the cache shrinks proportionally.

That's the entire motivation for MQA and GQA.

---

## 3. Multi-Query Attention (MQA)

### The idea

Keep Q per-head (each head asks its own question). Share a single K and V
across ALL heads (all heads look at the same keys and values).

```text
Standard MHA (32 heads):
    W_q[0], W_q[1], ..., W_q[31]   — 32 separate query projections
    W_k[0], W_k[1], ..., W_k[31]   — 32 separate key projections
    W_v[0], W_v[1], ..., W_v[31]   — 32 separate value projections

MQA (32 Q heads, 1 KV head):
    W_q[0], W_q[1], ..., W_q[31]   — still 32 query projections
    W_k                              — ONE key projection (shared)
    W_v                              — ONE value projection (shared)
```

### What attention looks like under MQA

```text
One token x (1, 4096):
    Q[h] = x × W_q[h]   → (1, 128)   for each of 32 heads
    K     = x × W_k      → (1, 128)   ONE key for ALL heads
    V     = x × W_v      → (1, 128)   ONE value for ALL heads

Head 0 attention:
    score[0] = Q[0] · K_cacheᵀ         (Q differs per head)
    output[0] = softmax(score[0]) × V_cache

Head 1 attention:
    score[1] = Q[1] · K_cacheᵀ         (different Q, same K_cache)
    output[1] = softmax(score[1]) × V_cache

Head 2 attention:
    score[2] = Q[2] · K_cacheᵀ         (different Q, same K_cache)
    output[2] = softmax(score[2]) × V_cache

...

Every head uses the SAME K_cache and V_cache.
But because Q differs per head, score[h] differs per head,
and so output[h] differs per head. The heads still diverge.
```

### Cache impact

```text
MHA cache per token per layer:
    32 K vectors × 128 dims + 32 V vectors × 128 dims = 8,192 values

MQA cache per token per layer:
    1 K vector  × 128 dims + 1 V vector  × 128 dims = 256 values

Reduction: 32× smaller cache.

At 128K tokens (LLaMA 70B scale):
    MHA:  163 GB
    MQA:  163 / 32 = 5.1 GB  ← fits easily
```

### The quality cost

```text
MHA: each head has its own W_k and W_v.
    Head 0's K projects tokens one way (e.g., syntactic role).
    Head 1's K projects tokens a different way (e.g., semantic similarity).
    The keys themselves are SPECIALISED per head.

MQA: all heads share one K and V.
    Every head asks its own question (Q differs).
    But every head is searching through the SAME key space.
    It's like 32 people asking different questions, but reading
    from the same index — the index wasn't built for all their questions.

Empirically: ~1-3% quality drop on most benchmarks.
Acceptable for some use cases, not for others.
Used by: PaLM (540B), Falcon. Not adopted by the LLaMA line.
```

---

## 4. Grouped-Query Attention (GQA)

### The idea

MQA goes too far — one K,V for 32 heads loses capacity. MHA is too expensive —
32 K,V sets for 32 heads. GQA is the middle: G groups, each sharing one K,V.

```text
MHA:  32 Q heads, 32 K heads, 32 V heads
GQA:  32 Q heads,  8 K heads,  8 V heads   (4 Q heads share each KV head)
MQA:  32 Q heads,  1 K head,   1 V head

With G=8 KV heads and 32 Q heads:
    Group 0:  Q[0],  Q[1],  Q[2],  Q[3]   share K[0], V[0]
    Group 1:  Q[4],  Q[5],  Q[6],  Q[7]   share K[1], V[1]
    Group 2:  Q[8],  Q[9],  Q[10], Q[11]  share K[2], V[2]
    Group 3:  Q[12], Q[13], Q[14], Q[15]  share K[3], V[3]
    Group 4:  Q[16], Q[17], Q[18], Q[19]  share K[4], V[4]
    Group 5:  Q[20], Q[21], Q[22], Q[23]  share K[5], V[5]
    Group 6:  Q[24], Q[25], Q[26], Q[27]  share K[6], V[6]
    Group 7:  Q[28], Q[29], Q[30], Q[31]  share K[7], V[7]
```

### Projection shapes

```text
d_model = 4096, n_q_heads = 32, n_kv_heads = 8, d_k = 128

W_q: (4096, 32 × 128) = (4096, 4096)    ← same as MHA
W_k: (4096,  8 × 128) = (4096, 1024)    ← 4× smaller than MHA
W_v: (4096,  8 × 128) = (4096, 1024)    ← 4× smaller than MHA

For one token x (1, 4096):
    Q = x × W_q   → (1, 4096)  → reshape to (32, 128): 32 query vectors
    K = x × W_k   → (1, 1024)  → reshape to (8, 128):  8 key vectors
    V = x × W_v   → (1, 1024)  → reshape to (8, 128):  8 value vectors
```

### Attention for one group

```text
Q heads 0-3 all use K[0] and V[0]:

    Head 0: score[0] = Q[0] · K_cache[0]ᵀ
            output[0] = softmax(score[0]) × V_cache[0]

    Head 1: score[1] = Q[1] · K_cache[0]ᵀ    ← same K_cache[0]
            output[1] = softmax(score[1]) × V_cache[0]    ← same V_cache[0]

    Head 2: score[2] = Q[2] · K_cache[0]ᵀ
            output[2] = softmax(score[2]) × V_cache[0]

    Head 3: score[3] = Q[3] · K_cache[0]ᵀ
            output[3] = softmax(score[3]) × V_cache[0]

score[0] ≠ score[1] ≠ score[2] ≠ score[3]   (Q differs)
output[0] ≠ output[1] ≠ output[2] ≠ output[3]  (attention weights differ)

The 4 heads within a group still diverge — just fewer distinct key spaces.
```

### Cache impact

```text
GQA cache per token per layer (G=8):
    8 K vectors × 128 dims + 8 V vectors × 128 dims = 2,048 values

vs MHA: 8,192 values per token per layer
Reduction: 4× smaller cache.

At 128K tokens (LLaMA 2 70B, 80 layers, FP16):
    MHA:  163 GB
    GQA:  163 / 4 = 41 GB   ← fits on one A100 (80GB) alongside weights
```

### Quality vs MQA

```text
MQA (1 KV head):
    All 32 Q heads share one key space.
    The single W_k must serve all 32 heads' needs.
    Overloaded — one projection can't specialise for all heads.

GQA (8 KV heads):
    Every 4 Q heads share one key space.
    The 4 heads in a group have similar enough concerns that one
    K,V representation serves them well.
    8 distinct key spaces still provides meaningful specialisation.

Empirically: GQA with G = n_heads/4 achieves quality nearly
indistinguishable from MHA on most benchmarks.
This is why it became the standard.

Used by: LLaMA 2 70B (G=8), LLaMA 3 (G=8), Mistral, Gemma.
Note: smaller models (LLaMA 2 7B) use MHA — the cache is small
enough that GQA isn't worth it at that scale.
```

---

## 5. Comparing All Three

```text
Setup: d_model = 4096, n_q_heads = 32, d_k = 128

                MHA         GQA (G=8)   MQA (G=1)
────────────────────────────────────────────────────
Q heads         32          32          32
KV heads        32          8           1
W_k shape       (4096,4096) (4096,1024) (4096,128)
W_v shape       (4096,4096) (4096,1024) (4096,128)
────────────────────────────────────────────────────
KV cache/token  8,192 vals  2,048 vals  256 vals
  (per layer)   (32 KV sets)(8 KV sets) (1 KV set)
Cache reduction 1×          4×          32×
────────────────────────────────────────────────────
Quality         Best        ~Same       Slight drop
Used by         BERT,GPT-2  LLaMA 2/3   PaLM, Falcon
                GPT-3       Mistral
```

---

## 6. Why This Matters for Inference

GQA is not just a memory trick — it directly determines what's feasible at serving time.

```text
Scenario: serve LLaMA 2 70B with 128K context window.

Model weights: 140 GB (FP16)
GPU: 1× H100 (80GB) or 2× A100 (80GB each)

With MHA:
    KV cache at 128K tokens: 163 GB
    Total memory needed: 140 + 163 = 303 GB
    Requires 4× A100s. Expensive to operate.

With GQA (G=8, as LLaMA 2 70B actually uses):
    KV cache at 128K tokens: 41 GB
    Total memory needed: 140 + 41 = 181 GB
    2× A100s. Practical for production.

GQA is what makes long-context serving economically viable.
Without it, the KV cache cost would dominate and long contexts
would require hardware that most teams can't afford.
```

KV cache, RoPE, and GQA form a natural chain:
- **KV cache** (file 01): avoid recomputing K,V — store them instead
- **RoPE** (file 02): encode positions into K so cached vectors remain valid
- **GQA** (this file): reduce how many K,V sets you have to store

Each one is independently useful, but together they're what makes a 70B
model with 128K context run on two GPUs instead of eight.
