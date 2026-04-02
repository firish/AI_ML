## Key Concepts: KV Cache, RoPE, GQA, MoE

These concepts come up constantly when reading about LLMs. Each one solves a specific practical problem with running or scaling transformers.

---

## 1. KV Cache — Why the First Token Is Slow

### The Problem

During generation, the model produces one token at a time. Naively, each new token requires reprocessing the ENTIRE sequence:

```text
Step 1: Process "The"                              → predict "cat"
Step 2: Process "The cat"                          → predict "sat"
Step 3: Process "The cat sat"                      → predict "on"
Step 4: Process "The cat sat on"                   → predict "the"
Step 5: Process "The cat sat on the"               → predict "mat"

At step 5, the model recomputes attention for "The", "cat", "sat", "on"
even though they haven't changed. This is wasted work.
```

### The Fix: Cache K and V

In attention, each token produces three vectors: Q (query), K (key), V (value).

```text
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V

Key observation:
    When generating token 5, the K and V vectors for tokens 1-4
    are IDENTICAL to what they were at step 4.
    Only the NEW token's Q, K, V need to be computed.
```

**The KV cache stores all previous K and V vectors:**

```text
Step 1: "The"
    Compute K₁, V₁ for "The"
    Cache: K = [K₁], V = [V₁]
    Use Q₁ to attend → predict "cat"

Step 2: "The cat"
    Compute K₂, V₂ for "cat" ONLY (not "The" again)
    Cache: K = [K₁, K₂], V = [V₁, V₂]
    Use Q₂ to attend to [K₁, K₂] → predict "sat"

Step 3: "The cat sat"
    Compute K₃, V₃ for "sat" ONLY
    Cache: K = [K₁, K₂, K₃], V = [V₁, V₂, V₃]
    Use Q₃ to attend to [K₁, K₂, K₃] → predict "on"

Each step: process 1 token through the network, append K,V to cache.
Without cache: process ALL tokens every step → O(n²) total work.
With cache: process 1 token per step → O(n) total work.
```

### Two Phases of Generation

```text
Phase 1: Prefill (process the prompt)
    Input: "What is the capital of France?" (8 tokens)
    Process all 8 tokens IN PARALLEL (one forward pass).
    Store all 8 K,V pairs in the cache.
    This is fast — parallelism.
    But it's a big computation (all tokens at once).

Phase 2: Decode (generate the response)
    Generate one token at a time.
    Each step: feed ONLY the new token through the model.
    Use KV cache for attention to all previous tokens.
    Each step is cheap (1 token), but sequential (can't parallelise).

This is why:
    - The first token takes longer (prefill: process entire prompt)
    - Subsequent tokens are faster (decode: just 1 token + cache lookup)
    - Long prompts take longer to start but don't slow down generation
```

### KV Cache Memory Cost

```text
KV cache size per token = 2 × n_layers × d_model × precision_bytes

LLaMA 2 (70B): 2 × 80 layers × 8192 dims × 2 bytes (FP16)
    = 2.6 MB per token

For a 4096-token context:
    = 2.6 MB × 4096 = 10.7 GB of KV cache

For a 128K-token context:
    = 2.6 MB × 128K = 333 GB of KV cache ← huge!

This is why long contexts are expensive — the KV cache
can exceed the model's own weight memory.
```

---

## 2. Positional Encodings — From Learned to RoPE

### The Problem

Transformers have no built-in sense of order (attention is permutation-invariant). We must explicitly tell the model where each token is. Three generations of solutions:

### Generation 1: Sinusoidal (Original Transformer, 2017)

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Fixed formula, not learned. Added to token embeddings.
Works well but doesn't extrapolate to lengths unseen during training.
```

### Generation 2: Learned Positions (BERT, GPT-2)

```text
A lookup table: position → vector (learned during training)
    position 0 → [0.01, 0.03, -0.02, ...]
    position 1 → [0.02, -0.01, 0.04, ...]
    ...
    position 1023 → [0.05, 0.02, -0.03, ...]

Simple and effective. But the table has a FIXED size.
GPT-2 learned 1024 positions → hard limit of 1024 tokens.
Can't handle position 1025 (no learned vector for it).
```

### Generation 3: RoPE — Rotary Position Embeddings (LLaMA, Mistral, GPT-NeoX)

The current standard. Instead of ADDING position info to embeddings, RoPE ROTATES the Q and K vectors based on position.

```text
Core idea:
    Multiply Q and K by a rotation matrix that depends on position.
    The rotation angle increases with position.

    At position 0: rotate by 0°
    At position 1: rotate by θ
    At position 2: rotate by 2θ
    At position 3: rotate by 3θ
    ...

    When computing attention (Q · K), the dot product between
    Q at position m and K at position n depends on (m - n),
    not on m and n individually.

    → The model learns RELATIVE positions, not absolute positions.
```

**Why relative positions matter:**

```text
"The cat sat on the mat"

Absolute positions:
    "cat" is at position 1, "sat" is at position 2.
    If you move the sentence: "I said the cat sat on the mat"
    now "cat" is at position 3, "sat" is at position 4.
    The relationship changed even though nothing meaningful changed.

Relative positions:
    "sat" is 1 position after "cat" — always.
    Regardless of where in the sentence they appear.
    The model only needs to learn "1 position apart = verb after noun"
    instead of memorising every possible absolute position combination.
```

**How the rotation works (simplified to 2D):**

```text
Take a 2D vector [x, y] at position m.
Rotate by angle m × θ:

    [x', y'] = [x·cos(mθ) - y·sin(mθ),  x·sin(mθ) + y·cos(mθ)]

For Q at position m and K at position n:
    Q_rotated · K_rotated depends on (m - n) × θ
    → relative position encoded automatically

In practice: the 768-d or 4096-d vector is split into pairs of dimensions,
and each pair is rotated independently at a different frequency.
Same idea as sinusoidal encoding, but applied multiplicatively to Q,K
instead of additively to embeddings.
```

**Why RoPE enables long contexts:**

```text
Learned positions: 1024 positions in the table → hard wall at 1024.
RoPE: rotation formula works for ANY position number.
    Position 100,000? Just rotate by 100,000 × θ.

In practice, models are trained on a certain max length (e.g., 4096).
Extending to longer sequences requires adjusting the base frequency:
    - "NTK-aware" scaling: change the 10000 base to a larger number
    - "YaRN": combines multiple scaling strategies

    LLaMA 2 trained on 4K → extended to 32K with position scaling
    LLaMA 3 trained on 8K → extended to 128K
```

---

## 3. Multi-Query and Grouped-Query Attention (MQA / GQA)

### The Problem

Standard multi-head attention has separate K, V, and Q for each head:

```text
Standard Multi-Head Attention (MHA):
    12 heads → 12 Q matrices, 12 K matrices, 12 V matrices
    KV cache stores: 12 × 2 = 24 sets of cached vectors per layer

    For LLaMA 70B (80 layers, 64 heads per layer):
    KV cache = 80 × 64 × 2 × head_dim × seq_len × 2 bytes
    At 4096 tokens: ~10.7 GB

    For 128K context: ~333 GB ← doesn't fit on most GPUs
```

### Multi-Query Attention (MQA)

```text
Idea: ALL heads share ONE set of K and V. Only Q is per-head.

Standard (MHA):     12 heads × {Q, K, V}  = 36 projections
Multi-Query (MQA):  12 heads × {Q} + 1×{K, V} = 14 projections

KV cache shrinks by 12× (from 12 K,V sets to 1)

    12 Q heads, each asking different questions
    All heads look at the SAME K and V
    Different heads still attend to different things
    (because Q differs, so the attention weights differ)

Used by: PaLM, Falcon
Downside: slight quality drop — with only 1 K,V set,
          the model has less capacity to represent different views.
```

### Grouped-Query Attention (GQA)

```text
The compromise between MHA and MQA:

MHA:  12 Q heads, 12 K heads, 12 V heads     (full, expensive)
GQA:  12 Q heads, 4 K heads, 4 V heads       (4 groups of 3 Q heads)
MQA:  12 Q heads, 1 K head, 1 V head         (minimal, cheapest)

GQA with 4 groups:
    Q heads 0,1,2  share K₀, V₀
    Q heads 3,4,5  share K₁, V₁
    Q heads 6,7,8  share K₂, V₂
    Q heads 9,10,11 share K₃, V₃

    KV cache shrinks by 3× (from 12 to 4 sets)
    Quality nearly matches full MHA
    Memory is much more manageable for long contexts

Used by: LLaMA 2 (70B), LLaMA 3, Mistral, Gemma
```

```text
Summary:
| Method | KV heads | KV cache size | Quality | Used by          |
| ------ | -------- | ------------- | ------- | ---------------- |
| MHA    | 12       | 1×            | Best    | GPT-2, BERT      |
| GQA    | 4        | 0.33×         | ~Same   | LLaMA 2/3, Mistral|
| MQA    | 1        | 0.08×         | Slight↓ | PaLM, Falcon     |
```

---

## 4. Context Window and Attention Patterns

### What the Context Window Is

```text
The context window is the maximum number of tokens the model can
process at once. Everything the model "knows" about the current
conversation must fit within this window.

    GPT-2:     1,024 tokens    (~750 words)
    GPT-3:     2,048 tokens    (~1,500 words)
    GPT-3.5:   4,096 tokens    (~3,000 words)
    GPT-4:     8,192 / 32K / 128K tokens
    Claude 3:  200K tokens     (~150,000 words / ~500 pages)
    Gemini:    1M+ tokens

What counts toward the window:
    System prompt + conversation history + your message + the response
    ALL of this must fit within the context window.

When you exceed it:
    Older messages get dropped (truncated from the beginning).
    The model literally can't see them anymore.
```

### Why Attention Is O(n²)

```text
Every token attends to every other token:
    n tokens → n × n attention scores to compute

    1K tokens:   1M attention operations      (fast)
    4K tokens:   16M attention operations     (fine)
    32K tokens:  1B attention operations      (getting slow)
    128K tokens: 16B attention operations     (expensive)
    1M tokens:   1T attention operations      (very expensive)

Memory is also O(n²): the full attention matrix must be stored.
For 128K tokens with 768-d: attention matrix = 128K × 128K × 4 bytes = 64 GB
(just for ONE layer, ONE head!)
```

### Sliding Window Attention (Mistral)

```text
Instead of every token attending to ALL previous tokens,
each token only attends to the last W tokens:

    W = 4096 (window size)

    Token at position 5000 attends to positions 904-5000
    (not positions 0-903 — they're outside the window)

Full attention:           Sliding window:
    ■ ■ ■ ■ ■ ■ ■            ■
    ■ ■ ■ ■ ■ ■ ■            ■ ■
    ■ ■ ■ ■ ■ ■ ■            ■ ■ ■
    ■ ■ ■ ■ ■ ■ ■            ■ ■ ■ ■
    ■ ■ ■ ■ ■ ■ ■              ■ ■ ■ ■
    ■ ■ ■ ■ ■ ■ ■                ■ ■ ■ ■
    ■ ■ ■ ■ ■ ■ ■                  ■ ■ ■ ■
    O(n²) attention           O(n × W) attention

But wait — doesn't this lose long-range information?
Not entirely: information propagates through layers.
Layer 1: token sees 4096 neighbours.
Layer 2: token sees 4096 neighbours, each of which already saw
         their own 4096 neighbours at layer 1.
After 32 layers: effective receptive field = 32 × 4096 = 131K tokens.

Used by: Mistral 7B
Benefit: O(n × W) attention instead of O(n²). Much faster for long sequences.
```

---

## 5. Mixture of Experts (MoE)

### The Problem

Bigger models are better (scaling laws), but they're also proportionally more expensive to run. A 70B model does 70B operations per token.

```text
What if most of the model's parameters existed but only a FRACTION
activated for each token? You'd get the knowledge capacity of a
large model with the compute cost of a small one.
```

### How MoE Works

Replace the single FFN in each transformer block with **multiple "expert" FFNs** and a **router** that picks which ones to use:

```text
Standard transformer block:
    Attention → FFN → output
    Every token goes through the SAME FFN.
    All parameters used for every token.

MoE transformer block:
    Attention → Router → {Expert₀, Expert₁, Expert₂, ..., Expert₇} → output
    The router picks 2 out of 8 experts for each token.
    Different tokens may use different experts.
    Only 2/8 = 25% of FFN parameters activated per token.
```

**Toy example:**

```text
8 experts, router picks top-2 per token.

Input: "The cat sat on the mat"

    "The"  → router scores: [0.8, 0.1, 0.5, 0.2, 0.1, 0.3, 0.1, 0.2]
              top-2: Expert 0 (0.8), Expert 2 (0.5)
              output = 0.62 × Expert₀("The") + 0.38 × Expert₂("The")

    "cat"  → router scores: [0.2, 0.7, 0.1, 0.6, 0.1, 0.1, 0.3, 0.1]
              top-2: Expert 1 (0.7), Expert 3 (0.6)
              output = 0.54 × Expert₁("cat") + 0.46 × Expert₃("cat")

    "sat"  → router scores: [0.1, 0.1, 0.3, 0.1, 0.8, 0.2, 0.1, 0.6]
              top-2: Expert 4 (0.8), Expert 7 (0.6)
              output = 0.57 × Expert₄("sat") + 0.43 × Expert₇("sat")

Different tokens use different experts!
Nouns might consistently route to experts 1,3.
Verbs might consistently route to experts 4,7.
Each expert can specialise.
```

### The Router

```text
The router is a small linear layer:

    router_scores = softmax(hidden × W_router)

    W_router shape: (d_model × n_experts) = (4096 × 8)
    Output: 8 scores, one per expert.
    Pick top-k (usually k=2).

    The router is LEARNED during training — it discovers which
    expert should handle which type of token.

Load balancing:
    Problem: the router might send all tokens to Expert 0
    (because it learned Expert 0 is "best") → other experts wasted.

    Fix: auxiliary load-balancing loss
    Penalise the model if expert usage is uneven.
    This encourages spreading tokens across all experts.
```

### MoE Dimensions

```text
Mixtral 8x7B (Mistral):
    Total parameters:    46.7B (8 experts × ~5.6B FFN params + shared attention)
    Active parameters:   12.9B (2 experts active per token)
    Performance:         ≈ LLaMA 2 70B quality
    Inference cost:      ≈ 13B model (only 2 experts compute)

    You get 70B-quality output for 13B-level compute cost.

GPT-4 (rumoured):
    Likely a very large MoE model.
    Total params: ~1.8T (rumoured)
    Active params: ~200-300B per token
    This would explain how it's so capable yet runs at
    practical speeds.

DeepSeek V3:
    Total: 671B parameters
    Active: 37B per token
    256 experts, top-8 routing
```

### Dense vs MoE Trade-offs

```text
| Aspect            | Dense (LLaMA)          | MoE (Mixtral)           |
| ----------------- | ---------------------- | ----------------------- |
| Params used/token | 100% (all params)      | 25-35% (top-k experts)  |
| Total params      | Smaller (7B, 70B)      | Larger (47B, 671B)      |
| Inference speed   | Predictable            | Faster per token        |
| Memory            | Just model weights     | ALL experts must be loaded|
|                   |                        | (even though only 2 used)|
| Training          | Straightforward        | Load balancing is tricky |
| Quality           | Strong at its size     | Matches larger dense    |

Key gotcha: MoE needs ALL parameters in memory, even though
only a fraction activate. A 47B MoE model needs as much GPU memory
as a 47B dense model, even though it computes like a 13B model.
```

---

## 6. Attention Sinks

A subtle phenomenon discovered in LLMs:

```text
Observation: the first token (position 0) gets disproportionately
high attention weight from ALL other tokens, across ALL layers.

    Attention weights for token at position 500:
    pos 0:   0.15   ← abnormally high (this is just "The" or <BOS>!)
    pos 1:   0.02
    pos 2:   0.03
    ...
    pos 498: 0.04
    pos 499: 0.08

The first token acts as an "attention sink" — a dumping ground
for attention weight that isn't needed elsewhere.

Why it happens:
    Softmax must assign weights that sum to 1.
    Sometimes no token in the context is truly relevant.
    But softmax CAN'T output all zeros.
    The model learns to dump "leftover" attention on position 0.
    Position 0 is always present, always in the same place — easy target.

Why it matters:
    If you implement sliding window attention and drop position 0,
    the model can break (no attention sink available).
    Fix: always keep the first few tokens in the window.

    StreamingLLM (MIT, 2023): keep the first 4 tokens + the recent
    window → stable generation for arbitrarily long sequences.
```

---

## 7. Flash Attention (Preview — covered more in file 08)

```text
Standard attention:
    1. Compute Q·Kᵀ → full n×n matrix in GPU memory
    2. Apply softmax
    3. Multiply by V

    Problem: the n×n matrix must be stored in HBM (GPU main memory).
    For n=128K: 128K × 128K × 2 bytes = 32 GB. Per layer. Per head.

Flash Attention:
    Never materialise the full n×n matrix.
    Compute attention in tiles (blocks), streaming through GPU SRAM
    (much faster but much smaller memory).

    Result: exact same output, but ~2-4× faster and uses O(n) memory
    instead of O(n²).

    This is what makes 128K+ context windows practical.
    Used by virtually every modern LLM framework.
```

---

## Summary: Which Models Use What

```text
| Concept        | What it solves              | Used by                    |
| -------------- | --------------------------- | -------------------------- |
| KV Cache       | Avoid recomputing past K,V  | Everything (standard)      |
| Learned pos    | Token ordering              | BERT, GPT-2                |
| RoPE           | Relative positions, extend  | LLaMA, Mistral, Gemma      |
| MHA            | Full per-head K,V           | BERT, GPT-2/3              |
| GQA            | Reduce KV cache size        | LLaMA 2/3, Mistral         |
| MQA            | Minimal KV cache            | PaLM, Falcon               |
| Sliding window | O(n×W) instead of O(n²)     | Mistral                    |
| MoE            | Large capacity, low compute | Mixtral, DeepSeek, GPT-4?  |
| Attention sinks| Stable long generation      | StreamingLLM               |
| Flash Attention| Fast, memory-efficient attn | Everything modern          |
```
