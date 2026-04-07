## Key Concepts: KV Cache

---

## 0. Recap — What Happens at Inference, Token by Token

Before understanding KV cache, you need to see exactly what the transformer does when generating each token of a response.

### The setup

```text
User sends: "What is the capital of France?"

The model will generate tokens one at a time:
    "The" → "capital" → "of" → "France" → "is" → "Paris" → "."

Let's trace what happens for EACH generated token.
```

### Generating the first token: "The"

```text
Input: the full prompt "What is the capital of France?" (8 tokens)

Step 1: Embed all 8 tokens
    Each token → 768-dim vector (from embedding table)
    Result: (8, 768)

Steps 2-4 happen ONCE PER LAYER (12 to 96 layers depending on model).
Each layer has its OWN W_q, W_k, W_v — so Q, K, V are recomputed
at every layer with different learned weights.

--- Layer 0 ---
Step 2: Compute Q, K, V for all 8 tokens
    Q = embedding × W_q₀     → one Q per token
    K = embedding × W_k₀     → one K per token
    V = embedding × W_v₀     → one V per token
    All 8 tokens processed IN PARALLEL (one matrix multiply).

Step 3: Attention
    scores = Q × Kᵀ / √d_k       (8, 8) — every token attends to every other
    weights = softmax(scores)       (8, 8) — with causal mask (can't look ahead)
    output = weights × V            (8, 768)

    What is √d_k?
        d_model = 768, split into 12 heads → d_k = 768/12 = 64 per head
        √d_k = √64 = 8

        We divide by 8 to keep dot products from getting too large.
        Without scaling: dot products grow with dimension count (~64).
        Large dot products → softmax becomes a hard argmax →
        model attends to only ONE token, ignores everything else.
        Dividing by √d_k keeps softmax in a useful range where
        attention can spread across multiple tokens.

Step 4: Feed through FFN → output of layer 0

--- Layer 1 ---
    Take output of layer 0 as input.
    Compute NEW Q, K, V using layer 1's own W_q₁, W_k₁, W_v₁
    Attention → FFN → output of layer 1

--- Layer 2 through N ---
    Same process. Each layer's W_q, W_k, W_v are different
    learned matrices, so each layer attends to different things:
        Early layers: syntax, word sense
        Middle layers: semantic relationships
        Late layers: high-level reasoning

--- After final layer ---
Step 5: Take the LAST token's output → predict next token
    The last position's hidden state → LM head → "The"

This entire forward pass happens ONCE for the whole prompt.
All 8 tokens computed in parallel. This is the "prefill" phase.
```

### Generating the second token: "capital"

```text
Input: just the NEW token "The" (1 token)

Step 1: Embed "The" → (1, 768)

Step 2: Compute Q, K, V for "The" ONLY
    Q_new = (1, 768) × W_q (768, 768) = (1, 768)
    K_new = (1, 768) × W_k (768, 768) = (1, 768)
    V_new = (1, 768) × W_v (768, 768) = (1, 768)

Step 3: Append to KV cache
    K_cached was (8, 768) from prefill.
    K_all = concat(K_cached, K_new) = (9, 768)     ← cache grows by 1 row
    V_all = concat(V_cached, V_new) = (9, 768)

Step 4: Attention (with shapes)
    scores  = Q_new × K_allᵀ  = (1, 768) × (768, 9) = (1, 9)
                                  ↑ one score per past token
    weights = softmax(scores / √d_k) = (1, 9)
                                  ↑ how much to attend to each of 9 tokens
    output  = weights × V_all = (1, 9) × (9, 768) = (1, 768)
                                  ↑ weighted blend of all 9 value vectors

Step 5: output (1, 768) → LM head → predict "capital"
```

**The data flow at each layer:**
```text
                    New token "The"
                         │
                    embed (1, 768)
                         │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
           Q_new      K_new      V_new
          (1,768)    (1,768)    (1,768)
              │          │          │
              │     ┌────┴────┐ ┌───┴────┐
              │     │  CACHE  │ │ CACHE  │
              │     │ K_old   │ │ V_old  │
              │     │(8,768)  │ │(8,768) │
              │     │+K_new   │ │+V_new  │
              │     │=(9,768) │ │=(9,768)│
              │     └────┬────┘ └───┬────┘
              │          │          │
              ↓          ↓          │
           Q × Kᵀ = (1,9) scores   │
              │                     │
           softmax → (1,9) weights  │
              │                     │
              └─── weights × V ─────┘
                        │
                   output (1,768)
```

**This happens at EVERY layer, each with its own cache:**
```text
Layer 0:  K_cache (8→9, 768),  V_cache (8→9, 768)   ← grows by 1
Layer 1:  K_cache (8→9, 768),  V_cache (8→9, 768)   ← grows by 1
...
Layer 11: K_cache (8→9, 768),  V_cache (8→9, 768)   ← grows by 1
```

### Generating the third token: "of"

```text
Input: just "capital" (1 token) → embed → (1, 768)

Compute Q, K, V for "capital" only → each (1, 768)
Append K, V to cache → cache now has 10 entries per layer

scores  = Q_new × K_allᵀ = (1, 768) × (768, 10) = (1, 10)
weights = softmax(scores / √d_k) = (1, 10)
output  = weights × V_all = (1, 10) × (10, 768) = (1, 768)

Output → predict "of"
```

### The pattern

```text
Token 1 ("The"):      Q(1,768) × K_allᵀ(768, 9) → scores(1, 9)  → output(1,768)
Token 2 ("capital"):   Q(1,768) × K_allᵀ(768,10) → scores(1,10)  → output(1,768)
Token 3 ("of"):        Q(1,768) × K_allᵀ(768,11) → scores(1,11)  → output(1,768)
Token 4 ("France"):    Q(1,768) × K_allᵀ(768,12) → scores(1,12)  → output(1,768)
Token 5 ("is"):        Q(1,768) × K_allᵀ(768,13) → scores(1,13)  → output(1,768)
Token 6 ("Paris"):     Q(1,768) × K_allᵀ(768,14) → scores(1,14)  → output(1,768)
Token 7 ("."):         Q(1,768) × K_allᵀ(768,15) → scores(1,15)  → output(1,768)

Cache grows by 1 row per token, per layer. Output is always (1, 768).

Each step: compute Q, K, V for ONE token.
    Q is used immediately for attention, then discarded.
    K and V are appended to the cache for all future tokens to use.

Without cache: recompute K, V for ALL tokens at every step → O(n²)
With cache: compute K, V for 1 token, reuse all previous → O(n)
```

This is exactly what the KV cache is. Let's look at it in detail.

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
