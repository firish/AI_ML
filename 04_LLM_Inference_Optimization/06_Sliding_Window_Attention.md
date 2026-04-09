## Key Concepts: Sliding Window Attention

---

## 0. Recap — The Cost of Full Attention

### What full (causal) attention does

```text
In standard causal attention, every token attends to ALL previous tokens:

    Token 0: attends to [0]
    Token 1: attends to [0, 1]
    Token 2: attends to [0, 1, 2]
    ...
    Token n: attends to [0, 1, 2, ..., n]

The attention matrix is lower-triangular (causal mask):

    Position:  0  1  2  3  4  5  6  7
    Token 0:  [✓  .  .  .  .  .  .  .]
    Token 1:  [✓  ✓  .  .  .  .  .  .]
    Token 2:  [✓  ✓  ✓  .  .  .  .  .]
    Token 3:  [✓  ✓  ✓  ✓  .  .  .  .]
    Token 4:  [✓  ✓  ✓  ✓  ✓  .  .  .]
    Token 5:  [✓  ✓  ✓  ✓  ✓  ✓  .  .]
    Token 6:  [✓  ✓  ✓  ✓  ✓  ✓  ✓  .]
    Token 7:  [✓  ✓  ✓  ✓  ✓  ✓  ✓  ✓]

    ✓ = attends to    . = masked out (can't look ahead)
```

### The costs

```text
Compute: O(n²) — every token attends to all previous tokens.
KV cache: O(n) — stores K, V for every token, grows linearly.

For long contexts, the KV cache is the practical bottleneck:

    LLaMA 2 7B, 32K context:
        KV cache = 2 × 32 layers × 4096 dims × 32K tokens × 2 bytes
                 = 16.8 GB  ← just the cache, on top of model weights

    LLaMA 2 7B, 128K context:
        KV cache = 67.2 GB  ← exceeds most single-GPU memory

The model weights are fixed, but the KV cache scales with context length.
For serving many users concurrently, each user needs their own KV cache.
This is what limits batch size at long contexts.
```

---

## 1. The Observation: Most Attention Is Local

### Attention patterns in practice

```text
If you visualize learned attention weights in a trained LLM,
a consistent pattern emerges:

    Most attention weight concentrates on NEARBY tokens.
    Token 500 attends heavily to tokens 495-500.
    It attends weakly to token 50, and barely at all to token 5.

This makes linguistic sense:
    "The cat sat on the mat and then it fell asleep"
                                     ^^
    "it" mainly needs to attend to nearby context ("mat", "then", "fell")
    to figure out what comes next. It rarely needs token 0 directly.

Not always true — some heads attend to distant tokens (retrieval heads).
But the MAJORITY of attention is local.
```

### The idea

```text
If most attention is local, why compute and cache attention for ALL past tokens?

    Full attention at position 10000:
        Attends to tokens [0, 1, 2, ..., 10000]
        KV cache holds 10001 entries
        Most attention weight is on tokens [9900, ..., 10000]
        Tokens [0, ..., 100] get ~0.01% of total attention weight

    What if we just... don't attend to tokens beyond some window?
```

---

## 2. Sliding Window Attention

### How it works

```text
Each token only attends to the W most recent tokens (window size W):

    Window size W = 4:

    Position:  0  1  2  3  4  5  6  7
    Token 0:  [✓  .  .  .  .  .  .  .]
    Token 1:  [✓  ✓  .  .  .  .  .  .]
    Token 2:  [✓  ✓  ✓  .  .  .  .  .]
    Token 3:  [✓  ✓  ✓  ✓  .  .  .  .]
    Token 4:  [.  ✓  ✓  ✓  ✓  .  .  .]  ← token 0 drops out
    Token 5:  [.  .  ✓  ✓  ✓  ✓  .  .]  ← token 1 drops out
    Token 6:  [.  .  .  ✓  ✓  ✓  ✓  .]
    Token 7:  [.  .  .  .  ✓  ✓  ✓  ✓]

    Compare to full attention — the lower-left triangle is gone.
    Each row has at most W checkmarks instead of growing to n.
```

### What this buys us

```text
                    Full Attention          Sliding Window (W)
Attention compute   O(n²)                   O(n × W)
KV cache size       O(n) — grows forever    O(W) — fixed cap
KV cache memory     proportional to n       constant after W tokens

Mistral 7B uses W = 4096:
    Full attention at 32K context: KV cache for 32K tokens
    Sliding window at 32K context: KV cache for 4096 tokens only

    The cache is 8× smaller. Same model, same quality for most tasks.
    And you can now serve 8× more concurrent users with the same GPU memory.
```

### The KV cache becomes a ring buffer

```text
With sliding window, the KV cache doesn't grow beyond W entries.
Old entries are overwritten — the cache works like a circular buffer:

    Window size W = 4, cache has 4 slots:

    After processing token 0:  cache = [K₀, __, __, __]
    After processing token 1:  cache = [K₀, K₁, __, __]
    After processing token 2:  cache = [K₀, K₁, K₂, __]
    After processing token 3:  cache = [K₀, K₁, K₂, K₃]   ← full
    After processing token 4:  cache = [K₄, K₁, K₂, K₃]   ← K₀ overwritten
    After processing token 5:  cache = [K₄, K₅, K₂, K₃]   ← K₁ overwritten
    After processing token 6:  cache = [K₄, K₅, K₆, K₃]
    After processing token 7:  cache = [K₄, K₅, K₆, K₇]

    Position in cache = token_position mod W
    Memory usage is constant regardless of sequence length.
```

---

## 3. But What About Long-Range Dependencies?

### The effective receptive field is larger than W

```text
Sliding window in ONE layer: token 8 can only see tokens [5, 6, 7, 8].
But the model has multiple layers, and each layer's output feeds into the next.

    Layer 0: token 8 attends to [5, 6, 7, 8]
    Layer 1: token 8 attends to [5, 6, 7, 8],
             but token 5 already contains info from tokens [2, 3, 4, 5] (from layer 0)

    So at layer 1, token 8 has INDIRECT access to tokens [2, 3, 4, 5]
    through token 5's representation.

Effective receptive field after L layers: L × W tokens

    Mistral 7B: 32 layers × 4096 window = 131,072 tokens
    The model can theoretically propagate information across 128K tokens,
    even though each layer only looks at 4096.

This is like a chain of people whispering:
    Person 8 can only hear person 5-8 directly.
    But person 5 heard from person 2-5.
    So person 8 can get information from person 2, via person 5.
    Each hop covers W positions. L hops cover L × W.
```

### The caveat: information degrades with hops

```text
Direct attention: token 8 attends to token 7 → strong, precise signal.
One hop away:     token 8 → token 5 → token 2 → indirect, lossy.
Two hops away:    even more diluted.

Sliding window preserves local relationships perfectly.
Long-range relationships are possible but weaker.

For most language tasks, this is fine — most dependencies ARE local.
For tasks requiring precise recall of distant information (e.g., "what was
the first word of this document?"), sliding window struggles.
```

---

## 4. Attention Sinks — Why the First Few Tokens Matter

### The problem with pure sliding window

```text
When researchers tested pure sliding window attention on long sequences,
they found quality dropped significantly when the FIRST few tokens
fell out of the window.

    Prompt: "Summarize this article: [5000 tokens of article]"
    Window size: 4096

    At token 5000: the window covers tokens [904, ..., 5000].
    Token 0 ("Summarize") and the first few tokens have been evicted.
    Quality drops noticeably — the model "forgets" the instruction.

This was surprising. Those early tokens get very little attention weight
in a normal forward pass. Why does evicting them hurt?
```

### Attention sinks (Xiao et al., 2023)

```text
The finding: in trained LLMs, the FIRST token (or first few tokens)
consistently receives a disproportionate amount of attention weight
across ALL layers, regardless of its content.

    Typical attention distribution for token 5000:
        Token 0:          8% of attention   ← "attention sink"
        Tokens 1-900:     ~0% each
        Tokens 4990-5000: 85% of attention  ← nearby tokens (expected)
        Tokens 901-4989:  ~7% spread thin

    Token 0 acts as a "sink" — a dumping ground for attention weight.

Why this happens:
    Softmax must distribute 100% of attention across all attended tokens.
    Sometimes a token doesn't need information from any specific position.
    But softmax can't output all zeros — it must attend SOMEWHERE.
    The model learns to dump "unused" attention on token 0.
    It's a learned no-op: "attend here when you have nothing better to attend to."

    If token 0 is evicted from the window, the model has no sink.
    The unused attention redistributes to other tokens randomly.
    This corrupts the output.
```

### The fix: sliding window + attention sinks

```text
Keep the first few tokens (typically 1-4) in the cache permanently,
plus the sliding window of recent tokens:

    Window size W = 4096, sink tokens = 4

    Cache layout:
        [K₀, K₁, K₂, K₃,  K_{n-4092}, K_{n-4091}, ..., K_{n}]
         ↑ always kept        ↑ sliding window of most recent 4092

    Token n attends to: {0, 1, 2, 3} ∪ {n-4092, ..., n}

    Total cache size: still ~W entries. Constant memory.
    But the sink tokens are never evicted.

This recovers nearly all of the quality of full attention
while keeping the fixed-size cache benefit.

Mistral 7B uses this approach: sliding window with sink tokens.
```

---

## 5. Sliding Window in Multi-Layer Architectures

### Not every layer needs the same strategy

```text
Some models (Gemma 2, Jamba) use a hybrid approach:

    Layer 0:  sliding window (W = 4096)
    Layer 1:  FULL attention
    Layer 2:  sliding window (W = 4096)
    Layer 3:  FULL attention
    ...

Alternating between local (sliding window) and global (full attention).

Why:
    Local layers handle syntax, nearby dependencies efficiently.
    Global layers handle long-range retrieval, instruction following.
    You get most of the KV cache savings (half the layers have fixed cache)
    without losing long-range capability.

Jamba (AI21, 2024) goes further:
    Some layers use attention (sliding window).
    Other layers use Mamba (a state-space model — no attention at all).
    SSM layers have O(1) cache per token — even cheaper than sliding window.
```

---

## Summary

```text
Full attention: every token attends to all previous tokens.
    KV cache grows linearly with context → memory bottleneck for long contexts.

Sliding window: each token attends to only the W most recent tokens.
    KV cache is fixed at W entries (ring buffer) → constant memory.
    Compute: O(n × W) instead of O(n²).
    Effective receptive field: L × W across layers (info propagates through hops).
    Trade-off: long-range dependencies are indirect and weaker.

Attention sinks: first few tokens act as learned no-ops for unused attention.
    Pure sliding window breaks when these are evicted.
    Fix: permanently keep first 1-4 tokens in cache alongside the window.

Used by: Mistral 7B (W=4096), Gemma 2 (hybrid sliding + full),
         Jamba (hybrid sliding + SSM).
```
