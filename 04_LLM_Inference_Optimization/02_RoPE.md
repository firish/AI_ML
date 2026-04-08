## Key Concepts: RoPE — Rotary Position Embeddings

---

## 0. Recap — Why Transformers Need Position Information

Attention is permutation-invariant. If you shuffle the tokens in the input, the model produces the exact same attention scores — just rearranged. Order is invisible.

```text
Sentence: "The cat sat on the mat"
Tokens:   ["The", "cat", "sat", "on", "the", "mat"]

Attention computes:
    score(i, j) = Q_i · K_j / √d_k

This is just a dot product between vectors.
Dot products don't know about order.

Proof: shuffle the tokens → ["mat", "sat", "The", "on", "cat", "the"]
    The same dot products get computed, just in a different arrangement.
    The model produces the same weighted combinations.
    "The cat sat on the mat" and "mat sat The on cat the" look IDENTICAL
    to attention.
```

This is a problem. "Dog bites man" and "Man bites dog" are very different sentences. Without position information, the model can't distinguish them.

```text
"Dog bites man":   ["Dog", "bites", "man"]   ← Dog is the subject
"Man bites dog":   ["Man", "bites", "dog"]   ← Man is the subject

Without positions: both produce the same set of Q, K, V vectors.
The attention scores between "Dog"/"Man" and "bites" are identical.
The model predicts the same output for both sentences.
```

**The fix:** inject position information into the token representations BEFORE attention runs.

---

## 1. Positional Embeddings — The Naive Fix

### What they are

A positional embedding is a vector added to each token's embedding. Its value depends on the token's position in the sequence, not its identity.

```text
Token embedding:       embed("cat")  = [0.2, -0.1, 0.5, ...]   (768-d)
Positional embedding:  PE(position=1) = [0.01, 0.03, -0.02, ...]  (768-d)

Input to transformer:  [0.2, -0.1, 0.5, ...] + [0.01, 0.03, -0.02, ...]
                     = [0.21, -0.07, 0.48, ...]

Now the same word at different positions produces different vectors.
"cat" at position 1 ≠ "cat" at position 5.
Attention CAN now distinguish them.
```

### When they are added (training and inference)

The same operation happens in both phases — positional embeddings are added immediately after token embedding, before the first transformer layer:

```text
TRAINING:
    for each training example (e.g., "The cat sat"):
        tokens = tokenise("The cat sat")     → [1532, 3857, 9241]
        embeds = embedding_table[tokens]      → (3, 768)
        pos_embeds = PE([0, 1, 2])            → (3, 768)
        x = embeds + pos_embeds               → (3, 768)  ← input to Layer 0
        ... forward pass, compute loss, backprop ...

INFERENCE:
    tokens = tokenise("The cat sat")          → [1532, 3857, 9241]
    embeds = embedding_table[tokens]          → (3, 768)
    pos_embeds = PE([0, 1, 2])                → (3, 768)
    x = embeds + pos_embeds                   → (3, 768)  ← input to Layer 0
    ... forward pass, take last token output, predict next ...

Identical operation. The positional embedding is always applied.
During decode (generating one token at a time), the new token gets
its position number and the corresponding PE is added.
```

### Generation 1: Fixed Sinusoidal (Original Transformer, 2017)

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

pos: position in the sequence (0, 1, 2, ...)
i:   dimension index (0, 1, 2, ..., d/2)
d:   total embedding size (e.g., 512 or 768)

Example for d=8, position=0 and position=1:
    PE(0) = [sin(0/1),    cos(0/1),    sin(0/100),  cos(0/100),
             sin(0/1000), cos(0/1000), sin(0/10000),cos(0/10000)]
           = [0,           1,           0,            1,
              0,           1,           0,            1]

    PE(1) = [sin(1/1),    cos(1/1),    sin(1/100),  cos(1/100),
             sin(1/1000), cos(1/1000), sin(1/10000),cos(1/10000)]
           = [0.841,       0.540,       0.010,        1.000,
              0.001,       1.000,       0.0001,        1.000]

The low-indexed dimensions oscillate quickly (distinguish nearby tokens).
The high-indexed dimensions oscillate slowly (distinguish far-apart tokens).
Together, each position gets a unique fingerprint.
```

No parameters. No learning. Just a formula. The same formula is used at both training and inference — even for positions the model has never seen.

**The catch:** while the formula works for any position number, the model's weights are trained only on the PATTERNS of PE values it saw. Positions far beyond the training distribution produce PE values that are unfamiliar combinations, degrading quality.

---

## 2. Generation 2: Learned Positional Embeddings (BERT, GPT-2)

### What changes

Instead of a fixed formula, you learn a lookup table during training:

```text
Learned PE table (GPT-2, trained with max_seq_len=1024):
    position 0    → [0.012, -0.034, 0.071, ...]    ← 768-dim vector
    position 1    → [0.028,  0.015, -0.043, ...]   ← 768-dim vector
    position 2    → [-0.007, 0.052, 0.038, ...]    ← 768-dim vector
    ...
    position 1023 → [0.041, -0.018, 0.063, ...]    ← 768-dim vector

Table size: 1024 × 768 = ~786K parameters.
These are initialized randomly and learned during training like any other weights.
```

### How training makes these useful

The model learns position embeddings by gradient descent, just like everything else:

```text
Training step:
    - Input: "The cat sat on the mat"   (positions 0-5)
    - Add PE[0..5] to token embeddings
    - Forward pass → compute loss
    - Backprop: gradient flows back through the PE table
    - PE[0] gets updated: "being at the start of a sentence" matters like this
    - PE[1] gets updated: "being second" matters differently

After millions of steps, each PE vector encodes what it means
to be at that position in context — not a formula, but a learned pattern.
```

### Why learned beats sinusoidal (in practice)

```text
Sinusoidal:
    The formula determines the vectors completely.
    Works "correctly" by design, but isn't optimized for the task.

Learned:
    The vectors are optimized to minimize loss.
    The model discovers what position information is ACTUALLY useful.
    Empirically: slightly better downstream task performance.
    The model isn't forced to use sinusoidal patterns — it finds its own.
```

### The hard limit problem

```text
GPT-2 table: 1024 rows. That's it.

Input: 1000-token sequence → positions 0-999 → all in table → works.
Input: 1025-token sequence → positions 0-1024 → position 1024 is missing.

What happens?
    Option 1: Index out of bounds → crash.
    Option 2: Clamp to max position (1023) → all tokens beyond 1023
              get PE[1023]. The model can't distinguish their positions.

Either way: broken. The model was never trained on position 1024.
It has no idea what it means to be at position 1024.
```

This is the fundamental problem learned positional embeddings introduce: they work great within training length, and fail completely outside it.

---

## 3. The Core Problem: Positions Are Absolute

Both sinusoidal and learned PE embed ABSOLUTE positions. This creates two issues:

### Issue 1: Hard length limits (just described)

### Issue 2: The subject position problem

```text
Consider a model trying to learn: "the subject of a sentence determines
what verb should follow."

With absolute positions:
    Example A: "The cat sat on the mat"
        "cat"  is at position 1 (absolute)
        "sat"  is at position 2 (absolute)
        Relationship: position 2 describes position 1 → learned for positions 1,2

    Example B: "I said the cat sat on the mat"
        "cat"  is at position 3 (absolute)
        "sat"  is at position 4 (absolute)
        Same relationship, but now it's about positions 3,4

The model must learn the same relationship separately for every possible
pair of absolute positions. "Adjacent token relationship" is learned
for (0,1), (1,2), (2,3), ... independently.

With relative positions:
    "sat" is 1 token after "cat" — always.
    Doesn't matter where in the sentence they appear.
    The model learns ONE rule: "1 token apart → verb relationship."

Relative positions generalize much better with fewer examples.
```

---

## 4. RoPE — Rotary Position Embeddings

RoPE is the answer to both problems. Instead of adding position info to token embeddings, it ROTATES the Q and K vectors before computing attention.

### The core idea

```text
Standard approach (additive):
    token_input = embed(token) + PE(position)
    Q = token_input × W_q
    K = token_input × W_k
    score = Q · K

RoPE approach (rotational):
    token_input = embed(token)                    ← no position added here
    Q = token_input × W_q
    K = token_input × W_k
    Q_rotated = rotate(Q, position)               ← rotation applied here
    K_rotated = rotate(K, position)               ← rotation applied here
    score = Q_rotated · K_rotated
```

### The math (2D first, then general)

Take a 2D query vector Q = [x, y] at position m. Apply a rotation of angle m × θ:

```text
R(m) × [x, y] = [x·cos(mθ) - y·sin(mθ),
                  x·sin(mθ) + y·cos(mθ)]

Written as a matrix:
    R(m) = | cos(mθ)  -sin(mθ) |
            | sin(mθ)   cos(mθ) |

This is just a standard 2D rotation matrix parameterized by mθ.
```

Now compute the attention score between Q at position m and K at position n:

```text
score = Q_rotated(m) · K_rotated(n)
      = [R(m)·q]ᵀ · [R(n)·k]
      = qᵀ · R(m)ᵀ · R(n) · k
      = qᵀ · R(n - m) · k      ← because rotation matrices compose: R(m)ᵀR(n) = R(n-m)

The score depends only on (n - m): the RELATIVE position.
Not on m alone, not on n alone. Only the distance between them.
```

This is the key property. The model is FORCED to learn relative distances, not absolute positions.

### Extending to d dimensions

For a 768-dim vector, we can't do one big rotation. Instead, split into d/2 pairs and rotate each pair independently at a different frequency.

Important: the d/2 pairs are pairs of DIMENSIONS within one token's vector, not pairs of tokens. Every token gets all 384 pairs rotated. The relative position between any two tokens (1 and 3, or 1 and 100) comes from the Q^T · K dot product — not from pairing tokens together.

```text
Vector: [x₀, x₁, x₂, x₃, x₄, x₅, ..., x₇₆₆, x₇₆₇]
Split:  [(x₀,x₁), (x₂,x₃), (x₄,x₅), ..., (x₇₆₆,x₇₆₇)]
                                                ↑ 384 pairs of DIMENSIONS

Each pair (x_{2i}, x_{2i+1}) rotated by angle: position × θᵢ
```

The rotation matrix is block-diagonal — each pair is an independent 2D rotation:

```text
[R(mθ₀)   0       0     ...  0      ]
[0       R(mθ₁)   0     ...  0      ]
[0       0       R(mθ₂) ...  0      ]
[...                                 ]
[0       0       0     ... R(mθ₃₈₃) ]

Each R is a 2×2 rotation matrix. The pairs don't interact with each other.
This is NOT the same as one big 768-dim rotation (which would be a single
angle in a single plane). Each pair rotates at its own frequency.
```

How Q^T · K recovers relative position in d dimensions: each pair independently contributes cos((m-n)θᵢ) to the dot product, just like the 2D case. The full dot product sums all pairs:

```text
Q_m^T · K_n = Σᵢ (terms involving cos((m-n) × θᵢ) and original q,k values)

Every pair encodes the SAME position difference (m-n), just at a different frequency θᵢ.
```

### Why multiple frequencies (not just one)

A single frequency breaks down:

```text
Fast frequency only (θ = 1.0 for all pairs):
    cos(0 × 1.0) = 1.0    ← position difference = 0
    cos(6.28 × 1.0) ≈ 1.0  ← position difference ≈ 6.28
    Model thinks 0 apart and ~6 apart are the SAME. Cosine wraps every 2π.

Slow frequency only (θ = 0.001 for all pairs):
    cos(1 × 0.001) = 0.9999995
    cos(2 × 0.001) = 0.999998
    Model can barely tell position 1 apart from position 2.

Single frequency forces a tradeoff:
    Fast → good at nearby positions, blind to distant ones (wraps around)
    Slow → good at distant positions, blind to nearby ones (everything looks the same)
```

The solution: use DIFFERENT frequencies for different pairs (the clock analogy).

```text
A clock has three hands spinning at different speeds. Each alone is ambiguous:
    Second hand: can't tell 0:00:05 from 1:00:05 — wraps every minute
    Minute hand: can't tell 0:00 from 1:00 — wraps every hour
    Hour hand:   can't tell 3:00:00 from 3:00:15 — moves too slowly to notice

But all three together uniquely identify any time.

RoPE does the same with position:
    Pair 0   (θ=1.0):    "second hand" — separates position 5 from 6,
                          but 5 and 5+2π look the same
    Pair 100 (θ=0.22):   "minute hand" — separates 5 from 20,
                          but can't distinguish 5 from 6
    Pair 383 (θ=0.0001): "hour hand" — separates 5 from 5000,
                          but 5 vs 6 looks identical
```

Concrete proof that multiple frequencies resolve ambiguity:

```text
Single frequency (θ=1.0 for all pairs):
    diff=1:    cos(1.0)  = 0.54   in ALL pairs
    diff=7.28: cos(7.28) = 0.54   in ALL pairs
    → identical. Model is blind to this difference.

Add a second frequency (θ₁=0.3):
    diff=1:    pair0: cos(1.0)=0.54,   pair1: cos(0.3)=0.96
    diff=7.28: pair0: cos(7.28)=0.54,  pair1: cos(2.18)=-0.57
                                                ^^^^^^^^^^^
                                                NOW DIFFERENT

Each additional frequency eliminates more ambiguities, until 384 frequencies
can uniquely encode any position difference the model will ever see.
```

### Frequency schedule

```text
θᵢ = 10000^(-2i/d):
    i=0:   θ₀ = 10000^(0)    = 1.0         (fast rotation)
    i=1:   θ₁ = 10000^(-2/d) ≈ 0.95        (slightly slower)
    i=100: θ₁₀₀ = 10000^(-200/768) ≈ 0.22  (moderate)
    i=383: θ₃₈₃ = 10000^(-1)  = 0.0001     (very slow rotation)

Low-index pairs:  rotate fast → sensitive to nearby tokens (short-range)
High-index pairs: rotate slowly → sensitive to distant tokens (long-range)
```

This is the same intuition as sinusoidal embeddings, but applied multiplicatively to Q,K rather than additively to embeddings.

### What the rotation looks like concretely

```text
Token "cat" at position 3, d=8 (so 4 pairs):

Raw query (from W_q): q = [0.5, -0.3, 0.8, 0.1, -0.2, 0.6, 0.3, -0.4]

Frequencies: θ₀=1.0, θ₁=0.1, θ₂=0.01, θ₃=0.001

At position 3:
    Pair (q₀,q₁) = (0.5, -0.3)  rotated by 3×1.0 = 3.0 rad
    Pair (q₂,q₃) = (0.8,  0.1)  rotated by 3×0.1 = 0.3 rad
    Pair (q₄,q₅) = (-0.2, 0.6)  rotated by 3×0.01 = 0.03 rad
    Pair (q₆,q₇) = (0.3, -0.4)  rotated by 3×0.001 = 0.003 rad

    Pair 0 has rotated 3 full radians — highly sensitive to position.
    Pair 3 has rotated 0.003 radians — barely changed from position 0.

Same token "cat" at position 50:
    Pair 0 rotated by 50 rad — totally different from position 3.
    Pair 3 rotated by 0.05 rad — barely different from position 3.

How each dimension type sees position differences:

              Close tokens (diff=2)    Far tokens (diff=100)
Fast dims     very different            very different (but wraps → unreliable)
Slow dims     barely different          more different (no wrap → reliable)

Fast dims change a lot for ANY distance, but wrap unpredictably (aliasing).
Slow dims change monotonically — small absolute change, but reliable.
Together, every position difference gets a unique fingerprint.
```

---

## 5. How RoPE Works at Inference (with KV Cache)

This is where RoPE ties directly to inference optimization.

### Where in the forward pass RoPE is applied

```text
For each token at position p, in each attention layer:

    x = embed(token) + ...                   ← NO position added here
    q = x × W_q                              ← raw query, no position
    k = x × W_k                              ← raw key, no position
    v = x × W_v                              ← value, never gets position

    q_rope = RoPE(q, position=p)             ← rotate Q by p
    k_rope = RoPE(k, position=p)             ← rotate K by p

    score = q_rope · k_rope                  ← dot product of rotated vectors

Note: V is NEVER rotated. Only Q and K.
Position only needs to affect the attention SCORE, not the values
being mixed. The output weighted sum uses un-rotated V.
```

### During prefill (processing the prompt)

```text
Prompt: "What is the capital of France?"   (8 tokens, positions 0-7)

For each token at position p (all in parallel):
    embed → q, k, v  (each 768-d)
    q_rope[p] = RoPE(q, p)
    k_rope[p] = RoPE(k, p)

KV cache stores: k_rope[0..7], v[0..7]
    ↑ the ROTATED keys are stored. Position is baked into the cache.

Attention:
    score[p, p'] = q_rope[p] · k_rope[p'] = f(p - p')  for all pairs
    weights = softmax(scores / √d_k)
    output = weights × V
```

### During decode (generating one token at a time)

```text
Generating token at position 8 ("The"):

    embed("The") → x (1, 768)
    q = x × W_q   → (1, 768)
    k = x × W_k   → (1, 768)
    v = x × W_v   → (1, 768)

    q_rope = RoPE(q, position=8)
    k_rope = RoPE(k, position=8)

    Append k_rope to cache: cache now has k_rope[0..8]
    Append v to cache:      cache now has v[0..8]

    score = q_rope[8] · K_cache_rotated[0..8]ᵀ
          = (1, 768) × (768, 9) = (1, 9)

    score[i] = q_rope[8] · k_rope[i] = f(8 - i)  for i in 0..8
               ↑ only the distance from position 8 matters

    weights = softmax(scores / √d_k) = (1, 9)
    output  = weights × V_cache = (1, 9) × (9, 768) = (1, 768)
    → predict next token
```

### Data flow showing where RoPE sits

```text
New token "The" (position 8)
         │
    embed (1, 768)
         │
    ┌────┴────────────────┐
    ↓         ↓           ↓
 q=x×W_q  k=x×W_k    v=x×W_v
 (1,768)  (1,768)    (1,768)
    │         │           │
 RoPE(q,8) RoPE(k,8)    │     ← position baked in here
    │         │           │
    │    ┌────┴────┐  ┌───┴────┐
    │    │  CACHE  │  │ CACHE  │
    │    │k_rope   │  │  v     │
    │    │[0..7]   │  │ [0..7] │
    │    │+k_rope8 │  │ +v8    │
    │    │=[0..8]  │  │=[0..8] │
    │    └────┬────┘  └───┬────┘
    │         │           │
    ↓         ↓           │
 q_rope ×K_cacheᵀ = (1,9)│
    │                     │
 softmax → (1,9) weights  │
    │                     │
    └──── weights × V ────┘
               │
          output (1,768)
```

**The critical insight:** because cached K vectors are already rotated (position baked in), each new Q needs only one RoPE application. The dot product naturally computes relative distance without extra bookkeeping.

---

## 6. Why RoPE Enables Long Contexts

### No fixed table → no hard limit

```text
Learned PE: a table with 1024 rows → hard wall at position 1024.
    Position 1025? No entry. Broken.

RoPE: a formula.
    Position 1025? Rotate by 1025 × θᵢ for each pair.
    Position 100,000? Rotate by 100,000 × θᵢ.
    The formula works for any integer.
```

### But training length still matters

The model is trained on sequences up to some max length (say, 4096 tokens). During training, it only sees relative distances up to 4096. At inference:

```text
Position difference within training range (0 to 4096):
    The rotation values for these distances are familiar.
    The model learned to interpret scores from these rotations.
    Works correctly.

Position difference beyond training range (e.g., 8000):
    RoPE can compute the rotation: no crash.
    But the resulting Q·K value comes from a (m-n) the model never trained on.
    The score may be out-of-distribution. Quality degrades.

It's like an extrapolation problem. The function is defined everywhere,
but the model only learned its behavior in a certain range.
```

### Context Length Extension

Two methods to extend RoPE beyond training length:

```text
Method 1: Position Interpolation (Chen et al., 2023)
    Compress all positions into the training range.

    Model trained on 4K. Want 32K.
    Scale factor = 32K / 4K = 8.

    Instead of position 31000 → use 31000 / 8 = 3875.
    All 32K positions now map into [0, 4096].
    Every position the model sees is within its trained range.

    Downside: nearby tokens look "closer" than they should.
    Position 1 and position 2 are now both mapped near 0.
    Fine-tuning for a few steps on longer sequences fixes this.

    LLaMA 2 (4K training) → extended to 32K using this method.

Method 2: NTK-Aware Scaling (bloc97, 2023 — "Neural Tangent Kernel")
    Change the base frequency instead of scaling positions.

    Original: θᵢ = 10000^(-2i/d)
    Scaled:   θᵢ = (10000 × s)^(-2i/d)   where s is the scale factor

    Effect:
        High-frequency dimensions (small i): barely change.
            They still distinguish nearby tokens fine.
        Low-frequency dimensions (large i): slow down significantly.
            They can now span the longer range without aliasing.

    Benefit over interpolation:
        Short-range relationships are preserved exactly.
        Long-range capacity is extended by slowing the low-freq dimensions.
        No fine-tuning required for many tasks.

Method 3: YaRN (Peng et al., 2023)
    Key insight: interpolation and NTK each break something.
        Interpolation: squashes ALL frequencies → hurts short-range (fast dims)
        NTK: scales ALL frequencies by the same base → doesn't help slow dims enough

    YaRN splits the 384 dimension pairs into three bands and treats each differently:

        Fast dims (low i, high θ):
            These already rotate many times within training length.
            Extending context doesn't push them out-of-distribution.
            → Leave them ALONE. No scaling needed.

        Slow dims (high i, low θ):
            These barely complete one rotation in training length.
            Extending context pushes them into unseen angles.
            → Apply interpolation (compress positions into trained range).

        Medium dims (middle i):
            → Blend between no-scaling and interpolation smoothly.

    This way: short-range attention (driven by fast dims) is preserved exactly,
    long-range attention (driven by slow dims) is extended without breaking.

    Attention temperature correction:
        When you extend context, each query attends over MORE keys.
        More keys → the dot product values spread out differently →
        softmax becomes less peaked (entropy increases).
        The model was trained with sharper attention distributions.

        Fix: scale the attention logits by a learned temperature factor
        (slightly > 1) to restore the sharpness the model expects.
        Without this, the model "spreads attention too thin" over the
        longer context and quality drops.

    LLaMA 3 (8K training) → extended to 128K using YaRN.
```

### Numbers

```text
| Model          | Training ctx | Extended ctx | Method         |
| -------------- | ------------ | ------------ | -------------- |
| LLaMA 2 7B     | 4,096        | 32,768       | Position interp|
| LLaMA 3 8B     | 8,192        | 128,000      | YaRN           |
| Mistral 7B     | 8,192        | 32,768       | NTK-aware      |
| GPT-NeoX 20B   | 2,048        | —            | (no extension) |

Compare: GPT-2 had a 1,024 learned PE table — hard stop.
LLaMA 3 with RoPE runs 128K tokens. 125× longer.
```

---

## 7. What RoPE Fixes vs What It Doesn't

```text
Problem                         | Sinusoidal | Learned PE | RoPE
--------------------------------|------------|------------|--------
No position info in attention   | Fixed      | Fixed      | Fixed
Hard length limit               | No limit   | BROKEN     | No limit
Quality beyond training length  | Degrades   | Broken     | Degrades (gracefully)
Relative position awareness     | Partial    | Partial    | Native
No extra parameters             | Yes        | No (table) | Yes
Works with KV cache natively    | Yes*       | Yes*       | Yes (baked in)
Context extension methods       | N/A        | N/A        | NTK, YaRN, interp

* Sinusoidal/learned PE: positions added to embeddings, which become K,V.
  Position IS in the cached K,V — but as an additive term in the embedding,
  not a rotation on K.
  The Q·K dot product still mixes absolute positions, not relative distances.
```

---

## Summary

```text
The progression:

1. No position info → attention is permutation invariant → broken for language.

2. Sinusoidal PE (2017):
    Add position vector to token embedding before any layer.
    Fixed formula: PE(pos, i) = sin/cos(pos / 10000^(2i/d)).
    Works for any position, but model quality degrades past training length.

3. Learned PE (BERT, GPT-2):
    Replace formula with a trained lookup table.
    Better in practice, but hard wall at max table size.
    GPT-2: 1024 hard limit.

4. RoPE (LLaMA, Mistral, Gemma):
    Don't add to embeddings. Rotate Q and K by angle proportional to position.
    Q_m · K_n = f(m - n): score depends only on relative position.
    No lookup table → no hard limit.
    Works with KV cache: cached K already contains its position via rotation.
    Extendable to 32K-128K with NTK / YaRN / interpolation.

Why it matters for inference:
    KV cache reuses K,V across decode steps.
    RoPE rotates K before it's cached — position baked in, no recomputation.
    New Q is rotated once (at its current position), then attends to all
    cached K vectors with correct relative distances automatically.
    Long contexts are possible (with extension methods) without changing
    the architecture or the cache mechanism.
```
