# Log Probabilities and Softmax

---

## 1. The Problem with Raw Probabilities

### Probabilities are between 0 and 1. Neural networks output any real number.

```
A neural network's final layer outputs raw scores (one per vocabulary token):

    Input: "The capital of France is"

    Final layer output (logits):
        z_Paris    =  8.2
        z_the      =  2.1
        z_Lyon     =  5.7
        z_banana   = -3.4
        z_zzz      = -9.1
        ... 127,995 more ...

These are NOT probabilities:
    - Some are negative
    - They don't sum to 1
    - They can be any real number from -∞ to +∞

We need to convert these into a valid probability distribution.
```

### Why not just normalize?

```
Naive attempt: divide by the sum.

    P(token) = z_token / Σ z_all

    Problem 1: negative values.
        z = [3, -2, 1]
        sum = 2
        P = [1.5, -1.0, 0.5]    ← negative probability. Broken.

    Problem 2: even if all positive, the model has no way to express
    strong preferences. z = [100, 101] → P ≈ [0.497, 0.503].
    Very similar scores give very similar probabilities.
    We want small differences in scores to create big differences
    in probability when the model is confident.

The fix: exponentiate first, then normalize. That's softmax.
```

---

## 2. Softmax — From Logits to Probabilities

### The formula

```
softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)

    1. Exponentiate every logit:    exp(zᵢ) — makes everything positive
    2. Divide by the sum:           normalize so they sum to 1

Example: z = [2.0, 1.0, 0.1]

    exp(2.0) = 7.389
    exp(1.0) = 2.718
    exp(0.1) = 1.105
    Sum = 11.212

    softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]
            = [0.659, 0.242, 0.099]

    Properties:
        All positive  ✓
        Sum to 1.0    ✓
        Order preserved: highest logit → highest probability  ✓
        Differences amplified: logit gap of 1.0 → probability ratio of e ≈ 2.7×
```

### Why exp() and not something else?

```
exp() has special properties:

    1. Always positive:  exp(x) > 0 for any x. No negative probabilities.

    2. Monotonic:        larger logit → larger probability. Always.

    3. Amplifies differences:
        z = [10.0, 10.1]  → softmax ≈ [0.475, 0.525]    (small gap → close probs)
        z = [10.0, 13.0]  → softmax ≈ [0.047, 0.953]    (gap of 3 → 20× ratio)

        exp(3) ≈ 20, so a logit gap of 3 means ~20× higher probability.

    4. Gradient-friendly: d/dx exp(x) = exp(x).
        Clean gradients for backpropagation.

    5. Maximum entropy: softmax is the unique function that maximizes
        entropy subject to matching expected logit values.
        It's the least biased way to convert scores to probabilities.
```

---

## 3. Numerical Stability — The Subtract-Max Trick

### The problem

```
Logits can be large. exp() of large numbers overflows:

    z = [1000, 1001, 999]

    exp(1000) = 1.97 × 10⁴³⁴    ← overflows float32 (max ~3.4 × 10³⁸)
    exp(1001) = overflow → inf
    inf / inf = NaN               ← computation breaks

Even FP16 overflows at exp(11.1). This happens in practice.
```

### The fix

```
Subtract the maximum logit before exponentiating:

    softmax(zᵢ) = exp(zᵢ - max(z)) / Σⱼ exp(zⱼ - max(z))

    This is mathematically IDENTICAL to the original softmax:
        exp(zᵢ - c) / Σ exp(zⱼ - c) = exp(zᵢ)exp(-c) / Σ exp(zⱼ)exp(-c)
                                      = exp(zᵢ) / Σ exp(zⱼ)

    The exp(-c) cancels top and bottom.

Example: z = [1000, 1001, 999]
    max = 1001
    z - max = [-1, 0, -2]

    exp(-1) = 0.368
    exp(0)  = 1.000
    exp(-2) = 0.135
    Sum = 1.503

    softmax = [0.245, 0.665, 0.090]

    No overflow. The largest exponent is always exp(0) = 1.

Every implementation of softmax does this. It's not optional.
```

---

## 4. Log Probabilities (Log-Probs)

### Why work in log-space

```
Three reasons to use log-probabilities instead of probabilities:

1. Numerical stability:
    Probabilities can be tiny:  P = 0.000000001 = 10⁻⁹
    Log-probabilities are moderate: log P = -20.7

    Products of tiny probabilities underflow to 0.
    Sums of log-probabilities stay in a reasonable range.

2. Products become sums:
    log(A × B) = log A + log B

    P(sentence) = P(t₁) × P(t₂|t₁) × P(t₃|t₁,t₂) × ...
    log P(sentence) = log P(t₁) + log P(t₂|t₁) + log P(t₃|t₁,t₂) + ...

    Addition is faster and more stable than multiplication.

3. Cross-entropy loss is already in log-space:
    Loss = -log P(correct token)

    If you have log-probs, the loss is just negation. No log needed.
```

### What log-probs look like

```
Token          Probability    Log-probability
─────────────────────────────────────────────
"Paris"        0.82           -0.198
"the"          0.03           -3.507
"Lyon"         0.02           -3.912
"banana"       0.0001         -9.210
"zzz"          0.0000001      -16.118

Properties:
    log(1.0) = 0       ← maximum: certain event
    log(0.5) = -0.693  ← fifty-fifty
    log(0.0) = -∞      ← impossible event

    Log-probs are always ≤ 0.
    Higher (closer to 0) = more probable.
    The cross-entropy loss for a token = negative of its log-prob.
```

### Log-softmax (combining the two)

```
In practice, you often want log P directly, not P.

    log softmax(zᵢ) = log(exp(zᵢ) / Σ exp(zⱼ))
                     = zᵢ - log(Σ exp(zⱼ))

    This avoids computing exp then log (which loses precision).
    PyTorch provides F.log_softmax() for exactly this reason.

    The cross-entropy loss in PyTorch:
        F.cross_entropy(logits, target)

    Internally computes log_softmax(logits) and picks the target's log-prob.
    It never materializes the full probability distribution.
```

---

## 5. Logits — What the Model Actually Outputs

### The name

```
"Logit" comes from the log-odds function in statistics:
    logit(p) = log(p / (1-p))

In deep learning, the meaning shifted to just mean "raw unnormalized scores"
— the output of the last linear layer BEFORE softmax.

    Logits:          z = x × W_vocab + bias     → (batch, vocab_size)
    Probabilities:   P = softmax(z)              → (batch, vocab_size)
    Log-probs:       log P = log_softmax(z)      → (batch, vocab_size)

    z can be any real number.
    P is in [0, 1] and sums to 1.
    log P is in (-∞, 0].
```

### The LLM output pipeline

```
Hidden state from last layer:  h = (1, d_model)     e.g., (1, 4096)

LM head (linear projection):
    logits = h × W_vocab                             e.g., (1, 4096) × (4096, 128000) = (1, 128000)

    This is a dot product of the hidden state with EVERY token's embedding.
    High dot product = hidden state is "similar" to that token → high logit.

Softmax:
    probs = softmax(logits)                           (1, 128000)
    Each entry is a probability. Sums to 1.

Sample or argmax:
    next_token = sample(probs)        ← stochastic
    next_token = argmax(probs)        ← greedy (deterministic)
```

---

## 6. Temperature — Reshaping the Distribution

### What temperature does

```
Divide logits by a temperature T before softmax:

    P(tokenᵢ) = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)

    T = 1.0:  standard softmax (no change)
    T → 0:    distribution becomes a spike at the highest logit (greedy)
    T → ∞:    distribution becomes uniform (random)
    T < 1.0:  sharper — high-probability tokens get more, low get less
    T > 1.0:  flatter — probabilities spread out more evenly
```

### Concrete example

```
Logits: z = [3.0, 1.0, 0.5]    (3 tokens for simplicity)

T = 1.0 (standard):
    exp([3.0, 1.0, 0.5]) = [20.09, 2.72, 1.65]
    softmax = [0.820, 0.111, 0.067]         ← peaked

T = 0.5 (sharp):
    z/T = [6.0, 2.0, 1.0]
    exp([6.0, 2.0, 1.0]) = [403.4, 7.39, 2.72]
    softmax = [0.976, 0.018, 0.007]         ← very peaked (almost greedy)

T = 2.0 (flat):
    z/T = [1.5, 0.5, 0.25]
    exp([1.5, 0.5, 0.25]) = [4.48, 1.65, 1.28]
    softmax = [0.605, 0.223, 0.173]         ← flatter (more random)

T = 100 (nearly uniform):
    z/T = [0.03, 0.01, 0.005]
    softmax ≈ [0.340, 0.333, 0.327]        ← almost uniform
```

### Why temperature matters

```
Low temperature (T = 0.1-0.5):
    Model is very confident. Picks the most likely token almost always.
    Use for: factual Q&A, code generation, structured output.
    Risk: repetitive, boring, can get stuck in loops.

Medium temperature (T = 0.7-1.0):
    Balanced. Some randomness, mostly sensible.
    Use for: general chat, creative writing with coherence.

High temperature (T = 1.5-2.0):
    Model takes risks. Surprising word choices.
    Use for: brainstorming, poetry, exploring diverse outputs.
    Risk: incoherent, nonsensical.
```

---

## 7. Top-k and Top-p Sampling

Temperature reshapes the distribution. Top-k and top-p TRUNCATE it.

### Top-k: only keep the k most likely tokens

```
Before sampling, zero out everything except the top k tokens.

    Full distribution: P = [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]

    Top-k = 3: keep only the top 3, renormalize:
        P_truncated = [0.40, 0.25, 0.15, 0, 0, 0, 0, 0, 0]
        After renormalization: [0.50, 0.3125, 0.1875]

    The model can ONLY sample from these 3 tokens.
    Prevents sampling extremely unlikely tokens ("banana" after "The capital of France is").

Problem with top-k:
    k is fixed. But sometimes the model is very confident (1 token has 95% prob),
    and sometimes it's uncertain (20 tokens each have ~5%).
    k=10 is too many when confident, too few when uncertain.
```

### Top-p (nucleus sampling): keep tokens until cumulative probability reaches p

```
Sort tokens by probability. Keep adding tokens until their cumulative
probability reaches the threshold p:

    Sorted: [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01]
    Cumulative: [0.40, 0.65, 0.80, 0.88, 0.93, 0.96, 0.98, 0.99, 1.00]

    Top-p = 0.9:
        Keep tokens until cumulative ≥ 0.9: first 4 tokens (cumulative = 0.88) → include 5th (0.93)
        P_truncated = [0.40, 0.25, 0.15, 0.08, 0.05]
        Renormalize.

    Adaptive: when confident (one token at 95%), only 1 token survives.
    When uncertain, many tokens survive. The set size adjusts automatically.
```

### How they work together in practice

```
Most serving systems apply them in sequence:

    1. Compute logits
    2. Apply temperature:     z' = z / T
    3. Compute softmax:       P = softmax(z')
    4. Apply top-k:           zero out all but top k
    5. Apply top-p:           zero out tokens beyond cumulative p
    6. Renormalize
    7. Sample from the resulting distribution

Typical settings:
    Factual/code:    T=0.0-0.3, top_p=0.9     (nearly deterministic)
    Chat:            T=0.7, top_p=0.9          (balanced)
    Creative:        T=1.0, top_p=0.95         (more variety)

    Note: T=0 means greedy decoding (always pick the max). No sampling.
```

---

## 8. Putting It Together

```
Concept         What it does                              Where
─────────────────────────────────────────────────────────────────────────
Logits          raw scores from last linear layer          model output
Softmax         logits → valid probability distribution    exp + normalize
Subtract-max    prevent overflow in softmax                every implementation
Log-softmax     logits → log-probabilities directly        loss computation
Log-probs       log of probabilities, avoids underflow     training, evaluation
Temperature     sharpen or flatten the distribution        inference control
Top-k           keep only k most likely tokens             prevent unlikely sampling
Top-p           keep tokens until cumulative prob = p      adaptive truncation
```

The pipeline from model output to generated token:

```
hidden state → linear layer → logits → (÷ temperature) → softmax → (top-k) → (top-p) → sample → token
                              ^^^^^^                      ^^^^^^^^
                              raw scores                  probabilities
```

Everything before softmax is in logit-space (unbounded reals). Softmax converts to probability-space ([0,1], sums to 1). Temperature, top-k, and top-p are knobs that control the tradeoff between coherence and diversity at generation time.

---

**Next:** This completes the core probability topics from the math roadmap. The information theory topics (entropy, cross-entropy, perplexity, KL divergence) build directly on these foundations.
