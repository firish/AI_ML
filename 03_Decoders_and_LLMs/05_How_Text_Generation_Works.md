## How Text Generation Works

### The Core Loop

A decoder generates text **one token at a time**, left to right. Each step:
1. Feed all tokens so far into the model
2. Get a probability distribution over the entire vocabulary for the next token
3. Pick one token from that distribution
4. Append it to the sequence
5. Repeat until a stop condition is met

```text
Step 1:  Input: "The"
         Model output: P(cat)=0.12, P(dog)=0.08, P(big)=0.06, ...
         Pick: "cat"

Step 2:  Input: "The cat"
         Model output: P(sat)=0.15, P(is)=0.10, P(ran)=0.09, ...
         Pick: "sat"

Step 3:  Input: "The cat sat"
         Model output: P(on)=0.25, P(down)=0.08, P(quietly)=0.05, ...
         Pick: "on"

Step 4:  Input: "The cat sat on"
         ...and so on until <EOS> or max length
```

This is called **autoregressive generation** — each prediction depends on all previous predictions. The model literally writes one word at a time, just like you do.

---

### What the Model Actually Outputs

At each step, the prediction head produces a **logit** (raw score) for every token in the vocabulary:

```text
Vocabulary size: 50,257 tokens (GPT-2)

Raw logits (before softmax):
    token 0  ("!")     : -1.2
    token 1  ("\"")    : -3.5
    ...
    token 2134 ("cat") :  4.8    ← high score
    token 2135 ("car") :  1.2
    ...
    token 8937 ("dog") :  3.9
    ...
    50,257 scores total — one for every possible next token

Apply softmax → probabilities:
    P("cat") = 0.12
    P("dog") = 0.08
    P("car") = 0.002
    ...
    All 50,257 probabilities sum to 1.0
```

The question is: **how do you pick one token from these 50,257 probabilities?** This is where sampling strategies come in.

---

### What "Sample from This Distribution" Actually Means

The sampling strategies below all end with "sample from the distribution." Here is what that means concretely.

```text
Sampling = rolling a weighted die.

After temperature scaling or top-k/top-p filtering gives you a
probability distribution, you randomly pick ONE token where each
token's probability is its chance of being selected.

Say after filtering you have:
    P("cat")  = 0.50
    P("dog")  = 0.30
    P("bird") = 0.15
    P("fish") = 0.05

This is a weighted die with 4 faces:
    ┌──────────────────────────────────────────────┐
    │  cat          dog       bird   fish           │
    │ ██████████  ██████    ███     █               │
    │   50%         30%      15%    5%              │
    └──────────────────────────────────────────────┘

Roll the die → land on one token.
50% of the time you get "cat", 30% "dog", etc.

In code, it's literally one line:

    import random
    tokens = ["cat", "dog", "bird", "fish"]
    probs  = [0.50,  0.30,  0.15,   0.05]
    next_token = random.choices(tokens, weights=probs, k=1)[0]

That's it. It's a random weighted selection. Nothing fancier.
```

```text
This is how temperature and top-k/top-p interact with sampling:

    Raw logits  →  temperature scaling  →  top-k/top-p filtering  →  SAMPLE
    [4.8, 3.9, ...]   (sharpen/flatten)     (cut the tail off)      (roll the die)

    Temperature changes the SHAPE of the die (how lopsided it is).
    Top-k/top-p changes how many FACES the die has.
    Sampling is the actual roll.
```

```text
Why this matters:

    Every single token in a response is a SEPARATE roll of this die.

    A 200-token response = 200 independent random selections,
    each conditioned on everything that came before.

    Token 1: roll die → "The"
    Token 2: (given "The") new die → roll → "capital"
    Token 3: (given "The capital") new die → roll → "of"
    ...

    Each roll creates a new context, which creates a new die
    (new probability distribution), which gets rolled again.

    This is why the same prompt gives different outputs.
    It's not a bug — it's 200 coin flips in sequence.
    One different flip early on → completely different response.
```

---

## Sampling Strategies

### 1. Greedy Decoding

**Always pick the highest-probability token.**

```text
Probabilities: P(cat)=0.12, P(dog)=0.08, P(the)=0.07, ...
Pick: "cat" (highest)

Next step probabilities: P(sat)=0.15, P(is)=0.10, ...
Pick: "sat" (highest)

Result: always the single most likely sequence.
```

```text
Pros:
    - Deterministic — same input always gives same output
    - Fast — no randomness to manage

Cons:
    - Boring and repetitive
    - Gets stuck in loops: "The cat is a cat is a cat is a cat..."
    - Misses good sequences that start with a lower-probability token
      (the globally best sentence might not start with the locally best word)
```

Used for: factual tasks where you want one definitive answer (rarely used alone in practice).

---

### 2. Temperature Sampling

**Scale the logits before softmax to control randomness.**

```text
probabilities = softmax(logits / T)

T = temperature parameter
```

**Toy example** — logits for 4 tokens: [4.0, 2.0, 1.0, 0.5]

```text
T = 1.0 (normal):
    softmax([4.0, 2.0, 1.0, 0.5]) = [0.64, 0.09, 0.03, 0.02]
    → model is fairly confident about the top token

T = 0.5 (low — sharper):
    softmax([8.0, 4.0, 2.0, 1.0]) = [0.93, 0.05, 0.01, 0.00]
    → model is very confident, almost greedy
    → output is predictable, safe, potentially repetitive

T = 2.0 (high — flatter):
    softmax([2.0, 1.0, 0.5, 0.25]) = [0.36, 0.18, 0.12, 0.10]
    → probabilities spread out, less confident
    → output is creative, surprising, potentially incoherent

T → 0:    equivalent to greedy (pick the top token with 100% probability)
T = 1.0:  use the model's raw probabilities as-is
T → ∞:    uniform distribution (every token equally likely = random garbage)
```

**Intuition:** Temperature controls how "creative" vs "focused" the model is.

```text
Low temperature (0.1-0.3):   "I'm very sure about the answer"
    → Use for: code generation, factual Q&A, math
    → The model sticks to what it's confident about

Medium temperature (0.5-0.8): "I have preferences but I'm open"
    → Use for: general conversation, summarisation
    → Good balance of quality and variety

High temperature (1.0-1.5):  "Let's explore possibilities"
    → Use for: creative writing, brainstorming
    → More surprising word choices, occasional brilliance and nonsense
```

---

### 3. Top-k Sampling

**Only consider the k most likely tokens, zero out everything else, redistribute.**

```text
Full distribution (50,257 tokens):
    P(cat)=0.12, P(dog)=0.08, P(the)=0.07, P(sat)=0.06, P(bird)=0.05,
    P(...)=0.02, P(...)=0.01, ... (50,252 more tokens with tiny probabilities)

Top-k with k=5:
    Keep only the top 5:
    P(cat)=0.12, P(dog)=0.08, P(the)=0.07, P(sat)=0.06, P(bird)=0.05
    Everything else → 0

    Redistribute so they sum to 1:
    Total = 0.12 + 0.08 + 0.07 + 0.06 + 0.05 = 0.38
    P(cat)=0.32, P(dog)=0.21, P(the)=0.18, P(sat)=0.16, P(bird)=0.13

    Sample from this filtered distribution.
```

**Why?** Without top-k, there's always a small chance of sampling a terrible token (the 50,000th most likely word). Top-k prevents this while keeping diversity among the top candidates.

```text
Problem with fixed k:
    Sometimes the model is confident: P(cat)=0.95, P(dog)=0.03, ...
        → k=50 includes 48 bad options that dilute the obvious answer

    Sometimes the model is uncertain: P(cat)=0.05, P(dog)=0.05, P(the)=0.04, ...
        → k=5 cuts off perfectly valid options

    Fixed k doesn't adapt to the model's confidence level.
    → This is why top-p was invented.
```

---

### 4. Top-p (Nucleus) Sampling

**Keep the smallest set of tokens whose cumulative probability exceeds p.** The number of tokens considered varies dynamically.

```text
p = 0.9 means: keep tokens until their cumulative probability reaches 90%.

Example 1 — model is confident:
    P(cat)=0.80, P(dog)=0.10, P(bird)=0.03, ...
    Cumulative: 0.80 → 0.90 → stop
    Keep: {cat, dog}  (only 2 tokens needed to reach 90%)
    The model stays focused.

Example 2 — model is uncertain:
    P(cat)=0.05, P(dog)=0.05, P(the)=0.04, P(sat)=0.04, ...
    Cumulative: 0.05 → 0.10 → 0.14 → 0.18 → ... → 0.90
    Keep: {cat, dog, the, sat, ...}  (maybe 40 tokens needed to reach 90%)
    The model explores more options.

Top-p adapts automatically:
    Confident → few tokens → focused output
    Uncertain → many tokens → diverse output
```

```text
Typical values:
    p = 0.9:   standard for most tasks
    p = 0.95:  more diverse
    p = 0.5:   quite focused
    p = 1.0:   no filtering (use full distribution)
```

---

### 5. Beam Search

**Track multiple candidate sequences in parallel, pick the best complete sequence.**

Unlike the previous methods (which pick one token at a time), beam search considers multiple paths:

```text
Beam size = 3 (track top 3 candidates at each step):

Step 1: "The" → top 3 next tokens:
    Beam 1: "The cat"     (score: 0.12)
    Beam 2: "The dog"     (score: 0.08)
    Beam 3: "The big"     (score: 0.06)

Step 2: Expand each beam by top 3 → 9 candidates, keep top 3:
    "The cat sat"         (score: 0.12 × 0.15 = 0.018)
    "The cat is"          (score: 0.12 × 0.10 = 0.012)
    "The dog ran"         (score: 0.08 × 0.12 = 0.010)

Step 3: Continue until all beams hit <EOS>.
    Pick the beam with the highest total score.
```

```text
Pros:
    - Finds higher-quality sequences than greedy
    - Good for tasks with one "correct" output (translation)

Cons:
    - Deterministic (like greedy, but better)
    - Still tends to produce generic, safe text
    - More expensive: beam_size × more computation per step
    - Not good for creative/conversational tasks

Used for: machine translation, speech recognition
NOT used for: ChatGPT-style conversation (too boring/safe)
```

---

### 6. Combining Strategies

In practice, you combine temperature with top-p or top-k:

```text
Typical ChatGPT-style setup:
    1. Compute logits
    2. Apply temperature (e.g., T=0.7)
    3. Apply top-p filtering (e.g., p=0.9)
    4. Sample from the filtered distribution

This gives you:
    - Temperature controls overall creativity
    - Top-p prevents sampling garbage tokens
    - Sampling (not greedy) keeps output interesting
```

```text
Common configurations:

| Task                  | Temperature | Top-p | Top-k | Strategy        |
| --------------------- | ----------- | ----- | ----- | --------------- |
| Code generation       | 0.0-0.2     | —     | —     | Near-greedy     |
| Factual Q&A           | 0.1-0.3     | 0.9   | —     | Focused         |
| General conversation  | 0.7         | 0.9   | —     | Balanced        |
| Creative writing      | 0.9-1.2     | 0.95  | —     | Exploratory     |
| Translation           | —           | —     | —     | Beam search (4) |
```

---

## Stop Conditions

How does the model know when to stop generating?

```text
1. <EOS> token
    The model outputs the special end-of-sequence token.
    This is learned during training — the model saw millions of examples
    that end with <EOS> and learned when a response is "complete."

2. Max length
    Hard limit on the number of tokens generated.
    Prevents infinite generation.
    "Max tokens = 4096" = stop after 4096 generated tokens.

3. Stop sequences
    Custom strings that trigger stopping.
    e.g., stop on "\n\nHuman:" in a chat format.
    The API checks the output text and stops when the pattern appears.
```

---

## The Full Generation Pipeline

Putting it all together — what happens when you send a message to ChatGPT/Claude:

```text
Your message: "What is the capital of France?"

Step 1: Tokenize
    System prompt + conversation history + your message
    → [token IDs]   (might be 500-2000 tokens)

Step 2: Forward pass (the "prefill")
    All input tokens processed in PARALLEL through the decoder.
    This is fast — one forward pass for all input tokens.
    Produces KV cache (stored for reuse).

Step 3: Generate first output token
    Take the last position's output → prediction head → softmax
    Apply temperature (T=0.7) → top-p (p=0.9) → sample
    → "Paris"

Step 4: Generate second token
    Append "Paris" to input.
    DON'T reprocess all previous tokens — use KV cache.
    Only process the ONE new token through all layers.
    → prediction head → sample → ","

Step 5: Continue until stop condition
    "Paris" → "," → " the" → " capital" → " of" → " France" → "." → <EOS>

Step 6: Detokenize
    [token IDs] → "Paris, the capital of France."

Total: 1 parallel forward pass (prefill) + 8 sequential forward passes (one per generated token).
This is why the first token takes longer than subsequent tokens.
```

---

## Why the Same Prompt Gives Different Answers

```text
If temperature > 0, sampling is RANDOM.

Prompt: "Tell me a fun fact about cats"

Run 1:  logits → softmax(T=0.7) → sample → "Cats"
Run 2:  logits → softmax(T=0.7) → sample → "Did"
Run 3:  logits → softmax(T=0.7) → sample → "A"

Each run samples from the same probability distribution,
but randomness means different tokens get picked.
One different token at step 1 → completely different continuation.

Setting T=0 (or greedy): same input always gives same output.
This is what "deterministic" mode means in API calls.
```

---

## Summary

```text
| Strategy        | How it picks          | Deterministic? | Best for              |
| --------------- | --------------------- | -------------- | --------------------- |
| Greedy          | Highest probability   | Yes            | Nothing (too boring)  |
| Temperature     | Scale then sample     | No             | Controlling creativity|
| Top-k           | Sample from top k     | No             | Simple filtering      |
| Top-p (nucleus) | Sample from top p%    | No             | Adaptive filtering    |
| Beam search     | Track N best paths    | Yes            | Translation, ASR      |
| Temp + Top-p    | Scale, filter, sample | No             | ChatGPT, Claude, etc. |
```
