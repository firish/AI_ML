## Key Concepts: Speculative Decoding

---

## 0. Recap — Why Autoregressive Decoding Is Slow

### The sequential bottleneck

```text
LLMs generate one token at a time. Each token requires a full forward pass:

    Step 1: feed token → read all model weights from HBM → compute → output token 1
    Step 2: feed token 1 → read all model weights from HBM → compute → output token 2
    Step 3: feed token 2 → read all model weights from HBM → compute → output token 3
    ...

Each step reads the ENTIRE model from memory.
LLaMA 2 70B in FP16: 140 GB read per token.

The GPU is memory-bound: it finishes the math before the next weights arrive.
Most of the GPU's compute capacity sits idle during decode.

    A100 GPU:
        Peak compute: 312 TFLOPS (FP16)
        HBM bandwidth: 2 TB/s

    LLaMA 2 70B decode:
        Reads ~140 GB per token → 140 GB / 2 TB/s = 70 ms per token
        Actual FLOPs per token: ~140 GFLOPs → 140 GFLOPs / 312 TFLOPS = 0.45 ms
        The GPU computes for 0.45 ms and waits for memory for 69.55 ms.
        Compute utilization: <1%

    The model weights are loaded from memory once per token, regardless of
    whether you're generating 1 token or 5 tokens in that forward pass.
```

### The key insight

```text
If you feed 5 tokens through the model at once (like during prefill),
the cost is almost the same as feeding 1 token.

    1 token:  read 140 GB of weights, do 140 GFLOPs of compute  → 70 ms
    5 tokens: read 140 GB of weights, do 700 GFLOPs of compute  → ~71 ms
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^
              same memory read — the bottleneck

    The extra compute for 4 more tokens is basically free.
    The memory read dominates, and it's the same either way.

But we can't feed 5 tokens at once during decode — we don't know
what the next 5 tokens ARE until we generate them sequentially.

Unless... we GUESS.
```

---

## 1. Speculative Decoding — The Core Idea

### Guess fast, verify in parallel

```text
Use a SMALL, fast model (the "draft model") to guess the next K tokens.
Then use the LARGE model (the "target model") to verify all K guesses in ONE forward pass.

    Draft model:  small (e.g., 1B params), fast, less accurate
    Target model: large (e.g., 70B params), slow, accurate — this is the model you're serving

    Step 1: Draft model generates K=5 guesses quickly
        "The" → "capital" → "of" → "France" → "is"

    Step 2: Target model processes all 5 tokens in ONE forward pass
        Feed: ["The", "capital", "of", "France", "is"]
        For each position, the target model outputs what IT would have generated.

    Step 3: Compare draft vs target, left to right
        Position 0: draft said "capital", target agrees  ✓ accept
        Position 1: draft said "of",      target agrees  ✓ accept
        Position 2: draft said "France",  target agrees  ✓ accept
        Position 3: draft said "is",      target says "is" ✓ accept
        Position 4: (bonus) target predicts "Paris"       → free extra token

    Result: 5 tokens verified + 1 bonus = 6 tokens from ONE target model forward pass.
    Without speculation: 6 tokens would need 6 sequential forward passes.
```

### What happens when the draft is wrong

```text
    Draft guesses: "The" → "capital" → "in" → "France" → "is"
                                        ^^
    Target model verification:
        Position 0: draft="capital", target="capital"  ✓ accept
        Position 1: draft="in",      target="of"       ✗ REJECT
        → Stop here. Discard "in", "France", "is".
        → Accept "capital", use target's "of" as the correct next token.
        → Restart drafting from "of".

    Result: 2 tokens accepted + 1 correction = 3 tokens from one target pass.
    Worst case: draft is wrong on the first token → 1 token per target pass (same as normal).
    Best case: all K tokens accepted → K+1 tokens per target pass.

The output is IDENTICAL to running the target model alone.
Accepted tokens are exactly what the target would have produced.
Speculative decoding is not an approximation — it's a lossless speedup.
```

---

## 2. Why This Is Lossless

### The verification step in detail

```text
The target model doesn't just check "did the draft get it right?"
It uses a precise acceptance criterion based on probability distributions.

    For each position i, the draft model has probability: q(token_i)
    The target model has probability: p(token_i)

    Acceptance rule:
        If p(token_i) >= q(token_i): always accept.
            The target model is at least as likely to generate this token.

        If p(token_i) < q(token_i): accept with probability p(token_i) / q(token_i).
            Randomly reject some tokens the draft was overconfident about.

        On rejection: sample from an adjusted distribution (p - q, normalized)
            to get the correct token.

    This guarantees the EXACT same output distribution as running
    the target model alone. Not approximately the same — exactly the same.

    The math: this is a form of rejection sampling.
    The draft model proposes, the target model accepts or corrects.
    The combined procedure samples from the target distribution exactly.
```

### Why not just use the draft model directly?

```text
The draft model is fast but worse. If you served the draft model directly:
    - Lower quality outputs
    - More hallucinations
    - Worse at complex reasoning

Speculative decoding gives you:
    - Exact target model quality (guaranteed by the acceptance criterion)
    - Speed closer to the draft model (when acceptance rate is high)

You're using the draft model's speed without accepting its quality loss.
```

---

## 3. The Speed Math

### When does speculative decoding help?

```text
Variables:
    K = number of draft tokens per round (typically 3-8)
    α = acceptance rate (fraction of draft tokens accepted by target)
    t_draft = time for draft model to generate one token
    t_target = time for one target model forward pass

Without speculation:
    1 token per t_target
    N tokens take N × t_target

With speculation:
    One round: K draft tokens + 1 target verification
    Time:      K × t_draft + t_target
    Expected accepted tokens: ~αK + 1  (accepted drafts + 1 correction/bonus)

    Speedup ≈ (αK + 1) / (K × t_draft/t_target + 1)

Example: LLaMA 70B (target) + LLaMA 1B (draft)
    K = 5, α = 0.7 (70% acceptance), t_draft = 2ms, t_target = 70ms

    Without: 1 token per 70 ms
    With:    (0.7 × 5 + 1) = 4.5 tokens per (5 × 2 + 70) = 80 ms
             = 4.5 / 80 ms = 56.25 tokens/sec
    Without: 1 / 70 ms = 14.3 tokens/sec

    Speedup: 56.25 / 14.3 ≈ 3.9×

The key factors:
    High acceptance rate (α) → more tokens per round → bigger win
    Fast draft model (small t_draft) → drafting overhead is negligible
    Slow target model (large t_target) → more idle compute to exploit
```

### When does it NOT help?

```text
1. Target model is small (already fast):
    LLaMA 7B target: t_target is already low.
    The draft overhead (K × t_draft) isn't negligible anymore.
    Speedup may be <1.5× or not worth the complexity.

2. Low acceptance rate:
    If the draft model frequently disagrees with the target,
    you reject most tokens and waste the draft computation.
    Happens with very different draft/target models,
    or on highly unpredictable text (code, math).

3. Batch serving (high batch size):
    With many concurrent requests, the GPU is already compute-bound
    (many tokens processed per weight read).
    There's no idle compute to exploit.
    Speculative decoding helps most at batch size 1 or small batches.
```

---

## 4. Draft Model Choices

### Separate small model

```text
The original approach: use a smaller model from the same family.

    Target: LLaMA 70B
    Draft:  LLaMA 1B or LLaMA 7B

    Pros: simple, draft model is pretrained and available.
    Cons: need to serve two models on the GPU (extra memory).
          Draft and target may have different tokenizers (tricky).
          Acceptance rate depends on how similar the models are.
```

### Self-drafting (Medusa, EAGLE)

```text
Instead of a separate model, add small "draft heads" to the target model itself.

Medusa (Cai et al., 2024):
    Add K extra prediction heads on top of the target model's last layer.
    Head 1 predicts token t+1 (same as normal LM head)
    Head 2 predicts token t+2
    Head 3 predicts token t+3
    ...

    These heads are small MLPs (~1% of model size).
    Trained separately (freeze the main model, only train the heads).

    One forward pass of the target model → K+1 candidate tokens.
    No separate draft model needed. No extra memory for a second model.
    But requires training the extra heads on representative data.

EAGLE (Li et al., 2024):
    Uses a lightweight draft head that takes the target model's
    hidden states as input to predict the next token's hidden state.
    More accurate than Medusa because it conditions on hidden states,
    not just the final layer output.
    Achieves higher acceptance rates.
```

### Prompt lookup decoding (no draft model at all)

```text
For tasks where the output contains text from the input (summarization,
code editing, retrieval-augmented generation):

    The "draft" is just n-gram matching against the input.

    Input:  "Summarize: The quick brown fox jumped over the lazy dog"
    Generating: "The quick brown fox..."

    When the model generates "The quick", look for "The quick" in the input.
    The input has "The quick brown fox jumped" → guess those as draft tokens.

    No draft model at all. Just string matching.
    Works surprisingly well for copy-heavy tasks.
```

---

## 5. Tree-Based Speculation

### Beyond linear drafting

```text
Standard speculation: draft K tokens in a SINGLE sequence.
    "The" → "capital" → "of" → "France" → "is"

    If token 2 ("of") is wrong, tokens 3-4 are wasted.

Tree speculation: draft multiple BRANCHES at each position.

                        "The"
                       /     \
                 "capital"   "city"
                 /    \         \
              "of"   "is"     "of"
              /        \         \
          "France"   "Paris"   "France"

    The target model verifies ALL branches in one forward pass
    (using careful attention masking so branches don't attend to each other).

    If "capital" → "of" fails, "capital" → "is" might still succeed.
    More total tokens verified per round.

    Medusa and EAGLE use tree-based verification.
    Typically 2-5 branches, verified with one target forward pass.
```

---

## 6. Practical Numbers

```text
Reported speedups (single-request latency):

    Method                Target model    Speedup
    Speculative (draft)   LLaMA 70B       2-3×
    Medusa                LLaMA 13B       2-2.5×
    EAGLE                 LLaMA 70B       2.5-3.5×
    Prompt lookup         Code models     1.5-3× (on copy-heavy tasks)

Acceptance rates vary by task:
    Conversational text:  70-85% (draft is often right)
    Code generation:      50-65% (more unpredictable tokens)
    Creative writing:     40-60% (many valid continuations → harder to guess)
    Translation:          75-90% (structured output → easier to predict)

Used in production:
    Google: speculative decoding in Gemini serving
    Anthropic, OpenAI: variations of speculative execution in their serving stacks
    llama.cpp: supports speculative decoding with draft models locally
```

---

## Summary

```text
LLM decode is memory-bound: each token reads the full model from HBM,
but the GPU's compute is barely used (<1% utilization).

Speculative decoding exploits this idle compute:
    1. A small, fast draft model guesses K tokens quickly
    2. The large target model verifies all K in ONE forward pass
       (costs barely more than generating 1 token — same memory read)
    3. Accept correct guesses, reject and correct the first wrong one

Key properties:
    - Lossless: output distribution is IDENTICAL to the target model alone
      (guaranteed by rejection sampling math)
    - Speedup depends on acceptance rate: 2-3.5× typical
    - Helps most when target is large, batch size is small, and text is predictable
    - Doesn't help when GPU is already compute-bound (large batches)

Draft model options:
    - Separate smaller model (original approach, simple)
    - Self-drafting heads: Medusa (extra heads), EAGLE (hidden state prediction)
    - Prompt lookup: n-gram matching against input (no model needed)
    - Tree-based: branch multiple guesses, verify all at once
```
