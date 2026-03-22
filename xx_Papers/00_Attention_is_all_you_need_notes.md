## Attention Is All You Need — Paper Notes

**Authors:** Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin
**From:** Google Brain / Google Research, 2017
**Published at:** NeurIPS 2017
**Why it matters:** Introduced the Transformer — the architecture behind BERT, GPT, LLaMA, ViT, CLIP, Whisper, and essentially every modern AI model.

---

## 1. The Problem

Before this paper, the best models for sequence tasks (translation, language modelling) were **RNNs** (Recurrent Neural Networks), specifically LSTMs and GRUs.

```text
How an RNN processes "The cat sat on the mat":

    Step 1: Read "The"    → h₁
    Step 2: Read "cat"    → h₂ (depends on h₁)
    Step 3: Read "sat"    → h₃ (depends on h₂)
    Step 4: Read "on"     → h₄ (depends on h₃)
    Step 5: Read "the"    → h₅ (depends on h₄)
    Step 6: Read "mat"    → h₆ (depends on h₅)

    Each step MUST wait for the previous step to finish.
```

**Three problems with RNNs:**

```text
1. Sequential — can't parallelise
    Each token depends on the previous token's output.
    You can't process tokens 1-6 simultaneously.
    GPUs are great at doing many things at once — RNNs can't use this.
    Training is slow.

2. Long-range dependencies are hard
    If token 1 ("The") is important for understanding token 100,
    the signal must pass through 99 intermediate steps.
    By the time it arrives, it's degraded (vanishing gradients).

3. Memory bottleneck
    The entire history of the sentence must be compressed into
    one fixed-size hidden state vector h.
    Long sentences lose information.
```

Some models added attention ON TOP of RNNs (Bahdanau attention, 2014), which helped with problem 2 and 3. But the sequential bottleneck (problem 1) remained — the RNN still processed tokens one at a time.

**The paper's question:** What if we remove recurrence entirely and use ONLY attention?

---

## 2. The Proposed Solution

**The Transformer:** A model built entirely from attention mechanisms, feed-forward networks, and residual connections. No recurrence. No convolutions.

```text
RNN:          Process tokens one by one, sequentially
Transformer:  Process ALL tokens simultaneously, in parallel

RNN:          Token 50 must wait for tokens 1-49 to be processed
Transformer:  Token 50 directly attends to any token (1, 25, 49) in one step
```

---

## 3. Architecture

The original Transformer is an **encoder-decoder** model designed for translation (English → German/French).

### 3.1 High-Level Structure

```text
┌───────────────┐              ┌───────────────┐
│   ENCODER      │              │   DECODER      │
│                │              │                │
│  6 identical   │──── K,V ───→│  6 identical   │
│  layers        │              │  layers        │
│                │              │                │
│  Input:        │              │  Input:        │
│  "The cat sat" │              │  "<BOS> Le chat"│
│  (English)     │              │  (French so far)│
│                │              │                │
│  Bidirectional │              │  Causal (left   │
│  attention     │              │  only) + cross- │
│  (sees all)    │              │  attention      │
│                │              │                │
│  Output:       │              │  Output:       │
│  3 context-    │              │  predict next   │
│  aware vectors │              │  French word    │
└───────────────┘              └───────────────┘
```

### 3.2 One Encoder Layer (repeated 6 times)

```text
Input (sequence of vectors)
    ↓
Multi-Head Self-Attention
    Every token attends to every other token.
    "cat" can see "The" and "sat" simultaneously.
    ↓
Add & Norm (residual connection + LayerNorm)
    output = LayerNorm(input + attention_output)
    ↓
Feed-Forward Network (FFN)
    Two linear layers with ReLU in between.
    Applied to each token independently.
    Expands: 512 → 2048 → 512
    ↓
Add & Norm (residual connection + LayerNorm)
    ↓
Output (same shape as input — sequence of vectors)
```

### 3.3 One Decoder Layer (repeated 6 times)

```text
Same as encoder, but with TWO changes:

1. Self-attention is MASKED (causal)
    Token at position i can only see positions ≤ i.
    Future tokens are masked with -∞ before softmax.
    This preserves the autoregressive property:
    predictions for position i depend only on positions < i.

2. An EXTRA attention layer: Cross-Attention
    Q comes from the decoder (what am I looking for?)
    K, V come from the encoder output (what does the input offer?)
    This is how the decoder "reads" the encoder's understanding
    of the input sentence.

Full decoder layer:
    Input → Masked Self-Attention → Add&Norm
          → Cross-Attention (to encoder) → Add&Norm
          → FFN → Add&Norm → Output
```

### 3.4 Model Dimensions

```text
| Parameter     | Base model | Big model |
| ------------- | ---------- | --------- |
| d_model       | 512        | 1024      |
| d_ff (FFN)    | 2048       | 4096      |
| Heads (h)     | 8          | 16        |
| Layers (N)    | 6          | 6         |
| d_k per head  | 64         | 64        |
| Dropout       | 0.1        | 0.3       |
| Parameters    | 65M        | 213M      |
```

---

## 4. Key Mechanisms Explained

### 4.1 Scaled Dot-Product Attention

```text
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V

Step by step:
    1. Q · Kᵀ           → raw similarity scores between all pairs
    2. ÷ √d_k            → scale down to prevent large values
    3. softmax            → convert to weights (sum to 1)
    4. × V               → weighted sum of values

Why scale by √d_k?
    Q and K have d_k dimensions. Their dot product has mean 0
    and variance d_k. When d_k = 64, dot products can be as
    large as ±16, pushing softmax into saturation (gradients → 0).
    Dividing by √64 = 8 brings values back to a reasonable range.
```

### 4.2 Multi-Head Attention

```text
Instead of one attention function with 512 dimensions,
split into 8 heads of 64 dimensions each:

    head_i = Attention(Q × W_Qi, K × W_Ki, V × W_Vi)

    Concatenate all 8 heads → multiply by W_O → output

Why multiple heads?
    Each head can learn a different type of relationship:
    - Head 1: syntactic (subject-verb agreement)
    - Head 2: coreference ("its" refers to "The Law")
    - Head 3: adjacent words
    - Head 4: long-distance dependencies ("making...more difficult")

    The paper's attention visualisations (pages 13-15) confirm
    that different heads learn different linguistic functions.

    Single-head attention averages all these patterns into one
    set of weights — multi-head keeps them separate.

Cost: same total compute as single-head (8 heads × 64 dims = 512 dims).
```

### 4.3 Positional Encoding

```text
Problem: Attention treats input as a SET, not a SEQUENCE.
    "cat sat the" and "the cat sat" would produce identical outputs.
    The model has no concept of word order.

Fix: Add positional encodings to the input embeddings.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Each position gets a unique pattern of sine/cosine values.
    Each dimension uses a different frequency (wavelengths form
    a geometric progression from 2π to 10000·2π).

Why sinusoidal (not learned)?
    The paper hypothesised that for any fixed offset k,
    PE(pos+k) can be expressed as a linear function of PE(pos).
    This would let the model learn relative positions easily.

    They also tested LEARNED positional embeddings — results
    were nearly identical (Table 3, row E). They chose sinusoidal
    because it might extrapolate to longer sequences than seen
    during training.

    (Later models like BERT and GPT-2 use LEARNED positions.
    Even later models like LLaMA use RoPE — rotary position
    embeddings — which encode relative positions more explicitly.)
```

### 4.4 Residual Connections + LayerNorm

```text
Every sub-layer (attention or FFN) is wrapped:
    output = LayerNorm(x + Sublayer(x))

Why residual connections?
    Without them, training 6+ layers would fail (vanishing gradients).
    The skip connection gives gradients a direct path backward.
    (Same idea as ResNet for images — see NN Basics.)

Why LayerNorm?
    Stabilises training by normalising the values at each layer.
    Prevents internal covariate shift (values drifting to extreme ranges).
    (See NN Basics file 05.)
```

### 4.5 Feed-Forward Network (FFN)

```text
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    = Linear(512 → 2048) → ReLU → Linear(2048 → 512)

Applied to each token position independently (no interaction between tokens).
Same weights at every position, different weights per layer.

Purpose: After attention mixes information across tokens,
FFN processes each token's enriched representation independently.
Think of attention as "gather info" and FFN as "process info."
```

---

## 5. Why Self-Attention Beats RNNs and CNNs

The paper's Table 1 — the core argument:

```text
| Layer Type     | Compute/layer | Sequential ops | Max path length |
| -------------- | ------------- | -------------- | --------------- |
| Self-Attention | O(n² · d)     | O(1)           | O(1)            |
| Recurrent      | O(n · d²)     | O(n)           | O(n)            |
| Convolutional  | O(k·n · d²)   | O(1)           | O(log_k(n))     |
```

**Three criteria and who wins:**

```text
1. Parallelisation (Sequential operations):
    RNN: O(n) — must process tokens one by one
    Attention: O(1) — all tokens processed simultaneously
    → Attention wins (massively faster training on GPUs)

2. Long-range dependencies (Max path length):
    RNN: O(n) — signal must traverse every intermediate token
    CNN: O(log n) — needs many layers to connect distant tokens
    Attention: O(1) — ANY token directly attends to ANY other
    → Attention wins (can model "making...more difficult" in one step)

3. Compute per layer:
    RNN: O(n · d²)
    Attention: O(n² · d)
    → RNN wins when n < d (rare for sentences), Attention wins otherwise
    → For typical sentence lengths (n ≈ 50-200, d = 512), similar cost
      but attention is fully parallel

Trade-off: Attention is O(n²) in sequence length — expensive for very long
sequences. This is why context windows exist (512 tokens originally,
now up to 128K+ with optimisations).
```

---

## 6. Training Details

```text
Dataset:
    English→German: WMT 2014, 4.5M sentence pairs
    English→French: WMT 2014, 36M sentence pairs

Tokenization:
    Byte Pair Encoding (BPE)
    ~37,000 token vocabulary (shared between source and target languages)

Hardware:
    8 NVIDIA P100 GPUs (one machine)

Training time:
    Base model: 100K steps = 12 hours
    Big model:  300K steps = 3.5 days

Optimizer:
    Adam (β₁=0.9, β₂=0.98, ε=10⁻⁹)

Learning rate schedule:
    Warmup + inverse square root decay:
    lr = d_model^(-0.5) × min(step^(-0.5), step × warmup_steps^(-1.5))
    Warmup = 4000 steps

    This means: lr increases linearly for 4000 steps,
    then decreases proportionally to 1/√step.

Regularisation:
    Dropout: P_drop = 0.1 (applied after each sub-layer and to embeddings)
    Label smoothing: ε = 0.1 (softens hard targets [1,0,0] to [0.9,0.05,0.05])
        → hurts perplexity (model is less "certain")
        → but improves BLEU (better translations in practice)

Weight tying:
    The embedding matrix and the pre-softmax linear transformation
    share the same weights. Saves ~25M parameters.
    (input embedding × √d_model to scale up)
```

---

## 7. Results

### 7.1 Machine Translation

```text
English → German (WMT 2014):
    Previous best (ensemble of multiple models): 26.36 BLEU
    Transformer (big, single model):             28.4  BLEU    ← +2.0 over ensembles
    Transformer (base):                          27.3  BLEU    ← still beats ensembles

English → French (WMT 2014):
    Previous best (single model): ~40.5 BLEU
    Transformer (big):            41.8  BLEU    ← new single-model SOTA

Training cost comparison:
    Previous best models:  ~1.0 × 10²⁰ FLOPs
    Transformer (big):      2.3 × 10¹⁹ FLOPs    (~4× cheaper)
    Transformer (base):     3.3 × 10¹⁸ FLOPs    (~30× cheaper)
```

**Translation:** The Transformer is both better AND cheaper to train than everything before it.

### 7.2 English Constituency Parsing (Generalisation Test)

```text
Task: Parse English sentences into grammatical tree structures.
Not a translation task — tests whether the architecture generalises.

    Transformer (4 layers, no task-specific tuning):
        WSJ-only:        91.3 F1  (competitive with specialised parsers)
        Semi-supervised:  92.7 F1  (beats most previous models)

Conclusion: The Transformer works well beyond translation,
even without any task-specific architecture changes.
```

---

## 8. Ablation Study (Table 3) — What Matters?

The paper systematically changed one thing at a time on the base model:

```text
| What they changed                     | Effect on BLEU | Takeaway                          |
| ------------------------------------- | -------------- | --------------------------------- |
| 8 heads → 1 head                      | -0.9           | Multi-head matters                |
| 8 heads → 32 heads                    | -0.4           | Too many heads hurts too          |
| Reduced d_k (attention key dimension) | worse          | Compatibility needs enough dims   |
| 2 layers instead of 6                 | -2.1           | Depth matters a lot               |
| 8 layers instead of 6                 | +0.2           | More layers help (diminishing)    |
| d_model 1024, d_ff 4096 (bigger)      | +0.6           | Bigger = better                   |
| Dropout 0.0 (no dropout)              | -1.2           | Dropout is critical               |
| Learned positions instead of sinusoidal | -0.1          | Nearly identical — doesn't matter |
```

**Key takeaways:**
- Multi-head attention is important (but there's a sweet spot — not 1, not 32)
- Depth matters more than width
- Dropout is essential
- Positional encoding method barely matters

---

## 9. The Paper's Conclusion (Verbatim Key Sentences)

> "We presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention."

> "For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers."

> "We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video."

That last sentence predicted everything that followed:
- ViT (2020) — Transformers for images
- Whisper (2022) — Transformers for audio
- CLIP (2021) — Transformers for image+text
- GPT series — Transformers for general intelligence

---

## 10. What This Paper Didn't Do (Left for Future Work)

```text
Things the paper acknowledged but didn't solve:

1. O(n²) attention cost
    Self-attention scales quadratically with sequence length.
    Processing 10,000 tokens = 100M attention computations.
    → Later solved by: Flash Attention, sparse attention, linear attention

2. Only tested on translation
    They showed parsing works too, but didn't test on
    language modelling, classification, summarisation, etc.
    → Later solved by: BERT (encoder), GPT (decoder), T5 (enc-dec)

3. Only encoder-decoder
    The paper didn't explore encoder-only or decoder-only variants.
    → Later: BERT stripped the decoder, GPT stripped the encoder

4. Fixed sequence length
    No mechanism for handling very long documents.
    → Later solved by: RoPE, ALiBi, sliding window attention

5. Images, audio, video
    They mentioned wanting to extend to other modalities.
    → Later solved by: ViT, AST, CLIP, Whisper, etc.
```

---

## 11. Impact — What This Paper Changed

```text
Before (2017):                         After (2017-present):
─────────────────────                  ─────────────────────
RNNs for sequences                     Transformers for everything
Sequential processing                  Parallel processing
Separate architectures per task        One architecture, many tasks
Small models (100M params)             Massive models (100B+ params)
Task-specific training                 Pre-train once, fine-tune for anything

Direct descendants:
    2018: BERT (encoder-only transformer → understanding)
    2018: GPT (decoder-only transformer → generation)
    2019: GPT-2, T5, BART
    2020: GPT-3 (175B params, in-context learning emerges)
    2020: ViT (transformers for images)
    2021: CLIP (transformers for multimodal)
    2022: ChatGPT, Whisper
    2023: GPT-4, LLaMA, Claude, Gemini
    2024-26: Claude 4, GPT-5, open-source explosion

Every model in your notes traces back to this 15-page paper.
```
