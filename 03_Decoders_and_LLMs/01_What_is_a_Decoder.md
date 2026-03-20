## What Is a Decoder?

### The One-Sentence Version

An encoder reads a full sentence and produces a **vector** (a summary).
A decoder reads tokens left-to-right and predicts the **next token** (generation).

```text
Encoder (BERT):     "The cat sat on the" → [0.12, -0.34, 0.56, ...]  (one vector)
Decoder (GPT):      "The cat sat on the" → "mat"                      (next word)
```

### Why Do We Need a Different Architecture?

Encoders see the **entire input at once** — every token attends to every other token. That's perfect for understanding, but it's cheating for generation.

```text
The problem with using an encoder for generation:

If the model is trying to predict what comes after "The cat sat on the",
it should NOT be able to see the answer while predicting.

Encoder attention (bidirectional):
    "The"  can see  →  "cat", "sat", "on", "the"     (sees everything)
    "cat"  can see  →  "The", "sat", "on", "the"     (sees everything)

If every token sees the future, the model just copies
instead of learning to predict. Useless for generation.
```

**The fix:** block each token from seeing anything to its right. This is the **only** architectural difference between an encoder and a decoder.

```text
Decoder attention (causal / left-to-right):
    "The"  can see  →  nothing to the left               (only itself)
    "cat"  can see  →  "The"                              (only past)
    "sat"  can see  →  "The", "cat"                       (only past)
    "on"   can see  →  "The", "cat", "sat"                (only past)
    "the"  can see  →  "The", "cat", "sat", "on"          (only past)
    ???    can see  →  "The", "cat", "sat", "on", "the"   → predict "mat"
```

---

## Architecture: Stage by Stage

The pipeline is nearly identical to a text encoder (Phase 1, file 06). The differences are marked with **[DIFFERENT]**.

```text
Text: "The cat sat on the"
    ↓
Stage 1: Tokenize                      → token IDs            (same as encoder)
    ↓
Stage 2: Embedding lookup              → token vectors        (same as encoder)
    ↓
Stage 3: Add position embeddings       → position-aware vecs  (same as encoder)
    ↓
Stage 4: N transformer blocks          → context-aware vecs   [DIFFERENT: causal mask]
    ↓
Stage 5: Prediction head               → next token probs     [DIFFERENT: not in encoder]
```

---

### Stage 1: Tokenize

Same as encoders — convert text to token IDs. (Tokenization details in the next file.)

```text
"The cat sat on the" → [464, 3797, 3332, 319, 262]

Decoder-specific detail:
    - No [CLS] or [SEP] tokens (those are BERT things)
    - May prepend a <BOS> (beginning of sequence) token
    - Generation stops when model outputs <EOS> (end of sequence)
```

---

### Stage 2: Embedding Lookup

Identical to encoder. Each token ID looks up a row in the embedding table.

```text
Embedding table (50,257 tokens × 768 dims for GPT-2):
    token 464  ("The") → [0.12, -0.08, 0.34, ...]
    token 3797 ("cat") → [0.45,  0.23, -0.11, ...]
    ...

This is a learned lookup table, exactly like the encoder's.
The same table is used throughout training and inference.
```

Toy example (5 tokens, 4 dims — real GPT-2 uses 768 dims):

```text
    "The"  → [0.10,  0.20, -0.15,  0.30]
    "cat"  → [0.40, -0.10,  0.25,  0.05]
    "sat"  → [0.15,  0.35,  0.10, -0.20]
    "on"   → [0.30,  0.05, -0.30,  0.25]
    "the"  → [0.10,  0.20, -0.15,  0.30]
```

---

### Stage 3: Add Position Embeddings

Identical to encoder. Add a learned position vector to each token.

```text
Position table (1024 positions × 768 dims for GPT-2):
    position 0 → [0.01, 0.00, 0.02, -0.01]
    position 1 → [0.00, 0.01, -0.01, 0.02]
    position 2 → [0.02, 0.01, 0.00, 0.01]
    position 3 → [0.01, 0.02, 0.01, 0.00]
    position 4 → [0.00, 0.01, 0.02, 0.01]

After adding positions:
    "The"  (pos 0) → [0.10+0.01, 0.20+0.00, -0.15+0.02, 0.30-0.01] = [0.11, 0.20, -0.13, 0.29]
    "cat"  (pos 1) → [0.40+0.00, -0.10+0.01, 0.25-0.01, 0.05+0.02] = [0.40, -0.09, 0.24, 0.07]
    "sat"  (pos 2) → [0.15+0.02, 0.35+0.01, 0.10+0.00, -0.20+0.01] = [0.17, 0.36, 0.10, -0.19]
    "on"   (pos 3) → [0.30+0.01, 0.05+0.02, -0.30+0.01, 0.25+0.00] = [0.31, 0.07, -0.29, 0.25]
    "the"  (pos 4) → [0.10+0.00, 0.20+0.01, -0.15+0.02, 0.30+0.01] = [0.10, 0.21, -0.13, 0.31]

Note: "The" (pos 0) and "the" (pos 4) have the SAME word embedding but
DIFFERENT position embeddings → the model knows they're at different positions.
```

GPT-2's context window is 1024 tokens — it has 1024 learned position vectors. Tokens beyond position 1024 can't be processed (this is the "context window limit").

---

### Stage 4: Transformer Blocks — The Key Difference

Each block has the same components as an encoder block:
- Multi-head self-attention
- Residual connection + LayerNorm
- Feed-forward network (FFN)
- Residual connection + LayerNorm

**The only difference: the causal attention mask.**

#### Causal (Masked) Self-Attention

In an encoder, every token attends to every other token. In a decoder, each token can only attend to tokens **at or before** its position.

This is enforced with a **triangular mask**:

```text
Attention scores BEFORE masking (Q · Kᵀ / √d):

              "The"   "cat"   "sat"    "on"   "the"
    "The"  [  2.1     0.5     0.3     0.1     0.8  ]
    "cat"  [  1.8     1.5     0.4     0.2     0.6  ]
    "sat"  [  0.3     2.0     1.9     0.5     0.4  ]
    "on"   [  0.1     0.4     1.7     1.3     0.2  ]
    "the"  [  0.9     0.3     0.6     1.8     2.0  ]

The causal mask: set all UPPER-RIGHT entries to -∞

              "The"   "cat"   "sat"    "on"   "the"
    "The"  [  2.1     -∞      -∞      -∞      -∞   ]
    "cat"  [  1.8     1.5     -∞      -∞      -∞   ]
    "sat"  [  0.3     2.0     1.9     -∞      -∞   ]
    "on"   [  0.1     0.4     1.7     1.3     -∞   ]
    "the"  [  0.9     0.3     0.6     1.8     2.0  ]

After softmax, -∞ becomes 0.0 (e^(-∞) = 0):

              "The"   "cat"   "sat"    "on"   "the"
    "The"  [  1.00    0.00    0.00    0.00    0.00  ]  ← only sees itself
    "cat"  [  0.57    0.43    0.00    0.00    0.00  ]  ← sees The, cat
    "sat"  [  0.10    0.56    0.34    0.00    0.00  ]  ← sees The, cat, sat
    "on"   [  0.06    0.08    0.29    0.57    0.00  ]  ← sees The..on
    "the"  [  0.10    0.06    0.08    0.26    0.50  ]  ← sees everything
```

**What the mask does in practice:**
- "The" at position 0 can only attend to itself → its output is just itself (no context yet)
- "cat" at position 1 sees "The" and itself → can learn "cat" appears after "The"
- "the" at position 4 sees all 5 tokens → has full left context

After softmax, multiply by V to get the weighted output — same as encoder attention, but the weights for future tokens are zero.

#### Why This Enables Generation

```text
During training:
    Input:  "The  cat  sat  on  the"
    Target: "cat  sat  on   the mat"
                                       (each position predicts the NEXT token)

    Because of the causal mask:
    - Position 0 sees only "The"         → must predict "cat"
    - Position 1 sees "The cat"          → must predict "sat"
    - Position 2 sees "The cat sat"      → must predict "on"
    - Position 3 sees "The cat sat on"   → must predict "the"
    - Position 4 sees "The cat sat on the" → must predict "mat"

    Every position is a training example!
    One sentence of 5 tokens gives us 5 prediction tasks simultaneously.

During inference:
    Start: "The cat sat on the"
    Position 4 output → predict "mat"
    Append: "The cat sat on the mat"
    Position 5 output → predict "."
    ...and so on until <EOS>
```

#### Multi-Head Attention

Identical to encoder. Split Q, K, V into multiple heads, each attending independently:

```text
GPT-2 (768 dims, 12 heads):
    Each head works with 768 / 12 = 64 dims
    12 heads run in parallel, each with the causal mask
    Concatenate → multiply by W_O → back to 768 dims

Same concept as encoder:
    Head 1 might track "which noun does this adjective modify?"
    Head 2 might track "is this part of a list?"
    Head 3 might track "does this word echo an earlier mention?"

All heads obey the causal mask — no head can cheat and look right.
```

#### Residual + LayerNorm + FFN

Identical to encoder. No changes.

```text
One decoder block:

    input
      ↓
    Causal Multi-Head Attention          [DIFFERENT: causal mask]
      ↓
    + input (residual)                   (same as encoder)
      ↓
    LayerNorm                            (same as encoder)
      ↓
    FFN: Linear(768 → 3072) → GELU → Linear(3072 → 768)   (same as encoder)
      ↓
    + previous (residual)                (same as encoder)
      ↓
    LayerNorm                            (same as encoder)
      ↓
    output
```

#### Stacking Blocks

GPT-2 stacks 12 blocks. GPT-3 stacks 96. Each block has its own separate weights.

```text
Embedding + position
    ↓
Block 1:  learns basic grammar ("articles before nouns")
    ↓
Block 2:  learns local patterns ("adjective + noun pairs")
    ↓
  ...
    ↓
Block 6:  learns medium-range patterns ("subject-verb agreement")
    ↓
  ...
    ↓
Block 12: learns long-range patterns ("the answer to the question asked 3 sentences ago")
    ↓
Final output vectors (5 × 768)
```

---

### Stage 5: Prediction Head — [DIFFERENT FROM ENCODER]

This is where encoder and decoder truly diverge. An encoder pools all token vectors into one embedding. A decoder takes each token's output vector and predicts the **next token**.

```text
The prediction head is a single linear layer:
    hidden vector (768-d) → vocabulary logits (50,257-d)

It maps from "what the model understands" to "probability of every possible next word."
```

**Toy walkthrough** (4-dim hidden, 8-word vocabulary):

```text
After 12 blocks, the output vector for position 4 ("the") is:
    h₄ = [0.82, -0.15, 0.63, 0.41]

Prediction head weight matrix W (4 × 8):
    Each column corresponds to one word in the vocabulary.

    h₄ × W + bias = logits:
    [2.1, 0.3, -0.5, 3.8, 0.1, -1.2, 1.5, 0.9]
     mat  dog  car   .    sat  the   on   cat

Apply softmax → probabilities:
    P(mat) = 0.14
    P(dog) = 0.02
    P(car) = 0.01
    P(".")  = 0.76    ← model thinks period is most likely
    P(sat) = 0.02
    P(the) = 0.01
    P(on)  = 0.03
    P(cat) = 0.01

The model picks one token (how it picks is covered in file 03 — sampling strategies).
```

**Weight tying (common trick):**

```text
The embedding table (Stage 2):         50,257 × 768   (token ID → vector)
The prediction head (Stage 5):         768 × 50,257   (vector → token probabilities)

These are transposes of each other!

Many models (GPT-2, LLaMA) share the same weight matrix:
    prediction_head.weight = embedding_table.weight.T

Why: saves ~39M parameters, and it makes sense —
"the vector for 'cat'" and "predicting 'cat' from a vector"
are related operations.
```

---

## The Full Picture: One Forward Pass

```text
Input: "The cat sat on the"
Goal:  predict the next token for every position

Step 1 — Tokenize:    [464, 3797, 3332, 319, 262]
Step 2 — Embed:       5 × 768 matrix (one row per token)
Step 3 — + Positions: 5 × 768 matrix (position-aware)
Step 4 — 12 blocks:   5 × 768 matrix (context-aware, causal)
                      Each token's vector now encodes everything
                      it saw to its LEFT (not right!)
Step 5 — Pred head:   Each row → 50,257 logits → softmax

    Position 0 ("The")  → predicts: "cat"  with P = 0.003  (hard with no context!)
    Position 1 ("cat")  → predicts: "sat"  with P = 0.02
    Position 2 ("sat")  → predicts: "on"   with P = 0.05
    Position 3 ("on")   → predicts: "the"  with P = 0.15
    Position 4 ("the")  → predicts: "mat"  with P = 0.14

Training loss = cross-entropy averaged across all 5 positions.
Backprop updates all weights: embeddings, positions, all 12 blocks, prediction head.
```

---

## What Gets Trained (All the Weights)

```text
| Component           | Shape (GPT-2)           | Count   | Learned how                  |
| ------------------- | ----------------------- | ------- | ---------------------------- |
| Token embeddings    | 50,257 × 768            | 38.6M   | Backprop (next-token loss)   |
| Position embeddings | 1,024 × 768             | 0.8M    | Backprop (next-token loss)   |
| Per block (×12):    |                         |         |                              |
|   W_Q, W_K, W_V     | 768 × 768 each          | 1.8M    | Backprop (next-token loss)   |
|   W_O               | 768 × 768               | 0.6M    | Backprop (next-token loss)   |
|   FFN up             | 768 × 3072              | 2.4M    | Backprop (next-token loss)   |
|   FFN down           | 3072 × 768              | 2.4M    | Backprop (next-token loss)   |
|   LayerNorm (×2)    | 768 × 2                 | 0.003M  | Backprop (next-token loss)   |
| Prediction head     | 768 × 50,257            | tied    | Shared with token embeddings |
| ─────────────────── | ─────────────────────── | ─────── | ──────────────────────────── |
| TOTAL               |                         | ~124M   | All trained with one game:   |
|                     |                         |         | predict the next token       |
```

Every single weight is trained with the same loss function: "given everything to the left, predict the next token." There is no separate game for attention vs FFN vs embeddings. One loss, one backward pass, all weights update together.

---

## How a Decoder Differs From an Encoder: Summary

```text
|                        | Encoder (BERT)            | Decoder (GPT)              |
| ---------------------- | ------------------------- | -------------------------- |
| Attention direction    | Bidirectional (sees all)  | Causal (sees only left)    |
| Output                 | One vector per token      | One vector per token       |
|                        | → pool into 1 embedding   | → predict next token       |
| Prediction head        | None (or MLM head)        | Linear → softmax over vocab|
| Training game          | Masked Language Model     | Next-token prediction      |
| What it's good at      | Understanding / retrieval | Generation / conversation  |
| Special tokens         | [CLS], [SEP], [MASK]     | <BOS>, <EOS>               |
| Positional constraint  | Max 512 tokens (BERT)     | Max 1024+ tokens           |
```

The transformer blocks themselves (attention → residual → LayerNorm → FFN → residual → LayerNorm) are **structurally identical**. The causal mask and prediction head are the only differences. That's it.
