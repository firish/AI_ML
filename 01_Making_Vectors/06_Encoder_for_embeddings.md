## What Is a Text Encoder?

### The problem with everything we've done so far

Pooling and Doc2Vec build sentence vectors from **frozen** word vectors. The word "bank" always has the same vector whether the sentence is:
- "I sat on the river **bank**"
- "I deposited money at the **bank**"

An encoder reads the **entire sentence at once** and produces context-aware vectors — "bank" gets a different representation depending on its neighbours. Then it collapses all those vectors into one fixed-length sentence vector.

```text
One-line summary:

    raw text in → one fixed-length vector out
    "The cat sat on the mat" → [0.12, -0.45, 0.78, ..., 0.33]   (e.g., 768 numbers)
```

### The progression so far

```text
One-hot / BoW / TF-IDF     → no learning, no semantics
Word2Vec / GloVe / FastText → learned word vectors, but static (one vector per word)
Pooling / Doc2Vec           → sentence vectors, but built from static word vectors
Encoder (this file)         → sentence vectors where every word is context-aware
```

---

## Architecture: The 6-Stage Pipeline

```text
Stage 1: Tokenizer          "The cat sat" → [101, 1996, 4937, 2938, 102]
Stage 2: Embedding lookup    each ID → a vector from a learned table
Stage 3: Positional encoding add position info so the model knows word order
Stage 4: Transformer blocks  self-attention + feed-forward, repeated 6-12 times
Stage 5: Pooling             collapse all token vectors into one sentence vector
Stage 6: Normalize           scale to unit length (for cosine similarity)
```

```text
| Stage                   | Shape (BERT-base)       | What happens                                    |
| ----------------------- | ----------------------- | ----------------------------------------------- |
| Tokenizer               | → sequence of IDs       | Splits text into subword tokens                 |
| Embedding lookup        | n_tokens × 768          | Each ID gets a 768-dim vector                   |
| + Positional encoding   | n_tokens × 768          | Position vectors added to each row              |
| Transformer blocks (×12)| n_tokens × 768          | Context mixing — shape doesn't change           |
| Pooling                 | 1 × 768                 | Mean of all rows (or take [CLS] row)            |
| Normalize               | 1 × 768                 | Scale to length 1                               |
```

Let's walk through each stage with real (toy) numbers.

---

## Toy Example: "The cat sat"

We'll use **4 tokens × 3 dimensions** (real BERT uses ~512 tokens × 768 dims, but the mechanics are identical).

### Stage 1: Tokenizer

The tokenizer splits text into **subword pieces** and maps each to an integer ID.

```text
Input:  "The cat sat"

BERT uses WordPiece tokenization:
    1. Add special tokens: [CLS] at start, [SEP] at end
    2. Split into subword pieces
    3. Look up each piece in a vocabulary of ~30,000 entries

Result: [CLS]  the   cat   sat   [SEP]
IDs:     101   1996  4937  2938   102
```

**What is [CLS]?** A special "blank" token prepended to every input. It has no linguistic meaning — it's a slot that the model learns to fill with a summary of the whole sentence. Some models use this as the sentence vector.

**What is [SEP]?** Marks the end of a sentence (or separates two sentences in pair tasks).

**Why subwords?** The vocabulary can't contain every possible word. WordPiece breaks rare words into known pieces:
```text
"unhappiness" → ["un", "##happiness"]
"transformers" → ["transform", "##ers"]
```
This means the model can handle any word, even ones it's never seen, by composing known pieces.

For our toy example, we'll drop [SEP] and work with 4 tokens:

```text
[CLS]   the     cat     sat
```

### Stage 2: Embedding Lookup

Each token ID is used to look up a row in a learned embedding table (exactly like Word2Vec's weight matrix, but this one gets further refined during training).

```text
Embedding table (learned during pre-training):
    ID 101  ([CLS]) → [0.10, 0.00, 0.10]
    ID 1996 (the)   → [0.00, 0.20, 0.00]
    ID 4937 (cat)   → [0.30, 0.10, 0.00]
    ID 2938 (sat)   → [0.20, 0.00, 0.20]

Stack into matrix X (4 rows × 3 cols):
    X = [[0.10, 0.00, 0.10],    ← [CLS]
         [0.00, 0.20, 0.00],    ← the
         [0.30, 0.10, 0.00],    ← cat
         [0.20, 0.00, 0.20]]    ← sat
```

### Stage 3: Positional Encoding

Self-attention treats its input as a **set**, not a sequence — it has no built-in notion of "first word" vs "last word." Positional encoding fixes this by adding a unique position vector to each row.

```text
Position vectors (one per position, either learned or computed via sine/cosine):
    pos_0 = [0.01, 0.00, 0.01]
    pos_1 = [0.00, 0.01, 0.00]
    pos_2 = [0.01, 0.01, 0.00]
    pos_3 = [0.00, 0.00, 0.01]

X = X + position_vectors:
    [CLS]: [0.10, 0.00, 0.10] + [0.01, 0.00, 0.01] = [0.11, 0.00, 0.11]
    the:   [0.00, 0.20, 0.00] + [0.00, 0.01, 0.00] = [0.00, 0.21, 0.00]
    cat:   [0.30, 0.10, 0.00] + [0.01, 0.01, 0.00] = [0.31, 0.11, 0.00]
    sat:   [0.20, 0.00, 0.20] + [0.00, 0.00, 0.01] = [0.20, 0.00, 0.21]
```

Now "cat" at position 2 has a different vector than "cat" at position 5 would — the model can distinguish them.

**Where do these position vectors come from?** Two approaches:

**Approach 1 — Sinusoidal (original Transformer, 2017):** Computed by a fixed formula, no learning involved:
```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

where:  pos = position in the sentence (0, 1, 2, ...)
        i   = dimension index
        d   = total dimensions (e.g., 768)
```
Each dimension uses a sine or cosine wave at a different frequency. Position 0 gets one pattern of values, position 1 gets a slightly different pattern — so every position has a unique fingerprint. The model doesn't learn these; it learns to *read* them during training.

**Approach 2 — Learned (BERT, GPT):** Just a lookup table — exactly like the word embedding table, but indexed by position instead of token ID:
```text
Position embedding table (randomly initialised, updated by backprop):
    position 0   → [0.01, 0.00, 0.01, ...]
    position 1   → [0.00, 0.01, 0.00, ...]
    ...
    position 511 → [0.03, -0.02, 0.01, ...]
```
The model learns "what does it mean to be at position 5" the same way it learns "what does 'cat' mean" — through gradient descent during pre-training. The values start random and settle into useful patterns.

In our toy example, the position values are just small made-up numbers to show the mechanics. In a real model they'd come from one of these two approaches — the rest of the pipeline works identically either way.

### Stage 4: Transformer Block (the core)

This is where the magic happens. Each block has two sub-steps:
1. **Self-attention**: let every token look at every other token
2. **Feed-forward network**: process each token independently to create new features

A full block:
```text
input → Self-Attention → Add & LayerNorm → Feed-Forward → Add & LayerNorm → output
              ↑                                    ↑
          residual skip                       residual skip
```

BERT-base stacks **12** of these blocks. Let's trace through one.

---

#### Step 4a: Self-Attention

**The question each token asks:** "Which other tokens should I pay attention to, and how much?"

Each token's vector is transformed into three roles:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I offer?"

```text
Three small weight matrices (learned during training):
    W_Q = [[1.0, 0.0, 0.0],      W_K = [[0.5, 0.0, 0.0],      W_V = [[0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],              [0.0, 0.5, 0.0],              [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0]]              [0.0, 0.0, 0.5]]              [0.0, 0.0, 0.5]]

Compute Q, K, V for each token:
    Q = X × W_Q    (each row × W_Q)
    K = X × W_K    (each row × W_K)
    V = X × W_V    (each row × W_V)
```

```text
Q (what each token is looking for):
    [CLS]: [0.11, 0.00, 0.11]
    the:   [0.00, 0.21, 0.00]
    cat:   [0.31, 0.11, 0.00]
    sat:   [0.20, 0.00, 0.21]

K (what each token advertises):
    [CLS]: [0.055, 0.00, 0.055]
    the:   [0.00,  0.105, 0.00]
    cat:   [0.155, 0.055, 0.00]
    sat:   [0.10,  0.00,  0.105]

V (what each token offers):
    (same as K in this toy example)
```

**Compute attention scores**: each token's Query dot-producted with every token's Key.

```text
scores = Q × K^T (4×3 times 3×4 = 4×4 matrix)

For token "cat" (row 2), its query [0.31, 0.11, 0.00] dotted with each key:
    cat·[CLS] = 0.31×0.055 + 0.11×0.00  + 0.00×0.055 = 0.017
    cat·the   = 0.31×0.00  + 0.11×0.105 + 0.00×0.00  = 0.012
    cat·cat   = 0.31×0.155 + 0.11×0.055 + 0.00×0.00  = 0.054
    cat·sat   = 0.31×0.10  + 0.11×0.00  + 0.00×0.105 = 0.031
```

**Scale by √d_k**: divide by √3 ≈ 1.73 to keep values stable (without this, large dot products push softmax into extreme regions where gradients vanish).

```text
cat's scaled scores: [0.010, 0.007, 0.031, 0.018]
```

**Softmax** to turn scores into weights that sum to 1:

```text
cat's attention weights: [0.24, 0.24, 0.27, 0.25]

Reading: "cat" pays 27% attention to itself, ~24-25% to each other token.
(In real models with more dimensions, these would be much more peaked.)
```

**Weighted sum of Values**: cat's new vector = weighted average of all Value vectors:

```text
cat_new = 0.24 × V_[CLS] + 0.24 × V_the + 0.27 × V_cat + 0.25 × V_sat

Before attention: cat = [0.31, 0.11, 0.00]  (knows only about itself)
After attention:  cat = [0.08, 0.04, 0.04]  (contains info from all tokens)
```

**The key insight**: after self-attention, "cat" is no longer just "cat" — it's "cat in the context of [CLS], the, and sat." This is how the encoder solves the "bank" problem: the word's representation changes based on its neighbours.

Do this for all 4 tokens → we get a new 4×3 matrix where every row is now context-aware.

---

#### Step 4b: Multi-Head Attention

The example above shows **single-head** attention — one set of Q, K, V matrices learning one pattern of "who should attend to whom."

Real transformers run **multiple heads in parallel**, each learning a different type of relationship:

```text
BERT-base: 12 heads, each working in 64 dimensions (12 × 64 = 768 total)

Head 1: might learn syntax         ("sat" attends to "cat" — subject-verb)
Head 2: might learn proximity      (each word attends to its neighbours)
Head 3: might learn coreference    ("it" attends to "the cat")
...
Head 12: might learn punctuation   (period attends to sentence start)
```

How it works:
```text
1. Split 768-dim space into 12 chunks of 64 dims each
2. Each head has its own W_Q, W_K, W_V (768 → 64)
3. Each head runs attention independently
4. Concatenate all outputs: 12 × 64 = 768 dims
5. Multiply by W_O (768 × 768) to mix head outputs
```

**Why not one big head?** A single head can only learn one attention pattern per layer. Multiple heads let the model attend to different types of relationships simultaneously.

---

#### Step 4c: Residual Connection + LayerNorm

**Residual (skip) connection**: add the attention output back to the original input:
```text
output = input + attention(input)
```

Why? In a 12-layer stack, gradients need to flow all the way back to layer 1 during training. Without skip connections, gradients shrink exponentially (vanishing gradient problem). The skip connection gives gradients a direct highway back.

**LayerNorm**: normalize each row to keep numbers stable:
```text
For each row:
    mean = average of all values in the row
    std  = standard deviation
    row  = (row - mean) / std
    row  = row × γ + β    (learned scale and shift)
```

This prevents values from growing or shrinking as they pass through many layers.

---

#### Step 4d: Feed-Forward Network (FFN)

A small two-layer neural net applied to **each token independently**:

```text
hidden = GELU(row × W1 + b1)     ← expand: 768 → 3072
output = hidden × W2 + b2         ← shrink: 3072 → 768
```

**Why expand then shrink?** The expansion gives the model room to compute new features in high-dimensional space (e.g., "this is a past-tense verb", "this is the subject"), then compress back down.

Another residual connection + LayerNorm after this.

---

#### Zoomed Out: The Full Encoder Architecture

Now let's step back and see how all the pieces fit together. This is the part that's easy to miss when you learn each component individually.

**What exactly repeats 12 times?**

One complete transformer block. Each block is its own self-contained unit with its **own separate weights**. Here's one block, fully expanded:

```text
┌─────────────────────────────────────────────────────────────┐
│ BLOCK 1 (has its own weights, independent of other blocks)  │
│                                                             │
│   input (n_tokens × 768)                                    │
│     │                                                       │
│     ├───────────────┐                                       │
│     ↓               │ (saved for residual)                  │
│   Multi-Head        │                                       │
│   Attention         │                                       │
│   (W_Q, W_K, W_V   │                                       │
│    × 12 heads,      │                                       │
│    + W_O)           │                                       │
│     │               │                                       │
│     ↓               │                                       │
│   ADD ←─────────────┘  ← residual: output + input           │
│     ↓                                                       │
│   LayerNorm (γ₁, β₁)                                       │
│     │                                                       │
│     ├───────────────┐                                       │
│     ↓               │ (saved for residual)                  │
│   Feed-Forward      │                                       │
│   (W1, b1, W2, b2) │                                       │
│     │               │                                       │
│     ↓               │                                       │
│   ADD ←─────────────┘  ← residual: output + input           │
│     ↓                                                       │
│   LayerNorm (γ₂, β₂)                                       │
│     │                                                       │
│   output (n_tokens × 768)  ← same shape as input            │
└─────────────────────────────────────────────────────────────┘
```

BERT-base stacks 12 of these, each with completely separate weights:

```text
Embedding + Positional Encoding → (n_tokens × 768)
    ↓
┌── Block 1 (own W_Q, W_K, W_V ×12 heads, W_O, W1, b1, W2, b2, γ₁, β₁, γ₂, β₂) ──┐
│   attention → add+norm → FFN → add+norm                                            │
└─── output: n_tokens × 768 ─────────────────────────────────────────────────────────┘
    ↓
┌── Block 2 (its OWN W_Q, W_K, W_V ×12 heads, W_O, W1, b1, W2, b2, γ₁, β₁, γ₂, β₂)┐
│   attention → add+norm → FFN → add+norm                                            │
└─── output: n_tokens × 768 ─────────────────────────────────────────────────────────┘
    ↓
    ... (blocks 3-11) ...
    ↓
┌── Block 12 (its OWN weights) ──────────────────────────────────────────────────────┐
│   attention → add+norm → FFN → add+norm                                            │
└─── output: n_tokens × 768 ─────────────────────────────────────────────────────────┘
    ↓
Pooling → (1 × 768)
    ↓
Normalize → final sentence vector
```

Each block has **two** residual connections (one after attention, one after FFN) and **two** LayerNorms. So 12 blocks = 24 residual connections + 24 LayerNorms.

The shape never changes — it's n_tokens × 768 at every stage. What changes is the *information content*: early blocks blend nearby tokens, later blocks capture whole-sentence meaning.

---

**What weights get trained during backprop?**

Every weight listed below is learned — updated by gradient descent during pre-training:

```text
Layer                          Trainable weights              Count (BERT-base)
─────────────────────────────────────────────────────────────────────────────────
Embedding table                one 768-d vector per token     30,522 × 768
Position embedding table       one 768-d vector per position  512 × 768

Per block (×12 blocks):
  Multi-head attention:
    W_Q (768×768)              12 heads × (768→64) per head   768 × 768
    W_K (768×768)              same                           768 × 768
    W_V (768×768)              same                           768 × 768
    W_O (768×768)              mixes head outputs             768 × 768
  LayerNorm 1:
    γ₁, β₁                    scale and shift (768 each)     768 + 768
  Feed-forward:
    W1 (768×3072) + b1         expand                         768 × 3072 + 3072
    W2 (3072×768) + b2         shrink                         3072 × 768 + 768
  LayerNorm 2:
    γ₂, β₂                    scale and shift (768 each)     768 + 768

Total: ~110 million parameters, all trained by backprop
```

Key point: **Block 1's W_Q is a completely different matrix from Block 2's W_Q.** They're not shared. Each block independently learns what to attend to and what features to extract at its depth in the stack.

---

**Residual connections: are they for backprop or forward pass?**

Both — and that's the beauty.

```text
FORWARD PASS (inference):
    The residual adds the original input back to the transformed output.
    This means the output is "original info + new info from attention/FFN."
    Even if a block's transformation is unhelpful, the original signal passes through.

    residual_output = input + attention(input)
                      ─────   ────────────────
                      "keep     "add whatever
                      what I     new context
                      had"       I learned"

BACKWARD PASS (training):
    Gradients need to flow from Block 12 all the way back to Block 1.
    Without residuals: gradient passes through 12 attention layers and
    12 FFN layers — it shrinks exponentially (vanishing gradient).

    With residuals: the gradient has a direct shortcut through the
    addition operation. The derivative of (x + f(x)) with respect to x
    is (1 + f'(x)) — that "1" means the gradient always has a path
    that doesn't shrink.

    Block 12 ←── gradient ──← Block 11 ←── ... ←── Block 1
         └──── shortcut ────────────────────────────────┘
              (gradient flows through the "+" without shrinking)
```

Without residual connections, training a 12-layer transformer would be nearly impossible — gradients would vanish before reaching the early layers. With them, even Block 1's weights get meaningful gradient updates.

---

### Stage 5: Pooling

After 12 blocks, we have a refined 4 × 768 matrix (one context-aware vector per token). We need **one** vector for the whole sentence.

Two common approaches:

```text
1. Mean pooling (most Sentence-BERT models):
    v_sentence = average of all token rows
    Simple, works well in practice.

2. [CLS] pooling (original BERT):
    v_sentence = just take the [CLS] row
    [CLS] was trained to summarise the sentence.
    Works okay, but mean pooling usually beats it for similarity tasks.
```

### Stage 6: Normalize

Scale the vector to unit length so cosine similarity works cleanly:

```text
v = v / ||v||

After this, cosine_similarity(a, b) = dot_product(a, b)
(because both vectors have length 1)
```

---

## How Is an Encoder Trained?

### How the architecture, the toy example, and the "games" relate

This is easy to get confused about, so let's be explicit. The architecture above (embedding → 12 transformer blocks → pooling) is **the machine**. It exists at all times. The toy example in Stage 4 shows what happens during **one forward pass** through that machine — and that exact computation happens during training, fine-tuning, AND inference. The difference between phases is only **what game drives the weight updates**.

```text
Timeline of a model like all-MiniLM-L6-v2:

Phase 0: Build the architecture
    → Define the structure (12 blocks, 768 dims, 12 heads, etc.)
    → Randomly initialise all ~110M weights
    → The model exists but knows nothing — outputs are garbage

Phase 1: Pre-training (BERT's game)
    → Feed millions of sentences through the SAME architecture
    → Play "fill in the blank" (MLM)
    → Backprop updates all 110M weights
    → After billions of steps: the model understands language
    → But sentence vectors aren't optimised for similarity yet

Phase 2: Fine-tuning (Sentence-BERT's game)
    → Take the SAME architecture with its pre-trained weights
    → Feed sentence pairs through it
    → Play "push similar close, push dissimilar far"
    → Backprop adjusts the SAME weights further
    → Now sentence vectors are good for similarity search

Phase 3: Inference (your code)
    → Freeze all weights (no more learning)
    → Feed your text through the SAME architecture
    → Get a sentence vector out
    → This is what model.encode("your text") does
```

The toy example (Stage 4) is showing what happens inside the box at **every** phase — the forward pass is always the same computation. The only thing that changes between phases is: what loss function compares the output to what target, and therefore what direction the gradients push the weights.

---

### Phase 1: Pre-training (BERT's game — learn language)

**Masked Language Model (MLM):** randomly mask 15% of tokens, make the model predict them.

```text
Input:  "The [MASK] sat on the mat"
Target: "cat"

What happens:
    1. Tokenize → embed → add positions                    (stages 1-3)
    2. Run through all 12 transformer blocks               (stage 4 — the toy example)
    3. The output vector for the [MASK] position now
       contains context from "the", "sat", "on", "the", "mat"
    4. A small prediction head on top: [MASK] vector → softmax over vocabulary
       → P("cat") should be high
    5. Loss = cross-entropy between prediction and "cat"
    6. Backprop through the entire architecture:
       prediction head → block 12 → block 11 → ... → block 1 → embeddings
    7. All ~110M weights get a small update
```

This is trained on massive text (Wikipedia + BookCorpus for BERT, even more for RoBERTa). After billions of masked predictions, the transformer blocks have learned grammar, word meaning, and world knowledge — they produce rich, context-aware token vectors.

**But there's a problem:** pre-training optimises for predicting individual words, not for comparing sentences. If you take two similar sentences and compare their pooled vectors, the similarity is mediocre (~0.4). The weights need one more round of adjustment.

### Phase 2: Fine-tuning for embeddings (Sentence-BERT's game — learn similarity)

**Contrastive learning:** teach the model that similar sentences should have similar vectors.

```text
Training triplets:
    Anchor:   "How do I sort a list in Python?"
    Positive: "Python list sorting methods"         ← should be close
    Negative: "What is the GDP of Canada?"           ← should be far

What happens:
    1. Run anchor through the architecture → vector_A       (all 6 stages)
    2. Run positive through SAME architecture → vector_B    (same weights)
    3. Run negative through SAME architecture → vector_C    (same weights)
    4. Compute: sim(A, B) and sim(A, C)
    5. Loss pushes sim(A, B) UP and sim(A, C) DOWN
    6. Backprop through the architecture (again, same blocks)
    7. Weights shift so that meaning-similarity = vector-closeness
```

```text
Before fine-tuning (after Phase 1 only):
    sim("How do I sort a list?", "Python list sorting") ≈ 0.4  (mediocre)

After fine-tuning:
    sim("How do I sort a list?", "Python list sorting") ≈ 0.85 (good!)
```

After fine-tuning on millions of pairs, the encoder produces vectors where **semantic closeness = vector closeness**. The model is then frozen and published.

**Key point:** Phase 2 doesn't add new layers or change the architecture. It's the same 12 blocks, same attention, same FFN. It just adjusts the existing weights so that the final pooled vector captures meaning in a way that's useful for similarity comparison.

---

## Using Encoders for Long Text

Most encoders have a **512-token limit** (some newer ones go to 8192). For longer documents:

```text
Strategy: Chunking with overlap

Document: 2000 tokens long
Chunk size: 256 tokens
Overlap: 50 tokens

Chunk 1: tokens 0-255      → encode → vector_1
Chunk 2: tokens 206-461    → encode → vector_2
Chunk 3: tokens 412-667    → encode → vector_3
...

Options:
  A. Store all chunk vectors in a vector DB (most common — used in RAG)
  B. Average all chunk vectors into one "document vector" (quick but lossy)
```

The overlap ensures no information is lost at chunk boundaries.

---

## Code Demo

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-d

sentences = [
    "How do I sort a list in Python?",
    "Python list sorting methods",
    "What is the GDP of Canada?",
]

vectors = model.encode(sentences, normalize_embeddings=True)  # shape: (3, 384)

print(f"sort↔sorting:  {util.cos_sim(vectors[0], vectors[1]).item():.3f}")  # ~0.85
print(f"sort↔GDP:      {util.cos_sim(vectors[0], vectors[2]).item():.3f}")  # ~0.05
```
