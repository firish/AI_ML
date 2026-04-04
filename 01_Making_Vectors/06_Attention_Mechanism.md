# The Attention Mechanism

File 05 covered RNNs and LSTMs — models that read sequences one token at a time. They work, but they have a fundamental bottleneck: information from early tokens must survive through every intermediate step to reach the end. Long-range dependencies get lost.

Attention solves this by letting every token look directly at every other token. No bottleneck. No forgetting. This file explains the mechanism from the ground up.

---

## 1. The Intuition: A Library Search

Before any formulas, understand what attention is doing with an analogy.

```
You walk into a library looking for information about "French geography."

Your QUERY: "French geography"
    What YOU are looking for.

Every book has two things:

    KEY: the title/label on the spine
        "History of France"
        "Italian Cooking"
        "European Geography"
        "French Wine Regions"

    VALUE: the actual content inside the book
        [500 pages about French history]
        [300 pages of Italian recipes]
        [400 pages of European maps]
        [200 pages about Bordeaux, Burgundy]

Step 1: Compare your QUERY against every KEY
    "French geography" vs "History of France"      → decent match
    "French geography" vs "Italian Cooking"        → bad match
    "French geography" vs "European Geography"     → good match
    "French geography" vs "French Wine Regions"    → partial match

Step 2: Use those match scores to grab VALUES
    Take 50% of European Geography's content
    + 30% of French History's content
    + 19% of French Wine's content
    + 1% of Italian Cooking's content

    = a weighted blend of the actual information you needed
```

That's attention. Every token in a sentence is simultaneously a book in the library AND a person searching it. Every token asks a question (Query) and every token offers an answer (Key + Value).

---

## 2. Q, K, V — What Each One Means

Every token generates three vectors from its embedding:

```
Query (Q):  "What am I looking for in other tokens?"
Key (K):    "What should I advertise so others can find me?"
Value (V):  "What information should I provide when someone matches with me?"
```

### Why THREE separate vectors?

A single token plays three roles simultaneously:

```
The word "bank" in "I sat on the river bank"

    As a Query: "I need context — what's around me?
                 Am I a financial bank or a river bank?"

    As a Key:   "I'm a noun, I'm about a physical location,
                 I'm related to nature/water."
                 (This is what "bank" advertises to OTHER tokens
                  looking for something relevant.)

    As a Value: "Here's my actual semantic content about being
                 a river bank."
                 (This is what gets sent when someone matches.)
```

### Why K and V are different

This is the most confusing part. Why not just have Q and one other vector?

```
Key = what you ADVERTISE for matching purposes
Value = what you actually CONTAIN

These serve different roles:

    Token "animal" in: "The animal didn't cross the street because it was too tired"

    Key of "animal":   "I'm a living thing, capable of being tired"
                        (this is the LABEL — helps "it" find its referent)

    Value of "animal": [the full semantic representation of "animal"]
                        (this is the CONTENT — gets blended into "it"'s output)

    Token "street":

    Key of "street":   "I'm a physical object, not capable of being tired"
                        (this label tells "it" that "street" is a bad match)

    Value of "street": [the full semantic representation of "street"]

    Q("it") × K("animal") → HIGH score   (tired matches living thing)
    Q("it") × K("street") → LOW score    (tired doesn't match an object)

    The Key helped MATCH. The Value provided the CONTENT.

    If K and V were the same vector, the model couldn't separate
    "what I advertise for matching" from "what I actually contain."
```

### Why W_v exists — why project the value at all?

Your raw 768-dim embedding contains EVERYTHING about a token:

```
The raw embedding of "bank" contains:
    - It's a noun
    - It can mean financial institution OR river bank
    - Its position in the sentence
    - Its phonetic properties (sounds like "blank", "band")
    - Its frequency in English
    - Dozens of other entangled features

ALL of this is packed into one 768-dim vector.
```

W_v is a filter — it extracts the subset of information useful for THIS layer:

```
Layer 3 might be resolving word sense:
    W_v extracts: location vs institution signal
    Suppresses: phonetics, frequency, visual similarity
    "If you attend to me, here's the relevant part: I'm a location"

Layer 8 might be about syntactic role:
    W_v extracts: noun, object of preposition
    Suppresses: semantic meaning, word sense
    "If you attend to me, here's the relevant part: I'm a noun"

Layer 15 might be about long-range reasoning:
    W_v extracts: connection to "river" from 5 tokens ago
    Suppresses: local grammar details

DIFFERENT layers need DIFFERENT information from the same token.
W_v lets each layer extract what IT needs.
```

Without W_v (just passing raw embeddings):

```
Every layer gets the SAME 768-dim blob when attending to "bank."
The RECEIVING token has to sort through all 768 dimensions to
find the 10 dimensions it actually needs.

With W_v: the SENDER filters before sending.
    "You matched with my key, so you want my value.
     Here — I've already extracted the relevant part for you."

Library analogy:
    Without W_v: the library gives you the entire book.
    With W_v:    the library gives you a summary relevant
                 to your section of the library (the layer's purpose).
```

Each layer has its own W_q, W_k, AND W_v. All three are learned during training. The model discovers:

```
Layer 5:
    W_q learns: "ask about word sense"
    W_k learns: "advertise word sense"
    W_v learns: "provide word sense information"

All three are aligned to the same PURPOSE for that layer.
Q and K find the right match. V provides the right content
for that match. All three projections are one coordinated system.
```

---

## 3. The Formulas — What Each Step Does

Now the formulas should feel obvious, not mysterious.

### Step 1: Project Q, K, V

```
For each token embedding x (768-dim):

    Q = x × W_q     (768 × 64) → "what am I looking for?"
    K = x × W_k     (768 × 64) → "what do I advertise?"
    V = x × W_v     (768 × 64) → "what do I provide?"

    768 → 64: each head works in a smaller subspace.
    (12 heads × 64 = 768, covering the full space.)

    W_q, W_k, W_v are LEARNED weight matrices.
    They're what makes attention powerful — the model discovers
    what to look for, what to advertise, and what to provide.
```

### Step 2: Compute attention scores — Q × Kᵀ

```
scores = Q × Kᵀ

Q: (seq, 64)     — one query per token
K: (seq, 64)     — one key per token
Kᵀ: (64, seq)    — transposed for the matrix multiply

Q × Kᵀ = (seq, seq)  — one score for every pair of tokens

    Entry [i, j] = dot product of Q_i and K_j
                  = "how relevant is token j to token i's query?"

The dot product measures SIMILARITY (from your linear algebra files).
    Q and K pointing in similar directions → large dot product → high relevance
    Q and K pointing in different directions → small dot product → low relevance

The transpose (Kᵀ) is just mechanics — it makes the matrix multiply
produce all pairwise dot products in one operation. Nothing deeper.
```

### Step 3: Scale — divide by √d_k

```
scores = Q × Kᵀ / √d_k

d_k = 64 (dimension per head)
√d_k = √64 = 8

Why divide?
    The dot product sums 64 multiplications.
    More dimensions → larger raw scores.

    Problem: softmax is sensitive to magnitude.
        softmax([1, 2, 3])    = [0.09, 0.24, 0.67]   — spread out, useful
        softmax([10, 20, 30]) = [0.00, 0.00, 1.00]   — all on one token

    Large scores → softmax becomes a hard argmax →
    model attends to ONLY ONE token, ignores everything else.
    Gradients vanish for other positions. Learning breaks.

    Dividing by √d_k normalizes the scale.
    Raw scores ~64 → divided by 8 → scores ~8.
    Softmax stays in a useful range where attention can
    spread across multiple tokens.
```

### Step 4: Softmax — turn scores into weights

```
weights = softmax(scores)

Converts raw scores into a probability distribution:
    - All values become positive
    - They sum to 1
    - Higher scores → higher weights

    Raw scores:  [2.1, -0.5, 8.3, 1.2]
    After softmax: [0.02, 0.00, 0.97, 0.01]

    "Almost all attention on the third token."

Why softmax specifically?
    We need a weighted average of values.
    Weights must be positive and sum to 1.
    Softmax does exactly this, and it's differentiable
    (so gradients can flow through it during training).
```

### Step 5: Weighted sum of values — weights × V

```
output = weights × V

    weights: (seq, seq)    — how much each token attends to each other
    V:       (seq, 64)     — what each token offers
    output:  (seq, 64)     — what each token receives

For token "it" with weights [0.02, 0.00, 0.97, 0.01]:

    output = 0.02 × V("The")
           + 0.00 × V("street")
           + 0.97 × V("animal")
           + 0.01 × V("didn't")

    Token "it" receives mostly "animal"'s value.
    It now KNOWS it refers to the animal, encoded in its vector.
```

### The complete formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V

Breaking it down:
    Q × Kᵀ           "compare every query to every key"     → similarity scores
    / √d_k           "normalize so softmax doesn't collapse" → scaled scores
    softmax(...)      "convert to weights that sum to 1"      → attention weights
    × V              "grab weighted blend of values"          → output
```

---

## 4. Multi-Head Attention — Why 12 Heads, Not 1?

One attention head can only focus on one type of relationship at a time. Multiple heads let the model attend to different things simultaneously.

```
Full embedding: 768 dims
Split into 12 heads: each head works with 64 dims

Head 0:  might learn to attend to syntactic subject
Head 1:  might learn to attend to the previous word
Head 3:  might learn to attend to semantic similarity
Head 7:  might learn to attend to the verb of the clause
Head 11: might learn to attend to punctuation/boundaries

All 12 heads run IN PARALLEL on the same input.
Each produces its own 64-dim output.
Concatenate all 12: 12 × 64 = 768 dims → back to full size.
```

### The multi-head reshape dance

```
1. Start:     (batch, seq, 768)
2. Project:   Q = input × W_q  → (batch, seq, 768)
              K = input × W_k  → (batch, seq, 768)
              V = input × W_v  → (batch, seq, 768)
3. Reshape:   (batch, seq, 12, 64)         — split into 12 heads
4. Transpose: (batch, 12, seq, 64)         — heads become a batch dim
5. Attention: each head independently:
              scores = Q × Kᵀ / √64        — (batch, 12, seq, seq)
              output = softmax(scores) × V  — (batch, 12, seq, 64)
6. Transpose: (batch, seq, 12, 64)         — undo step 4
7. Reshape:   (batch, seq, 768)            — merge heads back
8. Project:   output × W_o → (batch, seq, 768)   — final linear layer

Steps 3-4: split into heads.
Steps 6-7: merge heads back.
Step 5: the actual attention — runs independently per head.
```

---

## 5. Causal Mask — Why Decoders Can't Look Ahead

In a decoder (GPT, LLaMA), the model generates text left to right. Token 3 shouldn't be able to see token 5 — it hasn't been generated yet.

```
Without mask (encoder — BERT):
    Every token attends to every other token.
    "bank" can see both "river" (before) and "fishing" (after).

    Attention matrix (■ = can attend):
        ■ ■ ■ ■ ■
        ■ ■ ■ ■ ■
        ■ ■ ■ ■ ■
        ■ ■ ■ ■ ■
        ■ ■ ■ ■ ■

With causal mask (decoder — GPT):
    Each token can only attend to itself and previous tokens.
    Token 3 sees tokens 0, 1, 2, 3 — NOT 4 or 5.

    Attention matrix (■ = can attend, · = masked):
        ■ · · · ·
        ■ ■ · · ·
        ■ ■ ■ · ·
        ■ ■ ■ ■ ·
        ■ ■ ■ ■ ■

Implementation: set masked positions to -∞ before softmax.
    softmax(-∞) = 0, so those positions get zero attention weight.
```

Why this matters: during training, the model sees the full sequence but must predict each token using only past context. The mask enforces this. Without it, the model could "cheat" by looking at the answer.

---

## 6. Self-Attention vs Cross-Attention

```
Self-attention:
    Q, K, V all come from the SAME sequence.
    "Each token in this sentence attends to other tokens
     in the same sentence."
    Used in: encoders (BERT), decoders (GPT), every transformer.

Cross-attention:
    Q comes from one sequence, K and V from a DIFFERENT sequence.
    "Each token in the output attends to tokens in the input."
    Used in: encoder-decoder models (translation, T5),
             text-to-image (text tokens attend to image patches).

    Example — translation:
        Encoder processes: "Le chat est assis" (French)
        Decoder generates: "The cat is sitting" (English)

        When generating "cat", the decoder's Q asks:
            "What French word should I translate next?"
        The encoder's K and V for "chat" respond:
            K: "I'm the French word for a feline"
            V: [semantic content of "chat"]
        Q("cat") × K("chat") → high score → pull V("chat")
```

---

## 7. What Attention Replaced — and Why

```
Before attention (RNNs/LSTMs, file 05):
    Information flows left → right through hidden states.
    Token 1 must pass through tokens 2, 3, 4, ... to reach token 50.
    Like a game of telephone — information degrades over distance.
    This is the vanishing gradient problem.

    "The cat that the dog that the bird watched chased sat on the mat"
    By the time "sat" processes, it has weak signal about "cat."

With attention:
    "sat" directly looks at "cat" through Q × K matching.
    Distance doesn't matter. One hop, not 10.
    Token 1 and token 50 are equally accessible.

    This is why transformers killed RNNs:
        RNN: information travels through a chain (O(n) hops)
        Attention: information travels directly (O(1) hops)

Another advantage — parallelism:
    RNN: must process token 1 before token 2 before token 3...
         Sequential. Can't parallelize across tokens.

    Attention: every token's Q, K, V computed independently.
         All pairwise scores computed in one matrix multiply.
         Massively parallel. GPUs love this.

    This is why transformers train so much faster than RNNs.
```

---

## 8. Putting It All Together

```
Input: "The animal didn't cross the street because it was too tired"

Step 1: Each token gets an embedding (768-dim)

Step 2: Each token projects Q, K, V through learned matrices
    Q_"it" = embed("it") × W_q   → "looking for my referent"
    K_"animal" = embed("animal") × W_k → "I'm a living thing"
    K_"street" = embed("street") × W_k → "I'm a physical object"
    V_"animal" = embed("animal") × W_v → [filtered animal semantics]
    V_"street" = embed("street") × W_v → [filtered street semantics]

Step 3: Attention scores (dot products)
    Q_"it" · K_"animal" = 8.5     (high — "tired" matches "living thing")
    Q_"it" · K_"street" = 1.2     (low — "tired" doesn't match "object")
    Q_"it" · K_"the"    = 0.3     (very low — function word, irrelevant)

Step 4: Scale by √d_k
    8.5 / 8 = 1.06
    1.2 / 8 = 0.15
    0.3 / 8 = 0.04

Step 5: Softmax
    [1.06, 0.15, 0.04, ...] → [0.52, 0.09, 0.03, ...]
    "it" puts 52% attention on "animal"

Step 6: Weighted sum of values
    output_"it" = 0.52 × V_"animal" + 0.09 × V_"street" + 0.03 × V_"the" + ...

    The output vector for "it" now carries mostly "animal" information.
    The model resolved the pronoun — "it" = the animal.

This happens for EVERY token, across EVERY head, in EVERY layer.
12 heads × 12 layers = 144 attention operations.
Each one extracts different relationships.
By the final layer, every token's representation is rich with
context from the entire sequence.
```

---

## 9. Summary

```
The attention mechanism:
    Every token generates Q (what I need), K (what I advertise), V (what I offer).
    Match queries to keys (dot product) → get relevance scores.
    Softmax → convert to weights.
    Weighted sum of values → each token absorbs relevant context.

The formula:
    Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V

    Q × Kᵀ           compare queries to keys (similarity)
    / √d_k           prevent softmax from collapsing
    softmax           turn scores into weights (sum to 1)
    × V              weighted blend of values

Why three separate projections (W_q, W_k, W_v):
    W_q: extracts "what am I searching for?"
    W_k: extracts "how should I be found?"
    W_v: extracts "what should I provide?" (filtered per layer)
    Each layer learns its own set — different layers, different purposes.

Multi-head: 12 heads attend to different relationships in parallel.
Causal mask: decoders can only look backward (no peeking at future tokens).
Self vs cross: same-sequence attention vs across two sequences.

Why attention replaced RNNs:
    Direct access (O(1) hops, not O(n) chain)
    Fully parallel (all tokens computed simultaneously)
    No vanishing gradients across distance
```

---

**Previous:** `05_Sequential_Embeddings.md` — RNNs and LSTMs (the sequential approach attention replaced)
**Next:** `06_Attention_embeddings_OR_Encoder.md` — how attention is used inside encoder architectures (BERT) to produce context-aware embeddings
