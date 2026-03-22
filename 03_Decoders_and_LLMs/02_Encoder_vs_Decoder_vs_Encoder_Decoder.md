## Encoder vs Decoder vs Encoder-Decoder

Now that you understand both encoders (Phase 1, file 06) and decoders (file 01), let's see the full picture. There are three ways to use transformer blocks, and every major model falls into one of them.

---

## The Three Architectures

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        All use the SAME building blocks:                    │
│           Attention → Residual → LayerNorm → FFN → Residual → LayerNorm    │
│                                                                             │
│   The ONLY difference: what each token is allowed to attend to.             │
└─────────────────────────────────────────────────────────────────────────────┘

1. ENCODER-ONLY              2. DECODER-ONLY              3. ENCODER-DECODER
   (BERT, RoBERTa)              (GPT, LLaMA, Claude)        (T5, BART)

   Attention: see ALL           Attention: see LEFT          Encoder: see ALL
   tokens (bidirectional)       only (causal)                Decoder: see LEFT
                                                             + see encoder output
                                                               (cross-attention)

   Output: embeddings           Output: next token           Output: next token
   (understanding)              (generation)                 (conditioned generation)
```

---

## 1. Encoder-Only (BERT, RoBERTa, Sentence-BERT)

You already know this from Phase 1.

```text
Input:  "The bank is by the river"
        ↓
        Every token sees every other token (bidirectional)
        "bank" sees both "The" and "river" → knows it's a riverbank
        ↓
Output: One context-aware vector per token → pool into one embedding

Attention mask (all 1s — everything visible):
        The  bank  is   by   the  river
The   [  ✓    ✓    ✓    ✓    ✓    ✓  ]
bank  [  ✓    ✓    ✓    ✓    ✓    ✓  ]
is    [  ✓    ✓    ✓    ✓    ✓    ✓  ]
by    [  ✓    ✓    ✓    ✓    ✓    ✓  ]
the   [  ✓    ✓    ✓    ✓    ✓    ✓  ]
river [  ✓    ✓    ✓    ✓    ✓    ✓  ]
```

**Training game:** Masked Language Model — mask random tokens, predict them from context.

**Good at:** Understanding, classification, similarity, retrieval. Terrible at generation (can't generate left-to-right because it's used to seeing everything).

---

## 2. Decoder-Only (GPT, LLaMA, Claude, Gemini, Mistral)

Covered in detail in file 01.

```text
Input:  "The bank is by the"
        ↓
        Each token sees only itself and what came before (causal)
        ↓
Output: Predict the next token at each position

Attention mask (triangular — future blocked):
        The  bank  is   by   the
The   [  ✓    ✗    ✗    ✗    ✗  ]
bank  [  ✓    ✓    ✗    ✗    ✗  ]
is    [  ✓    ✓    ✓    ✗    ✗  ]
by    [  ✓    ✓    ✓    ✓    ✗  ]
the   [  ✓    ✓    ✓    ✓    ✓  ]  → predicts "river"
```

**Training game:** Next-token prediction — predict the next token from everything to the left.

**Good at:** Generation, conversation, reasoning, coding. The dominant architecture today — GPT-4, Claude, LLaMA, Gemini are all decoder-only.

---

## 3. Encoder-Decoder (T5, BART, original Transformer)

The **original** transformer architecture from "Attention Is All You Need" (2017). Uses BOTH an encoder and a decoder, connected by **cross-attention**.

```text
Task: Translate "The cat sat" → "Le chat assis"

┌─────────────────────┐         ┌─────────────────────┐
│       ENCODER        │         │       DECODER        │
│                      │         │                      │
│  Input: "The cat sat"│         │  Input: "<BOS> Le chat" │
│                      │         │                      │
│  Bidirectional       │  ──K,V──→  Causal attention     │
│  self-attention      │         │  + cross-attention   │
│  (sees all tokens)   │         │  (sees encoder output│
│                      │         │   + left context)    │
│  Output: 3 context-  │         │                      │
│  aware vectors       │         │  Output: predict     │
│                      │         │  "assis" (next word) │
└─────────────────────┘         └─────────────────────┘

The encoder reads and understands the input.
The decoder generates the output, one token at a time,
while looking at what the encoder understood.
```

### Cross-Attention: The Bridge

The decoder has an **extra** attention layer that encoder-only and decoder-only models don't have:

```text
A decoder block in an encoder-decoder model has THREE sub-layers:

    1. Causal self-attention     (decoder attends to its own previous tokens)
    2. Cross-attention           [UNIQUE to encoder-decoder]
    3. Feed-forward network

Cross-attention:
    Q = from decoder (what the decoder is looking for)
    K = from encoder (what the input contains)
    V = from encoder (the information to retrieve)

    The decoder QUERIES the encoder's output.
    "I'm about to generate the next French word — what part of the
     English input should I pay attention to?"
```

**Toy example: translating "The cat sat" → "Le chat assis"**

```text
Encoder output (3 vectors, one per English token):
    h_The  = [0.10, 0.82, -0.34]
    h_cat  = [0.75, -0.20, 0.15]
    h_sat  = [-0.30, 0.10, 0.90]

Decoder is generating token 3 ("assis").
It has already generated: "<BOS> Le chat"

Step 1: Causal self-attention
    "assis" position attends to <BOS>, Le, chat (not future tokens)
    → decoder_vec = [0.45, -0.12, 0.68]

Step 2: Cross-attention
    Q = decoder_vec × W_Q = [0.50, -0.10, 0.70]     (what am I looking for?)
    K = encoder outputs × W_K                          (what does the input offer?)
    V = encoder outputs × W_V                          (the actual information)

    Attention scores (Q · K):
        score(The) = 0.15   (low — "assis" doesn't need "The")
        score(cat) = 0.20   (low — "assis" is about sitting, not the cat itself)
        score(sat) = 0.85   (high! — "assis" is the French translation of "sat")

    After softmax:
        weight(The) = 0.08
        weight(cat) = 0.10
        weight(sat) = 0.82    ← decoder focuses on "sat"

    Output = 0.08 × V_The + 0.10 × V_cat + 0.82 × V_sat
    → enriched decoder vector that "knows" it should translate "sat"

Step 3: FFN → prediction head → predict next token
```

Cross-attention is what makes encoder-decoder models great for tasks where the output is a **transformation** of the input — translation, summarization, question answering where the answer comes from a given passage.

### Training Game

```text
Depends on the specific model:

T5 (Text-to-Text Transfer Transformer):
    Everything is framed as "text in → text out":
    Input:  "translate English to French: The cat sat"
    Target: "Le chat assis"

    Input:  "summarize: [long article]"
    Target: "[short summary]"

    Input:  "question: What color is the sky? context: The sky is blue."
    Target: "blue"

    Trained with teacher forcing: at each decoder step, feed the CORRECT
    previous token (not the model's prediction). Loss = cross-entropy
    on the target tokens.

BART:
    Pre-trained by corrupting text (mask, delete, shuffle, rotate)
    then reconstructing the original.
    Encoder reads the corrupted version, decoder generates the clean version.
```

---

## Side-by-Side Comparison

```text
|                    | Encoder-Only         | Decoder-Only          | Encoder-Decoder       |
| ------------------ | -------------------- | --------------------- | --------------------- |
| Attention          | Bidirectional        | Causal (left-only)    | Both + cross-attention|
| Sees future tokens | Yes                  | No                    | Encoder: yes          |
|                    |                      |                       | Decoder: no           |
| Output             | Embeddings           | Generated tokens      | Generated tokens      |
| Training game      | MLM (fill blanks)    | Next-token prediction | Varies (T5: text2text)|
| Example models     | BERT, RoBERTa        | GPT, LLaMA, Claude    | T5, BART, mBART       |
|                    | Sentence-BERT        | Gemini, Mistral       | Whisper, NLLB         |
| Best for           | Understanding        | Open-ended generation | Conditioned generation|
|                    | Embeddings           | Conversation          | Translation           |
|                    | Classification       | Reasoning, coding     | Summarization         |
|                    | Retrieval / search   | Creative writing      | Structured extraction |
| Params (typical)   | 110M - 330M          | 7B - 400B+            | 200M - 11B            |
```

---

## Why Decoder-Only Won

The original transformer (2017) was encoder-decoder. BERT (2018) showed encoder-only was great for understanding. GPT (2018) showed decoder-only was great for generation.

By 2023, decoder-only became the **dominant** architecture for almost everything. Why?

```text
1. Simplicity
    One architecture, one training game (next-token prediction).
    No need to design separate encoder/decoder or cross-attention.

2. Scaling
    Decoder-only models scale more predictably.
    Double the parameters → predictably better performance (scaling laws).
    Encoder-decoder has more hyperparameters to tune (encoder size vs decoder size).

3. Generality
    A decoder can do everything:
    - Generation: naturally (it's what it's designed for)
    - Understanding: "Is this sentence positive or negative? Answer:"
    - Translation: "Translate to French: The cat sat. Answer:"
    - Summarization: "Summarize this: [text]. Summary:"
    - Embeddings: take the last hidden state (not ideal, but works)

    The trick: frame every task as "complete this text."
    You lose some efficiency vs specialized architectures,
    but gain a single model that does everything.

4. Emergent abilities
    At large scale (100B+ params), decoder-only models develop
    abilities they weren't explicitly trained for:
    - Chain-of-thought reasoning
    - In-context learning (learn from examples in the prompt)
    - Tool use
    These emerged more clearly in decoder-only models.
```

---

## Encoder-Decoder: Where It Still Shines

Encoder-decoder isn't dead — it's the best choice when:

```text
- Translation (NLLB, mBART) — encoder understands source language,
  decoder generates target language. Cross-attention aligns them.

- Speech-to-text (Whisper) — encoder processes the audio spectrogram,
  decoder generates the transcript. The encoder "understands" the
  audio, the decoder "speaks" the text.

- Structured extraction — when output structure differs greatly from input
  (e.g., table extraction, code generation from specifications).

- Efficiency for short outputs — if the output is much shorter than
  the input (e.g., summarization), encoder-decoder processes the long
  input with efficient bidirectional attention, only using expensive
  autoregressive generation for the short output.
```

---

## The Big Picture

```text
"Attention Is All You Need" (2017)
    ↓
    Original Transformer = Encoder-Decoder (for translation)
    ↓
    ├── Strip the decoder → BERT (2018) — encoder-only
    │       Great at understanding, embeddings, classification
    │       → Phase 1 of your notes
    │
    ├── Strip the encoder → GPT (2018) — decoder-only
    │       Great at generation, scales massively
    │       → This phase of your notes
    │
    └── Keep both → T5, BART (2019-2020) — encoder-decoder
            Great at translation, summarization
            Still used in Whisper, NLLB

    By 2023: decoder-only won for general-purpose AI
             (GPT-4, Claude, LLaMA, Gemini, Mistral)

    But encoders still dominate for embeddings/retrieval (Phase 1)
    And encoder-decoder still dominates for translation/speech
```

The transformer is a family of architectures, not one model. All three variants use the same building blocks — the difference is just the attention mask and whether there's cross-attention.

---

## Additional Context: Encoder-Decoder Full Architecture Walkthrough

### The Complete Architecture (BART / T5 style)

```text
INPUT: "The cat sat"                          OUTPUT (generated): "Le chat assis"
      │                                                ▲
      ▼                                                │
┌─────────────────────────────┐          ┌─────────────────────────────────────┐
│         ENCODER              │          │              DECODER                 │
│                              │          │                                     │
│  ┌─────────────────────┐    │          │  ┌─────────────────────────────┐    │
│  │  Token Embeddings   │    │          │  │  Token Embeddings           │    │
│  │  + Position Embeds  │    │          │  │  + Position Embeds          │    │
│  └────────┬────────────┘    │          │  └──────────┬──────────────────┘    │
│           │                  │          │             │                       │
│  ┌────────▼────────────┐    │          │  ┌──────────▼──────────────────┐    │
│  │                      │    │          │  │                              │    │
│  │  Self-Attention      │    │          │  │  Causal Self-Attention       │    │
│  │  (bidirectional)     │    │          │  │  (masked, sees left only)    │    │
│  │  + Residual + Norm   │    │          │  │  + Residual + Norm           │    │
│  │                      │    │          │  │                              │    │
│  ├──────────────────────┤    │          │  ├──────────────────────────────┤    │
│  │                      │    │          │  │                              │    │
│  │  FFN                 │    │  K,V     │  │  Cross-Attention ◄───────────┼────┼── encoder
│  │  + Residual + Norm   │    │ ─────────┼──┤  Q = decoder, K,V = encoder │    │   output
│  │                      │    │          │  │  + Residual + Norm           │    │
│  └────────┬────────────┘    │          │  │                              │    │
│           │                  │          │  ├──────────────────────────────┤    │
│       × N layers             │          │  │                              │    │
│     (e.g. 6 or 12)          │          │  │  FFN                         │    │
│           │                  │          │  │  + Residual + Norm           │    │
│           ▼                  │          │  │                              │    │
│  ┌─────────────────────┐    │          │  └──────────┬───────────────────┘    │
│  │  Encoder Output      │    │          │             │                       │
│  │  (3 context-aware    │────┼──────────┼─→       × N layers                 │
│  │   vectors)           │    │          │         (same N as encoder)         │
│  └─────────────────────┘    │          │             │                       │
│                              │          │             ▼                       │
└─────────────────────────────┘          │  ┌──────────────────────────────┐    │
                                          │  │  Linear → Softmax           │    │
                                          │  │  (predict next token)       │    │
                                          │  └──────────────────────────────┘    │
                                          └─────────────────────────────────────┘

Key: each decoder block has 3 sub-layers (vs 2 in encoder, 2 in decoder-only)
     1. Causal self-attention   — decoder looks at its own past tokens
     2. Cross-attention         — decoder looks at the encoder output
     3. FFN                     — same as everywhere else
```

### Full Example: Translating "The cat sat" → "Le chat assis"

#### Phase 1: Encoder processes the input (happens ONCE)

```text
Input tokens: [The, cat, sat]

Embed:
    The → [0.50, -0.20, 0.80, 0.10]
    cat → [0.30, 0.70, -0.10, 0.40]
    sat → [-0.20, 0.10, 0.60, 0.90]

    + position embeddings added to each.

Encoder self-attention (bidirectional — every token sees every other):

    "The": attends to The, cat, sat
        Learns: "The" is a determiner for "cat"
    "cat": attends to The, cat, sat
        Learns: "cat" is the subject, "cat" that "sat" (not a cat that ran)
    "sat": attends to The, cat, sat
        Learns: "sat" is the action of "the cat"

    Each vector is now context-aware:
    h_The = [0.12, 0.78, -0.30, 0.22]    (knows it modifies "cat")
    h_cat = [0.70, -0.15, 0.18, 0.45]    (knows it's the sitter)
    h_sat = [-0.25, 0.08, 0.85, 0.92]    (knows the cat did the sitting)

    These pass through FFN + residual + norm (× N layers).

Encoder output: 3 vectors. These are FROZEN — computed once, reused
by the decoder at every generation step.
```

#### Phase 2: Decoder generates output token by token

```text
Generation step 1: produce the first output token
──────────────────────────────────────────────────
    Decoder input: [<BOS>]     (beginning-of-sequence token)

    Sub-layer 1 — Causal self-attention:
        <BOS> can only attend to itself (nothing to its left).
        → d₁ = [0.05, 0.02, -0.10, 0.30]

    Sub-layer 2 — Cross-attention:
        Q = d₁ × W_Q        → query: "what should the first French word be?"
        K = encoder output × W_K   → keys from "The", "cat", "sat"
        V = encoder output × W_V   → values from "The", "cat", "sat"

        Attention scores (Q · K for each encoder token):
            score(The) = 0.72   ← high! First word often aligns to first word
            score(cat) = 0.35
            score(sat) = 0.10

        After softmax:
            weight(The) = 0.58
            weight(cat) = 0.30
            weight(sat) = 0.12

        Output = 0.58 × V_The + 0.30 × V_cat + 0.12 × V_sat
        → enriched vector focused mostly on "The"

    Sub-layer 3 — FFN + residual + norm → final vector

    Prediction head (linear → softmax over French vocabulary):
        P("Le") = 0.65    ← highest
        P("La") = 0.20
        P("Un") = 0.05
        ...

    Output: "Le" ✓


Generation step 2: produce the second token
────────────────────────────────────────────
    Decoder input: [<BOS>, Le]

    Sub-layer 1 — Causal self-attention:
        "Le" attends to <BOS> and itself (causal mask).
        → d₂ = [0.40, -0.18, 0.55, 0.22]

    Sub-layer 2 — Cross-attention:
        Q = d₂ × W_Q        → "what should the second French word be?"
        K, V = same encoder output as before (reused, not recomputed)

        Attention scores:
            score(The) = 0.20
            score(cat) = 0.78   ← high! "Le ___" → the noun, which is "cat"
            score(sat) = 0.15

        After softmax:
            weight(The) = 0.12
            weight(cat) = 0.68   ← decoder focuses on "cat"
            weight(sat) = 0.20

        Output = weighted sum, focused on the "cat" vector

    Sub-layer 3 — FFN → prediction:
        P("chat") = 0.72   ← highest ("chat" = French for "cat")
        P("chien") = 0.05
        ...

    Output: "chat" ✓


Generation step 3: produce the third token
────────────────────────────────────────────
    Decoder input: [<BOS>, Le, chat]

    Sub-layer 1 — Causal self-attention:
        "assis" position attends to <BOS>, Le, chat

    Sub-layer 2 — Cross-attention:
        Q from decoder → "what should come after 'Le chat'?"

        Attention scores:
            score(The) = 0.05
            score(cat) = 0.10
            score(sat) = 0.88   ← high! Need the verb now

        After softmax:
            weight(The) = 0.04
            weight(cat) = 0.08
            weight(sat) = 0.88   ← decoder focuses on "sat"

    Prediction:
        P("assis") = 0.60  ← "assis" = French for "sat"

    Output: "assis" ✓


Generation step 4:
    Decoder input: [<BOS>, Le, chat, assis]
    → predicts <EOS> (end of sequence)
    → generation stops
```

### What Cross-Attention Actually Learns

```text
The cross-attention pattern across all 3 steps reveals an ALIGNMENT:

                    Encoder tokens
                    The     cat     sat
Decoder tokens    ┌───────┬───────┬───────┐
    Le            │ ██ 58 │ ░░ 30 │    12 │   "Le" mostly looks at "The"
    chat          │    12 │ ██ 68 │ ░░ 20 │   "chat" mostly looks at "cat"
    assis         │    04 │    08 │ ██ 88 │   "assis" mostly looks at "sat"
                  └───────┴───────┴───────┘

This is a soft alignment matrix. Unlike a dictionary lookup,
cross-attention can handle:

    Reordering:  "I like cats" → "Les chats me plaisent"
        (subject-verb-object → object-subject-verb)
        Cross-attention learns which output word should look at
        which input word, regardless of position.

    One-to-many: "kicked the bucket" → "est décédé" (died)
        Multiple input tokens map to different output tokens.
        Cross-attention scores distribute across the relevant inputs.

    Many-to-one: "United States of America" → "USA"
        One output token attends to multiple input tokens.

This is exactly what the original 2017 transformer was built for.
Cross-attention IS the translation mechanism.
```

### Why Decoder-Only Can Skip All This

```text
A decoder-only model does the same translation task like this:

    Input: "Translate to French: The cat sat → "

    The ENTIRE sequence (instruction + source + output) is ONE
    sequence in one decoder. No separate encoder, no cross-attention.

    Position 1-8:  "Translate to French: The cat sat →"
                   ↑ these tokens play the "encoder" role
                   (the model reads and "understands" them via
                    causal attention as it processes left to right)

    Position 9:    predicts "Le"    (attends to positions 1-8)
    Position 10:   predicts "chat"  (attends to positions 1-9)
    Position 11:   predicts "assis" (attends to positions 1-10)

    No cross-attention needed. The source text is just... earlier
    in the same sequence. Causal attention to earlier positions
    serves the same role as cross-attention to encoder output.

    Trade-off:
    ✗ Causal attention on source text (can't see future source tokens)
      vs bidirectional in an encoder (sees all source tokens)
    ✗ Reprocesses source tokens at every layer
      (encoder-decoder computes encoder output once)
    ✓ But with enough scale and context, these losses don't matter
    ✓ Simpler to train, scale, and serve
```
