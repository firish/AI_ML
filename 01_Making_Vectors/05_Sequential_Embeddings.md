## Sequential Embeddings: RNNs and LSTMs

### The Problem with Everything So Far

```text
File 03 (Word2Vec/GloVe): one fixed vector per word, no context.
    "bank" → same vector whether it's a river bank or a money bank.

File 04 (Pooling/Doc2Vec): sentence vectors, but still built from
    static word vectors. Order is lost or barely captured.
    "dog bites man" ≈ "man bites dog"

What we want:
    A model that READS the sentence left to right (or both directions),
    and at each word, outputs a vector that reflects EVERYTHING it has
    seen so far. The vector for "bank" should be DIFFERENT depending
    on whether "river" or "money" came before it.

This is what RNNs and LSTMs do.
They were THE dominant architecture for NLP from 2014-2018,
before transformers replaced them.
```

---

## The Recurrent Neural Network (RNN)

### The Core Idea

```text
An RNN processes tokens ONE AT A TIME, left to right.
At each step, it takes two inputs:
    1. The current word's embedding (from Word2Vec or a learned table)
    2. The hidden state from the PREVIOUS step (a summary of everything seen so far)

And produces one output:
    → A new hidden state (updated summary including this word)

    h₀ = zeros (start with blank memory)

    Step 1: h₁ = f(h₀, "The")     → h₁ knows: [The]
    Step 2: h₂ = f(h₁, "cat")     → h₂ knows: [The, cat]
    Step 3: h₃ = f(h₂, "sat")     → h₃ knows: [The, cat, sat]
    Step 4: h₄ = f(h₃, "on")      → h₄ knows: [The, cat, sat, on]
    Step 5: h₅ = f(h₄, "the")     → h₅ knows: [The, cat, sat, on, the]
    Step 6: h₆ = f(h₅, "mat")     → h₆ knows: [The, cat, sat, on, the, mat]

    h₆ is the "sentence vector" — it has seen every word in order.

What is f()?
    A simple matrix multiplication + activation:

    h_t = tanh(W_h × h_{t-1}  +  W_x × x_t  +  b)
           ↑                      ↑
           "what I knew before"   "what I'm reading now"

    W_h: weight matrix for the previous hidden state
    W_x: weight matrix for the current word embedding
    b:   bias
    tanh: squash output to [-1, +1]

The hidden state is like a person reading a book:
    After each word, they update their mental summary.
    By the end of the sentence, they have a compressed
    understanding of the whole thing.
```

### Toy Example with Numbers

```text
Setup:
    Embedding dim = 3, Hidden dim = 4
    Sentence: "cat sat"

    Word embeddings (from a lookup table):
        "cat" = [0.5, -0.2, 0.8]
        "sat" = [-0.3, 0.7, 0.1]

    W_x: 4×3 matrix (maps 3-dim embedding → 4-dim hidden)
    W_h: 4×4 matrix (maps 4-dim hidden → 4-dim hidden)
    b:   4-dim bias vector

── Step 0: Initialize ──
    h₀ = [0, 0, 0, 0]    (blank memory)

── Step 1: Read "cat" ──
    x₁ = [0.5, -0.2, 0.8]

    h₁ = tanh(W_h × h₀ + W_x × x₁ + b)
       = tanh([0,0,0,0] + W_x × [0.5, -0.2, 0.8] + b)
       = tanh([0,0,0,0] + [0.42, -0.15, 0.63, 0.28] + [0.1, 0.1, 0.1, 0.1])
       = tanh([0.52, -0.05, 0.73, 0.38])
       = [0.48, -0.05, 0.62, 0.36]

    h₁ = [0.48, -0.05, 0.62, 0.36]
    This vector encodes: "I've seen 'cat'"

── Step 2: Read "sat" ──
    x₂ = [-0.3, 0.7, 0.1]

    h₂ = tanh(W_h × h₁ + W_x × x₂ + b)
       = tanh(W_h × [0.48, -0.05, 0.62, 0.36] + W_x × [-0.3, 0.7, 0.1] + b)
       = tanh([0.31, 0.22, -0.08, 0.45] + [-0.12, 0.55, 0.18, -0.21] + [0.1, 0.1, 0.1, 0.1])
       = tanh([0.29, 0.87, 0.20, 0.34])
       = [0.28, 0.70, 0.20, 0.33]

    h₂ = [0.28, 0.70, 0.20, 0.33]
    This vector encodes: "I've seen 'cat sat'" — this is our sentence embedding.

Key: h₂ is influenced by BOTH "cat" (through h₁) and "sat" (through x₂).
    If the sentence were "dog sat", h₂ would be different because h₁ was different.
    Word ORDER matters — "cat sat" ≠ "sat cat" because the hidden states
    arrive in different sequences.
```

---

## The Problem with Vanilla RNNs: Vanishing Gradients

```text
RNNs work fine for short sentences. But for long ones:

    Token 1 → Token 2 → Token 3 → ... → Token 200 → Token 201

    To update the weights that process Token 1 based on a loss at Token 201,
    backpropagation must multiply gradients through 200 steps.

    At each step, the gradient gets multiplied by W_h (the hidden-state matrix).
    If the values in W_h are < 1: gradients SHRINK exponentially.
    If the values in W_h are > 1: gradients EXPLODE exponentially.

    200 steps of multiplying by 0.9: 0.9²⁰⁰ ≈ 0.0000000007 (vanished)
    200 steps of multiplying by 1.1: 1.1²⁰⁰ ≈ 190,000,000   (exploded)

    Result: the model "forgets" early tokens.
    By the time you're at Token 200, the hidden state has essentially
    lost all information about Token 1.

    This is the telephone game problem:
        Whisper a message through 200 people.
        By the end, the original message is garbled beyond recognition.

    For NLP this is devastating:
        "The author, who grew up in Paris and later moved to London
         where he studied at Oxford before returning to his hometown,
         wrote the book in ____"

        The answer depends on "Paris" (90 tokens ago).
        A vanilla RNN has forgotten "Paris" by the time it reaches "____".
```

---

## LSTMs: The Fix

### The Core Idea

```text
LSTM = Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)

The fix: add a SECOND pathway for information to flow through —
    the "cell state" — that doesn't get multiplied by W_h at every step.

Think of it like two parallel tracks:

    Vanilla RNN:
        One road with a tollbooth at every step.
        Each toll takes a cut → by step 200, you have nothing left.

    LSTM:
        Highway:  a straight road with no tolls. Information flows freely.
        Local road: tollbooths at every step (like the RNN hidden state).
        At each step, the model CHOOSES:
            - What to PUT onto the highway (from the local road)
            - What to REMOVE from the highway (no longer needed)
            - What to READ from the highway (into the local road)

    The highway is the "cell state" (c_t).
    The choices are made by "gates" — sigmoid layers that output
    values between 0 (block everything) and 1 (let everything through).
```

### The Three Gates

```text
At each time step, the LSTM has three gates:

1. FORGET GATE: "what should I erase from long-term memory?"
    f_t = sigmoid(W_f × [h_{t-1}, x_t])
    Output: values between 0 and 1 for each cell dimension.
    0 = forget completely, 1 = remember fully.

    Example: reading "She" after talking about "John" for 10 sentences.
    The forget gate might erase "subject = John" to make room for "She".

2. INPUT GATE: "what new information should I store?"
    i_t = sigmoid(W_i × [h_{t-1}, x_t])
    candidate = tanh(W_c × [h_{t-1}, x_t])

    i_t decides HOW MUCH of the candidate to store (0 to 1).
    candidate is WHAT to store (the new information).

    Example: reading "Paris" → input gate stores "location = Paris"
    onto the cell state highway.

3. OUTPUT GATE: "what should I output from long-term memory?"
    o_t = sigmoid(W_o × [h_{t-1}, x_t])
    h_t = o_t * tanh(c_t)

    Decides what part of the cell state to expose as the hidden state.
    Not everything in memory is relevant to the current output.

Updating the cell state (the highway):
    c_t = f_t * c_{t-1}  +  i_t * candidate
          ↑ keep/forget      ↑ add new info
          old memory          new memory

This is the key: c_t is updated by ADDITION, not multiplication.
    Addition preserves gradients. No vanishing gradient problem
    (or at least much less of one).
    Information stored at step 1 can survive to step 200
    as long as the forget gate keeps it (stays near 1).
```

### What It Looks Like in Practice

```text
Reading: "The cat, which was very fluffy, sat on the mat"

    Step 1 ("The"):     cell stores: [article detected]
    Step 2 ("cat"):     input gate stores: [subject = cat]
    Step 3 ("which"):   output gate: don't expose "cat" yet, we're in a clause
    Step 4 ("was"):     hidden state tracks local clause
    Step 5 ("very"):    hidden state tracks local clause
    Step 6 ("fluffy"):  input gate adds: [subject property = fluffy]
    Step 7 ("sat"):     output gate: NOW expose "cat" again — it's the subject of "sat"
                         forget gate: erase clause-tracking info

    The cell state carried "subject = cat" THROUGH the relative clause
    ("which was very fluffy") without losing it. A vanilla RNN would
    have overwritten "cat" by step 5 or 6.

    This is why it's called Long SHORT-TERM Memory:
    It can hold SHORT-TERM information (like the subject) for a LONG time.
```

---

## Bidirectional LSTMs

```text
A regular LSTM reads left → right.
    h₅ for "sat" knows: [The, cat, which, was, sat]
    But doesn't know: [on, the, mat] (hasn't seen it yet)

A Bidirectional LSTM runs TWO LSTMs:
    Forward:   The → cat → which → was → sat → on → the → mat
    Backward:  mat → the → on → sat → was → which → cat → The

    At each position, concatenate both hidden states:
    h_bidir("sat") = [h_forward("sat"); h_backward("sat")]
                    = [knows left context ; knows right context]
                    = knows FULL context

    If each direction has hidden_dim = 256,
    the bidirectional vector is 512-dim.

This is what ELMo (2018) used:
    A 2-layer bidirectional LSTM trained on language modelling.
    The concatenated hidden states at each position = contextual word embeddings.
    "bank" near "river" → different vector than "bank" near "money".
    This was a HUGE breakthrough — the first widely-used contextual embeddings.
```

---

## ELMo: The Peak of the LSTM Era

```text
ELMo (Embeddings from Language Models) — Peters et al., 2018

Architecture:
    2 layers of bidirectional LSTMs (so 4 LSTMs total: 2 forward, 2 backward)
    Each LSTM: hidden_dim = 4096 (large)
    Trained on: 1 Billion Word Benchmark (~800M tokens)
    Objective: predict next word (forward) + previous word (backward)

How ELMo produces embeddings:
    For each word, ELMo gives you THREE vectors:
        Layer 0: the static character-based embedding (context-free)
        Layer 1: hidden state from LSTM layer 1 (syntax-level context)
        Layer 2: hidden state from LSTM layer 2 (semantic-level context)

    The final embedding = learned weighted sum of all three layers:
        v_word = γ × (s₀ × L₀ + s₁ × L₁ + s₂ × L₂)

    The weights (s₀, s₁, s₂, γ) are learned per downstream task.
    Different tasks benefit from different layers:
        POS tagging → prefers layer 1 (syntax)
        Sentiment  → prefers layer 2 (semantics)

How ELMo was used (IMPORTANT — different from fine-tuning):
    1. Pre-train the bidirectional LSTM (once, expensive)
    2. FREEZE it
    3. For each downstream task, run text through frozen ELMo
    4. Use the output vectors as FEATURES — feed them into
       a SEPARATE task-specific model (could be another LSTM, CNN, etc.)

    The ELMo LSTM itself was never fine-tuned.
    It was a frozen feature extractor.

    GPT-1 (same year, 2018) challenged this:
        "Why freeze? Fine-tune the WHOLE model end-to-end."
        This became the standard approach.
```

---

## Why Transformers Replaced LSTMs

```text
LSTMs had three fundamental problems that transformers solved:

1. SEQUENTIAL = SLOW
    LSTMs process tokens one at a time: t₁ then t₂ then t₃...
    You can't compute h₃ until h₂ is done (it's an input).
    A 512-token sequence needs 512 sequential steps.

    Transformers use attention: every token attends to every other
    token IN PARALLEL. One step, not 512.
    On a GPU with 1000+ cores, this is dramatically faster.

    Training time comparison (same dataset):
        LSTM:        days to weeks
        Transformer: hours to days

2. LONG-RANGE DEPENDENCIES STILL DEGRADE
    LSTMs are BETTER than vanilla RNNs at long-range, but not great.
    The cell state helps, but information still degrades over 200+ tokens.

    In transformers, token 1 can directly attend to token 500.
    There's no chain of 500 hidden states between them.
    The "path length" between any two tokens is 1 (one attention step).

    Path length comparison:
        RNN:         O(n) — information passes through n hidden states
        LSTM:        O(n) — better gradient flow, but still n steps
        Transformer: O(1) — direct attention connection

3. NO BIDIRECTIONAL CONTEXT WITHOUT DOUBLING THE MODEL
    LSTMs need TWO separate models (forward + backward) for full context.
    That's 2× parameters, 2× compute.

    Transformer encoders (BERT) get bidirectional context for free —
    every token sees every other token through self-attention.

GPT-1 vs ELMo (both 2018) — the direct comparison:
    Same pre-train + downstream evaluation framework.
    GPT-1 (transformer): 74.7 average across benchmarks
    ELMo (LSTM):         69.1 average (5.6 points worse)

    The 5.6-point gap is ENTIRELY from swapping LSTM → transformer.
    Same data, same training approach, same tasks.
    The architecture won.
```

---

## The Timeline

```text
2014-2015: RNNs/LSTMs become dominant in NLP
            Seq2seq (Sutskever et al., 2014) — LSTMs for translation
            Attention mechanism added to seq2seq (Bahdanau et al., 2015)

2017:       "Attention Is All You Need" — transformers proposed
            Key insight: you don't need the LSTM at all,
            just use attention by itself

2018:       The transition year:
            ELMo (Feb) — peak LSTM, bidirectional contextual embeddings
            GPT-1 (Jun) — transformer decoder, fine-tuning paradigm
            BERT (Oct)  — transformer encoder, destroys ELMo on every benchmark

2019+:      LSTMs effectively dead for NLP
            Every major model is transformer-based
            (GPT-2, GPT-3, T5, RoBERTa, Claude, etc.)

LSTMs still used in:
    - Time series forecasting (finance, sensor data)
    - Small-scale embedded/mobile applications
    - Some speech processing pipelines
    Not in modern NLP.
```

---

## Summary: The Progression

```text
File 03: Word2Vec/GloVe    → static word vectors, no context
File 04: Pooling/Doc2Vec    → sentence vectors from static words, order mostly lost
File 05: RNNs/LSTMs (here) → sequential processing, context-aware, but slow and lossy
File 06: Transformers/BERT  → parallel attention, full context, the current standard

Each step solved the previous step's biggest problem:
    Static?     → Read sequentially (RNN/LSTM)
    Forgets?    → Add cell state highway (LSTM)
    Still slow? → Attend to everything at once (Transformer)
```
