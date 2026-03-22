## Training an LLM — Pre-training

### The Surprising Truth

Every LLM — GPT-4, Claude, LLaMA, Gemini — is trained with the **same single game**: predict the next token.

```text
Input:  "The cat sat on the"
Target: "mat"

That's it. No special reasoning training. No "learn to code" phase.
No "understand physics" module. Just: given these tokens, what comes next?

The model plays this game trillions of times on trillions of tokens.
Grammar, facts, reasoning, coding, humour, translation — all emerge
from this one objective, at sufficient scale.
```

---

## The Next-Token Prediction Game

### How One Training Step Works

```text
Training sentence: "The cat sat on the mat"

Step 1: Tokenize
    → [464, 3797, 3332, 319, 262, 2603]

Step 2: Create input-target pairs (shift right by 1)

    Input tokens:     [The,  cat,  sat,  on,   the ]
    Target tokens:    [cat,  sat,  on,   the,  mat ]

    Position 0: sees "The"                    → should predict "cat"
    Position 1: sees "The cat"                → should predict "sat"
    Position 2: sees "The cat sat"            → should predict "on"
    Position 3: sees "The cat sat on"         → should predict "the"
    Position 4: sees "The cat sat on the"     → should predict "mat"

    One sentence = 5 training examples simultaneously!
    (The causal mask ensures each position only sees tokens to its left.)

Step 3: Forward pass
    Feed all 5 input tokens through the decoder (in parallel).
    At each position, the model outputs a probability distribution
    over the vocabulary (50,257 probabilities).

Step 4: Compute loss
    At each position, cross-entropy between prediction and target:

    Position 0: model gives P("cat")  = 0.003  → loss = -log(0.003) = 5.81
    Position 1: model gives P("sat")  = 0.02   → loss = -log(0.02)  = 3.91
    Position 2: model gives P("on")   = 0.05   → loss = -log(0.05)  = 3.00
    Position 3: model gives P("the")  = 0.15   → loss = -log(0.15)  = 1.90
    Position 4: model gives P("mat")  = 0.001  → loss = -log(0.001) = 6.91

    Average loss = (5.81 + 3.91 + 3.00 + 1.90 + 6.91) / 5 = 4.31

Step 5: Backward pass (backpropagation)
    Compute gradients for ALL weights in the model:
    embeddings, all 96 transformer blocks, prediction head.

Step 6: Update weights (AdamW optimizer)
    Nudge every weight to make the loss slightly smaller.

Step 7: Repeat with the next batch of sentences.
```

### What the Loss Means

```text
Loss = cross-entropy = -log(P(correct token))

    Loss = 0.01  → P(correct) = 0.99  → model is very confident and right
    Loss = 2.30  → P(correct) = 0.10  → model gives 10% to the right answer
    Loss = 6.91  → P(correct) = 0.001 → model has almost no idea

Perplexity = 2^(average loss)    or equivalently e^(average loss)

    Perplexity = 5 means "the model is choosing between ~5 equally likely
    options at each step." Lower is better.

    GPT-3 perplexity ≈ 20 on diverse text
    Fine-tuned models on specific domains can get perplexity < 5
```

---

## Training Data

### What LLMs Train On

```text
Source                  Size             What it contains
──────────────────────────────────────────────────────────────
CommonCrawl             ~petabytes       Web pages (filtered, deduplicated)
Wikipedia               ~20GB           Encyclopaedia articles
Books (BookCorpus, etc) ~50-100GB       Fiction, non-fiction, textbooks
Code (GitHub)           ~100-500GB      Python, JavaScript, C++, etc.
ArXiv                   ~50GB           Scientific papers
StackOverflow           ~30GB           Q&A about programming
Reddit                  ~100GB          Discussions, comments
News articles           ~50GB           Journalism
Patents, legal docs     ~10-50GB        Formal/technical text

Total for modern LLMs:  1-15 TRILLION tokens
    (GPT-3: 300B tokens, LLaMA: 1-2T tokens, LLaMA 3: 15T tokens)
```

### Data Quality Matters More Than Quantity

```text
Raw CommonCrawl is mostly garbage:
    - Spam, SEO content, duplicates, porn, broken HTML
    - Boilerplate (cookie banners, nav menus, footers)

Filtering pipeline:
    1. Language detection (keep target language, discard others)
    2. Deduplication (exact and near-duplicate removal)
    3. Quality filtering (classifier trained on "high quality" vs "low quality")
    4. Toxicity filtering (remove harmful/hateful content)
    5. PII removal (personal information)
    6. Domain balancing (don't over-represent any one source)

Example: CommonCrawl has ~250B pages
    After filtering → ~3-5% survives as training data

The Chinchilla paper (DeepMind, 2022) showed:
    A smaller model trained on MORE high-quality data beats
    a larger model trained on less data.

    LLaMA 1 (7B params, 1T tokens) ≈ GPT-3 (175B params, 300B tokens)
    → 25× fewer parameters, 3× more data, same performance
```

---

## Training at Scale

### How Long Training Takes

```text
Model          Params    Tokens    GPUs          Training time
──────────────────────────────────────────────────────────────
GPT-2          1.5B      40B       32 V100s      ~1 week
GPT-3          175B      300B      ~1000 V100s   ~1 month
LLaMA 1 (65B)  65B      1.4T      2048 A100s    ~21 days
LLaMA 2 (70B)  70B      2T        ~2000 A100s   ~25 days
LLaMA 3 (405B) 405B     15T       16,000 H100s  ~54 days
```

### The Compute Equation

```text
Total compute (FLOPs) ≈ 6 × N × D

    N = number of parameters
    D = number of training tokens
    6 = constant (forward + backward pass per token)

Example: LLaMA 2 (70B)
    Compute = 6 × 70B × 2T = 8.4 × 10²³ FLOPs
    On 2000 A100s at ~300 TFLOPS each:
    Time = 8.4 × 10²³ / (2000 × 3 × 10¹⁴) ≈ 1.4 × 10⁶ seconds ≈ 16 days
    (Real: ~25 days due to communication overhead, checkpointing, etc.)
```

### Multi-GPU Training

A 70B parameter model doesn't fit on one GPU (even at 16-bit, that's 140GB — a single A100 has 80GB). Training requires splitting across many GPUs:

```text
Data Parallelism:
    Each GPU gets a different mini-batch of data.
    All GPUs have a full copy of the model.
    After each step, average the gradients across all GPUs.
    Problem: model must fit on one GPU.

Tensor Parallelism:
    Split individual weight matrices across GPUs.
    e.g., a 768 × 3072 FFN matrix → split into 4 GPUs of 768 × 768.
    Each GPU computes its shard, then they communicate partial results.
    Works within a single node (fast inter-GPU communication needed).

Pipeline Parallelism:
    Different layers on different GPUs.
    GPU 1: layers 1-8, GPU 2: layers 9-16, GPU 3: layers 17-24, ...
    Tokens flow through GPUs like an assembly line.
    Problem: "pipeline bubbles" where some GPUs sit idle.

In practice: all three combined.
    LLaMA 3 (405B): tensor parallel within nodes, pipeline parallel across nodes,
    data parallel across groups of nodes. 16,000 H100 GPUs working together.
```

---

## What Emerges from Next-Token Prediction

The remarkable thing: you never explicitly teach the model any skill. Everything emerges from "predict the next token" at sufficient scale.

### Grammar and Syntax

```text
Training data contains billions of grammatically correct sentences.
To predict the next token well, the model must learn:
    "The cats [are/is]" → "are" (subject-verb agreement)
    "She gave [him/he] the book" → "him" (case marking)
    "The [big/very] cat" → both valid, different probabilities

This isn't programmed. The model discovers grammar because
grammatical next-tokens get lower loss than ungrammatical ones.
```

### World Knowledge

```text
Training data: "The capital of France is Paris."
    To predict "Paris" after "The capital of France is",
    the model must STORE this fact in its weights.

Billions of such facts are encoded across the model's parameters.
Each fact isn't stored in one location — it's distributed across
many weights. This is why it's hard to "edit" a single fact.
```

### Reasoning (at Scale)

```text
Small models (100M params):
    Can complete sentences, basic grammar. No reasoning.

Medium models (1-10B params):
    Can follow simple patterns, basic Q&A. Shallow reasoning.

Large models (100B+ params):
    Something surprising happens — EMERGENT abilities appear:

    Chain-of-thought: "Let me think step by step..."
        The model generates intermediate reasoning steps.
        It WASN'T trained to do this. But it saw examples of
        step-by-step reasoning in its training data (math textbooks,
        forum explanations, etc.), and learned to produce similar patterns.

    In-context learning: give the model a few examples in the prompt,
        and it generalises to new cases — without any weight updates.
        [Example 1: cat → 猫, dog → 犬, fish → ?] → 魚
        Again, not programmed. Emerges at scale.

    Code generation: saw billions of lines of code + documentation.
        Learned to predict the next token of code.
        This implicitly requires understanding APIs, algorithms, logic.
```

### What Remains Broken After Pre-training

```text
The pre-trained model is a powerful text completer, but it's NOT yet useful:

Problem 1: It completes text, doesn't answer questions.
    Input:  "What is 2+2?"
    Output: "What is 2+3? What is 2+4? What is 2+5?"
    (It continues the PATTERN of questions, not answering.)

Problem 2: It has no concept of "helpful" or "harmful."
    It will happily generate toxic text if that's what the
    pattern suggests (it saw toxic text in training data).

Problem 3: It's verbose and unfocused.
    No preference for concise, direct answers.

These problems are fixed by the next two training stages:
    SFT (Supervised Fine-Tuning) → teach it to follow instructions
    RLHF / DPO → teach it to be helpful and harmless
    → covered in file 06
```

---

## Scaling Laws

### The Chinchilla Discovery (DeepMind, 2022)

Before Chinchilla, the assumption was: bigger model = better.

```text
GPT-3 approach (2020):
    Make the model as big as possible (175B params).
    Train on "enough" data (300B tokens).
    Assumption: parameter count is the main driver of quality.

Chinchilla finding:
    There's an OPTIMAL ratio of parameters to training tokens.
    For a given compute budget, you should balance model size and data:

    Optimal tokens ≈ 20 × parameters

    GPT-3:  175B params, 300B tokens  → ratio = 1.7   (severely undertrained!)
    Chinchilla: 70B params, 1.4T tokens → ratio = 20  (properly trained)

    Chinchilla (70B) matched GPT-3 (175B) with 4× less compute!
```

### The Scaling Laws Formula

```text
Loss ∝ 1/N^α + 1/D^β + constant

    N = number of parameters
    D = number of training tokens
    α, β ≈ 0.07 (approximately)

What this means:
    - Double the parameters → loss decreases by ~5%
    - Double the data → loss decreases by ~5%
    - These gains are PREDICTABLE and SMOOTH
    - No sudden jumps — you can forecast performance before training

This is why companies can estimate a model's quality before spending
$100M to train it. The scaling curve tells you what you'll get.
```

### The Practical Impact

```text
Before Chinchilla (2020-2022):
    "We want a better model → make it bigger"
    GPT-3: 175B params, 300B tokens

After Chinchilla (2022+):
    "We want a better model → train smaller model on more data"
    LLaMA 1:   65B params, 1.4T tokens  (≈ GPT-3 quality at 1/3 size)
    LLaMA 2:   70B params, 2T tokens
    LLaMA 3:   70B params, 15T tokens   (way beyond Chinchilla-optimal,
                                          but data quality improvements
                                          keep helping even past 20×)

The trend: models aren't getting much bigger in params (70B is common),
but training data keeps growing (1T → 2T → 15T tokens).
Over-training (ratio > 20×) on high-quality data works in practice,
even if the Chinchilla formula says it shouldn't help as much.
Smaller, well-trained models are cheaper to deploy.
```

---

## The Training Pipeline — End to End

```text
Phase 0: Data preparation (months)
    ├── Scrape the web, collect books, code, etc.
    ├── Filter, deduplicate, quality-score
    ├── Build the tokenizer (BPE on a subset of the data)
    └── Tokenize the entire dataset

Phase 1: Pre-training (weeks to months)
    ├── Initialise model weights randomly
    ├── Set up training across thousands of GPUs
    ├── For each batch:
    │     Forward pass → cross-entropy loss → backward pass → AdamW update
    ├── Learning rate: warmup → cosine decay
    ├── Save checkpoints every N steps
    ├── Monitor loss curve:
    │     Step 1:      loss = 11.0   (random model, ~50K equally likely tokens)
    │     Step 1K:     loss = 7.0    (learning basic word frequencies)
    │     Step 10K:    loss = 4.5    (learning grammar, common phrases)
    │     Step 100K:   loss = 3.2    (learning facts, patterns)
    │     Step 500K:   loss = 2.8    (learning nuance, reasoning)
    │     Step 1M+:    loss = 2.5    (diminishing returns)
    └── Final model: understands language but doesn't follow instructions

Phase 2: SFT (days) — covered in file 06
Phase 3: RLHF/DPO (days) — covered in file 06
```

---

## Pre-training Costs

```text
| Model          | Estimated cost     | GPUs              |
| -------------- | ------------------ | ----------------- |
| GPT-2 (1.5B)  | ~$50K              | 32 V100s          |
| GPT-3 (175B)  | ~$4-12M            | ~1000 V100s       |
| LLaMA 2 (70B) | ~$2-5M             | 2000 A100s        |
| LLaMA 3 (405B)| ~$50-100M          | 16,000 H100s      |
| GPT-4          | ~$100M+ (rumoured) | ~25,000 A100s     |

These costs are just for pre-training (Phase 1).
SFT and RLHF add relatively little cost on top.

The cost is dominated by GPU-hours:
    H100 rental ≈ $2-3/hour/GPU
    16,000 H100s × 54 days × 24 hours × $2.50 ≈ $52M
```

---

## Summary

```text
1. Pre-training = predict the next token, trillions of times
2. One objective (cross-entropy), one optimizer (AdamW), one architecture
3. Training data: web + books + code + wiki, heavily filtered
4. Scale: thousands of GPUs, weeks/months, millions of dollars
5. Scaling laws: performance is predictable from compute budget
6. Chinchilla: balance model size and data (tokens ≈ 20× params)
7. What emerges: grammar → facts → reasoning → coding → in-context learning
8. What's missing: helpfulness, safety → fixed by SFT + RLHF (file 06)
```
