## Popular Text Encoders and How to Choose

### The Family Tree

Every major text encoder descends from BERT. Each variant changed one thing:

```text
BERT (2018)
 ├── RoBERTa (2019)          — same architecture, trained better
 ├── Sentence-BERT (2019)    — added contrastive fine-tuning for similarity
 │    └── MiniLM (2020)      — distilled to be smaller and faster
 ├── E5 (2022)               — instruction-tuned ("query: ..." / "passage: ...")
 ├── GTE (2023)              — multi-stage training, large context
 └── BGE (2023)              — multi-task, multilingual

Commercial (architecture not public):
 ├── OpenAI text-embedding-3 (2024)
 ├── Cohere embed-v3 (2023)
 └── Voyage AI (2024)
```

---

### Model-by-Model Breakdown

#### BERT (2018) — The Foundation

**What it is:** The original encoder. 12 transformer blocks, 768-dim vectors, 110M parameters.

**Training game:** Masked Language Model — mask 15% of tokens, predict them. Also used Next Sentence Prediction (NSP) — given two sentences, predict if the second follows the first.

**Key facts:**
- Max 512 tokens (subword pieces, not words)
- Vocabulary: ~30,000 WordPiece tokens
- Pre-trained on Wikipedia + BookCorpus

**Limitation for embeddings:** BERT was trained for token-level tasks (fill in the blank), not sentence-level similarity. Its raw [CLS] vector is a poor sentence embedding out of the box.

---

#### RoBERTa (2019) — BERT, Trained Better

**What changed:** Nothing in the architecture. Everything in how it's trained:

```text
| Change                  | BERT                    | RoBERTa                   |
| ----------------------- | ----------------------- | ------------------------- |
| Training data           | 16 GB                   | 160 GB (10× more)         |
| Next Sentence Prediction| Yes                     | Removed (hurt performance)|
| Masking                 | Static (same mask)      | Dynamic (new mask each epoch)|
| Batch size              | 256                     | 8,000                     |
| Training steps          | 1M                      | 500K (but bigger batches) |
```

**Result:** Same architecture, consistently better performance across all tasks. RoBERTa proved that BERT was significantly undertrained.

**Takeaway:** If you see a model described as "RoBERTa-based," it just means BERT architecture with better training.

---

#### Sentence-BERT (2019) — Making BERT Useful for Similarity

**The problem:** To compare two sentences with raw BERT, you had to feed them together as a pair and run inference — O(n²) comparisons for n sentences. Comparing 10,000 sentences would take ~65 hours.

**The fix:** Fine-tune BERT (or RoBERTa) with contrastive learning so each sentence gets its own good vector. Then similarity = one dot product. Comparing 10,000 sentences: ~5 seconds.

```text
Training setup (Siamese network):

    Sentence A ──→ [BERT encoder] ──→ vector_A ──┐
                                                   ├── cosine similarity → loss
    Sentence B ──→ [same BERT]    ──→ vector_B ──┘

    If A and B are paraphrases: push similarity UP
    If A and B are unrelated:   push similarity DOWN
```

**This is the key innovation** that made encoder-based embeddings practical for search and retrieval.

---

#### MiniLM (2020) — Smaller, Faster, Nearly as Good

**The problem:** BERT is 110M parameters. That's too large and slow for many production use cases (mobile, edge, high-throughput APIs).

**The solution — knowledge distillation:**

You can't just make BERT smaller (fewer layers/dims) and re-train from scratch — the small model doesn't have enough capacity to learn well on its own. Instead, you use a **teacher-student** setup:

```text
Training game: Mimic the teacher's attention

Step 1: Take a fully trained BERT-base (the teacher)
Step 2: Create a smaller model — e.g., 6 layers, 384 dims (the student)
Step 3: Feed the same sentences through both

Teacher (12 layers, 768d):
    "The cat sat" → attention matrix A_teacher (how much each token attends to each other)

Student (6 layers, 384d):
    "The cat sat" → attention matrix A_student

Loss = difference between A_teacher and A_student
    → Backprop adjusts only the student's weights
    → Student learns to replicate the teacher's "understanding" in a smaller body

The key insight: the student doesn't learn from raw text (too hard for a small model).
It learns from the teacher's behaviour — a much easier target.
```

**MiniLM's specific trick:** Instead of mimicking the teacher's output or hidden states (what earlier distillation methods did), MiniLM mimics the **self-attention distributions** of the teacher's *last* layer. This captures the most refined relationship patterns between tokens.

```text
| Model                      | Layers | Dims | Parameters | Speed vs BERT |
| -------------------------- | ------ | ---- | ---------- | ------------- |
| BERT-base                  | 12     | 768  | 110M       | 1×            |
| all-MiniLM-L6-v2           | 6      | 384  | 22M        | 5× faster     |
| all-MiniLM-L12-v2          | 12     | 384  | 33M        | 2× faster     |
```

**The `all-` prefix** means it was then fine-tuned with contrastive learning (like Sentence-BERT) on a **large combined dataset** of over 1 billion sentence pairs — NLI, paraphrases, Q&A pairs, etc.

So `all-MiniLM-L6-v2` is actually: distill BERT → then contrastive fine-tune. Two training phases on top of BERT's own pre-training.

**Why it matters:** This is the default recommendation for most projects. Fast enough for CPU, small enough for edge devices, retains ~90-95% of full BERT quality.

---

#### E5 (2022) — Instruction-Tuned Embeddings

**The problem with Sentence-BERT's training:** It uses human-labelled sentence pairs (NLI datasets, paraphrase datasets). These are expensive to create and limited in size/diversity. Can we do better?

**E5's training game — two stages:**

```text
Stage 1: Contrastive pre-training on noisy web pairs (no human labels needed)

    Where do the pairs come from? Naturally occurring on the web:
    - (Reddit title, top comment)       — title and comment are about the same thing
    - (StackOverflow question, answer)  — question and answer are related
    - (Wikipedia section title, text)   — title describes the passage
    - (tweet, reply)                    — reply is contextually related

    These are noisy (not all pairs are great), but there are billions of them.
    Train with contrastive loss on ~1 billion such pairs.

Stage 2: Fine-tune on small, high-quality labelled data

    Use human-labelled NLI pairs and retrieval datasets (like MS MARCO).
    This cleans up what the model learned from noisy pairs.
```

**The instruction prefix innovation:** The same sentence should embed differently depending on the task. A search query is short and intent-focused; a document passage is long and content-rich. E5 handles this with prefixes:

```text
Search:     embed("query: How to sort in Python")          ← emphasises intent
Storage:    embed("passage: Python's sorted() function...")  ← emphasises content
Clustering: embed("cluster: ML algorithm comparison")       ← emphasises topic

Without prefix: the model doesn't know what task you're doing,
    so it picks a generic middle ground (worse for all tasks).

With prefix: the model shifts its embedding space to optimise for that task.
```

How? The prefix tokens flow through the same transformer blocks and influence all other tokens via self-attention. "query:" effectively tells the model "this is short, focus on intent," while "passage:" says "this is content, focus on coverage."

---

#### GTE / BGE / E5-Mistral (2023-2024) — Current Open-Source Leaders

These models combine multiple advances. Let's look at what each one actually does:

**BGE-M3 (BAAI, 2024) — Multi-everything**

```text
The "M3" stands for: Multi-lingual, Multi-granularity, Multi-functionality

Training game — 3 stages:
    1. Pre-train on massive multilingual text (like RoBERTa but for 100+ languages)
    2. Contrastive fine-tuning on large-scale retrieval pairs (like E5 Stage 1)
    3. Multi-task fine-tuning:
        - Dense retrieval (single vector per text, like Sentence-BERT)
        - Sparse retrieval (learns term importance weights, like learned TF-IDF)
        - Multi-vector retrieval (keeps multiple token vectors, compares token-to-token)

Why 3 retrieval modes?
    Dense:       fast, good for semantic similarity
    Sparse:      good for exact keyword matching (when "Python 3.11" must match "Python 3.11")
    Multi-vec:   most accurate but slowest (fine-grained token-level matching)
    → BGE-M3 does ALL THREE with one model, then you combine the scores
```

Base: XLM-RoBERTa (a multilingual version of RoBERTa). 1024 dims, 8K max tokens.

**E5-Mistral-7B (Microsoft, 2024) — Using an LLM as an encoder**

```text
The big idea: Why train a BERT-sized encoder when LLMs already understand
language much better? Take Mistral-7B (a decoder/LLM), and fine-tune it
to produce embeddings instead of generating text.

How it works:
    1. Take Mistral-7B (a pre-trained decoder with 7 billion parameters)
    2. Feed text through it: "Instruct: Retrieve passages about...\nQuery: How to sort in Python"
    3. Take the LAST token's hidden state as the embedding (instead of mean pooling)
       (decoders process left-to-right, so the last token has seen everything)
    4. Fine-tune with contrastive loss on retrieval pairs

Why last token instead of mean pooling?
    Decoder models use causal attention (each token only sees tokens before it).
    So early tokens have limited context, but the last token has seen the entire input.
    Mean pooling would dilute good representations with poor early ones.
```

Result: Much better quality than BERT-sized models, but ~30× larger and needs a GPU.

**GTE-Qwen2 (Alibaba, 2024) — Pushing all limits**

```text
Training game — multi-stage:
    1. Start from Qwen2 (Alibaba's LLM, decoder architecture)
    2. Contrastive pre-training on massive web pairs
    3. Fine-tune with instruction-tuning (like E5's prefixes)
    4. Additional training with Matryoshka loss (explained below in OpenAI section)

Key advance: 131,072 token context window
    → Can embed an entire book chapter as one vector
    → Mostly eliminates the need for chunking
```

```text
| Model              | Base           | Dims | Max tokens | Key innovation                        |
| ------------------ | -------------- | ---- | ---------- | ------------------------------------- |
| GTE-Qwen2-7B       | Qwen2          | 3584 | 131,072    | Decoder-based, massive context        |
| BGE-M3             | XLM-RoBERTa    | 1024 | 8,192      | Multilingual, multi-granularity       |
| E5-Mistral-7B      | Mistral-7B     | 4096 | 32,768     | LLM as encoder, instruction-tuned     |
| GTE-large-en-v1.5  | BERT           | 1024 | 8,192      | Multi-stage contrastive training      |
```

**The trend:** using full LLMs (decoders) as the backbone for encoders. The architecture we learned in file 06 (transformer blocks, attention, FFN) is the same — the decoder-based models just have more parameters and were pre-trained on more data. They're larger and slower but significantly more capable.

---

#### OpenAI text-embedding-3 (2024) — Commercial API

Architecture is not public — we don't know the exact model. What we do know:

```text
| Variant                   | Dims          | Max tokens | Price (per 1M tokens) |
| ------------------------- | ------------- | ---------- | --------------------- |
| text-embedding-3-small    | 1536          | 8,191      | $0.02                 |
| text-embedding-3-large    | 3,072 (or less)| 8,191     | $0.13                 |
```

**Key feature — Matryoshka embeddings:** Named after Russian nesting dolls. The model is trained so that the first N dimensions carry the most important information, and each additional dimension adds less.

```text
How Matryoshka training works:

    Normal contrastive training:
        Compute loss using the full 3072-dim vector → backprop

    Matryoshka training:
        Compute loss using dims 0-255   → backprop
        Compute loss using dims 0-511   → backprop
        Compute loss using dims 0-1023  → backprop
        Compute loss using dims 0-3071  → backprop
        Total loss = sum of all four

    This forces the model to pack the most important information
    into the earliest dimensions. Dim 0-255 alone must be decent.
    Adding dims 256-511 should only improve things, and so on.

Result: You can truncate the vector at any point and still get useful embeddings.
    Full 3072-d: best quality
    Truncated to 256-d: ~95% quality, 12× less storage
```

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="How to sort a list in Python",
    model="text-embedding-3-large",
    dimensions=256  # use fewer dims to save storage
)
vector = response.data[0].embedding  # 256-d instead of 3072-d
```

**Trade-off:** Closed-source, API-only, your data leaves your infrastructure. But very convenient and good quality.

---

#### Cohere embed-v3 (2023) — Task-Specific Embeddings

Like E5's instruction prefixes, but built into the API as an `input_type` parameter rather than text prefixes you write yourself.

```text
| Input type     | What it optimises for                    |
| -------------- | ---------------------------------------- |
| search_query   | Finding relevant passages                |
| search_document| Being found by queries                   |
| classification | Separating categories                    |
| clustering     | Grouping similar items                   |
```

**How this works under the hood:** Same idea as E5's prefixes — the input_type tells the model what to optimise for, and it shifts its embedding space accordingly. The difference is that Cohere handles the prefix mapping internally, so you just pass a parameter.

**Training game (what we know):**

```text
Cohere hasn't published the full details, but from their papers:

1. Pre-trained on large multilingual corpus
2. Contrastive fine-tuning with task-aware training:
    - For each batch, the model knows which task type (search, classification, etc.)
    - Different loss functions for different tasks:
        search → retrieval-focused contrastive loss (query must find its passage)
        classification → separability loss (classes should form tight clusters)
        clustering → cohesion loss (similar items should be close)
    - All tasks trained jointly, so one model learns all

3. Compression-aware training (like Matryoshka):
    - Supports int8 and binary quantisation with minimal quality loss
    - Binary: 1024 dims × 1 bit = 128 bytes per vector (vs 4096 bytes at float32)
    - Useful for massive-scale search (billions of documents)
```

1024-dim vectors, supports 100+ languages. API-only, data leaves your infra.

---

#### Voyage AI (2024) — Domain-Specific Models

**The problem:** General-purpose encoders (trained on web text) don't understand domain jargon well. "Motion to dismiss" has a specific legal meaning. "EBITDA margin" has a specific finance meaning. General models treat these as just more words.

**Voyage's approach — domain-specific fine-tuning:**

```text
Training game:
    1. Start from a strong general encoder (architecture not public)
    2. Continue contrastive training on domain-specific data:

    voyage-code-3:
        Trained on (code question, code snippet) pairs
        (natural language query, relevant function)
        Understands "binary search implementation" matches actual binary search code

    voyage-law-2:
        Trained on (legal question, relevant statute/case) pairs
        Understands "respondeat superior" matches employer liability cases

    voyage-finance-2:
        Trained on (financial query, relevant analysis) pairs
        Understands "DCF valuation" relates to discounted cash flow methods
```

Often tops the MTEB leaderboard for specific domains while being worse than GTE/BGE on general benchmarks. The trade-off is specialisation vs generality.

---

### MTEB Leaderboard — How Models Are Compared

The **Massive Text Embedding Benchmark (MTEB)** ranks models across 8 task types: classification, clustering, pair classification, reranking, retrieval, STS (semantic text similarity), summarisation.

As of early 2025, the leaderboard roughly looks like:

```text
Tier 1 (best quality, large):
    GTE-Qwen2-7B, E5-Mistral-7B, Voyage-large-2
    ~4096 dims, ~7B params, need GPU

Tier 2 (great quality, practical):
    BGE-M3, GTE-large-en-v1.5, OpenAI text-embedding-3-large
    ~1024 dims, manageable size or API

Tier 3 (good quality, fast):
    all-MiniLM-L6-v2, BGE-small, GTE-small
    ~384 dims, runs on CPU, good for prototyping
```

---

### How to Choose

```text
"I'm prototyping / learning"
    → all-MiniLM-L6-v2 (free, fast, works on CPU, good enough for most tasks)

"I need best quality and don't mind GPU / cost"
    → GTE-Qwen2 or E5-Mistral (open-source)
    → OpenAI text-embedding-3-large (API, easy)

"I need multilingual support"
    → BGE-M3 (open-source, 100+ languages)
    → Cohere embed-v3 (API)

"I need domain-specific (code, legal, medical)"
    → Voyage AI (has specialised models)
    → Fine-tune GTE or BGE on your domain data

"I can't send data to an external API"
    → Any open-source model (MiniLM, GTE, BGE, E5)
    → Run locally with sentence-transformers or HuggingFace

"I need very long documents (>8K tokens)"
    → GTE-Qwen2 (128K context)
    → Chunk + encode with any model (works fine for RAG)
```

---

### Key Trends (as of 2025)

1. **LLMs as encoders**: Using decoder models (Mistral, Qwen) as the backbone for encoders. Bigger but much more capable.
2. **Matryoshka embeddings**: Train once, use at any dimension — save storage without retraining.
3. **Instruction-tuning**: Tell the model what task you're doing (search vs clustering) for better results.
4. **Longer context**: Moving from 512 to 8K to 128K tokens, reducing the need for chunking.
5. **Multi-modal**: Models that embed text + images + code in the same space (CLIP, next file).

---

### Summary Table

```text
| Model                    | Year | Open? | Dims  | Max tokens | Best for                    |
| ------------------------ | ---- | ----- | ----- | ---------- | --------------------------- |
| BERT                     | 2018 | Yes   | 768   | 512        | Historical reference        |
| RoBERTa                  | 2019 | Yes   | 768   | 512        | Base for fine-tuning         |
| Sentence-BERT            | 2019 | Yes   | 768   | 512        | The idea that made it work   |
| all-MiniLM-L6-v2         | 2020 | Yes   | 384   | 512        | Prototyping, CPU inference   |
| E5-large-v2              | 2022 | Yes   | 1024  | 512        | Instruction-tuned search     |
| BGE-M3                   | 2023 | Yes   | 1024  | 8,192      | Multilingual retrieval       |
| GTE-Qwen2-7B             | 2023 | Yes   | 3584  | 131,072    | Best open-source quality     |
| OpenAI embed-3-large     | 2024 | No    | 3072  | 8,191      | Easy API, matryoshka dims    |
| Cohere embed-v3          | 2023 | No    | 1024  | 512        | Task-specific, multilingual  |
| Voyage-large-2           | 2024 | No    | 1024  | 16,000     | Domain-specific excellence   |
```
