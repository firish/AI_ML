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

**What it is:** A BERT model compressed via **knowledge distillation** — a large trained model (teacher) trains a smaller model (student) to mimic its attention patterns.

```text
| Model                      | Layers | Dims | Parameters | Speed vs BERT |
| -------------------------- | ------ | ---- | ---------- | ------------- |
| BERT-base                  | 12     | 768  | 110M       | 1×            |
| all-MiniLM-L6-v2           | 6      | 384  | 22M        | 5× faster     |
| all-MiniLM-L12-v2          | 12     | 384  | 33M        | 2× faster     |
```

**Why it matters:** `all-MiniLM-L6-v2` is the default recommendation for most projects. It's fast enough to run on a CPU, small enough to fit on edge devices, and retains ~90-95% of the quality of full BERT.

---

#### E5 (2022) — Instruction-Tuned Embeddings

**Key innovation:** Prefix your text with a task instruction so the same model produces different embeddings for different use cases.

```text
Search:     embed("query: How to sort in Python")
Storage:    embed("passage: Python's sorted() function returns a new sorted list...")
Clustering: embed("cluster: machine learning algorithm comparison")
```

The model adjusts its embedding space based on the prefix. A query embedding emphasises intent; a passage embedding emphasises content.

---

#### GTE / BGE / E5-Mistral (2023-2024) — Current Open-Source Leaders

These models push the state of the art by combining multiple improvements:

```text
| Model              | Base           | Dims | Max tokens | Key innovation                        |
| ------------------ | -------------- | ---- | ---------- | ------------------------------------- |
| GTE-Qwen2-7B       | Qwen2          | 3584 | 131,072    | Decoder-based, massive context        |
| BGE-M3             | XLM-RoBERTa    | 1024 | 8,192      | Multilingual, multi-granularity       |
| E5-Mistral-7B      | Mistral-7B     | 4096 | 32,768     | LLM as encoder, instruction-tuned     |
| GTE-large-en-v1.5  | BERT           | 1024 | 8,192      | Multi-stage contrastive training      |
```

**The trend:** using full LLMs (decoders) as the backbone for encoders. These are larger and slower but significantly more capable.

---

#### OpenAI text-embedding-3 (2024) — Commercial API

```text
| Variant                   | Dims          | Max tokens | Price (per 1M tokens) |
| ------------------------- | ------------- | ---------- | --------------------- |
| text-embedding-3-small    | 1536          | 8,191      | $0.02                 |
| text-embedding-3-large    | 3,072 (or less)| 8,191     | $0.13                 |
```

**Unique feature — Matryoshka embeddings:** You can request fewer dimensions (e.g., 256 instead of 3072) and still get useful vectors. The model is trained so that the first N dimensions carry the most information.

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

```text
| Input type     | What it optimises for                    |
| -------------- | ---------------------------------------- |
| search_query   | Finding relevant passages                |
| search_document| Being found by queries                   |
| classification | Separating categories                    |
| clustering     | Grouping similar items                   |
```

Like E5's instruction prefixes, but built into the API as an `input_type` parameter. 1024-dim vectors, supports 100+ languages.

---

#### Voyage AI (2024) — Domain-Specific Models

Offers specialised models for code, legal, finance, and multilingual text. Often tops the MTEB leaderboard for specific domains.

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
