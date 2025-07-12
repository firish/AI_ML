Continuation to part 5.

### 4. Sentence/-Paragraph Encoders

This referes to the modern transformer-based encoders that power most semantic-search & RAG stacks (Sentence-BERT, RoBERTa, MiniLM, OpenAI /embeddings, Cohere embed, etc.).

1. The structure of the encoders:
```bash
| Piece                   | What it holds                                                                                                                                 | Why it matters                                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Tokenizer               | Splits text into sub-word tokens and maps each to an integer ID.                                                                              | Lets one vocabulary handle *unseen* words by composing them from pieces (`▁quant`, `um`). |
| Transformer encoder     | A stack of self-attention layers (BERT, RoBERTa, MiniLM…).                                                                                    | Learns contextualised token embeddings that “know” their neighbours.                      |
| Pooling head            | Converts the sequence of token vectors into **one** fixed-length sentence vector.  Choices: `[CLS]`, mean-pool, or a small attention layer.  | This is the vector you’ll store.  Size is fixed: 384 / 768 / 1536 d.                      |

```

2. The learning game (contrastive fine-tuning)

- Positive pair: question -> duplicate question, ,image alt-text, back-translated sentence, etc.
- Negative: a random sentence from the batch.
- Encode each text → v₁, v₂, v₃ (dimension d).

example:
```bash
| Anchor (v₁)                            | Positive (v₂)               | Negative (v₃)                         | 
| -------------------------------------- | --------------------------- | ------------------------------------- |
| “How do I sort a list in Python?”      | “Python list sort methods?” | “What is the GDP of Canada?”          |
| “Symptoms of iron-deficiency anaemia?” | “Signs you’re low on iron”  | “Best way to waterproof hiking boots” |
```


- Compute cosine similarities: The loss pushes sim(v₁, v₂) up (they should be near) and sim(v₁, v₃) down.
- Back-prop through the whole stack: Gradients flow into every transformer layer and the pooling head.
- After a few epochs on millions of such pairs the model learns a space where “semantic closeness = vector closeness.”
- The transformer + pooling head is frozen and published as a single callable function: ``` embed("Superconductors expel magnetic fields.")  →  768-d numpy array ```
- No separate word or doc table is needed; everything is implicit in the weights.

Usage:
```bash
| Scenario                                                | Steps                                                                                                                                                                                         |
| ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Short text (≤ context window, typically 512 tokens)     | Tokenize → encode → store or compare the vector.                                                                                                                                              |
| Long text (book, PDF, chat log)                         | ① Split into overlapping chunks (e.g., 256 tokens). ② Encode each chunk. ③ Either: store chunk vectors directly (most common for RAG), or mean-/attention-pool them into one “book vector.” |
| Real-time query                                         | Encode the query sentence → cosine-search in vector DB → retrieve top-k chunks → optional re-ranking or pass to LLM.                                                                          |
```

### Demo
```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-d

docs = [
    "Quantum entanglement defies classical intuition.",
    "Superconductors expel magnetic fields below a critical temperature.",
    "The taco truck arrives every Tuesday.",
]

emb = model.encode(docs, normalize_embeddings=True)   # shape = (3, 384)

sim_qc = util.cos_sim(emb[0], emb[1]).item()
sim_qt = util.cos_sim(emb[0], emb[2]).item()
print(f"physics↔physics   cosine ≈ {sim_qc:.3f}")
print(f"physics↔tacos     cosine ≈ {sim_qt:.3f}")
```
Expected output: first similarity high (~0.6–0.8), second near zero or negative—showing topic separation.
