## How do we make Vectors for sentences, paragraphs, and documents?

In the previous parts, we saw how word embedding models are trained and used for generating vectors for words.
There are 4 major techniques of expanding these models to work for sentences, documents, books, etc. 

Think of a pre-trained word-embedding model as a dictionary → number lookup.
To turn many lookup results into one representation you have four common options, each trading speed, fidelity and code complexity.

### 1. Simple (un-weighted) pooling

Take every word in a span, look up its pre-trained embedding, then apply an element-wise mean, sum, or max across the resulting matrix:

```math
v_doc = (1 / n) * Σ_{i=1…n} v_word_i  
```

where is it used?
- Speed & memory: one pass through the text, one cheap NumPy operation.
- No extra training: works with any off-the-shelf word2vec/FastText table.
- Good enough for coarse tasks: quick clustering, duplicate-document detection, measuring topical drift over time.

What are its problems?
- Order lost: “dog bites man” ≈ “man bites dog.”
- Stop-word dilution: high-frequency words (“the”, “and”) pull vectors toward the corpus mean, blurring distinctions.
- Negation & syntax ignored: “not good” ≈ “good.”


### 2. Weighted pooling

**Same as simple pooling, but multiply each word vector by a weight that reflects its importance before pooling.**

```math
v_doc = (Σ_i w_i * v_word_i) / (Σ_i w_i)  
```

For the pool weighting, we mostly use

#### 1. TF-IDF
```math
w_i = tf(i, d) * log(N / df(i))
```
where,
- tf(i, d) — term frequency: how many times word i appears in document d.
- df(i) — document frequency: in how many documents word i occurs at least once.
- N — total number of documents in the corpus.
- log(N / df(i)) — inverse-document frequency; rare words get a big boost, ubiquitous words get < 1

When it shines
- You have a large, static corpus and can pre-compute df counts once.
- You need a quick baseline for search ranking or clustering.
- Word order is unimportant, but keyword salience matters (news headlines, product tags).

#### 2 Smooth Inverse Frequency (SIF)
```math
w_i = a / (a + freq(i)); where a ≈ 0.001  
```

where,
- freq(i) — the probability of word i in the entire corpus
- freq(i) = total_occurrences(i) / total_words
- a — small smoothing constant; keeps weights bounded even for extremely rare words.

Why it exists?
- TF-IDF weights can be unstable for very rare tokens in small corpora.
- TF-IDF doesn't do too well when document frequency varies widely.
- SIF scales each word by a smooth curve that monotonically down-weights common words without wild jumps.

When to use?
- You want a one-pass, streaming computation (need only running word frequencies, not per-doc df).
- Text length varies widely (tweets ↔ manuals) and you want weights that stay stable.

### Advantages
- You still need fast inference but plain averaging is too blunt; Captures keyword salience (“quantum” matters more than “the”).
- Retains interpretability: you can inspect which words carried the weight.
- Document length varies wildly; weighting tames the bias toward long texts.
- You’re building a baseline for semantic search before investing in transformers.

### Limitations
- Word order and compositional meaning are still missing.
- Requires pre-computing IDF statistics on your corpus.
- Negation/sarcasm problems still persist.

---

### 3. Doc2Vec — Learning a Vector per Document

Pooling (methods 1 and 2 above) just averages word vectors — the document doesn't get its own learned representation. Doc2Vec fixes this: it gives each document its own trainable vector that learns to capture the document's meaning as a whole.

**The key idea:** Add one extra row to the embedding table for each document. During training, that row participates in the word prediction game alongside the word vectors, so it learns to encode whatever the word vectors alone can't capture (topic, style, order within the context window).

#### Two lookup tables

```text
Word table (same as word2vec):
    cat:  [0.21, -0.45, 0.78]
    dog:  [0.33,  0.12, -0.55]
    sat:  [-0.10, 0.67, 0.44]
    ...

Document table (new, one row per document):
    doc_1: [0.05, 0.32, -0.11]    ← random at start, learned during training
    doc_2: [-0.22, 0.14, 0.58]
    doc_3: [0.41, -0.03, 0.27]
```

Both tables are the same dimensionality (e.g., 300). Both get updated by backprop.

#### Mode 1: PV-DM (Distributed Memory) — "CBOW with a document ID"

Feed the document vector AND context words into the network, predict the missing word.

```text
Document: doc_1 = "the cat sat on the mat"
Window size = 2

Training step:
    Input:  doc_1 vector + vectors for ["the", "cat", ___, "the"]
    Target: "sat"

    h = average(v_doc1, v_the, v_cat, v_the)
      = average([0.05, 0.32, -0.11], [0.12, 0.08, 0.11], [0.21, -0.45, 0.78], [0.12, 0.08, 0.11])
      = [0.125, 0.008, 0.223]

    → Network → softmax → P("sat") should be high

    Backprop updates: v_doc1, v_the, v_cat (all contributed to h)
```

This is basically CBOW from word2vec, but with the document vector thrown into the average. The document vector acts like a persistent "memory" of the document's topic — it helps predict words that the local context alone can't explain.

**Word order:** PV-DM has a limited sense of word order within its window (because the context words are position-aware), but it's not strong.

#### Mode 2: PV-DBOW (Distributed Bag of Words) — "Skip-gram with a document ID"

Feed ONLY the document vector, predict random words from the document. No context words at all.

```text
Document: doc_1 = "the cat sat on the mat"

Training step:
    Input:  doc_1 vector only
    Target: "cat" (a random word from doc_1)

    h = v_doc1 = [0.05, 0.32, -0.11]

    → Network → softmax → P("cat") should be high

    Backprop updates: only v_doc1
    (word vectors are NOT updated in pure PV-DBOW)
```

This is simpler and faster. The document vector alone must be enough to predict which words appear in it — so it must encode the document's entire topic.

**Word order:** None. PV-DBOW is truly bag-of-words — it doesn't see context, just "which words are in this document?"

**Counterintuitive fact:** PV-DBOW often outperforms PV-DM in practice, despite ignoring word order. The likely reason: it puts all the learning pressure on the document vector (since there are no context words to share the load), forcing it to become a better summary.

#### After training: using the model

```text
For documents seen during training:
    → just look up the row in the document table. Instant, no compute.

For a NEW document:
    1. Keep the word table frozen (already trained)
    2. Add a new row, initialise randomly
    3. Run 5-20 gradient steps so this row learns to predict the document's words
    4. That row is the new document's vector
    (fast — typically milliseconds for short documents)
```

#### When to use doc2vec vs pooling

| | Pooling (mean/TF-IDF) | Doc2Vec |
|---|---|---|
| Training needed? | No — just average pre-trained word vectors | Yes — train on your corpus |
| Captures document-level patterns? | No — just word averages | Yes — the doc vector learns topic/style |
| Handles word order? | No | PV-DM: slightly. PV-DBOW: no |
| Speed at inference | Instant | Instant (lookup) for seen docs, fast for new docs |
| Best for | Quick baselines, small data | Large fixed collections (news, legal, support tickets) |

#### Limitations

- If your documents change constantly, retraining is expensive
- For top-tier semantic accuracy, transformer-based sentence encoders (next files) are usually better
- Needs a decent corpus size (~10x more documents than vector dimensions)

#### Code demo (gensim)

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

raw_corpus = [
    ("doc_a", "deep learning models are data hungry"),
    ("doc_b", "the stock market crashed yesterday"),
    ("doc_c", "deep learning is revolutionising finance"),
]
tagged = [TaggedDocument(word_tokenize(text), [tag]) for tag, text in raw_corpus]

model = Doc2Vec(vector_size=100, window=5, min_count=1,
                workers=2, epochs=40, dm=1)   # dm=1 → PV-DM, dm=0 → PV-DBOW
model.build_vocab(tagged)
model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

print("Similarity doc_a ↔ doc_c:", model.dv.similarity("doc_a", "doc_c"))
# doc_a (deep learning) and doc_c (deep learning + finance) should be more similar
# than doc_a and doc_b (stock market)
```