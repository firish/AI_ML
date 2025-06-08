## How do we make Vectors for sentences, paragraphs, documents, vectors?

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
- TF-IDF can explode for ultra-rare tokens.
- TF-IDF doesnt do too well when document frequency varies widely.
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


```python
"""
Demo: salience vs. order vs. negation with *real* Word2Vec vectors
------------------------------------------------------------------
⏱  Run time depends on the first-time download of the model (~2 GB).
"""

import numpy as np
from numpy.linalg import norm
from collections import Counter
import gensim.downloader as api

# ─── 1. Load the pre-trained word2vec google-news vectors ───────────────────────
print("Loading Word2Vec Google-News… (one-time download)")
WV = api.load("word2vec-google-news-300")   # keyed-vectors object

# ─── 2. Build an IDF dictionary from a *tiny* background corpus ─────────────────
corpus = [
    "Deep learning models are amazing",
    "Models are amazing",
    "Dog bites man",
    "Man bites dog",
    "He hates traffic",
    "He does not hate traffic"
]
doc_freq = Counter()
for doc in corpus:
    seen = set()
    for tok in doc.lower().split():
        if tok in WV and tok not in seen:
            doc_freq[tok] += 1
            seen.add(tok)

N = len(corpus)
idf = {w: np.log((1 + N) / (1 + df)) + 1 for w, df in doc_freq.items()}

# default IDF for unseen words
def get_idf(w): return idf.get(w, 1.0)

# ─── 3. Embedding helpers (mean vs. tf-idf weighted) ────────────────────────────
def vec(tok):            # gracefully handle OOV
    return WV[tok] if tok in WV else np.zeros(WV.vector_size)

def embed(text, weighted=False):
    toks = [t.lower() for t in text.split() if t.lower() in WV]
    if not toks:
        return np.zeros(WV.vector_size)
    mats = np.stack([vec(t) for t in toks])
    if not weighted:
        return mats.mean(axis=0)
    wts = np.array([get_idf(t) for t in toks])
    return np.average(mats, axis=0, weights=wts)

def cos(a, b):
    d = norm(a) * norm(b)
    return 0.0 if d == 0 else (a @ b) / d

# ─── 4. Compare the same three sentence pairs ───────────────────────────────────
pairs = [
    ("Salience",
     "Deep learning models are amazing",
     "Models are amazing"),

    ("Order (position)",
     "Dog bites man",
     "Man bites dog"),

    ("Negation",
     "He hates traffic",
     "He does not hate traffic")
]

print("\nSimilarity scores (cosine):")
for title, s1, s2 in pairs:
    m  = cos(embed(s1),          embed(s2))
    tf = cos(embed(s1, True),    embed(s2, True))
    print(f"{title:10} — mean = {m:5.3f}   tf-idf = {tf:5.3f}")
```

| Pair         | Mean-pool sim             | TF-IDF sim                   | Interpretation                                                                                                                                                                                                  |
| ------------ | ------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Salience** | moderate-high (≈ 0.6–0.8) | noticeably lower (≈ 0.2–0.4) | TF-IDF down-weights the repeated high-frequency words, so the first sentence (with rare, descriptive “deep learning”) drifts away, while the plain mean is still pulled toward the common “models are amazing”. |
| **Order**    | very high (≈ 0.9+)        | very high (≈ 0.9+)           | Bag-of-words treatments see exactly the same token multiset, so both pooling schemes conclude the sentences are almost identical even though the semantics flip.                                                |
| **Negation** | high (≈ 0.8–1.0)          | high (≈ 0.8–1.0)             | The only difference is the low-content word “not”, which gets low IDF; both embeddings remain virtually unchanged, so the contradiction goes undetected.                                                        |

