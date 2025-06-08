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

where,
tf(i, d) — term frequency: how many times word i appears in document d.
df(i) — document frequency: in how many documents word i occurs at least once.
N — total number of documents in the corpus.
log(N / df(i)) — inverse-document frequency; rare words get a big boost, ubiquitous words get < 1
```

When it shines
- You have a large, static corpus and can pre-compute df counts once.
- You need a quick baseline for search ranking or clustering.
- Word order is unimportant, but keyword salience matters (news headlines, product tags).

#### 2 Smooth Inverse Frequency (SIF)
```math
w_i = a / (a + freq(i))       # a ≈ 0.001  

where,
freq(i) — the probability of word i in the entire corpus
freq(i) = total_occurrences(i) / total_words
a — small smoothing constant; keeps weights bounded even for extremely rare words.
```

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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = ["Deep learning models are data hungry",
          "The stock market crashed yesterday"]
vec = TfidfVectorizer(analyzer=str.split).fit(corpus)
idf = dict(zip(vec.get_feature_names_out(), vec.idf_))  # word ➜ IDF

def tfidf_pool(text, wv):
    tokens = [t for t in text.lower().split() if t in wv]
    weights = [idf.get(t, 0.0) for t in tokens]
    return np.average([wv[t] for t in tokens], axis=0, weights=weights)
```
