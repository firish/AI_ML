## How do we make Vectors for sentences, paragraphs, documents, vectors?

In the previous parts, we saw how word embedding models are trained and used for generating vectors for words.
There are 4 major techniques of expanding these models to work for sentences, documents, books, etc. 

Think of a pre-trained word-embedding model as a dictionary → number lookup.
To turn many lookup results into one representation you have four common options, each trading speed, fidelity and code complexity.

### 1. Simple (un-weighted) pooling

Take every word in a span, look up its pre-trained embedding, then apply an element-wise mean, sum, or max across the resulting matrix:

```ini
v_doc = (1 / n) * Σ_{i=1…n} v_word_i  
```
​
 
| Why people still use it |

Speed & memory: one pass through the text, one cheap NumPy operation.

No extra training: works with any off-the-shelf word2vec/FastText table.

Good enough for coarse tasks: quick clustering, duplicate-document detection, measuring topical drift over time.

| Blind spots |

Order lost: “dog bites man” ≈ “man bites dog.”

Stop-word dilution: high-frequency words (“the”, “and”) pull vectors toward the corpus mean, blurring distinctions.

Negation & syntax ignored: “not good” ≈ “good.”

| Tiny code snippet |

python
Copy
Edit
def mean_pool(text, wv):
    tokens = [t for t in text.lower().split() if t in wv]
    return np.mean([wv[t] for t in tokens], axis=0)   # shape (300,)
Same dimensionality as the original word vectors (e.g., 300 d).

