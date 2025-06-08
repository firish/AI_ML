Continuation of part 4.

### 3. doc2vec Method

How does it work?

1. Two lookup tables instead of one
- Word table – exactly the same 300-dimensional word vectors you already know from word2vec.
- Document table – one extra 300-dimensional row for each document in your training set. Think of it as a blank sticky-note you attach to every article, chat log, book, etc.
```bash
word-table  ← gradient step  (fine-tunes   general meanings to your domain)
doc-table   ← gradient step  (learns each document’s fingerprint)
```

Rules of thumb:
- Target at least 10× more documents than embedding dimensions (so ≥ 30 000 for 300 d) to avoid over-fitting.
- Make sure each document has a healthy length—dozens of sentences; extremely short items (< 5 tokens) are better merged or discarded.


2. The learning game
- At the start every row in both tables is just random noise.
- We still start with a trained word2Vec for the word embeddings as they have some semantics and grammar meaning encoded and help in faster convergence during retraining.

- You slide a small window across every document’s text, just like word2vec.
- For each position you play one of two flavours of “guess-the-word”:
| Flavour                                | What you feed into the tiny neural net                              | What the net must guess                                          |
| -------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **PV-DM** (Distributed Memory)         | *Doc-vector* **plus** the few context words around a missing target | The target word                                                  |
| **PV-DBOW** (Distributed Bag-of-Words) | *Doc-vector only*                                                   | Several random words that appear somewhere in that same document |

- During back-prop the model nudges both the word rows and that document’s row so the guess becomes a little less wrong.

3. What happens after many passes
- Because every “cat-sat-on-the” window inside Document 42 keeps pushing the Document 42 row, that row drifts to a position in space that helps predict its words better than any other document’s row.
- By the end: Rows of documents about the same subject land near each other.
- You save the two tables; the Document table is now your embedding index (one vector per doc).

4. Using the model in practice
- For documents the model saw during training, just look up row i in the saved table—instant, no compute.
- For a brand-new document:
  - Keep the word table frozen (it is already good).
  - Add a new row, initialise it randomly.
  - Run 5–20 very quick gradient steps so that this fresh row alone can predict the document’s own words.
  - Stop; that updated row is the new document’s vector.
  - Even on CPU this takes milliseconds, so you can process thousands per second in batch jobs.

5. Why bother instead of averaging word vectors?
- The model learns that word order and co-occurrence patterns matter (e.g., “stock market crash” differs from “market stock crash”).
- The doc vector acts like a topic fingerprint that cannot be reconstructed by simply averaging its words.
- Once trained, lookup is trivial—just read a row—so it is lighter than running a transformer encoder for every query.

6. Minimal code demo (gensim)
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
                workers=2, epochs=40, dm=1)   # dm=1 → PV-DM
model.build_vocab(tagged)
model.train(tagged, total_examples=model.corpus_count, epochs=model.epochs)

print("Vector for doc_a:", model.dv["doc_a"][:6])          # first 6 dims
print("Similarity doc_a ↔ doc_c:", model.dv.similarity("doc_a", "doc_c"))
```
After training you can call model.dv.most_similar("doc_c") to find articles with the closest doc vectors.

7. When to use doc2vec
- Closed, finite collection (news archives, legal cases, support tickets).
- Latency budget in milliseconds and no GPU on the serving path.
- Domain language so specialised that off-the-shelf sentence-BERT models perform poorly.

8. Limitations
- If your documents change constantly,
- You need top-tier semantic accuracy,
transformer-based sentence encoders are usually the better trade-off.
