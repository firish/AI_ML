# Before Embeddings: One-Hot, Bag of Words, TF-IDF

Before neural networks learned to create vectors, people represented text with hand-crafted methods. These are the simplest ways to turn words into numbers — no learning, no neural network, just counting and math.

They're worth knowing because:
1. They show WHY learned embeddings (Word2Vec, BERT) were such a big deal
2. TF-IDF is still used today inside BM25 (the keyword search in your hybrid search notes)
3. Every interview about NLP starts here

---

## 1. One-Hot Encoding

**Idea:** Give every unique word in your vocabulary its own position. Set that position to 1, everything else to 0.

```text
Vocabulary: [cat, dog, fish, bird, tree]   (5 words)

"cat"  → [1, 0, 0, 0, 0]
"dog"  → [0, 1, 0, 0, 0]
"fish" → [0, 0, 1, 0, 0]
"bird" → [0, 0, 0, 1, 0]
"tree" → [0, 0, 0, 0, 1]
```

**The problem:**

Every word is equally far from every other word. "cat" and "dog" (both animals) are just as different as "cat" and "tree" (unrelated).

```text
distance(cat, dog)  = distance(cat, tree)  = sqrt(2)
```

There's no concept of similarity. The vectors are also huge — if your vocabulary has 50,000 words, every vector is 50,000 dimensions with only a single 1. Extremely wasteful.

**Where it's still used:** As the INPUT to a neural network. Word2Vec starts by one-hot encoding each word, then multiplies it by a weight matrix to get a dense embedding. So one-hot is the "before" step, not the final representation.

---

## 2. Bag of Words (BoW)

**Idea:** Represent a document by counting how many times each word appears. Ignore the order completely.

```text
Vocabulary: [the, cat, sat, on, mat, dog, ran, park]

"the cat sat on the mat"  → [2, 1, 1, 1, 1, 0, 0, 0]
                              the=2, cat=1, sat=1, on=1, mat=1

"the dog ran in the park" → [2, 0, 0, 0, 0, 1, 1, 1]
                              the=2, dog=1, ran=1, park=1
```

**The problems:**

1. **No word order.** "Dog bites man" and "Man bites dog" produce the exact same vector — same words, same counts, completely different meaning.

2. **Common words dominate.** "the" appears twice in both sentences and gets the highest count, but it carries no meaning. Rare, meaningful words get drowned out.

3. **No similarity between words.** Just like one-hot — "cat" and "kitten" are as different as "cat" and "rocket." Each word is its own independent dimension.

**Where it's still used:** Sometimes as a quick baseline for document classification — surprisingly effective when you just need "is this document about sports or politics?" and don't care about nuance.

---

## 3. TF-IDF (Term Frequency - Inverse Document Frequency)

**Idea:** Like Bag of Words, but instead of raw counts, weight each word by how rare it is across all documents. Common words like "the" get low scores. Rare, distinctive words get high scores.

Two parts:

```text
TF (Term Frequency) = how often does this word appear in THIS document?
    TF("cat", doc) = count of "cat" in doc / total words in doc

IDF (Inverse Document Frequency) = how rare is this word across ALL documents?
    IDF("cat") = log(total documents / documents containing "cat")
```

**Walk-through:**

```text
3 documents:
  Doc 1: "the cat sat on the mat"
  Doc 2: "the dog sat on the rug"
  Doc 3: "the cat chased the dog"

For the word "cat":
  TF in Doc 1 = 1/6 = 0.167  (appears once out of 6 words)
  IDF = log(3/2) = 0.405     (appears in 2 of 3 docs)
  TF-IDF = 0.167 * 0.405 = 0.068

For the word "the":
  TF in Doc 1 = 2/6 = 0.333  (appears twice)
  IDF = log(3/3) = 0.000     (appears in ALL 3 docs → log(1) = 0)
  TF-IDF = 0.333 * 0.000 = 0.000  ← "the" gets zero weight!

For the word "mat":
  TF in Doc 1 = 1/6 = 0.167
  IDF = log(3/1) = 1.099     (appears in only 1 doc → very distinctive)
  TF-IDF = 0.167 * 1.099 = 0.183  ← highest score, most distinctive word
```

**What it fixes vs BoW:** Common words like "the" are automatically down-weighted. Distinctive words like "mat" get boosted. This is a significant improvement.

**What it still can't do:**

1. **Still no word meaning.** "happy" and "joyful" are treated as completely unrelated words
2. **Still no word order.** "dog bites man" = "man bites dog"
3. **Still sparse, high-dimensional.** Each unique word is its own dimension

---

## Why These Methods Were Replaced

| Method | Has similarity? | Understands order? | Understands meaning? | Vector size |
|---|---|---|---|---|
| One-hot | No | No | No | vocab size (50k+) |
| Bag of Words | No | No | No | vocab size (50k+) |
| TF-IDF | No | No | No (but weights important words) | vocab size (50k+) |
| **Word2Vec** | Yes | No | Some | 100-300 (dense) |
| **BERT** | Yes | Yes | Yes | 768 (dense) |

The breakthrough with Word2Vec (2013): instead of hand-crafting sparse vectors of size 50,000, train a neural network and let it learn dense vectors of size 300 where similar words are actually close together. That's the next file.

---

## Where They Survive Today

Despite being "old", these methods aren't dead:

- **One-hot:** Still the first step inside every neural network — the embedding layer takes a one-hot word ID and looks up its dense vector
- **TF-IDF → BM25:** The backbone of keyword search (Elasticsearch, Google's early search). In your hybrid search notes, the "sparse" side of dense+sparse fusion is essentially modernised TF-IDF
- **Bag of Words → SPLADE:** A learned sparse model that outputs a BoW-like vector but with learned weights. Better than TF-IDF, still sparse
