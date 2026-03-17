# Static Embeddings: Word2Vec, GloVe, FastText

## What Are Static Embeddings?

The previous file showed hand-crafted methods (one-hot, BoW, TF-IDF) where no learning happens — you just count. Static embeddings are the next step: **use a neural network (or matrix math) to learn a dense vector for each word**, where similar words end up close together.

"Static" means: each word gets ONE fixed vector, regardless of context.

```text
Hand-crafted:     "bank" → [0,0,0,1,0,0,...]    sparse, no meaning
                                                  50,000 dimensions, mostly zeros

Static embedding: "bank" → [0.21, -0.45, 0.78, ...]   dense, captures meaning
                                                        100-300 dimensions, all used

BUT: same vector for "river bank" and "bank account" — can't tell the difference
```

---

## What Changed vs One-Hot / TF-IDF

| Problem with hand-crafted | How static embeddings fix it |
|---|---|
| No similarity ("cat" and "dog" equally far) | Words that appear in similar contexts get similar vectors |
| Huge sparse vectors (50,000 dims) | Small dense vectors (100-300 dims) |
| No meaning captured | Vector arithmetic works: king - man + woman = queen |
| Can't generalise to unseen combos | Vectors compose: even unseen word pairs have meaningful distances |

What they DON'T fix (yet):
- Still **one vector per word** — no context sensitivity
- Still **no word order** — "dog bites man" and "man bites dog" have the same word vectors

---

## 1. Word2Vec (2013, Google)

The breakthrough. Two training strategies, same idea: **predict context from words, or words from context.**

### How do Neural Nets create embeddings? (High-level)

How a very popular vector embedding model like `word2vec` was trained?

**Goal:**
Teach a computer to turn every word into a list of numbers (a vector) so that words that appear in similar sentences,
like "cat" and "dog" — get similar vectors, while unrelated words — like "quantum" and "taco" — end up far apart.

**Neural Network Structure:**
- The model is just two tiny layers of weights.
- no convolutions
- no deep stacks
- The **input** weight matrix is the embedding table — each row becomes the vector for one word. This is not the output layer; the output layer is just a prediction head used during training and discarded afterwards.

```text
Input (one-hot word ID) → Hidden layer (300 neurons) → Output (predict neighbour)
                              ↑
                     THIS becomes the word vector
```

### Training Strategy 1: Skip-gram

Given a centre word, predict the surrounding words. The corpus itself is the teacher — no human labels needed.

Take a big text file and move a small window over it:
```bash
the  cat  sat  on  the  blue mat
     ^          window→ [cat sat on]
```
- For each centre word ("cat") record the words that sit nearby ("the", "sat", "on", "the").
- These pairs are your "good" examples. Positive example: ("cat" → "sat")
- Feed the centre word into the network and ask it to guess one of its real neighbours.
- To make the job meaningful, also give it a few negative guesses you know are wrong, e.g. ("cat" → "quantum").
- The network should score the true neighbour high and the random word low.
- While the network tries to win this guessing game, the numbers in that hidden layer keep adjusting.
- Row i is now the 100- or 300-dimensional vector for word i. Words that shared many contexts ended up with rows that point in similar directions — exactly what we wanted.

```text
Skip-gram input/output:

Sentence: "the cat sat on the mat"     window size = 2

Centre word: "sat"
The network is called ONCE PER PAIR:

    Input: "sat"  (one-hot vector) → Network → Output: probability for each word in vocab
    Target: "cat"   ← network should give "cat" a high probability

    Input: "sat"  (one-hot vector) → Network → Output: probability for each word in vocab
    Target: "on"    ← network should give "on" a high probability

    Input: "sat"  (one-hot vector) → Network → Output: probability for each word in vocab
    Target: "the"   ← network should give "the" a high probability

So for centre word "sat" with 3 neighbours, the network trains on 3 separate examples.
Each time: 1 word in → predict 1 word out.
```

### Training Strategy 2: CBOW (Continuous Bag of Words)

The reverse: given surrounding words, predict the centre word.

```text
CBOW input/output:

Sentence: "the cat sat on the mat"     window size = 2

Target word: "sat"
The network is called ONCE:

    Input: ["the", "cat", "on", "the"]  (4 one-hot vectors, averaged into 1)
          → Network →
    Output: probability for each word in vocab
    Target: "sat"   ← network should give "sat" a high probability

So CBOW takes MULTIPLE words in → predicts 1 word out.
This is why CBOW is faster: 1 training step per centre word, not 1 per neighbour pair.
```

CBOW is faster to train. Skip-gram works better for rare words (each rare word gets its own training examples as a centre word). In practice, skip-gram is more popular.

### End-to-End Toy Example (Skip-gram)

This traces through EVERYTHING — where the training data comes from, how the network produces output, how we tell it what's right, and how weights change.

```text
Setup:
    Vocabulary: [cat, dog, sat, the, mat]    (5 words, indexed 0-4)
    Vector size: 2 dimensions                (tiny, for illustration)
    Corpus: "the cat sat on the mat"
```

---

**STEP 0: Where does the training data come from?**

We slide a window over the text. The corpus tells us which words are neighbours — this is the ONLY supervision the model gets. No human labels "cat is similar to dog." The text itself is the teacher.

```text
Corpus: "the cat sat on the mat"
Window size = 1 (1 word to each side, to keep this small)

Slide the window:
    Centre="the" → neighbours: ["cat"]
    Centre="cat" → neighbours: ["the", "sat"]
    Centre="sat" → neighbours: ["cat", "on"]
    ... and so on

This gives us training pairs (skip-gram: centre → neighbour):
    ("the", "cat"),  ("cat", "the"),  ("cat", "sat"),  ("sat", "cat"), ...
```

Let's train on the pair: **input = "sat", target = "cat"**

The network's job: given "sat", output a high probability for "cat".

---

**STEP 1: Initialise weights randomly**

Two matrices, both filled with small random numbers at the start. The network knows nothing yet.

```text
W (input→hidden, 5 words x 2 dims) — this will become our embedding table:
    cat: [ 0.10,  0.30 ]     (row 0)
    dog: [-0.20,  0.15 ]     (row 1)
    sat: [ 0.40, -0.10 ]     (row 2)    ← we'll use this row
    the: [ 0.05,  0.25 ]     (row 3)
    mat: [-0.15,  0.35 ]     (row 4)

W' (hidden→output, 2 dims x 5 words) — this is the prediction head:
    W' = [ 0.20, -0.30,  0.10,  0.40, -0.10 ]
         [ 0.15,  0.25, -0.20,  0.05,  0.30 ]
         col:cat  col:dog col:sat col:the col:mat
```

These numbers are meaningless right now. Training will shape them.

---

**STEP 2: Forward pass — input**

Convert "sat" to one-hot, then look up its embedding.

```text
"sat" → one-hot: [0, 0, 1, 0, 0]
                   cat dog sat the mat

Multiply by W (or just pick row 2):
    [0, 0, 1, 0, 0] x W = [0.40, -0.10]

Hidden vector h = [0.40, -0.10]

(One-hot x matrix = just picks out that row. The "network" is a lookup table.)
```

---

**STEP 3: Forward pass — output scores**

Multiply the hidden vector by W' to get a raw score for every word in the vocabulary.

```text
h = [0.40, -0.10]

score_cat = h . W'_col_cat = (0.40)(0.20) + (-0.10)(0.15) = 0.08 - 0.015 = 0.065
score_dog = h . W'_col_dog = (0.40)(-0.30) + (-0.10)(0.25) = -0.12 - 0.025 = -0.145
score_sat = h . W'_col_sat = (0.40)(0.10) + (-0.10)(-0.20) = 0.04 + 0.02 = 0.060
score_the = h . W'_col_the = (0.40)(0.40) + (-0.10)(0.05) = 0.16 - 0.005 = 0.155
score_mat = h . W'_col_mat = (0.40)(-0.10) + (-0.10)(0.30) = -0.04 - 0.03 = -0.070

Raw scores: [0.065, -0.145, 0.060, 0.155, -0.070]
              cat     dog     sat     the     mat
```

These are just dot products — how "compatible" is sat's hidden vector with each output word's column.

---

**STEP 4: Forward pass — softmax (turn scores into probabilities)**

Softmax: take e^(score) for each, then divide by the sum so they add to 1.

```text
e^0.065  = 1.067
e^-0.145 = 0.865
e^0.060  = 1.062
e^0.155  = 1.168
e^-0.070 = 0.932

Sum = 1.067 + 0.865 + 1.062 + 1.168 + 0.932 = 5.094

Probabilities = each / sum:
    P(cat) = 1.067 / 5.094 = 0.209
    P(dog) = 0.865 / 5.094 = 0.170
    P(sat) = 1.062 / 5.094 = 0.209
    P(the) = 1.168 / 5.094 = 0.229
    P(mat) = 0.932 / 5.094 = 0.183

Output: [0.209, 0.170, 0.209, 0.229, 0.183]
          cat     dog     sat     the     mat
```

The network thinks "the" (23%) is most likely. But our target is "cat" (21%). Not terrible, but not confident either. All 5 words are around 20% — the network is basically guessing randomly. This is expected: we just started training.

---

**STEP 5: Compute the loss**

The target comes from our training pair: input = "sat", target = "cat". We told the network "cat" is the right answer because the TEXT said "cat" appears next to "sat." No human decided this — the corpus did.

```text
Target: "cat" (index 0)

The ideal output would be: [1.0, 0.0, 0.0, 0.0, 0.0]
                             cat   dog   sat   the   mat
    (100% sure it's "cat", 0% for everything else)

Actual output:             [0.209, 0.170, 0.209, 0.229, 0.183]

Cross-entropy loss = -log(P(cat)) = -log(0.209) = 1.566

This is the number we want to MINIMISE.
A perfect prediction (P(cat)=1.0) would give loss = -log(1.0) = 0.
```

---

**STEP 6: Backward pass — adjust weights**

Backprop computes: "how should each weight change to make P(cat) higher?"

```text
The error signal for each output word:
    error = predicted - target

    For "cat" (the target):  0.209 - 1.0 = -0.791   ← predicted too LOW, boost it
    For "dog":               0.170 - 0.0 = +0.170   ← predicted too HIGH, reduce it
    For "sat":               0.209 - 0.0 = +0.209   ← predicted too HIGH, reduce it
    For "the":               0.229 - 0.0 = +0.229   ← predicted too HIGH, reduce it
    For "mat":               0.183 - 0.0 = +0.183   ← predicted too HIGH, reduce it

Error vector e = [-0.791, 0.170, 0.209, 0.229, 0.183]
```

Now update both matrices. Learning rate = 0.5 (large, for visible changes).

**Update W' (output matrix):**

```text
Each column of W' gets nudged:
    W'_col_cat_new = W'_col_cat - lr * h * error_cat
                   = [0.20, 0.15] - 0.5 * [0.40, -0.10] * (-0.791)
                   = [0.20, 0.15] + [0.158, -0.040]
                   = [0.358, 0.110]    ← cat's column moves TOWARD sat's hidden vector

    W'_col_dog_new = [−0.30, 0.25] - 0.5 * [0.40, -0.10] * (0.170)
                   = [−0.30, 0.25] - [0.034, -0.009]
                   = [−0.334, 0.259]   ← dog's column moves AWAY from sat's hidden vector
```

The pattern: target word's column moves closer to the input's hidden vector. Non-target words' columns move away.

**Update W (embedding matrix) — only the row for "sat" changes:**

```text
Gradient for h = W' x error vector (simplified):
    dh = sum of (error_i * W'_col_i) for all i

    W_row_sat_new = W_row_sat - lr * dh
```

Sat's embedding vector gets nudged so that next time, when multiplied by the UPDATED W', it produces a higher score for "cat".

---

**STEP 7: What happens after thousands of training steps?**

```text
Step 1:    ("sat", "cat") → adjust weights to push P(cat) up
Step 2:    ("cat", "the") → adjust weights to push P(the) up
Step 3:    ("cat", "sat") → adjust weights to push P(sat) up
Step 4:    ("dog", "the") → adjust weights to push P(the) up
... millions more ...

After training:
    - "cat" and "dog" appeared near similar words ("the", "sat", "on")
    - So their rows in W got pulled in similar directions
    - Result: v_cat and v_dog point in similar directions = high cosine similarity

    - "cat" and "quantum" never appeared near similar words
    - Their rows got pulled in completely different directions
    - Result: v_cat and v_quantum point in very different directions
```

---

**STEP 8: After training is done**

```text
Throw away W' (the output matrix). It was only needed for training.

Keep W (the embedding matrix). Each row IS the word's vector:

W = [ 0.82, -0.31 ]   ← "cat"   ← these two are now similar
    [ 0.79, -0.25 ]   ← "dog"   ← because they had similar neighbours
    [-0.55,  0.71 ]   ← "sat"
    [ 0.12,  0.08 ]   ← "the"
    [-0.40,  0.65 ]   ← "mat"

cosine(cat, dog)  = 0.99   ← very similar (shared contexts)
cosine(cat, sat)  = -0.73  ← different (different roles in sentences)
```

This is your embedding table. To get any word's vector, just look up its row.

### How to Adapt the Toy Example for CBOW

The entire toy example above is skip-gram. For CBOW, **only Step 2 changes** — everything else (Steps 3-8) is identical.

```text
SKIP-GRAM Step 2 (what we did above):
    Training pair: input="sat", target="cat"
    h = W_row_sat = [0.40, -0.10]       ← look up 1 row

CBOW Step 2 (the only change):
    Training pair: input=["the","cat","on"], target="sat"
    h = average(W_row_the, W_row_cat, W_row_on)
      = average([0.05, 0.25], [0.10, 0.30], [some, values])
      = [averaged vector]                ← average multiple rows
```

Then from Step 3 onward: same dot products with W', same softmax, same loss, same backprop. Just the target word changes (now "sat" instead of "cat"), and the hidden vector `h` is an average instead of a single lookup.

The other difference is in Step 6 (backprop):

```text
Skip-gram backprop: update only 1 row of W (the input word "sat")
CBOW backprop:      update ALL context word rows ("the", "cat", "on")
                    (since all contributed to h, all share the blame)
```

**Summary of differences:**

```text
                    Skip-gram                    CBOW
Step 2: Input       1 word → lookup 1 row        N words → average N rows
Step 5: Target      1 neighbour ("cat")          1 centre word ("sat")
Step 6: W update    1 row updated                N rows updated
Training pairs      1 per neighbour              1 per centre word
Speed               Slower (more pairs)          Faster (fewer pairs)
Rare words          Better (each gets own pairs) Worse (diluted in average)
```

### Negative Sampling

Without a trick, the network would need to compute scores for all 50,000 words in the vocabulary at every step — too slow. Instead, for each real pair (sat → cat), sample 5-20 random words as "wrong" answers:

```text
Real pair:     (sat, cat)     → score should be HIGH
Fake pairs:    (sat, quantum) → score should be LOW
               (sat, guitar)  → score should be LOW
```

The model only needs to distinguish real neighbours from random words — much cheaper.

### Why do the models work for generating vectors across domains?

Because the learning signal is distributional ("You shall know a word by the company it keeps"),
the network captures surprisingly general syntactic and semantic regularities — gender pairs, country-capital analogies, verb tenses, etc. — that recur across corpora.
Even when you feed the trained model a sentence it never saw, each word still maps to the coordinate it learned from billions of earlier contexts,
and new combinations of those coordinates can be reasoned about through simple vector arithmetic.

Generalisation does break when you hit truly out-of-vocabulary tokens (medical jargon in a news-trained model),
which is why modern systems often fine-tune already trained models for specific domains.

---

## 2. GloVe (2014, Stanford)

**Global Vectors for Word Representation.** Same goal as Word2Vec (word → dense vector), completely different method.

### The Idea

Instead of a sliding window, GloVe uses a **global co-occurrence matrix**: count how often every pair of words appears near each other across the entire corpus.

```text
Co-occurrence matrix (simplified):

          cat    dog    sat    ran    mat    rug
cat        -      5      8      2      4      1
dog        5      -      3      7      1      3
sat        8      3      -      1      6      2
ran        2      7      1      -      0      1
mat        4      1      6      0      -      0
rug        1      3      2      1      0      -
```

"cat" and "sat" co-occur 8 times. "cat" and "ran" only 2 times.

### How Vectors Are Learned

GloVe then learns vectors such that:

```text
dot_product(v_cat, v_sat) ≈ log(co-occurrence count of cat and sat)
```

It's solving a matrix factorisation problem — decompose the big co-occurrence matrix into small dense vectors.

### Neural Network Structure

Unlike Word2Vec, GloVe is **not** a neural network. It:
1. Scans the entire corpus once to build the co-occurrence matrix
2. Uses weighted least squares to find vectors that best reconstruct the log-counts

No hidden layers, no backprop through a prediction task. Just optimisation of a matrix decomposition objective.

### Training Scale

| Pre-trained model | Corpus | Vectors | Dimensions |
|---|---|---|---|
| glove-wiki-gigaword | Wikipedia + Gigaword (6B tokens) | 400k words | 50, 100, 200, 300 |
| glove-twitter | 2B tweets (27B tokens) | 1.2M words | 25, 50, 100, 200 |
| glove-840B | Common Crawl (840B tokens) | 2.2M words | 300 |

### Word2Vec vs GloVe

| | Word2Vec | GloVe |
|---|---|---|
| Training signal | Local windows (predict neighbours) | Global co-occurrence counts |
| Training style | Neural network (gradient descent) | Matrix factorisation (weighted least squares) |
| Result quality | Very similar | Very similar |
| Speed | Faster on smaller corpora | Faster on very large corpora (counts computed once) |

In practice, the resulting vectors are almost interchangeable. Most people pick whichever pre-trained model is convenient.

---

## 3. FastText (2016, Facebook)

FastText solves a specific problem that Word2Vec and GloVe can't handle: **what do you do with a word you've never seen before?**

### The Problem

```text
Training vocabulary: ["happy", "unhappy", "happiness"]

New word at inference: "unhappiness"
Word2Vec/GloVe: "never seen this → UNK (unknown)" → garbage vector
```

### The Fix: Subword Embeddings

FastText breaks every word into character n-grams (pieces) and learns vectors for the pieces:

```text
"unhappiness" with n=3-6:

<un, unh, nha, hap, app, ppi, pin, ine, nes, ess, ss>
<unh, unha, nhap, happ, appi, ppin, pine, ines, ness, ess>
... and so on

The vector for "unhappiness" = average of all its piece vectors
```

Since "unhappy" and "unhappiness" share many pieces ("unh", "hap", "app", "ppi"), their vectors will be similar — even if "unhappiness" was never in the training data.

### Training

Under the hood, FastText is essentially Word2Vec's skip-gram — but instead of learning one vector per word, it learns vectors for each n-gram and sums them:

```text
Word2Vec:  v("unhappiness") = look up row in embedding table
                               (fails if word not in table)

FastText:  v("unhappiness") = v(<un) + v(unh) + v(nha) + ... + v(ss>) + v(whole_word)
                               (works even if "unhappiness" was never in training)
```

The training objective is the same: predict neighbouring words. The only change is how the input word's vector is constructed.

### When FastText Shines

```text
- Misspellings:     "happyness" → still shares pieces with "happiness"
- Morphology:       "running", "runner", "ran" → share "run" pieces
- Rare words:       Technical jargon composed of known parts
- Languages with rich morphology: Turkish, Finnish, German compound words
```

### Training Scale

| Pre-trained model | Corpus | Languages | Dimensions |
|---|---|---|---|
| fasttext-wiki-news | Wikipedia + news (16B tokens) | English | 300 |
| fasttext-crawl | Common Crawl (600B tokens) | English | 300 |
| fasttext-multi | Wikipedia | 157 languages | 300 |

### The Tradeoff

More memory — you're storing vectors for all n-gram pieces, not just whole words. A FastText model can be 2-3x larger than an equivalent Word2Vec model. But the ability to handle unseen words and typos is often worth it.

---

## Vector Arithmetic: The Magic Property

All three models exhibit this:

```text
king - man + woman ≈ queen

In vector space:
    v_king - v_man + v_woman = vector closest to v_queen

More examples:
    Paris - France + Italy ≈ Rome          (capital relationship)
    bigger - big + small ≈ smaller         (comparative form)
    walking - walk + swim ≈ swimming       (tense transformation)
```

This works because the training process encodes relationships as consistent directions in vector space. The "royalty" direction, the "capital city" direction, and the "gender" direction are all roughly linear.

---

## The Fundamental Limitation

All three produce **one vector per word, forever fixed.**

```text
"I deposited money at the bank"     →  bank = [0.2, -0.1, 0.8, ...]
"I sat on the river bank"           →  bank = [0.2, -0.1, 0.8, ...]
                                                SAME vector!
```

The model averages all possible meanings of "bank" into one vector. For most words this is fine, but for ambiguous words it's a real problem.

This is exactly what encoders (BERT, next files) solve — they produce a DIFFERENT vector for each word depending on the sentence it's in.

---
