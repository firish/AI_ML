## What is an encoder?

Think of the encoder as a factory line: raw input enters at one end, and a tidy fixed-length list of numbers (the vector) rolls out the other.

| Stage                                  | What goes in & out                                                                                           | Why it exists                                                         | Typical size notes                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------- |
| **1. Splitter**<br>(tokeniser)         | Raw text → a row of IDs                                                                                      | Breaks words into "pieces" so every symbol is in the dictionary.      | Up to 512 tokens for many models.                   |
| **2. Word-lookup table**               | Each ID → a small list of numbers (its *piece vector*).                                                      | Gives the model something numeric to work with.                       | 512 tokens × 768 numbers each = **512 × 768 grid**. |
| **3. Position stamps**                 | Adds a unique position vector to each row (see note below)                                                   | Tells the model which word came first, second, etc.                   | Same **512 × 768** grid (vectors just add on).      |
| **4. *Stack of 6-12 "mixing blocks"*** | Reads the grid, lets every word peek at every other word, then does a quick math stretch/shrink on each row. | Mixes context ("cat" sees "sat on the mat"), then refines it.         | Grid stays **512 × 768** the whole way.             |
| **5. Pooling**                         | Shrinks the grid to one row                                                                                  | Averages all rows, or keeps the first row, so we get a single vector. | **1 × 768** (or 384 / 1024, depending on model).    |
| **6. Normalise (optional)**            | Scales the vector to unit length.                                                                            | Makes cosine similarity behave nicely.                                | Still **1 × 768**.                                  |



### What are the mixing blocks?

Below is a walk-through with toy-sized numbers so you can see what each Transformer "mixing block" does to a short piece of text.

The sentence
```text
"The cat sat."
```

After tokenisation (we'll keep it to 4 tokens including [CLS] start tag):
```text
[CLS]   the     cat     sat
```

**What is [CLS]?** A special token prepended to every input. It has no linguistic meaning -- it's a "blank slot" that the model learns to fill with a summary of the entire sequence during training. Many models (BERT) use the final [CLS] row as the sentence vector. Others use mean pooling over all rows instead.

Word-lookup vectors (we'll use 3-dimensional toy embeddings):
```text
[CLS] : [0.1, 0.0, 0.1]
the   : [0.0, 0.2, 0.0]
cat   : [0.3, 0.1, 0.0]
sat   : [0.2, 0.0, 0.2]
```

Stack them into a 4 x 3 matrix called X
```text
X = [[0.1, 0.0, 0.1],   # row 0
     [0.0, 0.2, 0.0],   # row 1
     [0.3, 0.1, 0.0],   # row 2
     [0.2, 0.0, 0.2]]   # row 3

(In real models this would be 512 rows x 768 numbers.)
```

**Positional encoding:** Self-attention has no built-in notion of word order -- it sees a set of rows, not a sequence. Positional encoding fixes this by adding a unique position vector to each row before attention. Two common approaches:
- **Sinusoidal** (original Transformer): uses sine/cosine waves at different frequencies. Position 0 gets one pattern, position 1 gets another. No learned parameters.
- **Learned** (BERT, GPT): a trainable lookup table of position vectors, one per position. The model learns what "being at position 5" should mean.

Either way: `X = word_vectors + position_vectors`. After this addition, "cat" at position 2 and "cat" at position 7 have different row values, so the model can distinguish them.

#### Step 1 Self-attention (single head)

Every token row is split into three smaller vectors -- query, key, value -- via simple matrix multiplies.
Then each token asks "how much should I care about every other token?" by taking dot products of its query against all keys, **dividing by sqrt(d_k)** to keep values stable, normalising with softmax, and building a weighted average of the value rows.

For this,
Make queries, keys, values: We choose tiny weight matrices W_Q, W_K, W_V (3x3 each).
```text
W_Q = [[1,0,0],
        [0,1,0],
        [0,0,1]]
```

```math
W_K = W_V = 0.5 * W_Q

Q = X * W_Q
K = X * W_K
V = X * W_V
```

Similarity scores (with scaling):
```math
scores = (Q * K^T) / sqrt(d_k)
```
where d_k = dimension of key vectors (here d_k = 3, so sqrt(3) = 1.73).

**Why divide by sqrt(d_k)?** Without scaling, dot products grow with dimension size, pushing softmax into regions where gradients are tiny. Dividing by sqrt(d_k) keeps the variance of scores roughly 1, so softmax produces useful (non-extreme) attention weights. This is critical for training stability.

```text
scores (before scaling) =
    [[0.03, 0.01, 0.04, 0.03],
     [0.01, 0.04, 0.04, 0.02],
     [0.04, 0.04, 0.10, 0.07],
     [0.03, 0.02, 0.07, 0.08]]
```


Turn scores into weights (softmax row-wise):
For row 2 (token "cat") softmax turns the scaled scores into roughly [0.23, 0.23, 0.31, 0.23]. Do that for every row:
```math
weights = softmax(scores)

context = weights * V
```
Now each token row is a weighted average of all the value rows. For instance token "cat" (row 2) now contains bits of [CLS], "the", itself, and "sat" in proportions [0.23, 0.23, 0.31, 0.23].

#### Step 1B Multi-head attention (what real transformers do)

The example above shows **single-head** attention. Real transformers use **multi-head attention** -- they run several attention heads in parallel, each with its own W_Q, W_K, W_V matrices.

**How it works:**
1. Split the 768-dim space into H heads (e.g., H=12 heads of 64 dims each)
2. Each head has its own W_Q, W_K, W_V of size 768 x 64
3. Each head attends independently -- one head might focus on syntax, another on coreference, another on nearby words
4. Concatenate all head outputs: 12 heads x 64 dims = 768 dims
5. Multiply by a final W_O matrix (768 x 768) to mix head outputs

```text
head_1 = Attention(Q_1, K_1, V_1)    # 64-dim output
head_2 = Attention(Q_2, K_2, V_2)    # 64-dim output
...
head_12 = Attention(Q_12, K_12, V_12) # 64-dim output

MultiHead = Concat(head_1, ..., head_12) * W_O
```

**Why multiple heads?** A single head can only learn one attention pattern per layer. Multiple heads let the model attend to different types of relationships simultaneously -- one head for subject-verb, another for adjective-noun, another for long-range dependencies, etc.


#### Step 2 Residual connection + Feed-forward "rethink"

**Residual (skip) connection:** Before the feed-forward step, add the attention output back to the original input:
```math
row = row_original + attention_output
```
This "skip connection" lets gradients flow directly through the network during training, preventing the vanishing gradient problem in deep stacks. Without it, 12-layer transformers would be very hard to train.

Then run a tiny two-layer neural net on each row independently.
This lets the model invent new features ("subject ends here", "past-tense verb") from the mixed information.

Here, take a single row (say, [0.15, 0.03, 0.09]) and put it through a tiny 3->5->3 network.
```math
hidden = relu( row * W1 + b1 )
out    = row + (hidden * W2 + b2)    # another residual connection
```

#### Step 3 LayerNorm (keep numbers stable)

Rescale so numbers stay well-behaved.
Every row is normalised:
```math
mean = average(row)
std  = sqrt(average((row - mean)^2) + e)
row  = (row - mean) / std
```

The full sequence per block is:
```text
input -> Self-Attention -> Add & LayerNorm -> Feed-Forward -> Add & LayerNorm -> output
                ^                                    ^
                |                                    |
            residual skip                       residual skip
```

Do this six to twelve times.
Early blocks mostly blend close neighbours; later ones capture whole-sentence context.

#### Final pooling (vector for the whole sentence)

Average all the rows
```math
v_sentence = mean( refined_X , axis=0 )
```
