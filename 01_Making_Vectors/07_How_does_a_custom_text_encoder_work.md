## What is an encoder?

Think of the encoder as a factory line: raw input enters at one end, and a tidy fixed-length list of numbers (the vector) rolls out the other.

| Stage                                  | What goes in & out                                                                                           | Why it exists                                                         | Typical size notes                                  |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- | --------------------------------------------------- |
| **1. Splitter**<br>(tokeniser)         | Raw text → a row of IDs                                                                                      | Breaks words into “pieces” so every symbol is in the dictionary.      | Up to 512 tokens for many models.                   |
| **2. Word-lookup table**               | Each ID → a small list of numbers (its *piece vector*).                                                      | Gives the model something numeric to work with.                       | 512 tokens × 768 numbers each = **512 × 768 grid**. |
| **3. Position stamps**                 | Adds a unique tag to each row                                                                                | Tells the model which word came first, second, etc.                   | Same **512 × 768** grid (tags just add on).         |
| **4. *Stack of 6–12 “mixing blocks”*** | Reads the grid, lets every word peek at every other word, then does a quick math stretch/shrink on each row. | Mixes context (“cat” sees “sat on the mat”), then refines it.         | Grid stays **512 × 768** the whole way.             |
| **5. Pooling**                         | Shrinks the grid to one row                                                                                  | Averages all rows, or keeps the first row, so we get a single vector. | **1 × 768** (or 384 / 1024, depending on model).    |
| **6. Normalise (optional)**            | Scales the vector to unit length.                                                                            | Makes cosine similarity behave nicely.                                | Still **1 × 768**.                                  |



### What are the mixing block?

Below is a walk-through with toy-sized numbers so you can see what each Transformer “mixing block” does to a short piece of text. 

The sentence
```text
"The cat sat."
```

After tokenisation (we’ll keep it to 4 tokens including [CLS] start tag):
```text
[CLS]   the     cat     sat
```

Word-lookup vectors (we’ll use 3-dimensional toy embeddings):
```text
[CLS] : [0.1, 0.0, 0.1]
the   : [0.0, 0.2, 0.0]
cat   : [0.3, 0.1, 0.0]
sat   : [0.2, 0.0, 0.2]
```

Stack them into a 4 × 3 matrix called X
```text
X = [[0.1, 0.0, 0.1],   # row 0
     [0.0, 0.2, 0.0],   # row 1
     [0.3, 0.1, 0.0],   # row 2
     [0.2, 0.0, 0.2]]   # row 3

(In real models this would be 512 rows × 768 numbers.)
```

#### Step 1 Self-attention
Every token row is split into three smaller vectors—query, key, value—via simple matrix multiplies.
Then each token asks “how much should I care about every other token?” by taking dot products of its query against all keys, normalising with softmax, and building a weighted average of the value rows.

For this, 
Make queries, keys, values: We choose tiny weight matrices W_Q, W_K, W_V (3×3 each). 
```text
W_Q = [[1,0,0],
        [0,1,0],
        [0,0,1]]
```

```math
W_K = W_V = 0.5 * W_Q

Q = X · W_Q 
K = X · W_K
V = X · W_V
```

Similarity scores:
```math
scores = Q · Kᵀ                 
```
```text
scores = [[0.03, 0.01, 0.04, 0.03],
          [0.01, 0.04, 0.04, 0.02],
          [0.04, 0.04, 0.10, 0.07],
          [0.03, 0.02, 0.07, 0.08]]
```


Turn scores into weights (softmax row-wise): 
For row 2 (token “cat”) softmax turns [0.04, 0.04, 0.10, 0.07] into roughly [0.23, 0.23, 0.31, 0.23]. Do that for every row:
```math
weights = softmax(scores)

context = weights · V 
```
Now each token row is a weighted average of all the value rows. For instance token “cat” (row 2) now contains bits of [CLS], “the”, itself, and “sat” in proportions [0.23, 0.23, 0.31, 0.23].


#### Step 2 Feed-forward “rethink”

Run a tiny two-layer neural net on each row independently.
This lets the model invent new features (“subject ends here”, “past-tense verb”) from the mixed information.

Here, take a single row (say, [0.15, 0.03, 0.09]) and put it through a tiny 3-→-5-→-3 network.
```math
hidden = relu( row · W1 + b1 )  
out    = row + (hidden · W2 + b2)
```

#### Step 3 LayerNorm (keep numbers stable)

Add the original row back in and rescale so numbers stay well-behaved.
Every row is normalised:
```math
mean = average(row)  
std  = sqrt(average((row - mean)^2) + ε)  
row  = (row - mean) / std
```

Do Self-attention → Feed-forward → LayerNorm six to twelve times.
Early blocks mostly blend close neighbours; later ones capture whole-sentence context.

#### Final pooling (vector for the whole sentence)

Average all the rows
```math
v_sentence = mean( refined_X , axis=0 ) 
```






