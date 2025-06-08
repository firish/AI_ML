### How do Neural Nets create embeddingd? (High-level)

How a very popular vector embedding model like `word2vec` was trained?

1. Goal: 
Teach a computer to turn every word into a list of numbers (a vector) so that words that appear in similar sentences,
like “cat” and “dog”—get similar vectors, while unrelated words—like “quantum” and “taco”—end up far apart.

2. Neural Network Structure
- The model is just two tiny layers of weights (think of two Excel sheets full of numbers).
- no convolutions
- no deep stacks
- There is an embedding layer at the end which generates the embedding for the raw input.

3. Training data

How does training take place?
- Take a big text file and move a small window over it:
```bash
the  cat  sat  on  the  blue mat
     ^          window→ [cat sat on]
```
- For each centre word (“cat”) record the words that sit nearby (“the”, “sat”, “on”, “the”).
- These pairs are your “good” examples. Positive example: (“cat” → “sat”)
- Feed the centre word into the network and ask it to guess one of its real neighbours.
- To make the job meaningful, also give it a few negative guesses you know are wrong, e.g. (“cat” → “quantum”).
- The network should score the true neighbour high and the random word low.
- While the network tries to win this guessing game, the numbers in that hidden layer keep adjusting.
- row i is now the 100- or 300-dimensional vector for word i. Words that shared many contexts ended up with rows that point in similar directions—exactly what we wanted.

Note:
1. Vocabulary size = number of rows
- Exactly one row per unique token (word) that the model kept in its vocabulary.
- If you trained on a corpus whose cleaned-up word list ends up with 50 000 distinct words, the embedding matrix will have 50 000 rows. A large public word2vec model (e.g., trained on Google News) keeps about 3 million words, so its matrix has 3 million rows × 300 columns.

2. Vocab Management
During preprocessing you: 
- set a cutoff—usually drop infrequent words (e.g., anything that appears fewer than 5 times)
- stem words
so the table stays a manageable size.

3. Unseen Vocab
What about unseen words later?
Anything not in that original list maps to a special “UNK” (unknown) row, or you switch to a sub-word method (FastText) that can compose a vector from pieces.


### Training Scale
| Example corpus                                  | Words (tokens)                                 | Very rough sentence count\* | Resulting vocabulary†                                        |
| ----------------------------------------------- | ---------------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| Tiny tutorial data                              | 10 000 – 100 000                               | a few thousand              | a few k rows                                                 |
| Wikipedia dump                                  | \~2 billion                                    | \~100 million               | 2–4 million rows                                             |
| Google-News model (the classic pre-trained set) | **≈ 100 billion words**                        | hundreds of millions        | **≈ 3 million rows**                                         |

- Skip-gram treats each centre word/context word pair as one positive sample.
- window size is typicall small, k=5
- negative examples per center word is ~20 for small training pool, and ~5 for a large pool.



### Why do the models work for generating vectors across domains?

Because the learning signal is distributional (“You shall know a word by the company it keeps”), 
the network captures surprisingly general syntactic and semantic regularities—gender pairs, country-capital analogies, verb tenses, etc.—that recur across corpora. 
Even when you feed the trained model a sentence it never saw, each word still maps to the coordinate it learned from billions of earlier contexts, 
and new combinations of those coordinates can be reasoned about through simple vector arithmetic. 

Generalisation does break when you hit truly out-of-vocabulary tokens (medical jargon in a news-trained model), 
which is why modern systems often fine-tune already trained models for specific domains.

